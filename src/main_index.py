import argparse
import gc
import json
import os
import queue
import threading
import time
import traceback


import cv2
import numpy as np
import torch

import index_elastic as ind
import embeddings as emb
import keyframes as ky
from logger import setup_logger

logger = setup_logger()

# Configuração
INDEX_WORKERS    = 2
EMBEDDINGS_DIR   = "./data/embeddings"
VIDEO_DIR        = "./data/videos"

# Modo de processamento:
#   False = streaming (1 segmento por vez)
#   True  = batch    (todos frames na RAM, CLIP em 1 chamada GPU)
CONFIG = {"batch_mode": False, "index_workers": INDEX_WORKERS}

# embed_queue: (video_id, video_path, metadata_dict) - contem frames, maxsize controlado
# index_queue: (video_id, metadata_dict, video_json_path, audio_json_path) - apenas paths, sem limite
embed_queue = queue.Queue(maxsize=8)
index_queue = queue.Queue()

SENTINEL = object()


# Worker de embedding (GPU
def worker_embed(clip_model, clip_preprocess, clap_model, device, n_producers: int) -> None:
    sentinels_received = 0
    logger.info(f"[EMBED] worker iniciado. device={device} n_producers={n_producers}")

    while True:
        try:
            item = embed_queue.get(timeout=60)
        except queue.Empty:
            logger.warning("[EMBED] fila vazia há 60s.")
            continue

        if item is SENTINEL:
            sentinels_received += 1
            embed_queue.task_done()
            logger.info(f"[EMBED] sentinela recebida ({sentinels_received}/{n_producers})")
            if sentinels_received >= n_producers:
                for _ in range(INDEX_WORKERS):
                    index_queue.put(SENTINEL)
                logger.info("[EMBED] worker finalizando.")
                return
            continue

        video_id, video_path, meta = item
        logger.info(f"[EMBED] {video_id} ===== INICIANDO PROCESSAMENTO =====")
        logger.info(f"[EMBED] {video_id} video_path={video_path}")
        logger.info(f"[EMBED] {video_id} device={device} cuda_available={torch.cuda.is_available()}")

        try:
            video_json = os.path.join(EMBEDDINGS_DIR, f"{video_id}_video.json")
            audio_json = os.path.join(EMBEDDINGS_DIR, f"{video_id}_audio.json")
            logger.info(f"[EMBED] {video_id} video_json={video_json}")
            logger.info(f"[EMBED] {video_id} audio_json={audio_json}")

            # Cache: JSONs já existem -> pula extração
            cache_video = os.path.exists(video_json)
            cache_audio = os.path.exists(audio_json)
            logger.info(f"[EMBED] {video_id} cache_check: video_exists={cache_video} audio_exists={cache_audio}")
            if cache_video and cache_audio:
                logger.info(f"[EMBED] {video_id} cache encontrado - gerando só thumbnail.")
                try:
                    feature_thumb = ind.fetch_thumbnail_embedding(
                        video_id, clip_model, clip_preprocess, device
                    )
                except Exception:
                    feature_thumb = None
                index_queue.put((video_id, meta, video_json, audio_json, feature_thumb))
                continue

            # Detecção de cenas
            logger.info(f"[EMBED] {video_id} detectando cenas...")
            try:
                scenes = ind.detect_scenes(video_path)
                logger.info(f"[EMBED] {video_id} detect_scenes retornou {len(scenes)} cenas")
            except Exception as e:
                logger.warning(f"[EMBED] {video_id} erro detect_scenes: {e}")
                scenes = []

            if not scenes:
                logger.info(f"[EMBED] {video_id} sem cenas detectadas - 1 frame a cada 10s")
                cap   = cv2.VideoCapture(video_path)
                fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
                duration = total / fps
                scenes = [(t, t) for t in range(0, int(duration), 2)]
                logger.info(f"[EMBED] {video_id} fallback: fps={fps} duracao={duration:.1f}s {len(scenes)} frames")

            logger.info(f"[EMBED] {video_id} {len(scenes)} cenas para segmentar")

            # Segmenta + CLIP (stream ou batch)
            mode = "batch" if CONFIG["batch_mode"] else "stream"
            logger.info(f"[EMBED] {video_id} segmentando + CLIP ({mode}, device={device})...")
            try:
                if CONFIG["batch_mode"]:
                    video_embs = ky.batch_segments(
                        video_path, scenes, clip_model, clip_preprocess, device, max_frames=45,
                    )
                else:
                    video_embs = ky.stream_segments(
                        video_path, scenes, clip_model, clip_preprocess, device, max_frames=45,
                    )
                logger.info(f"[EMBED] {video_id} CLIP concluído: {len(video_embs)} embeddings")
            except Exception as e:
                logger.error(f"[EMBED] {video_id} erro {mode}_segments: {e}")
                import traceback
                logger.error(f"[EMBED] {video_id} traceback: {traceback.format_exc()}")
                continue

            if not video_embs:
                total_scenes = len(scenes)
                if total_scenes == 1 and scenes[0][0] == 0.0:
                    logger.warning(f"[EMBED] {video_id} video sem cenas e sem frames viaveis - "
                                   f"provavelmente arquivo corrompido ou vazio. Pulando.")
                else:
                    logger.warning(f"[EMBED] {video_id} {total_scenes} cenas mas 0 embeddings - "
                                   f"frames corrompidos ou codec nao suportado. Pulando.")
                continue

            # CLAP
            logger.info(f"[EMBED] {video_id} iniciando CLAP (segment_duration=1.5s)")
            try:
                audio_embs = emb.generate_audio_embeddings_from_segments(
                    video_path=video_path, segments=video_embs,
                    clap_model=clap_model, device=device, segment_duration=1.5,
                )
                logger.info(f"[EMBED] {video_id} CLAP concluído: {len(audio_embs)} embeddings")
                for i, ae in enumerate(audio_embs):
                    seg_ts = video_embs[i]["timestamp_sec"] if i < len(video_embs) else 0.0
                    ae["timestamp_sec"] = seg_ts
                    ae["scene_index"]   = video_embs[i].get("scene_index", -1) if i < len(video_embs) else -1
                    ae["part_index"]    = i
                    ae["center_frame"]  = -1
            except Exception as e:
                logger.warning(f"[EMBED] {video_id} erro CLAP: {e}")
                audio_embs = []

            logger.info(f"[EMBED] {video_id} resultados: {len(video_embs)} video | {len(audio_embs)} audio")

            # Thumbnail embedding
            logger.info(f"[EMBED] {video_id} gerando thumbnail embedding...")
            try:
                feature_thumb = ind.fetch_thumbnail_embedding(
                    video_id, clip_model, clip_preprocess, device
                )
            except Exception as e:
                logger.warning(f"[EMBED] {video_id} erro thumbnail: {e}")
                feature_thumb = None

            # Salva em disco e libera RAM
            os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
            logger.info(f"[EMBED] {video_id} salvando video_embs em {video_json}...")
            emb.save_embeddings_json(video_embs, path=video_json)
            if audio_embs:
                logger.info(f"[EMBED] {video_id} salvando audio_embs em {audio_json}...")
                emb.save_embeddings_json(audio_embs, path=audio_json)
            else:
                logger.info(f"[EMBED] {video_id} sem audio_embs - arquivo não salvo")

            logger.info(f"[EMBED] {video_id} liberando memoria...")
            del video_embs, audio_embs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            index_queue.put((video_id, meta, video_json, audio_json, feature_thumb))
            logger.info(f"[EMBED] {video_id} enviado para fila de indexação.")

        except Exception as e:
            logger.error(f"[EMBED] {video_id} erro no processamento: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"[EMBED] {video_id} traceback: {traceback.format_exc()}")
        finally:
            embed_queue.task_done()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"[EMBED] {video_id} task_done sinalizado para fila.")


# Worker de indexação
def worker_index(es, taxonomy_lookup: dict) -> None:
    while True:
        item = index_queue.get()
        try:
            if item is SENTINEL:
                logger.info("Index worker finalizando.")
                return

            video_id, meta, video_json, audio_json, feature_thumb = item

            with open(video_json) as f:
                video_embs = json.load(f)
            with open(audio_json) as f:
                audio_embs = json.load(f)

            for e_ in video_embs + audio_embs:
                if isinstance(e_["embedding"], list):
                    e_["embedding"] = np.array(e_["embedding"], dtype=np.float32)

            label = meta.get("anet_label", "")
            title = meta.get("title", "")
            feature_categorias = ind.get_feature_categorias(label, taxonomy_lookup)
            feature_desc = meta.get("feature_desc", "")
            keywords = meta.get("keywords", "")

            extra_meta = {
                "description":  meta.get("description", ""),
                "transcript":   meta.get("transcript", ""),
                "upload_date":  meta.get("upload_date", ""),
                "duration_sec": meta.get("duration", 0),
                "view_count":   meta.get("view_count", 0),
                "like_count":   meta.get("like_count", 0),
                "channel":      meta.get("channel", ""),
                "tags":         meta.get("tags", []),
                "categories":   meta.get("categories", []),
            }

            if video_embs:
                ind.index_embeddings_bulk(
                    es, video_embs, index_name="video_index",
                    video_id=video_id, title=title,
                    feature_categorias=feature_categorias,
                    feature_desc=feature_desc, keywords=keywords,
                    feature_thumb=feature_thumb, modality="video",
                    **extra_meta,
                )
            if audio_embs:
                ind.index_embeddings_bulk(
                    es, audio_embs, index_name="video_index",
                    video_id=video_id, title=title,
                    feature_categorias=feature_categorias,
                    feature_desc=feature_desc, keywords=keywords,
                    feature_thumb=feature_thumb, modality="audio",
                    **extra_meta,
                )

            logger.info(f"{video_id}: indexado")

            del video_embs, audio_embs
            gc.collect()

        except Exception as e:
            logger.error(f"Erro indexação {video_id}: {e}")
        finally:
            index_queue.task_done()


# Gera campos sintéticos se ausentes
def _ensure_synthetic_fields(meta: dict, taxonomy_lookup: dict) -> dict:
    if not meta.get("feature_desc") or not meta.get("keywords"):
        label = meta.get("anet_label", "")
        title = meta.get("title", "")
        fd, kw = ind.build_text_metadata(label, title, taxonomy_lookup)
        if not meta.get("feature_desc"):
            meta["feature_desc"] = fd
        if not meta.get("keywords"):
            meta["keywords"] = kw
    return meta


# Pipeline principal
def main_index_fast(window: int = 0, offset: int = 0, input_file: str | None = None) -> None:
    global_start = time.time()

    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    logger.info("Conectando ao Elasticsearch...")
    es = ind.connect_elasticsearch()
    ind.create_index(es, index_name="video_index", dims=512)

    logger.info("Carregando modelos CLIP e CLAP...")
    clip_model, clip_preprocess, clap_model, device = emb.load_all_models()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    anet_json = os.path.join(BASE_DIR, "..", "data", "activity_net.v1-3.min.json")
    taxonomy = ind.build_taxonomy_lookup(anet_json)

    # Arquivo de metadados
    if input_file is None:
        input_file = os.path.join(BASE_DIR, "..", "data", "metadata", "videos_metadata.json")

    if not os.path.exists(input_file):
        logger.warning(f"Arquivo de metadados não encontrado: {input_file}")
        logger.info(f"Criando arquivo vazio: {input_file}")
        os.makedirs(os.path.dirname(input_file), exist_ok=True)
        with open(input_file, "w") as f:
            json.dump({}, f)

    all_meta = ind.load_filtered_metadata(input_file)
    logger.info(f"Total de entradas em {os.path.basename(input_file)}: {len(all_meta)}")

    # Cruza com MP4s existentes em disco
    mp4_ids = set()
    if os.path.isdir(VIDEO_DIR):
        for fname in os.listdir(VIDEO_DIR):
            if fname.endswith(".mp4"):
                mp4_ids.add(fname.replace(".mp4", ""))

    candidates = sorted(
        vid for vid in all_meta if vid in mp4_ids
    )
    logger.info(f"Vídeos com MP4 em disco: {len(mp4_ids)} | Com metadados: {len(candidates)}")

    # Aplica offset na lista completa
    if offset > 0:
        candidates = candidates[offset:]
        logger.info(f"Offset aplicado: pulando {offset} videos")

    # Filtra já indexados (antes do window)
    pending = []
    for vid in candidates:
        if not ind.already_indexed(es, vid):
            meta = _ensure_synthetic_fields(all_meta[vid].copy(), taxonomy)
            pending.append((vid, meta))
        if window > 0 and len(pending) >= window:
            break

    logger.info(f"JA indexados: {len(candidates) - len(pending)} | NOVOS a processar: {len(pending)}")

    if not pending:
        logger.info("Nada para processar.")
        return

    # Embedding thread (1 ~ GPU)
    embed_thread = threading.Thread(
        target=worker_embed,
        args=(clip_model, clip_preprocess, clap_model, device, 1),
        daemon=False,
    )
    embed_thread.start()

    # Index threads (N ~ CPU, sem GPU)
    index_threads = [
        threading.Thread(
            target=worker_index,
            args=(es, taxonomy),
            daemon=False,
        )
        for _ in range(INDEX_WORKERS)
    ]
    for t in index_threads:
        t.start()

    # Alimenta fila de embedding com vídeos já baixados
    logger.info("Alimentando fila de embedding...")
    for vid, meta in pending:
        video_path = os.path.join(VIDEO_DIR, f"{vid}.mp4")
        if os.path.exists(video_path):
            embed_queue.put((vid, video_path, meta))
        else:
            logger.warning(f"{vid}: MP4 não encontrado em disco - pulando.")

    logger.info("Todos os vídeos enfileirados - sinalizando fim.")
    embed_queue.put(SENTINEL)

    embed_queue.join()
    embed_thread.join()
    index_queue.join()
    for t in index_threads:
        t.join()

    total = time.time() - global_start
    logger.info(f"Pipeline concluído em {total / 60:.1f} min ({len(pending)} vídeos)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de indexacao de videos no Elasticsearch",
    )
    parser.add_argument(
        "--window", "-w", type=int, default=0,
        help="Quantidade de NOVOS videos a indexar (0 = todos os pendentes)",
    )
    parser.add_argument(
        "--offset", "-o", type=int, default=0,
        help="Pula os primeiros N videos da lista completa",
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="Arquivo JSON de metadados (default: data/metadata/videos_metadata.json)",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Modo batch (carrega todos os frames na RAM, mais rapido mas consome mais memoria)",
    )

    args = parser.parse_args()

    if args.batch:
        CONFIG["batch_mode"] = True

    log_args = [
        f"window={args.window or 'todos'}",
        f"offset={args.offset}",
        f"input={args.input or 'videos_metadata.json'}",
        f"batch={CONFIG['batch_mode']}",
    ]
    logger.info(f"main_index iniciado com: {', '.join(log_args)}")

    main_index_fast(window=args.window, offset=args.offset, input_file=args.input)
