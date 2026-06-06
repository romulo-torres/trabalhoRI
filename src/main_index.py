import gc
import json
import os
import queue
import threading
import time
import concurrent.futures
from pathlib import Path

import numpy as np
import torch

import index_elastic as ind
import embeddings as emb
import keyframes as ky
from logger import setup_logger

logger = setup_logger()

# ─── Configuração ─────────────────────────────────────────────────────────────
MAX_VIDEOS       = 2000
DOWNLOAD_WORKERS = 2
INDEX_WORKERS    = 2
EMBEDDINGS_DIR   = "./data/embeddings"
VIDEO_DIR        = "./data/videos"
BROWSER          = "chrome"

# Filas passam apenas IDs/paths, não vetores
# download_queue: (video_id, meta, video_path)
# index_queue:    (video_id, label, title, video_json_path, audio_json_path)
download_queue = queue.Queue(maxsize=2)
index_queue    = queue.Queue(maxsize=4)

SENTINEL = object()


# ─── Worker de download ────────────────────────────────────────────────────────
def worker_download(video_id: str, meta: dict, video_dir: str) -> None:
    local_path = Path(video_dir) / f"{video_id}.mp4"
    try:
        if local_path.exists() and ind._is_valid_mp4(str(local_path)):
            video_path = str(local_path)
            logger.info(f"{video_id}: encontrado localmente.")
        else:
            logger.info(f"{video_id}: baixando...")
            video_path = ind.download_video(video_id, video_dir, browser=BROWSER)

        if not video_path:
            logger.warning(f"{video_id}: falha no download.")
            return

        while True:
            try:
                download_queue.put((video_id, meta, video_path), timeout=30)
                break
            except queue.Full:
                logger.warning(f"{video_id}: fila download cheia, aguardando...")

    except Exception as e:
        logger.error(f"{video_id}: erro download: {e}")


# ─── Worker de embedding (GPU) ────────────────────────────────────────────────
def worker_embed(clip_model, clip_preprocess, clap_model, device, n_producers: int) -> None:
    sentinels_received = 0

    while True:
        try:
            item = download_queue.get(timeout=60)
        except queue.Empty:
            logger.warning("Embedding: sem itens há 60s.")
            continue

        if item is SENTINEL:
            sentinels_received += 1
            download_queue.task_done()
            if sentinels_received >= n_producers:
                for _ in range(INDEX_WORKERS):
                    index_queue.put(SENTINEL)
                logger.info("Embedding: finalizando.")
                return
            continue

        video_id, meta, video_path = item

        try:
            annotations = meta.get("annotations", [])
            label = annotations[0].get("label", "") if annotations else ""
            title = meta.get("url", "")

            video_json = os.path.join(EMBEDDINGS_DIR, f"{video_id}_video.json")
            audio_json = os.path.join(EMBEDDINGS_DIR, f"{video_id}_audio.json")

            # ── Cache: JSONs já existem → pula extração ───────────────────────
            if os.path.exists(video_json) and os.path.exists(audio_json):
                logger.info(f"{video_id}: cache encontrado.")
                index_queue.put((video_id, label, title, video_json, audio_json))
                continue

            # ── Extrai embeddings ─────────────────────────────────────────────
            logger.info(f"{video_id}: extraindo embeddings...")

            try:
                scenes = ind.detect_scenes(video_path)
            except Exception as e:
                logger.warning(f"{video_id}: erro na detecção de cenas: {e}")
                scenes = []

            if not scenes:
                import cv2
                cap   = cv2.VideoCapture(video_path)
                fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
                scenes = [(0.0, total / fps)]

            logger.info(f"{video_id}: {len(scenes)} cenas detectadas.")

            # ── Segmentação sem limites ───────────────────────────────────────
            all_segments = []
            for scene_idx, (start, end) in enumerate(scenes):
                if end - start < 1.0:
                    continue
                # trunca cenas > 5min (geralmente erro de detecção)
                end = min(end, start + 300)
                try:
                    segs = ky.split_scene_into_segments(
                        video_path, start_time=start, end_time=end,
                        max_frames_per_segment=45,
                    )
                    for seg in segs:
                        seg["scene_index"]   = scene_idx
                        seg["segment_index"] = seg.get("segment_index", 0)
                        all_segments.append(seg)
                except Exception as e:
                    logger.warning(f"{video_id}: erro segmentação cena {scene_idx}: {e}")

            if not all_segments:
                logger.warning(f"{video_id}: nenhum segmento gerado.")
                continue

            logger.info(f"{video_id}: {len(all_segments)} segmentos.")

            # ── CLIP — libera frames imediatamente após cada segmento ─────────
            video_embs = []
            for idx, seg in enumerate(all_segments):
                try:
                    vector = emb.embed_window(
                        seg["frames"], clip_model, clip_preprocess, device, method="mean"
                    )
                    if vector is not None:
                        video_embs.append({
                            "scene_index":   seg["scene_index"],
                            "part_index":    seg["segment_index"],
                            "timestamp_sec": seg["timestamp_sec"],
                            "center_frame":  seg["center_frame"],
                            "embedding":     vector,
                        })
                except Exception as e:
                    logger.warning(f"{video_id}: erro CLIP seg {idx}: {e}")
                finally:
                    seg.pop("frames", None)  # libera frames imediatamente

                if idx % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # ── CLAP ──────────────────────────────────────────────────────────
            try:
                audio_embs = emb.generate_audio_embeddings_from_segments(
                    video_path=video_path, segments=all_segments,
                    clap_model=clap_model, device=device, segment_duration=1.5,
                )
                for i, ae in enumerate(audio_embs):
                    seg = all_segments[i] if i < len(all_segments) else None
                    ae["timestamp_sec"] = seg["timestamp_sec"] if seg else 0.0
                    ae["scene_index"]   = seg["scene_index"]   if seg else -1
                    ae["part_index"]    = seg.get("segment_index", i) if seg else i
                    ae["center_frame"]  = -1
            except Exception as e:
                logger.warning(f"{video_id}: erro CLAP: {e}")
                audio_embs = []

            logger.info(f"{video_id}: {len(video_embs)} vídeo | {len(audio_embs)} áudio")

            # ── Salva em disco e libera RAM ───────────────────────────────────
            os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
            emb.save_embeddings_json(video_embs, path=video_json)
            if audio_embs:
                emb.save_embeddings_json(audio_embs, path=audio_json)

            del video_embs, audio_embs, all_segments
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Coloca só os paths na fila — não os vetores
            index_queue.put((video_id, label, title, video_json, audio_json))
            logger.info(f"{video_id}: enviado para indexação.")

        except Exception as e:
            logger.error(f"{video_id}: erro embedding: {e}")
        finally:
            download_queue.task_done()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ─── Worker de indexação ──────────────────────────────────────────────────────
def worker_index(es, clip_model, clip_preprocess, device, taxonomy_lookup: dict) -> None:
    while True:
        item = index_queue.get()
        try:
            if item is SENTINEL:
                logger.info("Index worker finalizando.")
                return

            video_id, label, title, video_json, audio_json = item

            # Lê do disco — não carrega da RAM do worker_embed
            with open(video_json) as f:
                video_embs = json.load(f)
            with open(audio_json) as f:
                audio_embs = json.load(f)

            for e_ in video_embs + audio_embs:
                if isinstance(e_["embedding"], list):
                    e_["embedding"] = np.array(e_["embedding"], dtype=np.float32)

            feature_categorias = ind.get_feature_categorias(label, taxonomy_lookup)
            feature_desc, keywords = ind.build_text_metadata(label, title, taxonomy_lookup)
            feature_thumb = ind.fetch_thumbnail_embedding(
                video_id, clip_model, clip_preprocess, device
            )

            if video_embs:
                ind.index_embeddings_bulk(
                    es, video_embs, index_name="video_index",
                    video_id=video_id, title=title,
                    feature_categorias=feature_categorias,
                    feature_desc=feature_desc, keywords=keywords,
                    feature_thumb=feature_thumb, modality="video",
                )
            if audio_embs:
                ind.index_embeddings_bulk(
                    es, audio_embs, index_name="video_index",
                    video_id=video_id, title=title,
                    feature_categorias=feature_categorias,
                    feature_desc=feature_desc, keywords=keywords,
                    feature_thumb=feature_thumb, modality="audio",
                )

            logger.info(f"{video_id}: indexado ✓")

            del video_embs, audio_embs
            gc.collect()

        except Exception as e:
            logger.error(f"Erro indexação {video_id}: {e}")
        finally:
            index_queue.task_done()


# ─── Pipeline principal ───────────────────────────────────────────────────────
def main_index_fast() -> None:
    global_start = time.time()

    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    logger.info("Conectando ao Elasticsearch...")
    es = ind.connect_elasticsearch()
    ind.create_index(es, index_name="video_index", dims=512)

    logger.info("Carregando modelos CLIP e CLAP...")
    clip_model, clip_preprocess, clap_model, device = emb.load_all_models()

    BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(BASE_DIR, "..", "data", "activity_net.v1-3.min.json")
    ind.ensure_activitynet_json(json_path)

    dataset  = ind.load_activitynet(json_path)
    taxonomy = ind.build_taxonomy_lookup(json_path)

    validation = [
        (vid, meta) for vid, meta in dataset.items()
        if meta["subset"] == "validation"
    ][:MAX_VIDEOS]

    logger.info(f"Validação total: {len(validation)}")

    to_process = [
        (vid, meta) for vid, meta in validation
        if not ind.already_indexed(es, vid)
    ]

    logger.info(f"Para processar: {len(to_process)}")

    if not to_process:
        logger.info("Nada para processar.")
        return

    # Embedding thread (1 — GPU)
    embed_thread = threading.Thread(
        target=worker_embed,
        args=(clip_model, clip_preprocess, clap_model, device, 1),
        daemon=False,
    )
    embed_thread.start()

    # Index threads (N — CPU)
    index_threads = [
        threading.Thread(
            target=worker_index,
            args=(es, clip_model, clip_preprocess, device, taxonomy),
            daemon=False,
        )
        for _ in range(INDEX_WORKERS)
    ]
    for t in index_threads:
        t.start()

    # Downloads em paralelo
    logger.info(f"Iniciando downloads ({DOWNLOAD_WORKERS} workers)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = [
            pool.submit(worker_download, vid, meta, VIDEO_DIR)
            for vid, meta in to_process
        ]
        concurrent.futures.wait(futures)

    logger.info("Downloads concluídos — sinalizando fim.")
    download_queue.put(SENTINEL)

    download_queue.join()
    embed_thread.join()
    index_queue.join()
    for t in index_threads:
        t.join()

    total = time.time() - global_start
    logger.info(f"Pipeline concluído em {total / 60:.1f} min ({len(to_process)} vídeos)")


if __name__ == "__main__":
    main_index_fast()