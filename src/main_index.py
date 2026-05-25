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
DOWNLOAD_WORKERS = 4      # downloads simultâneos (CPU)
INDEX_WORKERS    = 2      # threads de indexação no ES (CPU)
EMBEDDINGS_DIR   = "./data/embeddings"
VIDEO_DIR        = "./data/videos"
BROWSER          = "chrome"

# Filas de comunicação entre os workers
# download_queue: (video_id, meta, video_path) — vídeos prontos para embeddar
# index_queue:    (video_id, label, title, video_embs, audio_embs) — prontos para indexar
download_queue = queue.Queue(maxsize=8)
index_queue    = queue.Queue(maxsize=16)

SENTINEL = None  # sinal de fim para os workers


# ─── Worker de download ────────────────────────────────────────────────────────
def worker_download(video_id: str, meta: dict, video_dir: str) -> None:
    """Baixa um vídeo e coloca na fila de embedding. Roda em pool de threads."""
    local_path = Path(video_dir) / f"{video_id}.mp4"

    if local_path.exists() and ind._is_valid_mp4(str(local_path)):
        video_path = str(local_path)
        logger.info(f"{video_id}: encontrado localmente.")
    else:
        logger.info(f"{video_id}: baixando...")
        video_path = ind.download_video(video_id, video_dir, browser=BROWSER)

    if video_path:
        download_queue.put((video_id, meta, video_path))
    else:
        logger.warning(f"{video_id}: falha no download — pulando.")


# ─── Worker de embedding (GPU — sempre 1 thread) ──────────────────────────────
def worker_embed(
    clip_model, clip_preprocess, clap_model, device,
    taxonomy_lookup: dict, n_producers: int
) -> None:
    """
    Consome vídeos da download_queue, gera embeddings CLIP+CLAP,
    salva os JSONs em disco e coloca na index_queue.
    Recebe n_producers sentinelas antes de terminar.
    """
    sentinels_received = 0

    while True:
        item = download_queue.get()

        if item is SENTINEL:
            sentinels_received += 1
            download_queue.task_done()
            if sentinels_received >= n_producers:
                # propaga sentinelas para os workers de indexação
                for _ in range(INDEX_WORKERS):
                    index_queue.put(SENTINEL)
                break
            continue

        video_id, meta, video_path = item

        try:
            annotations = meta.get("annotations", [])
            label = annotations[0].get("label", "") if annotations else ""
            title = meta.get("url", "")

            video_json = os.path.join(EMBEDDINGS_DIR, f"{video_id}_video.json")
            audio_json = os.path.join(EMBEDDINGS_DIR, f"{video_id}_audio.json")

            # ── Cache: se JSONs já existem, só carrega ────────────────────────
            if os.path.exists(video_json) and os.path.exists(audio_json):
                logger.info(f"{video_id}: embeddings em cache — carregando.")
                with open(video_json) as f:
                    video_embs = json.load(f)
                with open(audio_json) as f:
                    audio_embs = json.load(f)
                for item_ in video_embs + audio_embs:
                    if isinstance(item_["embedding"], list):
                        item_["embedding"] = np.array(item_["embedding"], dtype=np.float32)
            else:
                # ── Extrai embeddings do vídeo ────────────────────────────────
                logger.info(f"{video_id}: extraindo embeddings...")

                try:
                    scenes = ind.detect_scenes(video_path)
                except Exception as e:
                    logger.warning(f"{video_id}: falha na detecção de cenas: {e}. Usando inteiro.")
                    scenes = []

                if not scenes:
                    import cv2
                    cap   = cv2.VideoCapture(video_path)
                    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    cap.release()
                    scenes = [(0.0, total / fps)]

                all_segments = []
                for scene_idx, (start, end) in enumerate(scenes):
                    if end - start < 1.0:
                        continue
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
                        logger.warning(f"{video_id}: erro na cena {scene_idx}: {e}")

                if not all_segments:
                    logger.warning(f"{video_id}: nenhum segmento gerado — pulando.")
                    download_queue.task_done()
                    continue

                # CLIP
                video_embs = []
                for seg in all_segments:
                    try:
                        vector = emb.embed_window(
                            seg["frames"], clip_model, clip_preprocess, device, method="mean"
                        )
                        video_embs.append({
                            "scene_index":   seg["scene_index"],
                            "part_index":    seg["segment_index"],
                            "timestamp_sec": seg["timestamp_sec"],
                            "center_frame":  seg["center_frame"],
                            "embedding":     vector,
                        })
                    except Exception as e:
                        logger.warning(f"{video_id}: erro no segmento CLIP: {e}")

                # CLAP
                try:
                    audio_embs = emb.generate_audio_embeddings_from_segments(
                        video_path=video_path,
                        segments=all_segments,
                        clap_model=clap_model,
                        device=device,
                        segment_duration=1.5,
                    )
                    for i, ae in enumerate(audio_embs):
                        seg = all_segments[i] if i < len(all_segments) else None
                        ae["timestamp_sec"] = seg["timestamp_sec"] if seg else 0.0
                        ae["scene_index"]   = seg["scene_index"]   if seg else -1
                        ae["part_index"]    = seg.get("segment_index", i) if seg else i
                        ae["center_frame"]  = -1
                except Exception as e:
                    logger.warning(f"{video_id}: erro no CLAP: {e}")
                    audio_embs = []

                # Salva JSONs em disco
                os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
                emb.save_embeddings_json(video_embs, path=video_json)
                if audio_embs:
                    emb.save_embeddings_json(audio_embs, path=audio_json)

                logger.info(
                    f"{video_id}: {len(video_embs)} emb. vídeo | {len(audio_embs)} emb. áudio"
                )

            # Coloca na fila de indexação
            index_queue.put((video_id, label, title, video_embs, audio_embs))

        except Exception as e:
            logger.error(f"{video_id}: erro no embedding: {e}")
        finally:
            download_queue.task_done()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ─── Worker de indexação (ES — pode ter 2 threads) ────────────────────────────
def worker_index(es, clip_model, clip_preprocess, device, taxonomy_lookup: dict) -> None:
    """Consome embeddings da index_queue e indexa no Elasticsearch."""
    while True:
        item = index_queue.get()

        if item is SENTINEL:
            index_queue.task_done()
            break

        video_id, label, title, video_embs, audio_embs = item

        try:
            feature_categorias = ind.get_feature_categorias(label, taxonomy_lookup)
            feature_desc, keywords = ind.build_text_metadata(label, title, taxonomy_lookup)

            # Thumbnail
            feature_thumb = ind.fetch_thumbnail_embedding(
                video_id, clip_model, clip_preprocess, device
            )
            if not title:
                title = ind.fetch_video_title(video_id)
                feature_desc, keywords = ind.build_text_metadata(label, title, taxonomy_lookup)

            if video_embs:
                ind.index_embeddings_bulk(
                    es, video_embs,
                    index_name="video_index",
                    video_id=video_id, title=title,
                    feature_categorias=feature_categorias,
                    feature_desc=feature_desc, keywords=keywords,
                    feature_thumb=feature_thumb, modality="video",
                )
            if audio_embs:
                ind.index_embeddings_bulk(
                    es, audio_embs,
                    index_name="video_index",
                    video_id=video_id, title=title,
                    feature_categorias=feature_categorias,
                    feature_desc=feature_desc, keywords=keywords,
                    feature_thumb=feature_thumb, modality="audio",
                )
            logger.info(f"{video_id}: indexado ✓")

        except Exception as e:
            logger.error(f"{video_id}: erro na indexação: {e}")
        finally:
            index_queue.task_done()


# ─── Pipeline principal ────────────────────────────────────────────────────────
def main_index_fast() -> None:
    global_start = time.time()

    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    # Conexões e modelos
    logger.info("Conectando ao Elasticsearch...")
    es = ind.connect_elasticsearch()

    ind.create_index(es, index_name="video_index", dims=512)

    logger.info("Carregando modelos CLIP e CLAP...")
    clip_model, clip_preprocess, clap_model, device = emb.load_all_models()

    BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(BASE_DIR, "..", "data", "activity_net.v1-3.min.json")
    ind.ensure_activitynet_json(json_path)

    dataset        = ind.load_activitynet(json_path)
    taxonomy       = ind.build_taxonomy_lookup(json_path)

    # Filtra validação e já indexados
    validation = [
        (vid, meta) for vid, meta in dataset.items()
        if meta["subset"] == "validation"
    ][:MAX_VIDEOS]

    logger.info(f"Total de vídeos de validação: {len(validation)}")

    to_process = [
        (vid, meta) for vid, meta in validation
        if not ind.already_indexed(es, vid)
    ]

    logger.info(f"Para processar (não indexados): {len(to_process)}")

    if not to_process:
        logger.info("Todos os vídeos já estão indexados.")
        return

    # ── Inicia worker de embedding (1 thread GPU) ──────────────────────────────
    embed_thread = threading.Thread(
        target=worker_embed,
        args=(clip_model, clip_preprocess, clap_model, device, taxonomy, 1),
        daemon=True,
    )
    embed_thread.start()

    # ── Inicia workers de indexação (N threads CPU) ────────────────────────────
    index_threads = [
        threading.Thread(
            target=worker_index,
            args=(es, clip_model, clip_preprocess, device, taxonomy),
            daemon=True,
        )
        for _ in range(INDEX_WORKERS)
    ]
    for t in index_threads:
        t.start()

    # ── Downloads em paralelo (pool de threads CPU) ────────────────────────────
    logger.info(f"Iniciando {DOWNLOAD_WORKERS} workers de download...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = [
            pool.submit(worker_download, vid, meta, VIDEO_DIR)
            for vid, meta in to_process
        ]
        concurrent.futures.wait(futures)

    # Sinaliza fim para o worker de embedding
    download_queue.put(SENTINEL)

    # Aguarda todos terminarem
    embed_thread.join()
    for t in index_threads:
        t.join()

    total = time.time() - global_start
    logger.info(f"Pipeline concluído em {total / 60:.1f} min ({len(to_process)} vídeos)")


if __name__ == "__main__":
    main_index_fast()