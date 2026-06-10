import io
import json
import os
import random
import re
import subprocess
import urllib.request
import cv2

import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from elasticsearch import Elasticsearch, helpers
from PIL import Image
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

import embeddings as emb
import keyframes as ky
from logger import setup_logger

np.float_ = np.float64

logger = setup_logger()

os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "quiet"


# ==============================================================================
# ActivityNet — download do JSON de anotações
# ==============================================================================
def ensure_activitynet_json(json_path: str) -> None:
    if os.path.exists(json_path):
        print(f"Arquivo já existe: {json_path}")
        return

    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    url = "https://storage.googleapis.com/activitynet/annotations/activity_net.v1-3.min.json"
    print("Baixando ActivityNet JSON com wget...")

    try:
        subprocess.run(["wget", "-O", json_path, url], check=True)
    except Exception as e:
        raise RuntimeError("Falha ao baixar ActivityNet com wget") from e

    print("Download concluído!")


def load_activitynet(json_path: str) -> dict:
    with open(json_path) as f:
        data = json.load(f)
    return data["database"]


# ==============================================================================
# Taxonomia: nodeName → parentName
# ==============================================================================
def build_taxonomy_lookup(json_path: str) -> dict[str, str]:
    with open(json_path) as f:
        data = json.load(f)

    lookup: dict[str, str] = {}
    for node in data.get("taxonomy", []):
        node_name   = node.get("nodeName", "")
        parent_name = node.get("parentName", "")
        if node_name:
            lookup[node_name] = parent_name

    return lookup


def get_feature_categorias(label: str, taxonomy_lookup: dict[str, str]) -> str:
    if not label:
        return ""
    parent = taxonomy_lookup.get(label, "")
    return f"{label} > {parent}" if parent else label


def build_text_metadata(label: str, title: str, taxonomy_lookup: dict) -> tuple[str, str]:
    parent = taxonomy_lookup.get(label, "")
    if parent:
        feature_desc = f"Video about {label} (category: {parent}). Title: {title}" if title else f"Video about {label} (category: {parent})."
        keywords = f"{label}, {parent}, {title}" if title else f"{label}, {parent}"
    else:
        feature_desc = f"Video about {label}. Title: {title}" if title else f"Video about {label}."
        keywords = f"{label}, {title}" if title else label
    return feature_desc, keywords


# ==============================================================================
# Thumbnail — embedding via URL pública do YouTube
# ==============================================================================
def fetch_thumbnail_embedding(
    video_id:   str,
    model,
    preprocess,
    device:     str,
) -> np.ndarray | None:
    resolutions = ["maxresdefault", "sddefault", "hqdefault", "mqdefault"]

    for res in resolutions:
        url = f"https://img.youtube.com/vi/{video_id}/{res}.jpg"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                img_bytes = resp.read()

            image  = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            tensor = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model.encode_image(tensor)

            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy().flatten()

        except Exception:
            continue

    logger.warning(f"Thumbnail não disponível para {video_id}.")
    return None


# ==============================================================================
# Metadados completos do vídeo via yt-dlp
# ==============================================================================
def fetch_video_metadata(video_id: str) -> dict:
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-playlist", url],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning(f"yt-dlp retornou erro para {video_id}: {result.stderr.strip()}")
            return {}
        data = json.loads(result.stdout)
        return {
            "title":       data.get("title", ""),
            "description": data.get("description", ""),
            "upload_date": data.get("upload_date", ""),
            "duration":    data.get("duration", 0),
            "view_count":  data.get("view_count", 0),
            "like_count":  data.get("like_count", 0),
            "channel":     data.get("uploader", ""),
            "tags":        data.get("tags", []),
            "categories":  data.get("categories", []),
        }
    except Exception as e:
        logger.warning(f"Falha ao obter metadata de {video_id}: {e}")
        return {}


# ==============================================================================
# Legendas em inglês via yt-dlp
# ==============================================================================
def fetch_subtitles_en(video_id: str, output_dir: str = "./data/subs") -> str:
    os.makedirs(output_dir, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        subprocess.run([
            "yt-dlp",
            "--write-auto-sub",
            "--sub-lang", "en",
            "--sub-format", "vtt",
            "--skip-download",
            "--no-playlist",
            "-o", os.path.join(output_dir, f"{video_id}.%(ext)s"),
            url,
        ], check=True, capture_output=True, timeout=30)

        for suffix in ["en", "en-orig"]:
            vtt_path = os.path.join(output_dir, f"{video_id}.{suffix}.vtt")
            if os.path.exists(vtt_path):
                logger.info(f"Legenda EN encontrada para {video_id}: {vtt_path}")
                return _parse_vtt(vtt_path)

    except Exception as e:
        logger.warning(f"Legenda EN indisponível para {video_id}: {e}")
    return ""


def _parse_vtt(path: str) -> str:
    """Extrai texto limpo de um arquivo VTT, removendo timestamps, tags e duplicatas."""
    with open(path, encoding="utf-8") as f:
        content = f.read()

    content = re.sub(r"\d{2}:\d{2}:\d{2}\.\d+ --> .*\n", "", content)
    content = re.sub(r"<[^>]+>", "", content)
    content = re.sub(r"^WEBVTT.*\n", "", content, flags=re.MULTILINE)

    lines   = [l.strip() for l in content.splitlines() if l.strip()]
    deduped = [lines[0]] if lines else []
    for line in lines[1:]:
        if line != deduped[-1]:
            deduped.append(line)

    return " ".join(deduped)


# ==============================================================================
# Download de vídeo do YouTube via yt-dlp
# ==============================================================================
def download_video(
    video_id:   str,
    output_dir: str,
    browser:    str = "firefox",
) -> str | None:
    output_path = os.path.join(output_dir, f"{video_id}.mp4")

    if os.path.exists(output_path):
        if _is_valid_mp4(output_path):
            return output_path
        else:
            logger.warning(f"{video_id} corrompido — removendo e rebaixando.")
            os.remove(output_path)

    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        subprocess.run([
            "yt-dlp",
            "-f", "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]",
            "--merge-output-format", "mp4",
            "--cookies-from-browser", browser,
            "--no-write-info-json",
            "--no-write-thumbnail",
            "--no-playlist",
            "--retries", "5",
            "--fragment-retries", "5",
            "-o", output_path,
            url,
        ], check=True)
        return output_path if os.path.exists(output_path) else None
    except subprocess.CalledProcessError as e:
        logger.error(f"Falha ao baixar {video_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Erro inesperado ao baixar {video_id}: {e}")
        return None


def _is_valid_mp4(path: str) -> bool:
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ], capture_output=True, text=True, timeout=15)
        if result.returncode != 0 or not result.stdout.strip():
            return False
        if os.path.getsize(path) < 10_000:
            return False
        return True
    except Exception:
        return False


# ==============================================================================
# Detecção de cenas
# ==============================================================================
def detect_scenes(video_path: str, threshold: float = 30.0) -> list[tuple[float, float]]:
    video         = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    return [
        (scene[0].get_seconds(), scene[1].get_seconds())
        for scene in scene_manager.get_scene_list()
    ]


# ==============================================================================
# Conectar ao Elasticsearch
# ==============================================================================
def connect_elasticsearch(
    host:    str = "http://localhost:9200",
    timeout: int = 30,
) -> Elasticsearch:
    es = Elasticsearch(host, request_timeout=timeout)
    try:
        info = es.info()
        print(f"Conectado ao Elasticsearch {info['version']['number']} em {host}")
        return es
    except Exception as e:
        raise ValueError(f"Erro ao conectar ao Elasticsearch em '{host}': {e}") from e


# ==============================================================================
# Criar índice com mapeamento HNSW
# ==============================================================================
def create_index(es, index_name="video_index", dims=512):
    if es.indices.exists(index=index_name):
        print(f"Índice '{index_name}' já existe — nenhuma ação necessária.")
        return

    settings = {
        "analysis": {
            "analyzer": {
                "video_text_analyzer": {
                    "type": "standard",
                    "stopwords": "_english_"
                }
            }
        }
    }

    mapping = {
        "mappings": {
            "properties": {
                "video_id":           {"type": "keyword"},
                "title":              {"type": "text", "analyzer": "video_text_analyzer"},
                "scene_index":        {"type": "integer"},
                "part_index":         {"type": "integer"},
                "timestamp_sec":      {"type": "float"},
                "center_frame":       {"type": "integer"},
                "modality":           {"type": "keyword"},
                "feature_desc":       {"type": "text", "analyzer": "video_text_analyzer"},
                "keywords":           {"type": "text", "analyzer": "video_text_analyzer"},
                "feature_categorias": {"type": "keyword"},
                "description":        {"type": "text",    "analyzer": "video_text_analyzer"},
                "transcript":         {"type": "text",    "analyzer": "video_text_analyzer"},
                "upload_date":        {"type": "date",    "format": "basic_date"},
                "duration_sec":       {"type": "integer"},
                "view_count":         {"type": "long"},
                "like_count":         {"type": "long"},
                "channel":            {"type": "keyword"},
                "tags":               {"type": "keyword"},
                "categories":         {"type": "keyword"},
                "feature_thumb": {
                    "type": "dense_vector", "dims": dims, "index": True,
                    "similarity": "cosine",
                    "index_options": {"type": "hnsw", "m": 32, "ef_construction": 200}
                },
                "embedding": {
                    "type": "dense_vector", "dims": dims, "index": True,
                    "similarity": "cosine",
                    "index_options": {"type": "hnsw", "m": 32, "ef_construction": 200}
                }
            }
        }
    }

    es.indices.create(index=index_name, settings=settings, body=mapping)
    print(f"Índice '{index_name}' criado com suporte a texto.")


# ==============================================================================
# Indexar embeddings em bulk
# ==============================================================================
def index_embeddings_bulk(
    es,
    embeddings:         list[dict],
    index_name:         str               = "video_index",
    video_id:           str               = "video_1",
    title:              str               = "",
    feature_categorias: str               = "",
    feature_desc:       str               = "",
    keywords:           str               = "",
    feature_thumb:      np.ndarray | None = None,
    modality:           str               = "video",
    description:        str               = "",
    transcript:         str               = "",
    upload_date:        str               = "",
    duration_sec:       int               = 0,
    view_count:         int               = 0,
    like_count:         int               = 0,
    channel:            str               = "",
    tags:               list              = [],
    categories:         list              = [],
) -> None:
    if not embeddings:
        print(f"[WARN] Nenhum embedding para indexar (video_id={video_id}).")
        return

    thumb_list = (
        feature_thumb.tolist()
        if isinstance(feature_thumb, np.ndarray)
        else feature_thumb
    )

    def generate_actions():
        for item in embeddings:
            vector = (
                item["embedding"].tolist()
                if isinstance(item["embedding"], np.ndarray)
                else item["embedding"]
            )
            doc_id = f"{video_id}_{modality}_s{item.get('scene_index', 0)}_p{item['part_index']}"

            source = {
                "video_id":           video_id,
                "title":              title,
                "scene_index":        item.get("scene_index", 0),
                "part_index":         item["part_index"],
                "timestamp_sec":      item["timestamp_sec"],
                "center_frame":       item.get("center_frame", -1),
                "modality":           modality,
                "embedding":          vector,
                "feature_categorias": feature_categorias,
                "feature_desc":       feature_desc,
                "keywords":           keywords,
                "description":        description,
                "transcript":         transcript,
                "upload_date":        upload_date,
                "duration_sec":       duration_sec,
                "view_count":         view_count,
                "like_count":         like_count,
                "channel":            channel,
                "tags":               tags,
                "categories":         categories,
            }

            if thumb_list is not None and modality == "video":
                source["feature_thumb"] = thumb_list

            yield {"_index": index_name, "_id": doc_id, "_source": source}

    success, errors = helpers.bulk(
        es,
        generate_actions(),
        chunk_size=500,
        request_timeout=60,
        raise_on_error=False,
        stats_only=False,
    )

    print(f"Indexados {success} documentos ({modality}) para '{video_id}'.")
    if errors:
        print(f"[WARN] {len(errors)} erro(s) durante a indexação:")
        for err in errors[:5]:
            print(f"  {err}")


# ==============================================================================
# Atualizar feature_desc
# ==============================================================================
def update_feature_desc(es, video_id: str, description: str, index_name: str = "video_index") -> None:
    es.update_by_query(
        index=index_name,
        body={
            "script": {"source": "ctx._source.feature_desc = params.desc", "params": {"desc": description}},
            "query":  {"term": {"video_id": video_id}},
        },
        wait_for_completion=True,
    )
    logger.info(f"feature_desc atualizado para {video_id}.")


# ==============================================================================
# Atualizar title
# ==============================================================================
def update_title(es, video_id: str, title: str, index_name: str = "video_index") -> None:
    es.update_by_query(
        index=index_name,
        body={
            "script": {"source": "ctx._source.title = params.title", "params": {"title": title}},
            "query":  {"term": {"video_id": video_id}},
        },
        wait_for_completion=True,
    )
    logger.info(f"title atualizado para {video_id}: '{title}'")


# ==============================================================================
# Deletar índice
# ==============================================================================
def delete_index(es, index_name: str = "video_index") -> None:
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"Índice '{index_name}' deletado.")
    else:
        print(f"Índice '{index_name}' não existe — nenhuma ação.")


# ==============================================================================
# Verificar se vídeo já está indexado
# MELHORIA: usa aggregation — 1 roundtrip ao ES em vez de 2 queries separadas
# ==============================================================================
def already_indexed(es, video_id: str, index_name: str = "video_index") -> bool:
    try:
        res = es.search(
            index=index_name,
            size=0,
            query={"term": {"video_id": video_id}},
            aggs={"by_modality": {"terms": {"field": "modality", "size": 10}}},
        )
        buckets = {b["key"] for b in res["aggregations"]["by_modality"]["buckets"]}
        return {"video", "audio"}.issubset(buckets)
    except Exception as e:
        logger.warning(f"Erro ao verificar indexação de {video_id}: {e}")
        return False


# ==============================================================================
# Processar um único vídeo
# ==============================================================================
def process_video(
    video_path:      str,
    video_id:        str,
    clip_model,
    clip_preprocess,
    clap_model,
    device,
    es,
    taxonomy_lookup:        dict[str, str] = {},
    label:                  str            = "",
    title:                  str            = "",
    max_frames_per_segment: int            = 45,
    segment_duration:       float          = 1.5,
    embeddings_dir:         str            = "./data/embeddings",
) -> None:
    import gc

    logger.info(f"Processando {video_id}...")

    os.makedirs(embeddings_dir, exist_ok=True)

    video_json_path = os.path.join(embeddings_dir, f"{video_id}_video.json")
    audio_json_path = os.path.join(embeddings_dir, f"{video_id}_audio.json")

    feature_categorias = get_feature_categorias(label, taxonomy_lookup)

    meta       = fetch_video_metadata(video_id)
    title      = title or meta.get("title", "")
    transcript = fetch_subtitles_en(video_id)
    feature_desc, keywords = build_text_metadata(label, title, taxonomy_lookup)

    extra_meta = {
        "description":  meta.get("description", ""),
        "transcript":   transcript,
        "upload_date":  meta.get("upload_date", ""),
        "duration_sec": meta.get("duration", 0),
        "view_count":   meta.get("view_count", 0),
        "like_count":   meta.get("like_count", 0),
        "channel":      meta.get("channel", ""),
        "tags":         meta.get("tags", []),
        "categories":   meta.get("categories", []),
    }

    # ── Caso 1: embeddings já existem no disco ────────────────────────────
    if os.path.exists(video_json_path) and os.path.exists(audio_json_path):
        logger.info(f"{video_id}: embeddings encontrados no disco — carregando para indexar.")

        with open(video_json_path) as f:
            all_video_embs = json.load(f)
        with open(audio_json_path) as f:
            audio_embs = json.load(f)

        for item in all_video_embs:
            if isinstance(item["embedding"], list):
                item["embedding"] = np.array(item["embedding"], dtype=np.float32)
        for item in audio_embs:
            if isinstance(item["embedding"], list):
                item["embedding"] = np.array(item["embedding"], dtype=np.float32)

        feature_thumb_vec = fetch_thumbnail_embedding(video_id, clip_model, clip_preprocess, device)

        logger.info(f"{video_id}: {len(all_video_embs)} emb. vídeo | {len(audio_embs)} emb. áudio")

        index_embeddings_bulk(
            es, all_video_embs, index_name="video_index",
            video_id=video_id, title=title,
            feature_categorias=feature_categorias,
            feature_desc=feature_desc, keywords=keywords,
            feature_thumb=feature_thumb_vec, modality="video",
            **extra_meta,
        )
        index_embeddings_bulk(
            es, audio_embs, index_name="video_index",
            video_id=video_id, title=title,
            feature_categorias=feature_categorias,
            feature_desc=feature_desc, keywords=keywords,
            feature_thumb=feature_thumb_vec, modality="audio",
            **extra_meta,
        )
        logger.info(f"Indexado (cache): {video_id} (vídeo + áudio)")
        return

    # ── Caso 2: extrai do vídeo ───────────────────────────────────────────
    feature_thumb_vec = fetch_thumbnail_embedding(video_id, clip_model, clip_preprocess, device)

    if feature_thumb_vec is not None:
        logger.info(f"Thumbnail embutida para {video_id}.")
    else:
        logger.warning(f"Thumbnail indisponível para {video_id}.")

    try:
        scenes = detect_scenes(video_path)
    except Exception as e:
        logger.warning(f"Falha na detecção de cenas de {video_id}: {e}. Usando vídeo inteiro.")
        scenes = []

    if not scenes:
        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        scenes = [(0.0, total / fps)]

    logger.info(f"Cenas detectadas: {len(scenes)}")

    all_segments = []
    for scene_idx, (start, end) in enumerate(scenes):
        if end - start < 1.0:
            continue
        try:
            segments = ky.split_scene_into_segments(
                video_path,
                start_time=start,
                end_time=end,
                max_frames_per_segment=max_frames_per_segment,
            )
        except Exception as e:
            logger.warning(f"Erro ao segmentar cena {scene_idx} [{start:.2f}s–{end:.2f}s]: {e}")
            continue

        for seg in segments:
            seg["scene_index"]   = scene_idx
            seg["segment_index"] = seg.get("segment_index", 0)
            all_segments.append(seg)

    if not all_segments:
        logger.warning(f"Nenhum segmento gerado para {video_id}.")
        return

    all_video_embs = []
    for seg in all_segments:
        try:
            vector = emb.embed_window(
                seg["frames"], clip_model, clip_preprocess, device, method="mean",
            )
            all_video_embs.append({
                "scene_index":   seg["scene_index"],
                "part_index":    seg["segment_index"],
                "timestamp_sec": seg["timestamp_sec"],
                "center_frame":  seg["center_frame"],
                "embedding":     vector,
            })
        except Exception as e:
            logger.warning(f"Erro ao embeddar segmento cena {seg['scene_index']}: {e}")

    if all_video_embs:
        logger.info(f"Embeddings de vídeo gerados: {len(all_video_embs)} ({len(scenes)} cenas)")
        emb.save_embeddings_json(all_video_embs, path=video_json_path)
        index_embeddings_bulk(
            es, all_video_embs, index_name="video_index",
            video_id=video_id, title=title,
            feature_categorias=feature_categorias,
            feature_desc=feature_desc, keywords=keywords,
            feature_thumb=feature_thumb_vec, modality="video",
            **extra_meta,
        )

    try:
        audio_embs = emb.generate_audio_embeddings_from_segments(
            video_path=video_path,
            segments=all_segments,
            clap_model=clap_model,
            device=device,
            segment_duration=segment_duration,
        )

        if len(audio_embs) == len(all_segments):
            for ae, seg in zip(audio_embs, all_segments):
                ae["timestamp_sec"] = seg["timestamp_sec"]
                ae["scene_index"]   = seg["scene_index"]
                ae["part_index"]    = seg.get("segment_index", 0)
                ae["center_frame"]  = -1
        else:
            logger.warning(
                f"Tamanhos de audio_embs ({len(audio_embs)}) e all_segments ({len(all_segments)}) diferem."
            )
            for i, ae in enumerate(audio_embs):
                seg = all_segments[i] if i < len(all_segments) else None
                ae["timestamp_sec"] = seg["timestamp_sec"] if seg else 0.0
                ae["scene_index"]   = seg["scene_index"]   if seg else -1
                ae["part_index"]    = seg.get("segment_index", i) if seg else i
                ae["center_frame"]  = -1

        logger.info(f"Embeddings de áudio gerados: {len(audio_embs)}")
        if audio_embs:
            emb.save_embeddings_json(audio_embs, path=audio_json_path)
            index_embeddings_bulk(
                es, audio_embs, index_name="video_index",
                video_id=video_id, title=title,
                feature_categorias=feature_categorias,
                feature_desc=feature_desc, keywords=keywords,
                feature_thumb=feature_thumb_vec, modality="audio",
                **extra_meta,
            )
    except Exception as e:
        logger.warning(f"Falha ao processar áudio para {video_id}: {e}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Indexado: {video_id} (vídeo + áudio)")


# ==============================================================================
# Processar todos os vídeos locais
# MELHORIA: ThreadPoolExecutor para paralelizar metadados/download enquanto
#           o embedding na GPU permanece sequencial (evita contenção de VRAM).
# ==============================================================================
def process_local_videos(
    video_dir:       str,
    clip_model,
    clip_preprocess,
    clap_model,
    device,
    es,
    taxonomy_lookup: dict[str, str] = {},
    anet_database:   dict           = {},
    max_workers:     int            = 4,
) -> None:
    filenames = [f for f in sorted(os.listdir(video_dir)) if f.endswith(".mp4")]

    if not filenames:
        logger.warning(f"Nenhum arquivo .mp4 encontrado em '{video_dir}'.")
        return

    # Pré-filtra vídeos já indexados em paralelo (1 roundtrip por vídeo, mas I/O bound)
    def check_indexed(filename: str) -> tuple[str, bool]:
        video_id = filename.replace(".mp4", "")
        return video_id, already_indexed(es, video_id)

    logger.info(f"Verificando {len(filenames)} vídeos no índice...")
    pending: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_indexed, f): f for f in filenames}
        for future in as_completed(futures):
            video_id, indexed = future.result()
            if indexed:
                logger.info(f"{video_id} já indexado — pulando.")
            else:
                pending.append(video_id)

    logger.info(f"{len(pending)} vídeo(s) para processar.")

    # Embedding e indexação permanecem sequenciais (GPU não é thread-safe)
    for video_id in sorted(pending):
        video_path = os.path.join(video_dir, f"{video_id}.mp4")

        label = ""
        title = ""
        if video_id in anet_database:
            entry       = anet_database[video_id]
            annotations = entry.get("annotations", [])
            if annotations:
                label = annotations[0].get("label", "")
            title = entry.get("url", "")

        try:
            process_video(
                video_path, video_id,
                clip_model, clip_preprocess, clap_model, device, es,
                taxonomy_lookup=taxonomy_lookup, label=label, title=title,
            )
        except Exception as e:
            logger.error(f"Erro fatal no vídeo {video_id}: {e}")


# ==============================================================================
# Contagem de vídeos locais
# ==============================================================================
def count_videos(video_dir: str) -> int:
    if not os.path.isdir(video_dir):
        return 0
    return sum(1 for f in os.listdir(video_dir) if f.endswith(".mp4"))


# ==============================================================================
# Índices aleatórios fixos
# ==============================================================================
def get_fixed_random_indices(
    n_samples: int = 10,
    max_range: int = 1000,
    seed:      int = 42,
) -> list[int]:
    random.seed(seed)
    return sorted(random.sample(range(max_range), n_samples))