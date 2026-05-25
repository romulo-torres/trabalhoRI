import io
import json
import os
import random
import subprocess
import urllib.request
import cv2

import numpy as np
import torch
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
# Título do vídeo via yt-dlp (sem baixar o vídeo)
# ==============================================================================
def fetch_video_title(video_id: str) -> str:
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        result = subprocess.run(
            ["yt-dlp", "--get-title", "--no-playlist", url],
            capture_output=True,
            text=True,
            timeout=15,
        )
        title = result.stdout.strip()
        if title:
            logger.info(f"Título obtido para {video_id}: '{title}'")
        return title
    except Exception as e:
        logger.warning(f"Não foi possível obter título para {video_id}: {e}")
        return ""


# ==============================================================================
# Download de vídeo do YouTube via yt-dlp (com cookies do navegador)
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
# Processar um único vídeo
# Lógica de cache:
#   1. Se embeddings JSON já existem no disco → carrega direto (pula extração)
#   2. Se não existem → extrai dos frames/áudio e salva
#   3. Em ambos os casos → indexa no ES (se ainda não indexado)
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
    feature_desc, keywords = build_text_metadata(label, title, taxonomy_lookup)

    # ── Caso 1: embeddings já existem no disco ────────────────────────────
    if os.path.exists(video_json_path) and os.path.exists(audio_json_path):
        logger.info(f"{video_id}: embeddings encontrados no disco — carregando para indexar.")

        with open(video_json_path) as f:
            all_video_embs = json.load(f)
        with open(audio_json_path) as f:
            audio_embs = json.load(f)

        # Converte listas de volta para numpy (necessário para index_embeddings_bulk)
        for item in all_video_embs:
            if isinstance(item["embedding"], list):
                item["embedding"] = np.array(item["embedding"], dtype=np.float32)
        for item in audio_embs:
            if isinstance(item["embedding"], list):
                item["embedding"] = np.array(item["embedding"], dtype=np.float32)

        # Thumbnail ainda precisa ser gerada (não é salva em disco)
        feature_thumb_vec = fetch_thumbnail_embedding(video_id, clip_model, clip_preprocess, device)

        if not title:
            title = fetch_video_title(video_id)
            feature_desc, keywords = build_text_metadata(label, title, taxonomy_lookup)

        logger.info(f"{video_id}: {len(all_video_embs)} emb. vídeo | {len(audio_embs)} emb. áudio")

        index_embeddings_bulk(
            es, all_video_embs, index_name="video_index",
            video_id=video_id, title=title,
            feature_categorias=feature_categorias,
            feature_desc=feature_desc, keywords=keywords,
            feature_thumb=feature_thumb_vec, modality="video",
        )
        index_embeddings_bulk(
            es, audio_embs, index_name="video_index",
            video_id=video_id, title=title,
            feature_categorias=feature_categorias,
            feature_desc=feature_desc, keywords=keywords,
            feature_thumb=feature_thumb_vec, modality="audio",
        )
        logger.info(f"Indexado (cache): {video_id} (vídeo + áudio)")
        return

    # ── Caso 2: extrai do vídeo ───────────────────────────────────────────
    feature_thumb_vec = fetch_thumbnail_embedding(video_id, clip_model, clip_preprocess, device)

    if not title:
        title = fetch_video_title(video_id)
        feature_desc, keywords = build_text_metadata(label, title, taxonomy_lookup)

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

    # Segmentação
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

    # Embeddings de vídeo (CLIP)
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
        )

    # Embeddings de áudio (CLAP)
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
            )
    except Exception as e:
        logger.warning(f"Falha ao processar áudio para {video_id}: {e}")

    # Libera memória após cada vídeo
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Indexado: {video_id} (vídeo + áudio)")


# ==============================================================================
# Verificar se vídeo já está indexado
# ==============================================================================
def already_indexed(es, video_id: str) -> bool:
    res_video = es.search(
        index="video_index",
        query={"bool": {"must": [
            {"term": {"video_id": video_id}},
            {"term": {"modality": "video"}}
        ]}},
        size=1,
    )
    has_video = len(res_video["hits"]["hits"]) > 0

    res_audio = es.search(
        index="video_index",
        query={"bool": {"must": [
            {"term": {"video_id": video_id}},
            {"term": {"modality": "audio"}}
        ]}},
        size=1,
    )
    has_audio = len(res_audio["hits"]["hits"]) > 0

    return has_video and has_audio


# ==============================================================================
# Processar todos os vídeos locais
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
) -> None:
    for filename in sorted(os.listdir(video_dir)):
        if not filename.endswith(".mp4"):
            continue

        video_id   = filename.replace(".mp4", "")
        video_path = os.path.join(video_dir, filename)

        if already_indexed(es, video_id):
            logger.info(f"{video_id} já indexado — pulando.")
            continue

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