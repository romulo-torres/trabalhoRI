import os
import subprocess
import random
import numpy as np
import json

from elasticsearch import Elasticsearch, helpers

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
import urllib.request

import keyframes as ky
import embeddings as emb
from logger import setup_logger

# Compatibilidade com versões antigas do NumPy
np.float_ = np.float64

logger = setup_logger()


def ensure_activitynet_json(json_path: str):
    if os.path.exists(json_path):
        print(f"Arquivo já existe: {json_path}")
        return

    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    url = "https://storage.googleapis.com/activitynet/annotations/activity_net.v1-3.min.json"

    print("Baixando ActivityNet JSON com wget...")

    try:
        subprocess.run([
            "wget",
            "-O", json_path,
            url
        ], check=True)
    except Exception as e:
        raise RuntimeError("Falha ao baixar ActivityNet com wget") from e

    print("Download concluído!")

def download_video(video_id, output_dir):
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_path = os.path.join(output_dir, f"{video_id}.mp4")

    if os.path.exists(output_path):
        return output_path

    try:
        subprocess.run([
            "yt-dlp",
            "-f", "mp4",
            "-o", output_path,
            url
        ], check=True)

        return output_path

    except Exception:
        return None


def load_activitynet(json_path):
    with open(json_path) as f:
        data = json.load(f)

    return data["database"]

# ==============================================================================
# Detecção de cenas
# ==============================================================================
def detect_scenes(video_path: str, threshold: float = 30.0) -> list[tuple[float, float]]:
    """Retorna lista de (start_sec, end_sec) para cada cena detectada."""
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
def create_index(
    es,
    index_name: str = "video_index",
    dims:       int = 512,
) -> None:
    if es.indices.exists(index=index_name):
        print(f"Índice '{index_name}' já existe — nenhuma ação necessária.")
        return

    mapping = {
        "mappings": {
            "properties": {
                "video_id":      {"type": "keyword"},
                "timestamp_sec": {"type": "float"},
                "center_frame":  {"type": "integer"},
                "embedding": {
                    "type":        "dense_vector",
                    "dims":        dims,
                    "index":       True,
                    "similarity":  "cosine",
                    "index_options": {
                        "type":            "hnsw",
                        "m":               32,
                        "ef_construction": 200,
                    },
                },
            }
        }
    }

    es.indices.create(index=index_name, body=mapping)
    print(f"Índice '{index_name}' criado com {dims} dimensões.")


# ==============================================================================
# Indexar embeddings em bulk
# ==============================================================================
def index_embeddings_bulk(
    es,
    embeddings: list[dict],
    index_name: str = "video_index",
    video_id:   str = "video_1",
) -> None:
    if not embeddings:
        print(f"[WARN] Nenhum embedding para indexar (video_id={video_id}).")
        return

    def generate_actions():
        for item in embeddings:
            vector = (
                item["embedding"].tolist()
                if isinstance(item["embedding"], np.ndarray)
                else item["embedding"]
            )
            yield {
                "_index": index_name,
                "_id":    f"{video_id}_{item['center_frame']}",
                "_source": {
                    "video_id":      video_id,
                    "timestamp_sec": item["timestamp_sec"],
                    "center_frame":  item["center_frame"],
                    "embedding":     vector,
                },
            }

    success, errors = helpers.bulk(
        es,
        generate_actions(),
        chunk_size=500,
        request_timeout=60,
        raise_on_error=False,
        stats_only=False,
    )

    print(f"Indexados {success} documentos para '{video_id}'.")
    if errors:
        print(f"[WARN] {len(errors)} erro(s) durante a indexação:")
        for err in errors[:5]:
            print(f"  {err}")


# ==============================================================================
# Busca por similaridade (kNN)
# ==============================================================================
def search_similar(
    es,
    query_embedding: np.ndarray,
    index_name:      str        = "video_index",
    video_id:        str | None = None,
    k:               int        = 10,
) -> list[dict]:
    norm = np.linalg.norm(query_embedding)
    if norm == 0:
        raise ValueError("query_embedding é um vetor nulo — impossível normalizar.")
    query_vector = (query_embedding / norm).tolist()

    body: dict = {
        "knn": {
            "field":          "embedding",
            "query_vector":   query_vector,
            "k":              k,
            "num_candidates": k * 10,
        },
        "size": k,
    }

    if video_id is not None:
        body["knn"]["filter"] = {"term": {"video_id": video_id}}

    response = es.search(index=index_name, body=body)

    return [
        {
            "score":         hit["_score"],
            "video_id":      hit["_source"]["video_id"],
            "timestamp_sec": hit["_source"]["timestamp_sec"],
            "center_frame":  hit["_source"]["center_frame"],
        }
        for hit in response["hits"]["hits"]
    ]


# ==============================================================================
# Deletar índice (utilitário)
# ==============================================================================
def delete_index(es, index_name: str = "video_index") -> None:
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"Índice '{index_name}' deletado.")
    else:
        print(f"Índice '{index_name}' não existe — nenhuma ação.")


# ==============================================================================
# Processar um único vídeo
# ==============================================================================
def process_video(
    video_path: str,
    video_id:   str,
    model,
    preprocess,
    device,
    es,
) -> None:
    logger.info(f"🎬 Processando {video_id}...")

    try:
        scenes = detect_scenes(video_path)
    except Exception as e:
        logger.warning(f"Falha na detecção de cenas de {video_id}: {e}. Usando vídeo inteiro.")
        scenes = []

    logger.info(f"Cenas detectadas: {len(scenes)}")

    windows = []
    for start, end in scenes:
        if end - start < 1.0:
            continue
        try:
            windows.extend(ky.generate_windows_stream_centered(
                video_path, start_time=start, end_time=end, k_seconds=0.5,
            ))
        except Exception as e:
            logger.warning(f"Erro ao gerar janela [{start:.2f}s – {end:.2f}s]: {e}")

    if not windows:
        logger.warning("Nenhuma janela por cena — fallback para janelas fixas no vídeo inteiro.")
        try:
            windows = ky.generate_windows_stream_centered(video_path, k_seconds=0.5)
        except Exception as e:
            logger.error(f"Fallback também falhou para {video_id}: {e}")
            return

    if not windows:
        logger.error(f"Nenhuma janela gerada para {video_id}. Pulando.")
        return

    logger.info(f"Janelas geradas: {len(windows)}")

    embeddings = emb.generate_embeddings(windows, model, preprocess, device)

    emb_dir = "./data/embeddings"
    os.makedirs(emb_dir, exist_ok=True)
    emb.save_embeddings_json(embeddings, path=os.path.join(emb_dir, f"{video_id}.json"))
    logger.info(f"Embeddings gerados: {len(embeddings)}")

    index_embeddings_bulk(es, embeddings, index_name="video_index", video_id=video_id)
    logger.info(f"✅ Indexado: {video_id}")

# ==============================================================================
# Verificar se vídeo já está indexado
# ==============================================================================
def already_indexed(es, video_id: str) -> bool:
    res = es.search(
        index="video_index",
        query={"term": {"video_id": video_id}},
        size=1,
    )
    return len(res["hits"]["hits"]) > 0


# ==============================================================================
# Processar todos os vídeos locais
# ==============================================================================
def process_local_videos(video_dir: str, model, preprocess, device, es) -> None:
    for filename in sorted(os.listdir(video_dir)):
        if not filename.endswith(".mp4"):
            continue

        video_id   = filename.replace(".mp4", "")
        video_path = os.path.join(video_dir, filename)

        if already_indexed(es, video_id):
            logger.info(f"{video_id} já indexado — pulando.")
            continue

        try:
            process_video(video_path, video_id, model, preprocess, device, es)
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
# Índices aleatórios fixos (utilitário de amostragem)
# ==============================================================================
def get_fixed_random_indices(
    n_samples: int = 10,
    max_range: int = 1000,
    seed:      int = 42,
) -> list[int]:
    random.seed(seed)
    return sorted(random.sample(range(max_range), n_samples))