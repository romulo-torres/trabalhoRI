import cv2
import json
from collections import defaultdict
import numpy as np
import os

import index_elastic as ind
from embeddings import embed_frame
from logger import setup_logger

logger = setup_logger()


# ==============================================================================
# 1. Buscar por frame (array NumPy BGR)
# ==============================================================================
def search_by_frame(
    es,
    frame:       np.ndarray,
    model,
    preprocess,
    device:      str,
    index_name:  str       = "video_index",
    video_id:    str | None = None,
    k:           int        = 5,
) -> list[dict]:
    """
    Gera o embedding do frame e busca os k vizinhos mais próximos no índice.
    """
    query_embedding = embed_frame(frame, model, preprocess, device)
    return ind.search_similar(
        es,
        query_embedding,
        index_name=index_name,
        video_id=video_id,
        k=k,
    )


# ==============================================================================
# 2. Buscar por caminho de imagem
# ==============================================================================
def search_by_image_path(
    es,
    image_path:  str,
    model,
    preprocess,
    device:      str,
    index_name:  str       = "video_index",
    video_id:    str | None = None,
    k:           int        = 5,
) -> list[dict]:
    """
    Lê uma imagem do disco e delega para search_by_frame.
    Lança ValueError se o arquivo não puder ser aberto pelo OpenCV.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(
            f"Não foi possível carregar a imagem: '{image_path}'. "
            "Verifique se o caminho existe e o formato é suportado pelo OpenCV."
        )
    return search_by_frame(
        es, frame, model, preprocess, device,
        index_name=index_name, video_id=video_id, k=k,
    )


# ==============================================================================
# 3. Buscar vídeo mais relevante a partir de múltiplos embeddings de consulta
#
# Estratégia:
#   - Para cada embedding da query, busca os top-50 candidatos no ES.
#   - Limita a max_hits_per_video contribuições por vídeo (evita que um vídeo
#     longo domine só por ter mais janelas indexadas).
#   - Agrega por média de score e retorna os top_k vídeos.
# ==============================================================================
def search_video(
    es,
    query_embeddings:     list[dict],
    top_k:                int   = 10,
    candidates_per_query: int   = 50,
    threshold:            float = 0.2,
    use_order_bonus:      bool  = False,
) -> list[tuple[str, float]]:

    best_scores  = defaultdict(list)
    used_frames  = defaultdict(set)
    timestamps   = defaultdict(list)   # só se use_order_bonus=True

    N = len(query_embeddings)
    if N == 0:
        return []

    for item in query_embeddings:
        vector = item["embedding"]
        if isinstance(vector, np.ndarray):
            norm = np.linalg.norm(vector)
            if norm == 0:
                continue
            vector = (vector / norm).tolist()

        try:
            results = es.search(
                index="video_index",
                body={
                    "knn": {
                        "field":          "embedding",
                        "query_vector":   vector,
                        "k":              candidates_per_query,
                        "num_candidates": max(candidates_per_query * 20, 1000),
                    },
                    "size": candidates_per_query,
                },
            )
        except Exception as e:
            logger.error(f"Erro na busca kNN: {e}")
            continue

        # Melhor hit por vídeo para este q_i
        best_per_video: dict[str, tuple[float, int, float]] = {}
        for hit in results["hits"]["hits"]:
            vid       = hit["_source"]["video_id"]
            raw_score = hit["_score"]
            cosine    = 2 * raw_score - 1          # converte para cosine real
            center    = hit["_source"]["center_frame"]
            ts        = hit["_source"]["timestamp_sec"]

            if cosine < threshold:
                continue

            if vid not in best_per_video or cosine > best_per_video[vid][0]:
                best_per_video[vid] = (cosine, center, ts)

        for vid, (cosine, center, ts) in best_per_video.items():
            if center not in used_frames[vid]:
                best_scores[vid].append(cosine)
                used_frames[vid].add(center)
                if use_order_bonus:
                    timestamps[vid].append(ts)
            # sem else: não penaliza duplicata — só ignora

    if not best_scores:
        return []

    final_scores = {}
    for vid, scores in best_scores.items():
        base = sum(scores) / N

        if use_order_bonus and len(timestamps[vid]) >= 2:
            ts_list = timestamps[vid]
            ordered = sum(a <= b for a, b in zip(ts_list, ts_list[1:]))
            bonus   = 0.7 + 0.3 * (ordered / (len(ts_list) - 1))
            base   *= bonus

        final_scores[vid] = base

    return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]



### =========================================================================
### Essas duas funções aqui abaixo são ideias do gpt pra fazer a busca de videos (ideia maluca que provavelmente vai ser tirada)
### =========================================================================

def load_embeddings_matrix(path: str) -> tuple[np.ndarray, list[dict]]:
    """
    Lê o JSON de embeddings e retorna:
    - matrix: np.ndarray shape (n_frames, 512), normalizada
    - metadata: lista de dicts com center_frame e timestamp_sec
    """
    with open(path, "r") as f:
        data = json.load(f)

    matrix = np.array([item["embedding"] for item in data], dtype=np.float32)
    
    # Normaliza cada vetor (necessário para dot product = cosine similarity)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)   # evita divisão por zero
    matrix = matrix / norms

    metadata = [{"center_frame": d["center_frame"], "timestamp_sec": d["timestamp_sec"]} for d in data]
    return matrix, metadata


def chamfer_score(query_matrix: np.ndarray, candidate_matrix: np.ndarray) -> float:
    """
    Chamfer similarity entre dois conjuntos de embeddings.
    
    Para cada vetor da query, acha o mais similar no candidato.
    Score = média desses máximos, dividido pelo N efetivo
    (apenas matches acima do threshold contam no denominador).
    
    Retorna valor entre 0 e 1.
    """
    THRESHOLD = 0.2

    # Matriz de similaridade coseno: shape (N_query, N_candidate)
    # Como os vetores já estão normalizados, dot product = cosine similarity
    sim_matrix = query_matrix @ candidate_matrix.T   # (N_q, N_c)

    # Para cada vetor da query: score do vizinho mais próximo no candidato
    max_scores = sim_matrix.max(axis=1)   # shape (N_q,)

    # Só conta os matches acima do threshold
    valid = max_scores[max_scores > THRESHOLD]

    if len(valid) == 0:
        return 0.0

    return float(valid.mean())


def search_by_embeddings(
    query_embeddings_path: str,
    index_dir:             str   = "./data/embeddings",
    top_k:                 int   = 10,
) -> list[tuple[str, float]]:
    """
    Compara o vídeo de consulta contra todos os vídeos indexados localmente.

    query_embeddings_path — JSON gerado por save_embeddings_json para o vídeo de entrada
    index_dir             — diretório com os JSONs de todos os vídeos indexados
    """
    query_matrix, _ = load_embeddings_matrix(query_embeddings_path)

    scores = []

    for filename in sorted(os.listdir(index_dir)):
        if not filename.endswith(".json"):
            continue

        video_id        = filename.replace(".json", "")
        candidate_path  = os.path.join(index_dir, filename)

        # Não compara o vídeo consigo mesmo
        query_id = os.path.splitext(os.path.basename(query_embeddings_path))[0]
        if video_id == query_id:
            continue

        try:
            candidate_matrix, _ = load_embeddings_matrix(candidate_path)
            score = chamfer_score(query_matrix, candidate_matrix)
            scores.append((video_id, score))
        except Exception as e:
            print(f"[WARN] Erro ao processar {video_id}: {e}")

    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    return ranked[:top_k]