import cv2
import json
from collections import defaultdict
from math import inf
from typing import Optional

import numpy as np
import os

import index_elastic as ind
from embeddings import embed_frame
from logger import setup_logger

logger = setup_logger()


# ==============================================================================
# Utilitários internos
# ==============================================================================

def _normalize(vec) -> list[float]:
    """Normaliza um vetor para comprimento unitário (L2). Retorna lista."""
    v = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(v)
    return (v / norm).tolist() if norm > 0 else v.tolist()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Similaridade cosseno entre dois arrays 1-D."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ==============================================================================
# 1. Buscar por frame (array NumPy BGR)
# ==============================================================================

def search_by_frame(
    es,
    frame:       np.ndarray,
    model,
    preprocess,
    device:      str,
    index_name:  str           = "video_index",
    video_id:    Optional[str] = None,
    k:           int           = 5,
) -> list[dict]:
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
    index_name:  str           = "video_index",
    video_id:    Optional[str] = None,
    k:           int           = 5,
) -> list[dict]:
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
# Melhorias:
#   - msearch: todas as queries em 1 roundtrip ao ES (era N roundtrips).
#   - Normalização L2 consistente antes de enviar ao ES.
#   - Threshold adaptativo por query (percentil 60 dos scores retornados).
#   - Agregação combina max + média ponderada por cobertura.
#   - order_bonus contínuo via Kendall tau (scipy) com fallback sem scipy.
# ==============================================================================

def search_video(
    es,
    query_embeddings:     list[dict],
    top_k:                int   = 10,
    candidates_per_query: int   = 50,
    threshold:            float = 0.2,
    use_order_bonus:      bool  = False,
    index_name:           str   = "video_index",
) -> list[tuple[str, float]]:

    N = len(query_embeddings)
    if N == 0:
        return []

    # MELHORIA: monta todas as queries e dispara em 1 roundtrip via msearch
    body = []
    vectors = []
    for item in query_embeddings:
        vector = _normalize(item["embedding"])
        vectors.append(vector)
        body.append({"index": index_name})
        body.append({
            "knn": {
                "field":          "embedding",
                "query_vector":   vector,
                "k":              candidates_per_query,
                "num_candidates": max(candidates_per_query * 20, 1000),
            },
            "size": candidates_per_query,
        })

    try:
        responses = es.msearch(body=body)
    except Exception as e:
        logger.error(f"Erro no msearch: {e}")
        return []

    best_scores  = defaultdict(list)
    used_frames  = defaultdict(set)
    timestamps   = defaultdict(list)

    for response in responses["responses"]:
        if "error" in response:
            logger.warning(f"Erro numa sub-query do msearch: {response['error']}")
            continue

        hits = response["hits"]["hits"]
        if not hits:
            continue

        # Threshold adaptativo por query
        raw_scores = [2 * h["_score"] - 1 for h in hits]
        adaptive_threshold = max(threshold, float(np.percentile(raw_scores, 60)))

        best_per_video: dict[str, tuple[float, int, float]] = {}
        for hit in hits:
            vid    = hit["_source"]["video_id"]
            cosine = 2 * hit["_score"] - 1
            center = hit["_source"]["center_frame"]
            ts     = hit["_source"]["timestamp_sec"]

            if cosine < adaptive_threshold:
                continue

            if vid not in best_per_video or cosine > best_per_video[vid][0]:
                best_per_video[vid] = (cosine, center, ts)

        for vid, (cosine, center, ts) in best_per_video.items():
            if center not in used_frames[vid]:
                best_scores[vid].append(cosine)
                used_frames[vid].add(center)
                if use_order_bonus:
                    timestamps[vid].append(ts)

    if not best_scores:
        return []

    final_scores: dict[str, float] = {}
    for vid, scores in best_scores.items():
        n_hits   = len(scores)
        coverage = n_hits / N
        base     = (
            max(scores) * 0.4
            + float(np.mean(scores)) * 0.6
        ) * (0.7 + 0.3 * coverage)

        if use_order_bonus and len(timestamps[vid]) >= 2:
            try:
                from scipy.stats import kendalltau
                tau, _ = kendalltau(range(len(timestamps[vid])), timestamps[vid])
                bonus  = 0.7 + 0.3 * ((tau + 1) / 2)
            except ImportError:
                ts_list = timestamps[vid]
                ordered = sum(a <= b for a, b in zip(ts_list, ts_list[1:]))
                bonus   = 0.7 + 0.3 * (ordered / (len(ts_list) - 1))
            base *= bonus

        final_scores[vid] = base

    return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ==============================================================================
# 4. MMR — Maximal Marginal Relevance
# ==============================================================================

def mmr_rerank(
    results:        list[tuple[str, float]],
    embeddings_map: dict[str, np.ndarray],
    lambda_:        float = 0.7,
    top_k:          int   = 10,
) -> list[tuple[str, float]]:
    """
    Re-ranqueia resultados penalizando vídeos redundantes entre si.
    embeddings_map: {video_id: np.ndarray} — embedding representativo por vídeo.
    lambda_: 1.0 = só relevância, 0.0 = só diversidade.
    """
    if not results:
        return []

    selected:   list[tuple[str, float]] = []
    candidates: list[tuple[str, float]] = list(results)

    while candidates and len(selected) < top_k:
        best_item:  Optional[tuple[str, float]] = None
        best_score: float = -inf

        for vid, rel in candidates:
            if not selected or vid not in embeddings_map:
                mmr_score = rel
            else:
                redundancy = max(
                    _cosine_sim(embeddings_map[vid], embeddings_map[s])
                    for s, _ in selected
                    if s in embeddings_map
                ) if any(s in embeddings_map for s, _ in selected) else 0.0
                mmr_score = lambda_ * rel - (1 - lambda_) * redundancy

            if mmr_score > best_score:
                best_score = mmr_score
                best_item  = (vid, rel)

        if best_item is None:
            break
        selected.append(best_item)
        candidates = [(v, s) for v, s in candidates if v != best_item[0]]

    return selected


# ==============================================================================
# 5. Carregar matriz de embeddings de um JSON
# ==============================================================================

def load_embeddings_matrix(path: str) -> tuple[np.ndarray, list[dict]]:
    with open(path, "r") as f:
        data = json.load(f)

    matrix = np.array([item["embedding"] for item in data], dtype=np.float32)
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms  = np.where(norms == 0, 1.0, norms)
    matrix = matrix / norms

    metadata = [
        {"center_frame": d["center_frame"], "timestamp_sec": d["timestamp_sec"]}
        for d in data
    ]
    return matrix, metadata


# ==============================================================================
# 6. Chamfer similarity
# ==============================================================================

def chamfer_score(query_matrix: np.ndarray, candidate_matrix: np.ndarray) -> float:
    """
    Chamfer similarity entre dois conjuntos de embeddings normalizados.
    Pondera pela cobertura (fração de queries com match acima do threshold).
    """
    THRESHOLD = 0.2

    sim_matrix = query_matrix @ candidate_matrix.T
    max_scores = sim_matrix.max(axis=1)

    valid = max_scores[max_scores > THRESHOLD]
    if len(valid) == 0:
        return 0.0

    coverage = len(valid) / len(max_scores)
    return float(valid.mean()) * (0.7 + 0.3 * coverage)


# ==============================================================================
# 7. Busca local por embeddings (Chamfer sobre arquivos JSON)
# Ideal como re-ranker sobre os top-K retornados pelo ES, não como busca primária.
# ==============================================================================

def search_by_embeddings(
    query_embeddings_path: str,
    index_dir:             str = "./data/embeddings",
    top_k:                 int = 10,
) -> list[tuple[str, float]]:
    query_matrix, _ = load_embeddings_matrix(query_embeddings_path)
    query_id        = os.path.splitext(os.path.basename(query_embeddings_path))[0]

    scores: list[tuple[str, float]] = []

    for filename in sorted(os.listdir(index_dir)):
        if not filename.endswith(".json"):
            continue

        video_id = filename.replace(".json", "")
        if video_id == query_id:
            continue

        candidate_path = os.path.join(index_dir, filename)
        try:
            candidate_matrix, _ = load_embeddings_matrix(candidate_path)
            score = chamfer_score(query_matrix, candidate_matrix)
            scores.append((video_id, score))
        except Exception as e:
            logger.warning(f"Erro ao processar {video_id}: {e}")

    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]


# ==============================================================================
# 8. Busca híbrida texto + vetor
# MELHORIA: normalização L2, tipagem moderna (list/tuple).
# ==============================================================================

def search_hybrid_text_vector(
    es,
    query_vector:  np.ndarray,
    query_text:    str,
    index_name:    str   = "video_index",
    top_k:         int   = 10,
    weight_vector: float = 0.6,
    weight_text:   float = 0.4,
    modality:      str   = "video",
) -> list[tuple[str, float]]:
    vector = _normalize(query_vector)

    res_knn = es.search(
        index=index_name,
        body={
            "knn": {
                "field":          "embedding",
                "query_vector":   vector,
                "k":              top_k * 10,
                "num_candidates": 200,
            },
            "query": {"term": {"modality": modality}},
            "_source": ["video_id"],
        },
        size=top_k * 10,
    )

    res_text = es.search(
        index=index_name,
        body={
            "query": {
                "bool": {
                    "must": [{"term": {"modality": modality}}],
                    "should": [
                        {"match": {"feature_desc": {"query": query_text, "boost": 2}}},
                        {"match": {"keywords":     {"query": query_text, "boost": 1.5}}},
                        {"match": {"title":        {"query": query_text, "boost": 1}}},
                    ],
                    "minimum_should_match": 1,
                }
            },
            "_source": ["video_id"],
        },
        size=top_k * 10,
    )

    scores: dict[str, float] = defaultdict(float)

    knn_hits  = res_knn["hits"]["hits"]
    text_hits = res_text["hits"]["hits"]

    max_knn  = max((h["_score"] for h in knn_hits),  default=1.0)
    max_text = max((h["_score"] for h in text_hits), default=1.0)

    for hit in knn_hits:
        scores[hit["_source"]["video_id"]] += weight_vector * (hit["_score"] / max_knn)
    for hit in text_hits:
        scores[hit["_source"]["video_id"]] += weight_text * (hit["_score"] / max_text)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ==============================================================================
# 9. Busca híbrida vídeo + áudio
# MELHORIA: normalização L2 consistente, lógica de query extraída em helper,
#           msearch para enviar embeddings de vídeo e áudio juntos.
# ==============================================================================

def search_hybrid(
    es,
    query_video_embeddings: list[dict],
    query_audio_embeddings: list[dict],
    index_name:    str   = "video_index",
    top_k:         int   = 10,
    weight_video:  float = 0.5,
    weight_audio:  float = 0.5,
) -> list[tuple[str, float]]:

    score_video: dict[str, float] = defaultdict(float)
    score_audio: dict[str, float] = defaultdict(float)
    count_video: dict[str, int]   = defaultdict(int)
    count_audio: dict[str, int]   = defaultdict(int)

    # Monta msearch com vídeo + áudio juntos em 1 roundtrip
    body = []
    meta: list[tuple[str, dict, dict]] = []  # (modality, score_acc, count_acc)

    for q in query_video_embeddings:
        vector = _normalize(q["embedding"])
        body.append({"index": index_name})
        body.append({
            "knn": {
                "field": "embedding", "query_vector": vector,
                "k": 50, "num_candidates": 200,
            },
            "query": {"term": {"modality": "video"}},
            "_source": ["video_id"],
        })
        meta.append(("video", score_video, count_video))

    for q in query_audio_embeddings:
        vector = _normalize(q["embedding"])
        body.append({"index": index_name})
        body.append({
            "knn": {
                "field": "embedding", "query_vector": vector,
                "k": 50, "num_candidates": 200,
            },
            "query": {"term": {"modality": "audio"}},
            "_source": ["video_id"],
        })
        meta.append(("audio", score_audio, count_audio))

    if not body:
        return []

    try:
        responses = es.msearch(body=body)
    except Exception as e:
        logger.error(f"Erro no msearch híbrido: {e}")
        return []

    for response, (modality, score_acc, count_acc) in zip(responses["responses"], meta):
        if "error" in response:
            logger.warning(f"Erro numa sub-query híbrida ({modality}): {response['error']}")
            continue
        for hit in response["hits"]["hits"]:
            vid = hit["_source"]["video_id"]
            score_acc[vid] += hit["_score"]
            count_acc[vid] += 1

    combined: dict[str, float] = {}
    all_vids = set(score_video) | set(score_audio)
    for vid in all_vids:
        avg_v = score_video[vid] / count_video[vid] if count_video[vid] > 0 else 0.0
        avg_a = score_audio[vid] / count_audio[vid] if count_audio[vid] > 0 else 0.0
        combined[vid] = weight_video * avg_v + weight_audio * avg_a

    return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]