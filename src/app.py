import os
import tempfile
from collections import defaultdict

import numpy as np
import streamlit as st
import torch

import embeddings as emb
import index_elastic as ind
import keyframes as ky
import search as sc

st.set_page_config(page_title="VideoSearch AI", layout="wide")


# ==============================================================================
# Cache de recursos
# ==============================================================================
def get_es():
    return ind.connect_elasticsearch()


def get_models():
    return emb.load_all_models()


es                                            = get_es()
clip_model, clip_preprocess, clap_model, device = get_models()


# ==============================================================================
# Expansão de query com sinônimos (WordNet via NLTK)
# ==============================================================================
def expand_query(text: str) -> tuple[str, list[str]]:
    """
    Retorna (query_expandida, lista_de_sinonimos).
    Usa WordNet para buscar sinônimos e hiperônimos dos tokens relevantes.
    """
    try:
        import nltk
        from nltk.corpus import wordnet
        from nltk.tokenize import word_tokenize

        for resource in ["wordnet", "punkt", "stopwords"]:
            try:
                nltk.data.find(f"corpora/{resource}" if resource != "punkt" else f"tokenizers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

        from nltk.corpus import stopwords
        stop_words = set(stopwords.words("english"))

        tokens = [t.lower() for t in word_tokenize(text) if t.isalpha() and t.lower() not in stop_words]
        synonyms: set[str] = set()

        for token in tokens:
            for syn in wordnet.synsets(token):
                for lemma in syn.lemmas():
                    clean = lemma.name().replace("_", " ").lower()
                    if clean != token:
                        synonyms.add(clean)
                # Hiperônimos (1 nível) — ampliam o contexto
                for hypernym in syn.hypernyms():
                    for lemma in hypernym.lemmas():
                        clean = lemma.name().replace("_", " ").lower()
                        synonyms.add(clean)

        # Limita a 10 sinônimos mais curtos (menos ruído)
        sorted_syns = sorted(synonyms, key=len)[:10]
        expanded    = text + " " + " ".join(sorted_syns)
        return expanded.strip(), sorted_syns

    except Exception as e:
        st.warning(f"Expansão de query indisponível: {e}")
        return text, []


# ==============================================================================
# Agregação de scores com LogSumExp (estável numericamente)
#
# Comportamento: domina pelo pico do melhor segmento, mas sobe ligeiramente
# quando múltiplos segmentos são relevantes — sem favorecer vídeos longos
# linearmente como a soma faria.
#
# Fórmula: LSE(s) = max(s) + log( Σ exp(s_i − max(s)) )
# Subtrair o máximo antes do exp evita overflow em float64.
# ==============================================================================
def _logsumexp_scores(score_lists: dict[str, list[float]]) -> dict[str, float]:
    result = {}
    for vid, scores in score_lists.items():
        arr = np.array(scores, dtype=np.float64)
        m   = arr.max()
        lse = m + np.log(np.sum(np.exp(arr - m)))
        result[vid] = float(lse)
    return result


# ==============================================================================
# Busca por TEXTO — híbrida: CLIP + CLAP (ANN) + BM25 + tags/categories
#
# Pesos:
#   - ANN visual  (CLIP texto → embedding vídeo)  : 5
#   - ANN áudio   (CLAP texto → embedding áudio)  : 5
#   - BM25 legenda (transcript)                   : 4
#   - BM25 tags/categories                        : 2
# ==============================================================================
def search_by_text(query: str, top_k: int) -> list[tuple[str, float]]:
    expanded_query, synonyms = expand_query(query)

    # --- embeddings de texto ---
    vec_clip = emb.embed_text_clip(query, clip_model, device)
    vec_clap = emb.embed_text_clap(query, clap_model)

    # --- tags/categories: query original + sinônimos ---
    tag_terms = [query] + synonyms

    body = {
        "size": top_k * 5,  # busca mais para re-ranquear depois
        "query": {
            "bool": {
                "should": [
                    # BM25 — transcript (legenda)
                    {
                        "match": {
                            "transcript": {
                                "query":    expanded_query,
                                "boost":    4.0,
                                "operator": "or",
                                "fuzziness": "AUTO",
                            }
                        }
                    },
                    # BM25 — title + feature_desc
                    {
                        "multi_match": {
                            "query":  expanded_query,
                            "fields": ["title^2", "feature_desc", "keywords", "description"],
                            "boost":  2.0,
                        }
                    },
                    # Tags e categories com sinônimos
                    {
                        "terms": {
                            "tags":       tag_terms,
                            "boost":      2.0,
                        }
                    },
                    {
                        "terms": {
                            "categories": tag_terms,
                            "boost":      2.0,
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        },
        # KNN híbrido — canal visual (CLIP)
        "knn": [
            {
                "field":          "embedding",
                "query_vector":   vec_clip.tolist(),
                "k":              top_k * 3,
                "num_candidates": top_k * 10,
                "boost":          5.0,
                "filter":         {"term": {"modality": "video"}},
            },
            # KNN híbrido — canal áudio (CLAP)
            {
                "field":          "embedding",
                "query_vector":   vec_clap.tolist(),
                "k":              top_k * 3,
                "num_candidates": top_k * 10,
                "boost":          5.0,
                "filter":         {"term": {"modality": "audio"}},
            },
        ],
    }

    try:
        resp = es.search(index="video_index", body=body)
        hits = resp["hits"]["hits"]
        score_lists: dict[str, list[float]] = defaultdict(list)
        for h in hits:
            vid = h["_source"].get("video_id", "")
            if vid:
                score_lists[vid].append(h["_score"])
        aggregated = _logsumexp_scores(score_lists)
        return sorted(aggregated.items(), key=lambda x: x[1], reverse=True)[:top_k]
    except Exception as e:
        st.error(f"Erro na busca por texto: {e}")
        return []


# ==============================================================================
# Busca por VÍDEO — híbrida: segmentos ANN + comparação com thumbnail
#
# Pesos:
#   - ANN visual  (CLIP frame → embedding vídeo)  : 2
#   - ANN áudio   (CLAP áudio → embedding áudio)  : 2
#   - Thumbnail   (CLIP frame → feature_thumb)    : 5
# ==============================================================================
def search_by_video(video_path: str, top_k: int) -> list[tuple[str, float]]:
    # --- cenas e segmentos (mesmo pipeline da indexação) ---
    scenes = ind.detect_scenes(video_path)
    if not scenes:
        st.warning("Nenhuma cena detectada no vídeo enviado.")
        return []

    all_segments: list[dict] = []
    for start, end in scenes:
        try:
            segs = ky.split_scene_into_segments(
                video_path, start, end, max_frames_per_segment=45
            )
            all_segments.extend(segs)
        except Exception as e:
            st.warning(f"Erro ao segmentar cena [{start:.1f}s–{end:.1f}s]: {e}")

    if not all_segments:
        st.warning("Nenhum segmento gerado a partir do vídeo.")
        return []

    # --- embeddings de vídeo (CLIP) ---
    video_vecs: list[np.ndarray] = []
    for seg in all_segments:
        try:
            vec = emb.embed_window(
                seg["frames"], clip_model, clip_preprocess, device, method="mean"
            )
            video_vecs.append(vec)
        except Exception:
            pass

    # --- embeddings de áudio (CLAP) ---
    audio_vecs: list[np.ndarray] = []
    try:
        audio_results = emb.generate_audio_embeddings_from_segments(
            video_path, all_segments, clap_model, device
        )
        audio_vecs = [r["embedding"] for r in audio_results]
    except Exception as e:
        st.warning(f"Áudio não processado: {e}")

    # --- thumbnail do vídeo enviado (frame central) ---
    thumb_vec: np.ndarray | None = None
    try:
        mid_seg   = all_segments[len(all_segments) // 2]
        mid_frame = mid_seg["frames"][len(mid_seg["frames"]) // 2]
        thumb_vec = emb.embed_frame(mid_frame, clip_model, clip_preprocess, device)
    except Exception:
        pass

    if not video_vecs and thumb_vec is None:
        st.warning("Não foi possível gerar nenhum embedding do vídeo enviado.")
        return []

    # --- média dos vetores por canal ---
    mean_video = np.mean(video_vecs, axis=0) if video_vecs else None
    mean_audio = np.mean(audio_vecs, axis=0) if audio_vecs else None

    # --- normaliza ---
    def _norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)

    knn_clauses = []

    if mean_video is not None:
        knn_clauses.append({
            "field":          "embedding",
            "query_vector":   _norm(mean_video).tolist(),
            "k":              top_k * 3,
            "num_candidates": top_k * 10,
            "boost":          2.0,
            "filter":         {"term": {"modality": "video"}},
        })

    if mean_audio is not None:
        knn_clauses.append({
            "field":          "embedding",
            "query_vector":   _norm(mean_audio).tolist(),
            "k":              top_k * 3,
            "num_candidates": top_k * 10,
            "boost":          2.0,
            "filter":         {"term": {"modality": "audio"}},
        })

    if thumb_vec is not None:
        knn_clauses.append({
            "field":          "feature_thumb",
            "query_vector":   _norm(thumb_vec).tolist(),
            "k":              top_k * 3,
            "num_candidates": top_k * 10,
            "boost":          5.0,
        })

    if not knn_clauses:
        st.warning("Nenhum vetor disponível para busca.")
        return []

    body = {
        "size": top_k * 5,
        "knn":  knn_clauses,
    }

    try:
        resp = es.search(index="video_index", body=body)
        hits = resp["hits"]["hits"]
        score_lists: dict[str, list[float]] = defaultdict(list)
        for h in hits:
            vid = h["_source"].get("video_id", "")
            if vid:
                score_lists[vid].append(h["_score"])
        aggregated = _logsumexp_scores(score_lists)
        return sorted(aggregated.items(), key=lambda x: x[1], reverse=True)[:top_k]
    except Exception as e:
        st.error(f"Erro na busca por vídeo: {e}")
        return []


# ==============================================================================
# Renderização dos resultados
# ==============================================================================
def render_results(results: list[tuple[str, float]]) -> None:
    if not results:
        st.info("Nenhum resultado encontrado.")
        return

    for vid, score in results:
        if not vid:
            continue
        col_thumb, col_info = st.columns([1, 3])
        with col_thumb:
            st.image(
                f"https://img.youtube.com/vi/{vid}/hqdefault.jpg",
                use_container_width=True,
            )
        with col_info:
            st.markdown(f"**{vid}**")
            st.caption(f"https://www.youtube.com/watch?v={vid}")
            st.progress(min(score / 20.0, 1.0), text=f"score: {score:.3f}")
        st.divider()


# ==============================================================================
# Interface
# ==============================================================================
st.title("VideoSearch AI")

with st.sidebar:
    st.header("Configurações")
    mode  = st.radio("Modo de busca", ["Texto", "Vídeo"])
    top_k = st.slider("Número de resultados", 5, 30, 10)

    if mode == "Texto":
        st.caption(
            "**Pesos aplicados:**\n"
            "- ANN visual (CLIP): 5\n"
            "- ANN áudio (CLAP): 5\n"
            "- BM25 legenda: 4\n"
            "- BM25 tags/categories: 2"
        )
    else:
        st.caption(
            "**Pesos aplicados:**\n"
            "- ANN visual (CLIP): 2\n"
            "- ANN áudio (CLAP): 2\n"
            "- Thumbnail: 5"
        )

# --- modo TEXTO ---
if mode == "Texto":
    query = st.text_input("Descreva a cena que quer encontrar")

    if query:
        _, synonyms = expand_query(query)
        if synonyms:
            with st.expander("Sinônimos utilizados na busca", expanded=False):
                st.write(", ".join(synonyms))

        with st.spinner("Buscando..."):
            results = search_by_text(query, top_k)

        st.subheader(f"{len(results)} resultado(s)")
        render_results(results)

# --- modo VÍDEO ---
else:
    uploaded = st.file_uploader("Envie um vídeo MP4", type=["mp4"])

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.video(tmp_path)

        if st.button("🔍 Buscar vídeos similares"):
            with st.spinner("Processando vídeo (segmentação + embeddings)..."):
                results = search_by_video(tmp_path, top_k)

            st.subheader(f"{len(results)} resultado(s)")
            render_results(results)

        os.unlink(tmp_path)