import streamlit as st
import os
import tempfile
import numpy as np
import torch

import index_elastic as ind
import embeddings as emb
import keyframes as ky
import search as sc

st.set_page_config(page_title="VideoSearch AI", layout="wide")

# Conexão e modelos (cache global)
@st.cache_resource
def get_es():
    return ind.connect_elasticsearch()

@st.cache_resource
def load_clip_only():
    """Carrega apenas o CLIP (sem áudio) para reduzir uso de RAM."""
    return emb.load_all_models()  # ainda carrega CLAP, mas se quiser evitar, faça uma versão só com CLIP
    # Alternativa: chame emb.load_model() se tiver

es = get_es()
clip_model, clip_preprocess, clap_model, device = load_clip_only()  # aqui ainda carrega CLAP, se não quiser, crie load_clip_only

def text_to_vector(text):
    with torch.no_grad():
        tokens = clip.tokenize([text]).to(device)
        vec = clip_model.encode_text(tokens)
        vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.cpu().numpy().flatten()

def search_uploaded_video(video_path, top_k, w_vid, w_aud):
    """
    Processa o vídeo enviado usando a mesma segmentação do indexador:
    detecção de cenas → segmentos consecutivos de até 45 frames.
    Gera embeddings de vídeo (CLIP) e áudio (CLAP) e chama a busca híbrida.
    """
    # 1. Detectar cenas (mesma função usada na indexação)
    scenes = ind.detect_scenes(video_path)
    if not scenes:
        st.warning("Nenhuma cena detectada no vídeo.")
        return []

    # 2. Extrair segmentos (idêntico ao que foi indexado)
    segments = []
    for start, end in scenes:
        segments.extend(
            ky.split_scene_into_segments(
                video_path, start, end, max_frames_per_segment=45
            )
        )
    if not segments:
        st.warning("Nenhum segmento gerado.")
        return []

    # 3. Embeddings de vídeo (CLIP) para cada segmento
    video_embs = []
    for seg in segments:
        emb_vec = emb.embed_window(
            seg["frames"], clip_model, clip_preprocess, device, method="mean"
        )
        video_embs.append({
            "embedding": emb_vec,
            "timestamp_sec": seg["timestamp_sec"],
        })

    # 4. Embeddings de áudio (CLAP) – tratamento robusto
    audio_embs = []
    try:
        # Prepara timestamps para a função de áudio
        timestamps = [{"timestamp_sec": seg["timestamp_sec"]} for seg in segments]
        audio_embs = emb.generate_audio_embeddings_from_windows(
            video_path, timestamps, clap_model, device
        )
    except Exception as e:
        st.warning(f"Áudio não processado: {e}")

    # 5. Busca híbrida
    return sc.search_hybrid(
        es,
        query_video_embeddings=video_embs,
        query_audio_embeddings=audio_embs,
        top_k=top_k,
        weight_video=w_vid,
        weight_audio=w_aud,
    )

# Interface (mesma, mas com verificações)
st.title("Busca Inteligente de Vídeos")
with st.sidebar:
    mode = st.radio("Modo", ["Texto", "Vídeo"])
    top_k = st.slider("Resultados", 5, 30, 10)
    if mode == "Texto":
        weight_vec = st.slider("Peso visual (CLIP)", 0.0, 1.0, 0.6)
        weight_bm25 = 1.0 - weight_vec
        modality_filter = st.selectbox("Modalidade", ["video", "audio", "ambos"])
    else:
        w_vid = st.slider("Peso vídeo", 0.0, 1.0, 0.5)
        w_aud = st.slider("Peso áudio", 0.0, 1.0, 0.5)

if mode == "Texto":
    query = st.text_input("Descrição da cena")
    if query:
        with st.spinner("Buscando..."):
            if modality_filter == "ambos":
                res_v = sc.search_hybrid_text_vector(es, text_to_vector(query), query, modality="video", top_k=top_k, weight_vector=weight_vec, weight_text=weight_bm25)
                res_a = sc.search_hybrid_text_vector(es, text_to_vector(query), query, modality="audio", top_k=top_k, weight_vector=weight_vec, weight_text=weight_bm25)
                combined = {}
                for vid, score in res_v:
                    combined[vid] = combined.get(vid, 0) + score
                for vid, score in res_a:
                    combined[vid] = combined.get(vid, 0) + score * 0.7
                sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
            else:
                sorted_results = sc.search_hybrid_text_vector(es, text_to_vector(query), query, modality=modality_filter, top_k=top_k, weight_vector=weight_vec, weight_text=weight_bm25)
        if sorted_results:
            for vid, score in sorted_results:
                if not vid: continue
                st.image(f"https://img.youtube.com/vi/{vid}/hqdefault.jpg", use_container_width=True)
                st.write(f"**{vid}** - score: {score:.3f}")
                st.caption(f"https://www.youtube.com/watch?v={vid}")
                st.divider()
else:
    uploaded_file = st.file_uploader("Vídeo MP4", type=["mp4"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        st.video(tmp_path)
        if st.button("Buscar similares"):
            with st.spinner("Processando (pode demorar um pouco)..."):
                results = search_uploaded_video(tmp_path, top_k, w_vid, w_aud, max_keyframes=20)
            if results:
                for vid, score in results:
                    if not vid: continue
                    st.image(f"https://img.youtube.com/vi/{vid}/hqdefault.jpg", use_container_width=True)
                    st.write(f"**{vid}** - score: {score:.3f}")
                    st.caption(f"https://www.youtube.com/watch?v={vid}")
                    st.divider()
        os.unlink(tmp_path)