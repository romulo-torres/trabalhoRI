import os

import index_elastic as ind
import keyframes as ky
import embeddings as emb
import search as sc
from embeddings import load_all_models
from logger import setup_logger

logger = setup_logger()


def main():
    # --- Conexão ---
    logger.info("Conectando ao Elasticsearch...")
    es = ind.connect_elasticsearch()

    # --- Modelos (CLIP + CLAP) ---
    logger.info("Carregando modelos CLIP e CLAP...")
    clip_model, clip_preprocess, clap_model, device = load_all_models()

    # --- Vídeo de consulta ---
    video_path = "./data/videos/_MWyhJS4KbM.mp4"
    if not os.path.exists(video_path):
        logger.error(f"Vídeo não encontrado: '{video_path}'")
        return

    # --- Janelas de vídeo (keyframes) ---
    logger.info("Gerando janelas de vídeo...")
    query_windows = ky.generate_windows_stream_centered(video_path, k_seconds=0.5)
    if not query_windows:
        logger.error("Nenhuma janela de vídeo gerada.")
        return
    logger.info(f"Janelas de vídeo: {len(query_windows)}")

    # --- Embeddings de vídeo (CLIP) ---
    logger.info("Gerando embeddings de vídeo...")
    video_embs = emb.generate_embeddings(
        query_windows, clip_model, clip_preprocess, device, method="mean"
    )
    if not video_embs:
        logger.error("Nenhum embedding de vídeo gerado.")
        return

    # --- Embeddings de áudio (CLAP) ---
    logger.info("Gerando embeddings de áudio...")
    audio_embs = emb.generate_audio_embeddings_from_windows(
        video_path, query_windows, clap_model, device
    )
    if not audio_embs:
        logger.warning("Nenhum embedding de áudio gerado (vídeo sem som?).")

    # --- Busca híbrida ---
    logger.info("Executando busca híbrida...")
    results = sc.search_hybrid(
        es,
        query_video_embeddings=video_embs,
        query_audio_embeddings=audio_embs,
        top_k=10,
        weight_video=0.5,
        weight_audio=0.5,
    )

    if not results:
        logger.warning("Nenhum resultado encontrado.")
        return

    query_id = os.path.splitext(os.path.basename(video_path))[0]
    print("\nTop resultados:")
    for i, (vid, score) in enumerate(results, start=1):
        marker = "  ← consulta" if vid == query_id else ""
        print(f"  {i:>2}. {vid}  score={score:.4f}{marker}")


if __name__ == "__main__":
    main()