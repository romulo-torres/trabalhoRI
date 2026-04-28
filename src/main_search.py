import os

import index_elastic as ind
import keyframes as ky
import embeddings as emb
import search as sc
from embeddings import load_model
from logger import setup_logger

logger = setup_logger()


def main():
    # --- Conexão ---
    logger.info("Conectando ao Elasticsearch...")
    es = ind.connect_elasticsearch()

    # --- Modelo ---
    logger.info("Carregando modelo CLIP...")
    model, preprocess, device = load_model()

    # --- Vídeo de consulta ---
    video_path = "./data/videos/_MWyhJS4KbM.mp4"

    if not os.path.exists(video_path):
        logger.error(f"Vídeo não encontrado: '{video_path}'")
        return

    # --- Janelas e embeddings ---
    logger.info(f"Gerando janelas para '{video_path}'...")
    query_windows = ky.generate_windows_stream_centered(video_path, k_seconds=0.5)

    if not query_windows:
        logger.error("Nenhuma janela gerada. Verifique o vídeo.")
        return

    logger.info(f"Janelas geradas: {len(query_windows)}")

    query_embeddings = emb.generate_embeddings(query_windows, model, preprocess, device)

    if not query_embeddings:
        logger.error("Nenhum embedding gerado.")
        return

    logger.info(f"Embeddings gerados: {len(query_embeddings)}")

    # --- Busca ---
    logger.info("Buscando vídeos similares...")
    results = sc.search_video(es, query_embeddings, top_k=10)

    if not results:
        logger.warning("Nenhum resultado encontrado.")
        return

    print("\nTop resultados:")
    for i, (vid, score) in enumerate(results, start=1):
        # Exclui o próprio vídeo de consulta dos resultados exibidos
        query_id = os.path.splitext(os.path.basename(video_path))[0]
        marker = "  ← consulta" if vid == query_id else ""
        print(f"  {i:>2}. {vid}  score={score:.4f}{marker}")


if __name__ == "__main__":
    main()