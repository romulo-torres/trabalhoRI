import index_elastic as ind
from embeddings import load_model

import keyframes as ky
import embeddings as emb

import index_elastic as ind
from logger import setup_logger
import search as sc


    

# USO

def main():
    es = ind.connect_elasticsearch()
    print("Carregando modelo...")
    model, preprocess, device = load_model()

    video_path = "./data/videos/video_93.mp4"

    query_windows = ky.generate_windows_stream_centered(video_path, 0.5)

    query_embeddings = emb.generate_embeddings(
        query_windows,
        model,
        preprocess,
        device
    )

    print("Qtd janelas:", len(query_windows))
    print("Qtd embeddings:", len(query_embeddings))

    results = sc.search_video(es, query_embeddings)

    print("\nTop resultados:")
    for i, (vid, score) in enumerate(results):
        print(f"{i+1}. {vid} -> {score:.4f}")

main()
