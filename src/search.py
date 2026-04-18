import index_elastic as ind
from embeddings import embed_frame

import keyframes as ky
import embeddings as emb
import index_elastic as ind
from logger import setup_logger
from collections import defaultdict



# ==============================
# 4. Buscar por imagem/frame
# ==============================
def search_by_frame(
    es,
    frame,
    model,
    preprocess,
    device,
    index_name="video_index",
    video_id=None,
    k=5
):
    query_embedding = embed_frame(frame, model, preprocess, device)
    
    return ind.search_similar(
        es,
        query_embedding,
        index_name=index_name,
        video_id=video_id,
        k=k
    )


# ==============================
# 5. Buscar por arquivo de imagem
# ==============================
def search_by_image_path(
    es,
    image_path,
    model,
    preprocess,
    device,
    index_name="video_index",
    video_id=None,
    k=5
):
    import cv2
    
    frame = cv2.imread(image_path)
    
    if frame is None:
        raise ValueError("Erro ao carregar imagem")
    
    return search_by_frame(
        es,
        frame,
        model,
        preprocess,
        device,
        index_name=index_name,
        video_id=video_id,
        k=k
    )



def search_video(es, query_embeddings,top_k=10):
    scores = defaultdict(float)
    counts = defaultdict(int)
    unique_videos = set()

    max_hits_per_video = 5 # ou 3

    video_hits = defaultdict(int)

    for item in query_embeddings:
        emb = item["embedding"]

        if not isinstance(emb, list):
            emb = emb.tolist()

        results = es.search(
            index="video_index",
            knn={
                "field": "embedding",
                "query_vector": emb,
                "k": 50,
                "num_candidates": 500
            }
        )

        for hit in results["hits"]["hits"]:
            video_id = hit["_source"]["video_id"]
            unique_videos.add(video_id)
            if video_hits[video_id] >= max_hits_per_video:
                continue

            score = hit["_score"]

            scores[video_id] += score
            counts[video_id] += 1
            video_hits[video_id] += 1
            

    # ✅ média
    final_scores = {
        vid: scores[vid] / counts[vid]
        for vid in scores
    }

    print("Vídeos únicos encontrados:", len(unique_videos))
    print(unique_videos)

    # ✅ ordenar pela média (correto)
    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    return ranked[:top_k]
    



# # ==============================
# # 6. Exemplo de uso
# # ==============================
# if __name__ == "__main__":
#     es = ind.connect_elasticsearch()
    
#     print("Carregando modelo...")
#     model, preprocess, device = load_model()
    
#     # exemplo com imagem
#     image_path = "query.jpg"
    
#     print("Buscando...")
#     results = search_by_image_path(
#         es,
#         image_path,
#         model,
#         preprocess,
#         device,
#         k=5
#     )
    
#     print("\nResultados:")
#     for r in results:
#         print(f"Score: {r['score']:.4f}")
#         print(f"Vídeo: {r['video_id']}")
#         print(f"Timestamp: {r['timestamp_sec']}s")
#         print(f"Frame: {r['center_frame']}")
#         print("-" * 30)