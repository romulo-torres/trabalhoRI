import numpy as np
import embeddings as emb
import index_elastic as ind
import keyframes as key
from elasticsearch import Elasticsearch

from embeddings import load_model, embed_frame


# ==============================
# 2. Normalizar embedding
# ==============================
def normalize_embedding(embedding):
    return embedding / np.linalg.norm(embedding)



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


# ==============================
# 6. Exemplo de uso
# ==============================
if __name__ == "__main__":
    es = ind.connect_elasticsearch()
    
    print("Carregando modelo...")
    model, preprocess, device = load_model()
    
    # exemplo com imagem
    image_path = "query.jpg"
    
    print("Buscando...")
    results = search_by_image_path(
        es,
        image_path,
        model,
        preprocess,
        device,
        k=5
    )
    
    print("\nResultados:")
    for r in results:
        print(f"Score: {r['score']:.4f}")
        print(f"Vídeo: {r['video_id']}")
        print(f"Timestamp: {r['timestamp_sec']}s")
        print(f"Frame: {r['center_frame']}")
        print("-" * 30)