from elasticsearch import Elasticsearch, helpers
import numpy as np


# ==============================
# 1. Conectar ao Elasticsearch
# ==============================
def connect_elasticsearch(host="http://localhost:9200"):
    es = Elasticsearch(
        host,
        request_timeout=30,
        max_retries=3,
        retry_on_timeout=True
        )
    
    if not es.ping():
        raise ValueError("Erro ao conectar ao Elasticsearch")
    
    print("Conectado ao Elasticsearch!")
    return es


# ==============================
# 2. Criar índice (com vetor)
# ==============================
def create_index(es, index_name="video_index", dims=512):
    if es.indices.exists(index=index_name):
        print("Índice já existe")
        return
    
    mapping = {
        "mappings": {
            "properties": {
                "video_id": {"type": "keyword"},
                "timestamp_sec": {"type": "float"},
                "center_frame": {"type": "integer"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                    "index_options": {
                        "type": "hnsw",
                        "m": 32,
                        "ef_construction": 200
                    }
                }
            }
        }
    }
    
    es.indices.create(index=index_name, body=mapping)
    print("Índice criado!")


# ==============================
# 3. Indexar embeddings
# ==============================
def index_embeddings_bulk(es, embeddings, index_name="video_index", video_id="video_1"):
    def generate_actions():
        for item in embeddings:
            yield {
                "_index": index_name,
                "_id": f"{video_id}_{item['center_frame']}",
                "_source": {
                    "video_id": video_id,
                    "timestamp_sec": item["timestamp_sec"],
                    "center_frame": item["center_frame"],
                    "embedding": item["embedding"].tolist()
                }
            }

    helpers.bulk(
        es,
        generate_actions(),
        chunk_size=500,
        request_timeout=60,
        raise_on_error=False
    )

# ==============================
# 4. Busca por similaridade
# ==============================
def search_similar(es, query_embedding, index_name="video_index", video_id=None,k=10):
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    query = {
        "size": k,
        "query": {
            "bool": {
                "filter": [
                    {"term": {"video_id": "video_1"}}
                ],
                "must": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding.tolist(),
                            "k": k,
                            "num_candidates": k*10
                        }
                    }
                }
            }
        }
    }

    # filtro opcional por vídeo
    if video_id is not None:
        query["query"]["bool"]["filter"] = [
            {"term": {"video_id": video_id}}
        ]


    response = es.search(index=index_name, body=query)

    return [
        {
            "score": hit["_score"],
            "timestamp_sec": hit["_source"]["timestamp_sec"],
            "center_frame": hit["_source"]["center_frame"],
            "video_id": hit["_source"]["video_id"]
        }
        for hit in response["hits"]["hits"]
    ]


# ==============================
# 5. Exemplo de uso completo
# ==============================
# if __name__ == "__main__":
#     from embeddings import load_model, generate_embeddings
#     from video_windows import generate_windows_stream_centered
    
#     video_path = "video.mp4"
    
#     # 1. Conectar
#     es = connect_elasticsearch()
    
#     # 2. Criar índice
#     create_index(es, index_name="video_index", dims=512)
    
#     # 3. Gerar embeddings
#     print("Carregando modelo...")
#     model, preprocess, device = load_model()
    
#     print("Gerando janelas...")
#     windows = generate_windows_stream_centered(video_path, k_seconds=0.5)
    
#     print("Gerando embeddings...")
#     embeddings = generate_embeddings(windows, model, preprocess, device)
    
#     # 4. Indexar
#     print("Indexando...")
#     index_embeddings(es, embeddings, index_name="video_index", video_id="video_1")
    
#     # 5. Teste de busca
#     print("Testando busca...")
    
#     query_embedding = embeddings[0]["embedding"]
    
#     results = search_similar(es, query_embedding)
    
#     print("Resultados:")
#     for r in results:
#         print(r)