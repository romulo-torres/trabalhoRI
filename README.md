---

# 📘 README.md

## 🎯 Visão Geral

Este projeto implementa um sistema de **busca de vídeos por similaridade** (video-to-video search) utilizando:

* Embeddings visuais (frames)
* Contexto temporal (janelas de frames)
* Busca vetorial com Elasticsearch

O objetivo é recuperar trechos de vídeo semelhantes com base no conteúdo visual (e opcionalmente áudio).

---

## ALGUMAS CONFIGURAÇÕES

* Configurar o .env com a chave do hugginface para usar

--- 

## 🧠 Arquitetura

```
Vídeo → Extração de Frames → Embeddings → Elasticsearch → Busca por Similaridade
```

### Etapas do Pipeline

1. Extrair frames ou keyframes do vídeo
2. Gerar embeddings para cada frame
3. Agregar contexto temporal (janela de frames)
4. Armazenar embeddings no Elasticsearch
5. Realizar busca k-NN (vizinhos mais próximos)

---

## 📁 Estrutura do Projeto

```
trabalhoRI/
│
├── data/
│   ├── videos/
│   └── embeddings/
│
├── src/
│   ├── extract_frames.py
│   ├── detect_keyframes.py
│   ├── embeddings.py
│   ├── index_elastic.py
│   ├── search.py
│
├── docker/
│   └── docker-compose.yml
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuração

### 1. Clonar o repositório

```bash
git clone <seu-repositorio>
cd trabalhoRI
```

---

### 2. Criar ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

---

### 4. Subir o Elasticsearch com Docker

```bash
docker-compose up -d
```

Teste no navegador:

```
http://localhost:9200
```

---

## 🧩 Índice no Elasticsearch

Crie um índice com suporte a vetores:

```json
PUT video_index
{
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
```

---

## 🎥 Estratégia de Embeddings

### Contexto Temporal

Em vez de usar apenas um frame isolado:

```
[t-3, t-2, t-1, t, t+1, t+2, t+3]
```

Agregamos os embeddings para obter uma representação mais rica.

### Métodos de Agregação

* Média (baseline)
* Média ponderada
* Attention (mais avançado)

---

## 🔍 Busca

Exemplo de consulta:

```json
POST video_index/_search
{
        "size": k,
        "query": {
            "bool": {
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
```

---

## 🚀 Funcionalidades

* Extração de embeddings por frame
* Agregação temporal
* Busca por similaridade vetorial
* Escalável com Elasticsearch
* Pronto para multimodal (áudio/texto)

---

## 🔮 Melhorias Futuras

* Detecção de cortes de cena (shot detection)
* Integração com embeddings de áudio
* Fusão multimodal (vídeo + áudio + texto)
* Busca híbrida (vetorial + texto)
* Integração com FAISS para maior desempenho

---

## 🧪 Ideias de Experimentos

* Comparar frame isolado vs contexto temporal
* Avaliar diferentes tamanhos de janela (k=3, 5, 7)
* Medir precisão de recuperação (precision@k)

---

## ⚠️ Observações

* Elasticsearch é usado por simplicidade; FAISS pode ser mais rápido em larga escala
* A dimensão do embedding depende do modelo (ex: CLIP = 512)

---

## 🤝 Contribuição

Sinta-se livre para contribuir com:

* novos modelos
* melhorias na indexação
* otimizações de performance

---

## 📌 Resumo

Este projeto demonstra uma implementação prática de:

> Recuperação de vídeos baseada em conteúdo (Content-Based Video Retrieval) com embeddings sensíveis ao contexto temporal

---
