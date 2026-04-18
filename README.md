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

1. Extrair os frames em um segundo do vídeo
2. Gerar embeddings para cada frame com o contexto do segundo que ele está dentro
3. Armazenar embeddings no Elasticsearch
4. Realizar busca k-NN (vizinhos mais próximos)

---

## 📁 Estrutura do Projeto

```
trabalhoRI/
│
├── data/
│   ├── videos/
│   └── embeddings/ (tem que colocar para salvar os embeddings)
│
├── src/
│   ├── consultas.sh
│   ├── embeddings.py
│   ├── index_elastic.py
│   ├── indexar.sh
│   ├── keyframes.py
│   ├── logger.py
│   ├── main_index.py
│   ├── main_search.py
│   └── search.py
│
├── docker/
│   └── docker-compose.yml
|
├── .env
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

### 2. Dar permissão para ambos os arquivos `.sh`

```bash
chmod +x src/indexar.sh
chmod +x src/consultas.sh
```

---

### 3. Executar primeiro o `indexar.sh` (espere ele terminar tudo)

```bash
src/indexar.sh
```

---

### 4. Agora executar o `consultas.sh` (futuramente adicionar mais consultas)

```bash
src/consultas.sh
```
---

## 🧩 Índice no Elasticsearch

Crie um índice com suporte a vetores:

```json
função create_index do index_elastic.py
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

* Média (baseline) (atualmente esse é usado)
* Média ponderada
* Attention (mais avançado)

---

## 🔍 Busca

Exemplo de consulta:

```json
parte da função search_video do search.py
index="video_index",
            knn={
                "field": "embedding",
                "query_vector": emb,
                "k": 50,
                "num_candidates": 500
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


## 📌 Resumo

Este projeto demonstra uma implementação prática de:

> Recuperação de vídeos baseada em conteúdo (Content-Based Video Retrieval) com embeddings sensíveis ao contexto temporal

---

## Comentários sobre os arquivos


## Arquivos `.sh`

Os scripts `.sh` foram criados para automatizar a execução do pipeline completo.

### Funções principais:
- Subir o Elasticsearch via Docker
- Aguardar o serviço iniciar corretamente
- Instalar dependências
- Executar os scripts Python na ordem correta

Isso evita problemas de execução manual e garante consistência do ambiente.

---

## `embeddings.py`

Este módulo é responsável por **gerar embeddings vetoriais a partir de vídeos**.

---

###  Modelo utilizado

- Utiliza o modelo **CLIP (Contrastive Language–Image Pretraining)** da OpenAI
- Arquitetura: `ViT-B/32`

### Motivo da escolha:
- Já estava pronto
- Excelente desempenho em tarefas de similaridade visual
- Não requer treinamento adicional
---

## `index_elastic.py`

Responsável pela **integração com o Elasticsearch**.

### Funcionalidades:

- **Conexão com Elasticsearch**
  - `connect_elasticsearch()`
  - Testa conexão com `localhost:9200`

- **Criação do índice**
  - `create_index()`
  - Define o schema com:
    - `video_id` (keyword)
    - `timestamp_sec` (float)
    - `center_frame` (int)
    - `embedding` (dense_vector - 512 dimensões)
  - Usa **HNSW** para busca vetorial eficiente

- **Indexação em lote**
  - `index_embeddings_bulk()`
  - Usa `helpers.bulk` para performance
  - ID único: `video_id + center_frame`

- **Busca por similaridade**
  - `search_similar()`
  - Usa KNN com similaridade de cosseno
  - Permite filtro opcional por `video_id`

---

## `keyframes.py`

Responsável pela **extração de frames e geração de janelas temporais**.

### Funcionalidades:

- **Extração de frames**
  - `extract_all_frames()`
  - Retorna todos os frames + FPS

- **Sincronização temporal**
  - `get_sync_indices()`
  - Seleciona frames (ex: 1 por segundo)

- **Criação de janelas**
  - `get_window()`
  - Cria contexto ao redor de um frame

- **Pipeline principal**
  - `generate_windows_stream_centered()`
  - Processa vídeo em **streaming (baixo uso de memória)**
  - Gera janelas com:
    - frame central
    - timestamp
    - contexto temporal

 Ideia chave:
> Cada embedding representa um momento do vídeo + contexto ao redor

---

## `embeddings.py`

Responsável por gerar **representações vetoriais (embeddings)**.

### Funcionalidades:

- **Carregar modelo CLIP**
  - `load_model()`
  - Usa `ViT-B/32`

- **Conversão de frame**
  - `frame_to_pil()`
  - OpenCV (BGR) → PIL (RGB)

- **Embedding de frame**
  - `embed_frame()`
  - Normalização L2 (essencial para cosine similarity)

- **Embedding de janela**
  - `embed_window()`
  - Métodos:
    - `mean` (usado)
    - `max`
    - `center`

- **Embeddings de vídeo**
  - `generate_embeddings()`
  - Gera embedding por janela

- **Salvar embeddings**
  - `save_embeddings_json()`

Ideia chave:
> O embedding representa o conteúdo visual do vídeo

---

## `logger.py`

Sistema de **log estruturado**.

### Funcionalidades:

- Cria pasta de logs automaticamente
- Salva logs em arquivo
- Exibe logs no terminal
- Evita duplicação de handlers


---

## `main_index.py`

Pipeline principal de **indexação de vídeos**.

### Etapas:

1. Conecta ao Elasticsearch
2. Verifica pasta de vídeos
3. Baixa vídeos se necessário
4. Cria índice
5. Carrega modelo CLIP
6. Processa vídeos locais

### Funcionalidades importantes:

- `count_videos()`
  - Conta vídeos locais

- `download_videos()`
  - Baixa até atingir N vídeos

- `already_indexed()`
  - Evita reprocessar vídeos já indexados

- `process_video()`
  - Pipeline completo:
    - gerar janelas
    - gerar embeddings
    - salvar JSON
    - indexar no Elasticsearch

Ideia chave:
> Pipeline robusto e incremental (evita retrabalho)

---

## `main_search.py`

Script de **consulta (busca por vídeo)**.

### Fluxo:

1. Conecta ao Elasticsearch
2. Carrega modelo CLIP
3. Gera embeddings do vídeo query
4. Busca vídeos similares
5. Exibe ranking

Saída:

```bash
Top resultados:

1. video_90 -> 0.9920
```

---

## `search.py`

Responsável pelas **estratégias de busca**.

### Funcionalidades:

#### Busca por frame
- `search_by_frame()`
- Gera embedding e consulta Elasticsearch

#### Busca por imagem
- `search_by_image_path()`

#### Busca por vídeo (principal)
- `search_video()`

### Estratégia usada:

- Para cada embedding da query:
  - Faz busca KNN
- Agrega resultados por `video_id`

### Melhorias implementadas:

- **Score médio por vídeo**
- **Limite de hits por vídeo**
  - evita um único vídeo dominar o ranking
- **Diversificação**
- controle de:
  - `k`
  - `num_candidates`

Ideia chave:
> Ranking baseado na **similaridade média entre janelas**

---

# Conceitos Importantes

## Embedding
Representação vetorial de uma imagem/frame.

## Cosine Similarity
Mede similaridade entre vetores.

## HNSW
Algoritmo eficiente para busca aproximada em vetores.

## Janela Temporal
Contexto ao redor de um frame (melhora semântica).

---

# Pipeline Geral

```
Vídeo → Frames → Janelas → Embeddings → Elasticsearch → Busca
```
---

# Observações

- `np.float_ = np.float64` usado por compatibilidade com NumPy 2.0 (mas não deu certo mesmo assim então nos `requeriments.txt` está usando `numpy<2.0`)
- Normalização dos embeddings é essencial
- Elasticsearch mantém dados mesmo após desligar (se persistência ativa)

---

# Possíveis Melhorias

- Hybrid Search (CLIP + BM25) [talvez BM25 não funcione pois não há palavras, apenas embeddings]
- Re-ranking com modelo mais forte
- PCA para reduzir dimensionalidade (sei nem o que significa, GPT que escreveu)
- Indexar áudio/transcrição
- Usar modelos multimodais mais recentes (pesquisar possivelmente)
- Detecção de cortes de cena (shot detection)
- Integração com embeddings de áudio
- Fusão multimodal (vídeo + áudio + texto)
- Busca híbrida (vetorial + texto)
- Integração com FAISS para maior desempenho (GPT falou muitas vezes que isso é melhor que elastic pra busca entre vetores, sei não se é vdd)

---

# Status do Projeto

✔ Indexação funcionando  
✔ Busca vetorial funcionando  
✔ Pipeline completo funcional  
✔ Evita duplicação  
✔ Ranking com score médio  

---


