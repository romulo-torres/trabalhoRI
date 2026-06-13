# 📘 README.md

## 🎯 Visão Geral

Este projeto implementa um sistema de **busca multimodal de vídeos** que combina:

- **Embeddings visuais** (CLIP) a partir de segmentos de frames.
- **Embeddings de áudio** (CLAP) extraídos das mesmas janelas temporais.
- **Busca textual** com suporte a descrições (CLIP texto + BM25).
- **Indexação e busca vetorial** utilizando Elasticsearch com HNSW.
- Interface interativa via **Streamlit** para consultas por texto ou por upload de vídeo.

O sistema é capaz de recuperar vídeos semelhantes considerando tanto o conteúdo visual quanto o sonoro, e também permite buscas puramente textuais (ex.: “pessoa tocando violão”).

---

## ALGUMAS CONFIGURAÇÕES

- Configure o arquivo `.env` com a chave do HuggingFace caso utilize datasets que exijam autenticação.

---

## 🧠 Arquitetura


```
trabalhoRI/
│
├── data/
│   ├── videos/
│   └── embeddings/ (tem que colocar para salvar os embeddings)
│
├── src/
│   ├── app.py
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

### 1. Clonar o repositório e Dependências

```bash
git clone <seu-repositorio>
cd trabalhoRI
pip install -r requirements.txt
```

---

### 2. Dar permissão para indexar.sh `.sh`

```bash
chmod +x src/indexar.sh
```

---

### 3. Executar primeiro o `indexar.sh` (espere ele terminar tudo)

```bash
src/indexar.sh
```

### Para uma procura rápida de somente um vídeo use o `main_index.py`
```bash
python src/main_index.py
```

### Para usar a interface do streamlit use 
```bash
streamlit run src/app.py
```

---

## 🧩 Índice no Elasticsearch

Crie um índice com suporte a vetores:

```json
função create_index do index_elastic.py
{
  "mappings": {
    "properties": {
      "video_id":          {"type": "keyword"},
      "title":             {"type": "text", "analyzer": "video_text_analyzer"},
      "scene_index":       {"type": "integer"},
      "part_index":        {"type": "integer"},
      "timestamp_sec":     {"type": "float"},
      "center_frame":      {"type": "integer"},
      "modality":          {"type": "keyword"},
      "feature_desc":      {"type": "text", "analyzer": "video_text_analyzer"},
      "keywords":          {"type": "text", "analyzer": "video_text_analyzer"},
      "feature_categorias":{"type": "keyword"},
      "feature_thumb":     {"type": "dense_vector", "dims": 512, …},
      "embedding":         {"type": "dense_vector", "dims": 512, …}
    }
  }
}
```
- `embedding`: vetor denso de 512 dimensões (CLIP ou CLAP, dependendo da modality).

- `feature_thumb`: embedding da thumbnail do YouTube (mesmo vídeo, mesmo vetor).

- Campos de texto com analisador personalizado (video_text_analyzer) para suporte a stemming e stopwords em inglês.


---

## Estratégia de Embeddings

## Videos

1. Detecção de cenas: PySceneDetect identifica mudanças de conteúdo.

2. Segmentação: Cada cena é dividida em segmentos consecutivos de no máximo 45 frames (configurável via max_frames_per_segment). Isso garante que não haja uma divisão arbitrária em N partes fixas, mas sim uma varredura completa da cena.

3. Extração: Para cada segmento, os frames são convertidos para embeddings CLIP (ViT‑B/32) e agregados por média (padrão) [também possui a opção de média ponderada e attention]. O vetor resultante é normalizado (L2).

## Áudio

- O áudio completo do vídeo é extraído (ffmpeg → WAV mono 16kHz).

- Para cada segmento temporal correspondente a um segmento de vídeo (mesmo intervalo de tempo), um embedding CLAP é gerado.

- Se o vídeo não possuir áudio, os embeddings de áudio são ignorados (sem quebrar o pipeline).

## Metadados Textuais

- feature_desc: descrição gerada a partir da categoria do ActivityNet e do título.

- keywords: palavras‑chave extraídas da taxonomia e título.

- title: título do vídeo (obtido via yt‑dlp ou anotação).

---

### Módulos Principais

`embeddings.py`

    load_all_models(): carrega CLIP e CLAP com cache global.

    embed_frame() / embed_window(): embeddings de frames/janelas com agregação (mean, max, center).

    generate_embeddings_from_scenes(): gera embeddings de vídeo a partir de cenas.

    generate_audio_embeddings_from_windows() / from_scenes(): embeddings de áudio.

    extract_audio_from_video(): extrai áudio via ffmpeg.

    save_embeddings_json() / load_embeddings_json(): persistência dos embeddings.

`keyframes.py`

    generate_windows_stream_centered(): janelas temporais centradas nos keyframes (1 por segundo).

    split_scene_into_segments(): divide uma cena em segmentos consecutivos de até N frames.

    extract_all_frames(), get_sync_indices(), get_window(): funções auxiliares.

`index_elastic.py`

    connect_elasticsearch(): conexão com o cluster.

    create_index(): define o mapeamento (vetores + texto) e cria o índice.

    index_embeddings_bulk(): indexação em lote com metadados.

    process_video(): pipeline completo (cenas → segmentos → embeddings → indexação).

    process_local_videos(): varre uma pasta de MP4 e indexa os ainda não processados.

    update_feature_desc(), update_title(): atualizações parciais.

`search.py`

    search_hybrid(): busca combinada de vídeo + áudio.

    search_hybrid_text_vector(): busca combinada de vetor (CLIP) + BM25 textual.

    search_by_frame(), search_by_image_path(): busca por frame/imagem.

    search_video(): busca por múltiplos embeddings agregando por vídeo.

`app.py (Streamlit)`

    Interface gráfica com seleção de modo, pesos, upload de vídeo e exibição de resultados (thumbnails + links).

### Melhorias Futuras

    Integração com FAISS para busca vetorial de alta performance.

    Modelos mais recentes (ex.: CLIP ViT‑L, ImageBind, Video‑LLaMA).

    Legendagem automática do vídeo de consulta para busca textual mais precisa.

    Suporte a outros idiomas nos analisadores de texto.

    Re‑ranking com modelos mais fortes (cross‑encoders).

    Detecção de cortes de cena mais robusta (shot detection).

    Frontend aprimorado com visualização dos segmentos correspondentes.

📌 Status do Projeto

✔ Indexação multimodal (vídeo + áudio + texto)
✔ Busca vetorial com HNSW
✔ Busca híbrida (vídeo + áudio) e textual (CLIP + BM25)
✔ Interface Streamlit funcional
✔ Cache de modelos para evitar recargas
✔ Tratamento robusto de vídeos sem áudio
✔ Pipeline incremental (pula vídeos já indexados)
✔ Metadados enriquecidos com taxonomia ActivityNet

Este projeto demonstra um sistema completo de Content‑Based Video Retrieval com suporte a múltiplas modalidades e busca híbrida, pronto para ser estendido e integrado a aplicações reais.


