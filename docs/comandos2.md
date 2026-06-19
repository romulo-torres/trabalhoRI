# Coleção de Referência — ActivityNet Video Search

Sistema de busca texto-vídeo + geração de coleção de referência no formato BEIR, usando o dataset ActivityNet.

---

## 1. Visão Geral

O projeto tem **dois subsistemas**:

| Sistema | Descrição | Tecnologias |
|---------|-----------|-------------|
| **Search App** | Busca híbrida (ANN + BM25) em vídeos do ActivityNet | Docker, CUDA, Elasticsearch (HNSW), CLIP, CLAP, Streamlit |
| **Coleção de Referência** | Pipeline BEIR: corpus → consultas → pooling → julgamento → qrels + hard negatives | Python, LLMs (OpenAI/Anthropic/Gemini/etc.), BM25 (rank_bm25) |

---

## 2. Search App

Requer **Docker com suporte a GPU** (CUDA) e Elasticsearch.

### 2.1 Comandos

```powershell
# Windows
.\docker\run.ps1 build           # Constrói imagem (torch + CUDA)
.\docker\run.ps1 up              # Sobe Elasticsearch
.\docker\run.ps1 index -w 10     # Indexa 10 vídeos
.\docker\run.ps1 app             # Streamlit em localhost:8501
.\docker\run.ps1 stats           # Estatísticas do índice
.\docker\run.ps1 logs            # Logs do container
.\docker\run.ps1 shell           # Bash dentro do container
.\docker\run.ps1 down            # Para tudo
```

```bash
# Linux
./docker/run.sh build
./docker/run.sh up
./docker/run.sh index -w 10
./docker/run.sh app
./docker/run.sh stats
./docker/run.sh logs
./docker/run.sh shell
./docker/run.sh down
```

### 2.2 Indexação

```powershell
.\docker\run.ps1 index                           # Todos pendentes
.\docker\run.ps1 index -w 10                     # 10 novos
.\docker\run.ps1 index -o 100 -w 50              # 50 a partir do 100
.\docker\run.ps1 index --corpus                  # Usa corpus.jsonl como fonte
.\docker\run.ps1 index --batch                   # Modo batch (mais RAM, mais rápido)
.\docker\run.ps1 index -i data/metadata/videos_metadata.json  # JSON específico
```

Flags: `--window/-w` (qtde), `--offset/-o` (pular N), `--input/-i` (JSON metadata), `--batch`, `--corpus`

### 2.3 Estatísticas

```powershell
.\docker\run.ps1 stats                           # Visão geral
.\docker\run.ps1 stats --verbose                 # Lista detalhada por vídeo
```

Flags: `--index` (nome do índice, default: video_index), `--verbose/-v`

---

## 3. Coleção de Referência

Gera arquivos BEIR: `corpus.jsonl`, `queries.jsonl`, `qrels.tsv`, `hard_negatives.jsonl`.

### 3.1 Pipeline Completo

#### Windows (PowerShell)

```powershell
# 1. Corpus
.\docker\run.ps1 corpus

# 2. Consultas (2 por vídeo: factoid + keyword)
.\docker\run.ps1 queries --provider openrouter -k SUA_KEY -w 2001

# 3. Pooling BM25 (requer container ES rodando)
.\docker\run.ps1 up
.\docker\run.ps1 pooling --top-k 20 --merge

# 4. Julgamento de relevância via LLM
.\docker\run.ps1 judge --provider openrouter -k SUA_KEY --batch-size 20

# 5. Montar coleção final (qrels + hard negatives)
.\docker\run.ps1 build-collection

# 6. Avaliar métricas TREC
.\docker\run.ps1 evaluate

# 7. Opcional: desligar ES
.\docker\run.ps1 down
```

#### Linux (bash)

```bash
# 1. Corpus
./docker/run.sh corpus

# 2. Consultas
./docker/run.sh queries --provider openrouter -k SUA_KEY -w 2001

# 3. Pooling
./docker/run.sh up
./docker/run.sh pooling --top-k 20 --merge

# 4. Julgamento
./docker/run.sh judge --provider openrouter -k SUA_KEY --batch-size 20

# 5. Montar
./docker/run.sh build-collection

# 6. Avaliar
./docker/run.sh evaluate

# 7. Desligar
./docker/run.sh down
```

### 3.2 Comandos Detalhados

#### `corpus`

```powershell
.\docker\run.ps1 corpus
.\docker\run.ps1 corpus --only-with-video          # Só vídeos com MP4
```

Flags: `--metadata` (default: data/metadata/videos_metadata.json), `--corpus-dir`, `--corpus-file`, `--only-with-video`, `--video-dir`

#### `queries`

Gera **2 consultas por vídeo** (factoid + keyword) via LLM.

```powershell
.\docker\run.ps1 queries --provider openrouter -k SUA_KEY -w 10
.\docker\run.ps1 queries --provider lm-studio -k http://localhost:1234/v1 -w 10
.\docker\run.ps1 queries --provider openrouter -k SUA_KEY --force -w 5
.\docker\run.ps1 queries --provider openrouter -k SUA_KEY -o 50 -w 10
.\docker\run.ps1 queries --provider openrouter -k SUA_KEY --base-url https://openrouter.ai/api/v1
```

Flags: `--provider`, `--model`, `--api-key/-k`, `--base-url`, `--window/-w`, `--offset/-o`, `--force/-f`, `--sleep`, `--no-export`, `--input`, `--output`

#### `pooling`

BM25 pooling (Okapi, L, Plus). Roda dentro do container (não requer ES).

```powershell
.\docker\run.ps1 pooling                           # top-20, 3 variantes
.\docker\run.ps1 pooling --top-k 50 --merge        # top-50 com merge
.\docker\run.ps1 pooling --variants bm25,bm25l     # só 2 variantes
```

Flags: `--top-k` (default: 20), `--variants` (default: bm25,bm25l,bm25plus), `--workers`, `--merge`, `--corpus`, `--queries`, `--output`

#### `judge`

Julgamento de relevância (escala 0-3) para cada par (query, candidato) via LLM.

```powershell
.\docker\run.ps1 judge --provider openrouter -k SUA_KEY -w 5 --batch-size 20
.\docker\run.ps1 judge --provider openrouter -k SUA_KEY --force --query-ids q_001,q_005
.\docker\run.ps1 judge --provider openrouter -k SUA_KEY -w 1 --batch-size 10
.\docker\run.ps1 judge --provider openrouter -k SUA_KEY --base-url https://openrouter.ai/api/v1
```

Flags: `--provider`, `--model`, `--api-key/-k`, `--base-url`, `--window/-w`, `--offset/-o`, `--force`, `--query-ids`, `--batch-size` (default: 0 = query inteira), `--max-doc-chars` (default: 300), `--sleep`, `--no-progress`

#### `build-collection`

Monta qrels.tsv e hard_negatives.jsonl a partir dos julgamentos LLM + auto qrels.

```powershell
.\docker\run.ps1 build-collection
```

Flags: `--hard-top-k` (default: 100), `--hard-per-query` (default: 10), `--validate`, `--corpus`, `--queries`, `--judgments`, `--pools`, `--qrels`, `--output-qrels`, `--hard-negatives`

#### `evaluate`

Calcula métricas TREC (P@K, R@K, MRR, MAP, nDCG@10).

```powershell
.\docker\run.ps1 evaluate
.\docker\run.ps1 evaluate --k 5,10,20,100
.\docker\run.ps1 evaluate --pool-source merged
```

Flags: `--k` (default: 10,100), `--pool-source` (default: merged), `--pools`, `--qrels`, `--queries`, `--output`

---

## 4. Comandos do Run Script

### 4.1 Resumo Geral

```
Uso: .\docker\run.ps1 {comando} [args]    (Windows)
Uso: ./docker/run.sh {comando} [args]     (Linux)

--- Search App (requer Docker + ES) ---
  up       Inicia Elasticsearch
  down     Para todos os servicos
  build    Reconstroi a imagem Docker
  index    Pipeline de indexacao (--window, --offset, --input, --batch, --corpus)
  app      Inicia interface Streamlit (localhost:8501)
  shell    Abre bash interativo no container
  stats    Estatisticas dos videos indexados (--index, --verbose)
  logs     Segue os logs do container

--- Colecao de Referencia (roda local ou container) ---
  corpus   Gera corpus.jsonl dos metadados (--only-with-video, --metadata)
  queries  Gera 2 consultas por video via LLM (--provider, --window, --offset, --force, --api-key)
  pooling  BM25 pooling dentro do container (--top-k, --variants, --merge)
  judge    Julga relevancia via LLM (--provider, --window, --batch-size, --api-key)
  build-collection  Monta qrels + hard negatives
  evaluate Avalia metricas TREC (--k, --pool-source)
```

### 4.2 Mapa de Execução (Onde Roda Cada Comando)

| Comando | Roda no | Container | Requer ES | Requer GPU |
|---------|---------|-----------|-----------|------------|
| `up` | Host (Docker) | elasticsearch | — | ❌ |
| `down` | Host (Docker) | — | — | ❌ |
| `build` | Host (Docker) | — | — | ❌ |
| `index` | Container `system` | sim | ✅ | ✅ |
| `app` | Container `system` | sim | ✅ | ❌ |
| `shell` | Container `system` | sim | ❌ | ❌ |
| `stats` | Container `system` | sim | ✅ | ❌ |
| `logs` | Host (Docker) | — | — | ❌ |
| `corpus` | Host (Python) | não | ❌ | ❌ |
| `queries` | Host (Python) | não | ❌ | ❌ |
| `pooling` | Container `system` | sim | ❌ | ❌ |
| `judge` | Host (Python) | não | ❌ | ❌ |
| `build-collection` | Host (Python) | não | ❌ | ❌ |
| `evaluate` | Host (Python) | não | ❌ | ❌ |

---

## 5. Workflows

### 5.1 Workflow Completo: Search App (Windows)

```powershell
# 1. Build da imagem (primeira vez ou após mudanças)
.\docker\run.ps1 build

# 2. Subir Elasticsearch
.\docker\run.ps1 up

# 3. Indexar vídeos
.\docker\run.ps1 index -w 10

# 4. Abrir interface web
.\docker\run.ps1 app
# → http://localhost:8501

# 5. Ver estatísticas
.\docker\run.ps1 stats

# 6. Logs (opcional)
.\docker\run.ps1 logs

# 7. Parar
.\docker\run.ps1 down
```

### 5.2 Workflow Completo: Coleção de Referência (Windows)

```powershell
# 1. Corpus
.\docker\run.ps1 corpus

# 2. Gerar consultas para todos os vídeos
.\docker\run.ps1 queries --provider openrouter -k SUA_KEY -w 2001

# 3. Subir container + Pooling BM25
.\docker\run.ps1 up
.\docker\run.ps1 pooling --top-k 20 --merge

# 4. Julgar relevância
.\docker\run.ps1 judge --provider openrouter -k SUA_KEY --batch-size 20

# 5. Montar coleção final
.\docker\run.ps1 build-collection

# 6. Avaliar
.\docker\run.ps1 evaluate

# 7. Opcional: desligar ES
.\docker\run.ps1 down
```

### 5.3 Workflow: Coleção de Referência (Linux)

```bash
# 1. Corpus
./docker/run.sh corpus

# 2. Gerar consultas
./docker/run.sh queries --provider openrouter -k SUA_KEY -w 2001

# 3. Pooling
./docker/run.sh up
./docker/run.sh pooling --top-k 20 --merge

# 4. Julgar
./docker/run.sh judge --provider openrouter -k SUA_KEY --batch-size 20

# 5. Montar
./docker/run.sh build-collection

# 6. Avaliar
./docker/run.sh evaluate

# 7. Desligar
./docker/run.sh down
```

---

## 6. Scripts Independentes

Scripts que **não têm atalho** nos run scripts — executar diretamente com Python.

### `download_workflow.py`

Download de vídeos do YouTube a partir do dataset ActivityNet.

```bash
python src/download_workflow.py -n 10 --subset validation
python src/download_workflow.py -n 50 --subset training --browser chrome
python src/download_workflow.py -n 100 --window              # Modo janela móvel
python src/download_workflow.py -n 10 --parity                # Corrigir inconsistências
python src/download_workflow.py -t                             # Preencher transcripts (backfill)
```

Flags: `--n-videos/-n`, `--subset` (training/validation/testing), `--json-path`, `--output-dir`, `--metadata-dir`, `--browser` (chrome/firefox/edge/none), `--force`, `--window/-w` (modo janela móvel), `--parity/-p`, `--transcript/-t`

### `filter_valid.py`

Filtra metadados, embeddings e vídeos para apenas aqueles que possuem consultas geradas.

```bash
python src/filter_valid.py
```

### `sort_queries.py`

Ordena arquivos de consultas por ID.

```bash
python src/sort_queries.py
```

### `check_queries.py`

Valida e deduplica arquivos de consultas.

```bash
python src/check_queries.py
```

### `main_search.py`

Demo de busca via terminal (legado — prefira `app.py` + Streamlit).

---

## 7. Provedores LLM

| Provider | Scripts | Como usar |
|----------|---------|-----------|
| `openai` | queries, judge | `--provider openai -k sk-...` |
| `anthropic` | queries, judge | `--provider anthropic -k sk-ant-...` |
| `gemini` | queries, judge | `--provider gemini -k AIza...` |
| `deepseek` | queries, judge | `--provider deepseek -k sk-...` |
| `openrouter` | queries, judge | `--provider openrouter -k sk-or-...` |
| `lm-studio` | queries, judge | `--provider lm-studio -k http://localhost:1234/v1` |
| `ollama` | queries, judge | `--provider ollama -k http://localhost:11434` |

A chave também pode ser definida via variável de ambiente (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, `LMSTUDIO_BASE_URL`, `OLLAMA_BASE_URL`). Use `--base-url` para sobrescrever a URL base do provedor.

### Exemplo OpenRouter

```powershell
.\docker\run.ps1 queries --provider openrouter -k CHAVE -w 10
.\docker\run.ps1 judge --provider openrouter -k CHAVE -w 5 --batch-size 20
```
