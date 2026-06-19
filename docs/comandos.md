# Coleção de Referência — ActivityNet Video Search

Sistema de busca texto-vídeo + geração de coleção de referência no formato BEIR, usando o dataset ActivityNet.

---

## 2. Search App

Requer **Docker com suporte a GPU** (CUDA) e Elasticsearch.

### 2.2 Comandos

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

### 2.3 Indexação

```powershell
.\docker\run.ps1 index                           # Todos pendentes
.\docker\run.ps1 index -w 10                     # 10 novos
.\docker\run.ps1 index -o 100 -w 50              # 50 a partir do 100
.\docker\run.ps1 index --corpus                  # Usa corpus.jsonl como fonte
.\docker\run.ps1 index --batch                   # Modo batch (mais RAM, mais rápido)
```

---

## 3. Coleção de Referência

Gera arquivos BEIR: `corpus.jsonl`, `queries.jsonl`, `qrels.tsv`, `hard_negatives.jsonl`.

### 3.2 Executando Passo a Passo

#### Windows (PowerShell)

```powershell
# 1. Corpus
.\docker\run.ps1 corpus

# 2. Consultas
.\docker\run.ps1 queries --provider openrouter -k SUA_KEY -w 2001

# 3. Pooling (requer ES rodando)
.\docker\run.ps1 up
.\docker\run.ps1 pooling --top-k 20 --merge

# 4. Julgamento
.\docker\run.ps1 judge --provider openrouter -k SUA_KEY --batch-size 20

# 5. Montar coleção
.\docker\run.ps1 build-collection

# 6. Avaliar
.\docker\run.ps1 evaluate
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

# 5. Montar coleção
./docker/run.sh build-collection

# 6. Avaliar
./docker/run.sh evaluate
```

#### `corpus`

```powershell
.\docker\run.ps1 corpus
.\docker\run.ps1 corpus --only-with-video          # Só vídeos com MP4
```

#### `queries`

```powershell
.\docker\run.ps1 queries --provider openrouter -k SUA_KEY -w 10
.\docker\run.ps1 queries --provider lm-studio -k http://localhost:1234/v1 -w 10
.\docker\run.ps1 queries --provider openrouter -k SUA_KEY --force -w 5
.\docker\run.ps1 queries --provider openrouter -k SUA_KEY -o 50 -w 10
```

Flags: `--provider`, `--model`, `--api-key/-k`, `--window/-w`, `--offset/-o`, `--force/-f`, `--sleep`, `--no-export`

#### `pooling`

```powershell
.\docker\run.ps1 pooling                           # top-20, 3 variantes
.\docker\run.ps1 pooling --top-k 50 --merge        # top-50 com merge
.\docker\run.ps1 pooling --variants bm25,bm25l     # só 2 variantes
```

Flags: `--top-k` (default: 20), `--variants`, `--workers`, `--merge`, `--corpus`, `--queries`

#### `judge`

```powershell
.\docker\run.ps1 judge --provider openrouter -k SUA_KEY -w 5 --batch-size 20
.\docker\run.ps1 judge --provider openrouter -k SUA_KEY --force --query-ids q_001,q_005
.\docker\run.ps1 judge --provider openrouter -k SUA_KEY -w 1 --batch-size 10
```

Flags: `--provider`, `--model`, `--api-key/-k`, `--window/-w`, `--offset/-o`, `--force`, `--query-ids`, `--batch-size` (default: 0 = query inteira), `--max-doc-chars` (default: 300), `--sleep`

#### `build-collection`

```powershell
.\docker\run.ps1 build-collection
```

Flags: `--hard-top-k` (default: 100), `--hard-per-query` (default: 10), `--validate`, `--corpus`, `--queries`, `--judgments`, `--pools`, `--qrels`

#### `evaluate`

```powershell
.\docker\run.ps1 evaluate
.\docker\run.ps1 evaluate --k 5,10,20,100
.\docker\run.ps1 evaluate --pool-source bm25
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
  index    Pipeline de indexacao
  app      Inicia interface Streamlit (localhost:8501)
  shell    Abre bash interativo no container
  stats    Estatisticas dos videos indexados
  logs     Segue os logs do container

--- Colecao de Referencia (roda local) ---
  corpus   Gera corpus.jsonl dos metadados
  queries  Gera consultas via LLM
  pooling  BM25 pooling dentro do container
  judge    Julga relevancia via LLM
  build-collection  Monta qrels + hard negatives
  evaluate Avalia metricas TREC
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

# 3. Subir ES + Pooling BM25
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


## 6. Scripts Independentes

Scripts que **não têm atalho** nos run scripts — executar diretamente com Python.

### `download_workflow.py`

Download de vídeos do YouTube a partir do dataset ActivityNet.

```bash
python src/download_workflow.py --n-videos 10 --subset validation
python src/download_workflow.py --window 50 --subset training --browser chrome
```

Flags: `--n-videos`, `--subset` (training/validation/testing), `--window`, `--parity`, `--transcript`, `--browser`, `--force`

### `main_search.py`

Demo de busca via terminal (não usar, prefira `app.py` + Streamlit).

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

### Exemplo OpenRouter

```powershell
.\docker\run.ps1 queries --provider openrouter -k CHAVE -w 10
.\docker\run.ps1 judge --provider openrouter -k CHAVE -w 5 --batch-size 20
```
