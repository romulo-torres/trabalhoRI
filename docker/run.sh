#!/bin/bash
set -e

COMPOSE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$COMPOSE_DIR"

COMPOSE_FILES="-f docker-compose.yml"
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    COMPOSE_FILES="-f docker-compose.yml -f docker-compose.gpu.yml"
fi

case "${1:-help}" in
    up)
        echo "Iniciando Elasticsearch..."
        docker compose up -d elasticsearch
        echo "Elasticsearch pronto em localhost:9200"
        ;;
    down)
        echo "Parando todos os servicos..."
        docker compose down
        ;;
    index)
        shift
        echo "Executando pipeline de indexacao... args: $@"
        docker compose $COMPOSE_FILES run --rm system python src/main_index.py "$@"
        ;;
    corpus)
        shift
        echo "Gerando corpus.jsonl a partir dos metadados..."
        cd "$COMPOSE_DIR/.."
        python src/build_corpus.py "$@"
        ;;
    app)
        echo "Iniciando Streamlit em http://localhost:8501..."
        docker compose $COMPOSE_FILES run --rm -p 8501:8501 system streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0
        ;;
    shell)
        echo "Abrindo shell interativo..."
        docker compose run --rm system bash
        ;;
    build)
        build_args=""
        if [ -f "../.env" ]; then
            url=$(grep -o '^PYTORCH_INDEX_URL=[^ ]*' "../.env" 2>/dev/null | head -1 | cut -d= -f2-)
            if [ -n "$url" ]; then
                build_args="--build-arg PYTORCH_INDEX_URL=$url"
                echo "Usando PYTORCH_INDEX_URL=$url"
            fi
        fi
        echo "Construindo imagem..."
        docker compose build $build_args system
        ;;
    queries)
        shift
        echo "Gerando 4 consultas por video via LLM... args: $@"
        cd "$COMPOSE_DIR/.."
        python src/generate_queries.py "$@"
        ;;
    pooling)
        shift
        echo "Executando BM25 pooling... args: $@"
        docker compose $COMPOSE_FILES run --rm system python src/pooling.py "$@"
        ;;
    judge)
        shift
        echo "Executando julgamento de relevancia via LLM... args: $@"
        cd "$COMPOSE_DIR/.."
        python src/judge_relevance.py "$@"
        ;;
    build-collection)
        shift
        echo "Montando colecao BEIR final..."
        cd "$COMPOSE_DIR/.."
        python src/build_collection.py "$@"
        ;;
    evaluate)
        shift
        echo "Avaliando metricas TREC..."
        cd "$COMPOSE_DIR/.."
        python src/evaluate.py "$@"
        ;;
    stats)
        shift
        echo "Estatisticas do indice ES..."
        docker compose $COMPOSE_FILES run --rm system python src/stats_index.py "$@"
        ;;
    logs)
        docker compose logs -f
        ;;
    *)
        echo "Uso: $0 {comando} [args]"
        echo ""
        echo "--- Search App (requer Docker + ES) ---"
        echo "  up       Inicia Elasticsearch"
        echo "  down     Para todos os servicos"
        echo "  build    Reconstroi a imagem Docker"
        echo "  index    Pipeline de indexacao (--window, --offset, --input, --batch, --corpus)"
        echo "  app      Inicia interface Streamlit (localhost:8501)"
        echo "  shell    Abre bash interativo no container"
        echo "  stats    Estatisticas dos videos indexados (--index, --verbose)"
        echo "  logs     Segue os logs do container"
        echo ""
        echo "--- Colecao de Referencia (roda local, sem Docker) ---"
        echo "  corpus   Gera corpus.jsonl dos metadados (--only-with-video, --metadata)"
        echo "  queries  Gera consultas via LLM (--provider, --window, --offset, --force, --api-key)"
        echo "  pooling  BM25 pooling dentro do container (--top-k, --variants, --merge)"
        echo "  judge    Julga relevancia via LLM (--provider, --window, --batch-size, --api-key)"
        echo "  build-collection  Monta qrels + hard negatives"
        echo "  evaluate Avalia metricas TREC (--k, --pool-source)"
        ;;
esac
