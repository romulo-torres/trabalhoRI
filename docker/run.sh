#!/bin/bash
set -e

COMPOSE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$COMPOSE_DIR"

COMPOSE_FILES="-f docker-compose.yml"

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
        echo "Executando gerador de queries (LLM)..."
        cd "$COMPOSE_DIR/.."
        python src/generate_queries.py "$@"
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
        echo "Uso: $0 {up|down|index|app|shell|build|queries|stats|logs}"
        echo ""
        echo "  up       - Inicia Elasticsearch"
        echo "  down     - Para todos os servicos"
        echo "  index    - Executa pipeline de indexacao (aceita: --window, --offset, --input, --batch)"
        echo "  app      - Inicia interface Streamlit"
        echo "  shell    - Abre bash interativo no container"
        echo "  build    - Reconstroi a imagem Docker"
        echo "  queries  - Gera queries de busca via LLM (local, fora do Docker)"
        echo "  stats    - Estatisticas dos videos indexados (aceita: --index, --verbose)"
        echo "  logs     - Segue os logs"
        ;;
esac
