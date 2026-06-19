#!/bin/bash
set -e

mkdir -p /app/data/videos /app/data/metadata /app/data/embeddings /app/logs

if [ -n "$ES_HOST" ]; then
    echo "Aguardando Elasticsearch em $ES_HOST..."
    until curl -s "$ES_HOST" >/dev/null 2>&1; do
        sleep 2
    done
    echo "Elasticsearch disponivel!"
fi

exec "$@"