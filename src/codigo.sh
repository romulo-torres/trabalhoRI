#!/bin/bash

echo "🚀 Iniciando pipeline de vídeos..."

# ==============================
# IR PARA RAIZ DO PROJETO
# ==============================
cd "$(dirname "$0")/.." || exit

# ==============================
# 1. Subir Docker (Elasticsearch)
# ==============================
echo "🐳 Subindo Elasticsearch via Docker..."
docker-compose -f docker/docker-compose.yml up -d

# ==============================
# 2. Esperar Elasticsearch iniciar
# ==============================
echo "⏳ Aguardando Elasticsearch iniciar..."

until curl -s http://localhost:9200 >/dev/null; do
  echo "Esperando Elasticsearch..."
  sleep 5
done

echo "✅ Elasticsearch está pronto!"

# ==============================
# 4. Ativar ambiente virtual (se existir)
# ==============================
if [ -d ".venv" ]; then
  echo "🔧 Ativando ambiente virtual..."
  source .venv/bin/activate
fi

# ==============================
# 5. Instalar dependências (opcional)
# ==============================
echo "📦 Instalando dependências..."
pip install -r requirements.txt

# ==============================
# 6. Rodar pipeline
# ==============================
echo "🎬 Executando main.py..."
python src/main.py

echo "🎉 Pipeline finalizado!"