#!/bin/bash

echo "🚀 Iniciando pipeline de vídeos..."

# ==============================
# 1. Subir Docker (Elasticsearch)
# ==============================
echo "🐳 Subindo Elasticsearch via Docker..."
cd docker || exit
docker-compose up -d

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
# 3. Voltar pro projeto
# ==============================
cd ..

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