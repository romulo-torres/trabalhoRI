FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG PYTORCH_INDEX_URL=""
COPY requirements.txt .
RUN if [ -n "$PYTORCH_INDEX_URL" ]; then \
        echo "Instalando PyTorch de $PYTORCH_INDEX_URL"; \
        pip install --no-cache-dir torch torchvision torchaudio --index-url "$PYTORCH_INDEX_URL"; \
    else \
        pip install --no-cache-dir torch torchvision torchaudio; \
    fi
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

COPY data/activity_net.v1-3.min.json data/activity_net.v1-3.min.json

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]