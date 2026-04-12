import os
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
import random
import time

import keyframes as ky
import embeddings as emb
import index_elastic as ind
from logger import setup_logger

logger = setup_logger()

# ==============================
# 1. Setup (env + login)
# ==============================
def setup_huggingface():
    load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")

    if not token:
        raise ValueError("Token não encontrado no .env")

    login(token=token)


# ==============================
# 2. Baixar dataset 
# ==============================
def load_video_dataset():
    return load_dataset(
        "facebook/PE-Video",
        split="test",
        streaming=True
    )

# ==============================
# 3. Salvar vídeo localmente
# ==============================
def save_video(sample, output_dir, idx):
    video_bytes = sample["mp4"]
    filename = f"video_{idx}.mp4"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "wb") as f:
        f.write(video_bytes)

    return filepath, filename


# ==============================
# 4. Processar UM vídeo
# ==============================
def process_video(video_path, video_id, model, preprocess, device, es):
    logger.info(f"🎬 Processando {video_id}...")

    windows = ky.generate_windows_stream_centered(
        video_path,
        k_seconds=0.5
    )

    logger.info(f"Janelas geradas: {len(windows)}")

    embeddings = emb.generate_embeddings(
        windows,
        model,
        preprocess,
        device
    )

    emb.save_embeddings_json(
        embeddings,
        path=f"../data/embeddings/{video_id}.json"
    )

    logger.info(f"Embeddings gerados: {len(embeddings)}")

    ind.index_embeddings_bulk(
        es,
        embeddings,
        index_name="video_index",
        video_id=video_id
    )

    logger.info(f"✅ Indexado: {video_id}")

    
def get_fixed_random_indices(n_samples=10, max_range=1000, seed=42):
    random.seed(seed)
    return sorted(random.sample(range(max_range), n_samples))

def download_videos(dataset, output_dir, limit=10):
    os.makedirs(output_dir, exist_ok=True)

    for i, sample in enumerate(dataset):
        if i >= limit:
            break

        try:
            filepath = os.path.join(output_dir, f"video_{i}.mp4")

            if os.path.exists(filepath):
                logger.info(f"Vídeo {i} já existe, pulando...")
                continue

            logger.info(f"Baixando vídeo {i}...")

            with open(filepath, "wb") as f:
                f.write(sample["mp4"])

        except Exception as e:
            logger.error(f"Erro ao baixar vídeo {i}: {e}")


def process_local_videos(video_dir, model, preprocess, device, es):
    for filename in os.listdir(video_dir):
        if not filename.endswith(".mp4"):
            continue

        video_path = os.path.join(video_dir, filename)
        video_id = filename.replace(".mp4", "")

        try:
            process_video(
                video_path,
                video_id,
                model,
                preprocess,
                device,
                es
            )

        except Exception as e:
            logger.error(f"Erro no vídeo {video_id}: {e}")


# ==============================
# 5. Pipeline principal
# ==============================
def main():
    # elasticsearch
    logger.info("Tentativa de conexão com o elasticsearch")
    es = ind.connect_elasticsearch()

    # setup
    setup_huggingface()

    # pasta
    output_dir = "../data/videos"
    os.makedirs(output_dir, exist_ok=True)

    ind.create_index(es, index_name="video_index", dims=512)

    # dataset
    dataset = load_video_dataset()
    download_videos(dataset, "../data/videos", limit=100)

    # modelo
    logger.info("Carregando modelo CLIP...")
    model, preprocess, device = emb.load_model()

    process_local_videos(
        "../data/videos",
        model,
        preprocess,
        device,
        es
    )
    selected_indices = list(range(10))

    logger.info(f"Índices escolhidos: {selected_indices}")
    logger.info("\n🚀 Pipeline finalizado!")


# ==============================
# 6. Executar
# ==============================
if __name__ == "__main__":
    main()