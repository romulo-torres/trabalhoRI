import os
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
import random

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

def download_videos(dataset, output_dir, target_total=100):
    os.makedirs(output_dir, exist_ok=True)

    existing = count_videos(output_dir)

    if existing >= target_total:
        logger.info("Já tem vídeos suficientes.")
        return

    for i, sample in enumerate(dataset):
        if existing >= target_total:
            break

        filepath = os.path.join(output_dir, f"video_{existing}.mp4")

        if os.path.exists(filepath):
            existing += 1
            continue

        logger.info(f"Baixando vídeo {existing}...")

        try:
            with open(filepath, "wb") as f:
                f.write(sample["mp4"])

            existing += 1

        except Exception as e:
            logger.error(f"Erro ao baixar vídeo {existing}: {e}")

def already_indexed(es, video_id):
    res = es.search(
        index="video_index",
        query={"term": {"video_id": video_id}},
        size=1
    )
    return len(res["hits"]["hits"]) > 0


def process_local_videos(video_dir, model, preprocess, device, es):
    for filename in sorted(os.listdir(video_dir)):
        if not filename.endswith(".mp4"):
            continue

        video_id = filename.replace(".mp4", "")

        if already_indexed(es, video_id):
            logger.info(f"{video_id} já indexado, pulando...")
            continue

        video_path = os.path.join(video_dir, filename)

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


def count_videos(video_dir):
    return len([
        f for f in os.listdir(video_dir)
        if f.endswith(".mp4")
    ])



# ==============================
# 5. Pipeline principal
# ==============================
def main():
    # 1. Elasticsearch
    logger.info("Tentativa de conexão com o elasticsearch")
    es = ind.connect_elasticsearch()

    # 2. Pasta de vídeos
    video_dir = "./data/videos"
    os.makedirs(video_dir, exist_ok=True)

    # 3. Verificar quantidade
    num_videos = count_videos(video_dir)

    if num_videos < 100:
        logger.info(f"Só existem {num_videos} vídeos. Baixando até completar 100...")
        
        setup_huggingface()
        dataset = load_video_dataset()
        download_videos(dataset, video_dir, target_total=100)
    else:
        logger.info(f"Já existem {num_videos} vídeos. Pulando download.")

    # 4. Criar índice
    ind.create_index(es, index_name="video_index", dims=512)

    # 5. Modelo
    logger.info("Carregando modelo CLIP...")
    model, preprocess, device = emb.load_model()

    # 6. Processar vídeos locais
    process_local_videos(
        video_dir,
        model,
        preprocess,
        device,
        es
    )


# ==============================
# 6. Executar
# ==============================
if __name__ == "__main__":
    main()