import os
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
import random

import keyframes as ky
import embeddings as emb
import index_elastic as ind


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
# 2. Baixar dataset (streaming)
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
    print(f"\n🎬 Processando {video_id}...")

    # 1. Gerar janelas
    windows = ky.generate_windows_stream_centered(
        video_path,
        k_seconds=0.5
    )

    print(f"Janelas geradas: {len(windows)}")

    # 2. Gerar embeddings
    embeddings = emb.generate_embeddings(
        windows,
        model,
        preprocess,
        device
    )

    embeddings = emb.generate_embeddings(...)

    emb.save_embeddings_json(embeddings, video_id)

    print(f"Embeddings gerados: {len(embeddings)}")

    # 3. Indexar no Elasticsearch
    ind.index_embeddings_bulk(
        es,
        embeddings,
        index_name="video_index",
        video_id=video_id
    )

    print(f"✅ Indexado: {video_id}")

    
def get_fixed_random_indices(n_samples=10, max_range=1000, seed=42):
    random.seed(seed)
    return sorted(random.sample(range(max_range), n_samples))


# ==============================
# 5. Pipeline principal
# ==============================
def main():
    # setup
    setup_huggingface()

    # dataset
    dataset = load_video_dataset()

    # pasta
    output_dir = "../data/videos"
    os.makedirs(output_dir, exist_ok=True)

    # elasticsearch
    es = ind.connect_elasticsearch()
    ind.create_index(es, index_name="video_index", dims=512)

    # modelo
    print("Carregando modelo CLIP...")
    model, preprocess, device = emb.load_model()

    selected_indices = get_fixed_random_indices(n_samples=10, max_range=1000)

    print(f"Índices escolhidos: {selected_indices}")

    for i, sample in enumerate(dataset):
        if i > max(selected_indices):
            break

        if i not in selected_indices:
            continue

        try:
            video_path, filename = save_video(sample, output_dir, i)
            video_id = f"video_{i}"

            process_video(
                video_path,
                video_id,
                model,
                preprocess,
                device,
                es
            )

        except Exception as e:
            print(f"Erro no vídeo {i}: {e}")

            # limitar (IMPORTANTE!)
            if i >= 4:
                break

    print("\n🚀 Pipeline finalizado!")


# ==============================
# 6. Executar
# ==============================
if __name__ == "__main__":
    main()