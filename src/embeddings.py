import torch
import numpy as np
from PIL import Image
import cv2
import clip  # OpenAI CLIP
import keyframes as ky
import json
import os


# ==============================
# 1. Carregar modelo CLIP
# ==============================
def load_model(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    return model, preprocess, device


# ==============================
# 2. Converter frame (OpenCV → PIL)
# ==============================
def frame_to_pil(frame):
    # OpenCV usa BGR, PIL usa RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


# ==============================
# 3. Embedding de um frame
# ==============================
def embed_frame(frame, model, preprocess, device):
    image = frame_to_pil(frame)
    image = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model.encode_image(image)
    
    # normalizar (muito importante pra busca vetorial)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    
    return embedding.cpu().numpy().flatten()


# ==============================
# 4. Embedding de uma janela (agregação)
# ==============================
def embed_window(window, model, preprocess, device, method="mean"):
    frame_embeddings = []
    
    for frame in window:
        emb = embed_frame(frame, model, preprocess, device)
        frame_embeddings.append(emb)
    
    frame_embeddings = np.array(frame_embeddings)
    
    # ==========================
    # Métodos de agregação
    # ==========================
    
    if method == "mean":
        final_embedding = np.mean(frame_embeddings, axis=0)
    
    elif method == "max":
        final_embedding = np.max(frame_embeddings, axis=0)
    
    elif method == "center":
        # usa só o frame central
        center_idx = len(frame_embeddings) // 2
        final_embedding = frame_embeddings[center_idx]
    
    else:
        raise ValueError("Método inválido")
    
    # normalizar de novo (IMPORTANTE)
    final_embedding = final_embedding / np.linalg.norm(final_embedding)
    
    return final_embedding


# ==============================
# 5. Gerar embeddings para todas janelas
# ==============================
def generate_embeddings(windows, model, preprocess, device):
    results = []
    
    for w in windows:
        embedding = embed_window(
            w["window"],
            model,
            preprocess,
            device,
            method="mean"
        )
        
        results.append({
            "center_frame": w["center_frame"],
            "timestamp_sec": w["timestamp_sec"],
            "embedding": embedding
        })
    
    return results


def save_embeddings_json(embeddings, path="../data/embeddings.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    serializable = []

    for item in embeddings:
        serializable.append({
            "center_frame": item["center_frame"],
            "timestamp_sec": item["timestamp_sec"],
            "embedding": item["embedding"].tolist()
        })

    with open(path, "w") as f:
        json.dump(serializable, f)

    print(f"Embeddings salvos em {path}")


# ==============================
# 6. Exemplo de uso
# ==============================
# if __name__ == "__main__":
    
#     video_path = "video.mp4"
    
#     print("Carregando modelo...")
#     model, preprocess, device = load_model()
    
#     print("Gerando janelas...")
#     windows = ky.generate_windows_stream_centered(video_path, k_seconds=0.5)
    
#     print("Gerando embeddings...")
#     embeddings = generate_embeddings(windows, model, preprocess, device)
    
#     print(f"Total de embeddings: {len(embeddings)}")
    
#     first = embeddings[0]
    
#     print("Exemplo:")
#     print(f"Timestamp: {first['timestamp_sec']}")
#     print(f"Dimensão do embedding: {len(first['embedding'])}")