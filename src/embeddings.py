import os
import json

import torch
import numpy as np
from PIL import Image
import cv2
import clip  # OpenAI CLIP

import keyframes as ky

# Compatibilidade com versões antigas do NumPy que usavam np.float_
np.float_ = np.float64


# ==============================
# 1. Carregar modelo CLIP
# ==============================
def load_model(device: str | None = None):
    """
    Carrega o modelo CLIP ViT-B/32.
    Se device não for informado, usa CUDA quando disponível.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()  # garante modo inferência (desativa dropout etc.)

    return model, preprocess, device


# ==============================
# 2. Converter frame OpenCV → PIL
# ==============================
def frame_to_pil(frame: np.ndarray) -> Image.Image:
    """OpenCV armazena em BGR; PIL espera RGB."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


# ==============================
# 3. Embedding de um único frame
# ==============================
def embed_frame(
    frame:      np.ndarray,
    model,
    preprocess,
    device:     str,
) -> np.ndarray:
    """
    Gera o embedding normalizado de um frame (array NumPy BGR).
    Retorna vetor 1-D float32 de dimensão 512.
    """
    image = frame_to_pil(frame)
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image)

    # Normaliza para busca por cosseno
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().flatten()


# ==============================
# 4. Embedding de uma janela (agregação de frames)
# ==============================
def embed_window(
    window,
    model,
    preprocess,
    device:  str,
    method:  str = "mean",
) -> np.ndarray:
    """
    Agrega os embeddings de todos os frames de uma janela.

    Métodos disponíveis:
        "mean"   — média dos embeddings (padrão, mais robusto)
        "max"    — max-pooling por dimensão
        "center" — usa apenas o frame central
    """
    if not window:
        raise ValueError("A janela está vazia — nenhum frame para processar.")

    frame_embeddings = np.array([
        embed_frame(frame, model, preprocess, device)
        for frame in window
    ])  # shape: (n_frames, 512)

    if method == "mean":
        aggregated = np.mean(frame_embeddings, axis=0)
    elif method == "max":
        aggregated = np.max(frame_embeddings, axis=0)
    elif method == "center":
        aggregated = frame_embeddings[len(frame_embeddings) // 2]
    else:
        raise ValueError(f"Método de agregação inválido: '{method}'. Use 'mean', 'max' ou 'center'.")

    # Renormaliza após agregação (essencial para busca por cosseno)
    norm = np.linalg.norm(aggregated)
    if norm == 0:
        raise ValueError("Embedding agregado resultou em vetor nulo.")

    return aggregated / norm


# ==============================
# 5. Gerar embeddings para todas as janelas
# ==============================
def generate_embeddings(
    windows:    list[dict],
    model,
    preprocess,
    device:     str,
    method:     str = "mean",
) -> list[dict]:
    """
    Recebe a lista de janelas produzida por keyframes.py e retorna
    lista de dicts com chaves: center_frame, timestamp_sec, embedding.
    """
    results = []

    for w in windows:
        try:
            embedding = embed_window(
                w["window"],
                model,
                preprocess,
                device,
                method=method,
            )
            results.append({
                "center_frame":  w["center_frame"],
                "timestamp_sec": w["timestamp_sec"],
                "embedding":     embedding,          # np.ndarray — serializar ao salvar
            })
        except Exception as e:
            # Loga o problema mas continua processando as outras janelas
            print(f"[WARN] Janela ignorada (center_frame={w.get('center_frame')}): {e}")

    return results


# ==============================
# 6. Salvar embeddings em JSON
# ==============================
def save_embeddings_json(embeddings: list[dict], path: str = "../data/embeddings.json") -> None:
    """
    Serializa a lista de embeddings para JSON.
    Cria o diretório pai automaticamente se não existir.
    """
    # Proteção: dirname retorna '' quando path não tem diretório
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    serializable = [
        {
            "center_frame":  item["center_frame"],
            "timestamp_sec": item["timestamp_sec"],
            "embedding":     item["embedding"].tolist(),
        }
        for item in embeddings
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f)

    print(f"Embeddings salvos em '{path}' ({len(serializable)} itens).")


# ==============================
# 7. Carregar embeddings do JSON
# ==============================
def load_embeddings_json(path: str) -> list[dict]:
    """
    Lê embeddings salvos por save_embeddings_json.
    Converte as listas de volta para np.ndarray.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item["embedding"] = np.array(item["embedding"], dtype=np.float32)

    return data