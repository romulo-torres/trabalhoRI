import os
import json
import subprocess
import tempfile

import torch
import numpy as np
from PIL import Image
import cv2
import clip  # OpenAI CLIP
import laion_clap  # CLAP
import librosa
import soundfile as sf

import keyframes as ky

# Compatibilidade com versões antigas do NumPy que usavam np.float_
np.float_ = np.float64

# Cache de modelos (evita reload)
_clip_model      = None
_clip_preprocess = None
_clap_model      = None
_device          = None


# ==============================================================================
# Carregar todos os modelos (CLIP + CLAP) — com cache global
# ==============================================================================
def load_all_models(device: str | None = None):
    """
    Carrega CLIP e CLAP uma única vez. Usa cache global.
    Retorna (clip_model, clip_preprocess, clap_model, device).
    """
    global _clip_model, _clip_preprocess, _clap_model, _device

    if _clip_model is not None and _clap_model is not None:
        return _clip_model, _clip_preprocess, _clap_model, _device

    if device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        _device = device

    # CLIP — ViT-B/32 produz embeddings de 512 dimensões para imagem e texto
    _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=_device)
    _clip_model.eval()

    # CLAP — embeddings de 512 dimensões para áudio e texto
    _clap_model = laion_clap.CLAP_Module(enable_fusion=False)
    _clap_model.load_ckpt()
    _clap_model.to(_device)
    _clap_model.eval()

    return _clip_model, _clip_preprocess, _clap_model, _device


def clear_model_cache():
    global _clip_model, _clip_preprocess, _clap_model, _device
    _clip_model      = None
    _clip_preprocess = None
    _clap_model      = None
    _device          = None
    torch.cuda.empty_cache()


# ==============================================================================
# Utilitários de frame
# ==============================================================================
def frame_to_pil(frame: np.ndarray) -> Image.Image:
    """OpenCV armazena em BGR; PIL espera RGB."""
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# ==============================================================================
# Embedding de imagem (CLIP)
# ==============================================================================
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
    image = preprocess(frame_to_pil(frame)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().flatten()


def embed_window(
    window,
    model,
    preprocess,
    device:  str,
    method:  str = "mean",
) -> np.ndarray:
    """
    Agrega os embeddings de todos os frames de uma janela (até 45 frames).

    Métodos:
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
        raise ValueError(f"Método inválido: '{method}'. Use 'mean', 'max' ou 'center'.")

    norm = np.linalg.norm(aggregated)
    if norm == 0:
        raise ValueError("Embedding agregado resultou em vetor nulo.")
    return (aggregated / norm).astype(np.float32)


# ==============================================================================
# Embedding de texto (CLIP) — busca semântica texto → vídeo
# ==============================================================================
def embed_text_clip(text: str, model, device: str) -> np.ndarray:
    """
    Gera o embedding de texto usando CLIP.
    Retorna vetor 1-D float32 de dimensão 512, normalizado.
    Compatível com os embeddings de imagem gerados por embed_frame/embed_window.
    """
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        embedding = model.encode_text(tokens)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().flatten().astype(np.float32)


# ==============================================================================
# Embedding de texto (CLAP) — busca semântica texto → áudio
# ==============================================================================
def embed_text_clap(text: str, clap_model) -> np.ndarray:
    """
    Gera o embedding de texto usando CLAP.
    Retorna vetor 1-D float32 de dimensão 512, normalizado.
    Compatível com os embeddings de áudio gerados por embed_audio_segment.
    """
    embedding = clap_model.get_text_embedding([text])
    if embedding.ndim == 2:
        embedding = embedding.flatten()
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.astype(np.float32)


# ==============================================================================
# Gerar embeddings para todas as janelas (modo legado — por keyframes)
# ==============================================================================
def generate_embeddings(
    windows:    list[dict],
    model,
    preprocess,
    device:     str,
    method:     str = "mean",
) -> list[dict]:
    results = []
    for w in windows:
        try:
            embedding = embed_window(w["window"], model, preprocess, device, method=method)
            results.append({
                "center_frame":  w["center_frame"],
                "timestamp_sec": w["timestamp_sec"],
                "embedding":     embedding,
                "modality":      "video",
            })
        except Exception as e:
            print(f"[WARN] Janela ignorada (center_frame={w.get('center_frame')}): {e}")
    return results


# ==============================================================================
# Gerar embeddings de vídeo a partir de cenas (modo novo — por segmentos)
# ==============================================================================
def generate_embeddings_from_scenes(
    video_path:  str,
    scenes:      list[tuple[float, float]],
    model,
    preprocess,
    device:      str,
    n_parts:     int = 4,
    max_frames:  int = 45,
    method:      str = "mean",
) -> list[dict]:
    """
    Para cada cena, extrai N partes (até max_frames=45 por segmento) e gera
    um embedding agregado da janela correspondente.
    """
    results = []
    for scene_start, scene_end in scenes:
        parts = ky.split_scene_into_parts(
            video_path,
            scene_start,
            scene_end,
            n_parts=n_parts,
            max_window_frames=max_frames,
        )
        for part in parts:
            try:
                embedding = embed_window(
                    part["frames"], model, preprocess, device, method=method
                )
                results.append({
                    "scene":         (scene_start, scene_end),
                    "part_index":    part["part_index"],
                    "center_frame":  part["center_frame"],
                    "timestamp_sec": part["timestamp_sec"],
                    "embedding":     embedding,
                    "modality":      "video",
                })
            except Exception as e:
                print(
                    f"[WARN] Embedding ignorado cena={scene_start:.1f}-{scene_end:.1f}, "
                    f"parte={part['part_index']}: {e}"
                )
    return results


# ==============================================================================
# Serialização de embeddings
# ==============================================================================
def save_embeddings_json(embeddings: list[dict], path: str = "../data/embeddings.json") -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    serializable = []
    for item in embeddings:
        d = {"embedding": item["embedding"].tolist()}
        for key in ["center_frame", "timestamp_sec", "scene", "part_index", "modality"]:
            if key in item:
                d[key] = item[key]
        serializable.append(d)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f)

    print(f"Embeddings salvos em '{path}' ({len(serializable)} itens).")


def load_embeddings_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        item["embedding"] = np.array(item["embedding"], dtype=np.float32)
        if "modality" not in item:
            item["modality"] = "video"
    return data


# ==============================================================================
# Utilitários de áudio
# ==============================================================================
def extract_audio_from_video(video_path: str, sr: int = 16000) -> tuple[np.ndarray, int]:
    """Extrai o áudio do vídeo como array NumPy mono via ffmpeg + librosa."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
            tmp_path,
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        audio_np, _ = librosa.load(tmp_path, sr=sr, mono=True)
        return audio_np, sr
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ==============================================================================
# Embedding de áudio (CLAP)
# ==============================================================================
def embed_audio_segment(
    audio:      np.ndarray,
    sr:         int,
    clap_model,
    device:     str,
) -> np.ndarray:
    """
    Salva o segmento em arquivo WAV temporário e usa
    get_audio_embedding_from_filelist do CLAP.
    Retorna vetor 1-D float32 de dimensão 512, normalizado.
    """
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if len(audio) == 0:
        raise ValueError("Segmento de áudio vazio.")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        sf.write(tmp_path, audio, sr)
        embedding = clap_model.get_audio_embedding_from_filelist([tmp_path])
        if embedding.ndim == 2:
            embedding = embedding.flatten()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.astype(np.float32)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ==============================================================================
# Gerar embeddings de áudio alinhados com os segmentos de vídeo
# ==============================================================================
def generate_audio_embeddings_from_segments(
    video_path:       str,
    segments:         list[dict],
    clap_model,
    device:           str,
    segment_duration: float = 1.5,
    sr:               int   = 16000,
) -> list[dict]:
    """
    Para cada segmento de vídeo (com 'timestamp_sec' como centro),
    recorta a janela de áudio correspondente e gera embedding CLAP.
    Alinhado ao fluxo de 45 frames por segmento de indexing.py.
    """
    full_audio, audio_sr = extract_audio_from_video(video_path, sr=sr)
    results = []

    for seg in segments:
        t_center = seg["timestamp_sec"]
        t_start  = max(0.0, t_center - segment_duration / 2)
        t_end    = t_center + segment_duration / 2

        start_sample = int(t_start * audio_sr)
        end_sample   = min(len(full_audio), int(t_end * audio_sr))
        audio_chunk  = full_audio[start_sample:end_sample]

        # Descarta segmentos muito curtos (menos de 10 ms)
        if len(audio_chunk) < int(0.01 * audio_sr):
            continue

        try:
            embedding = embed_audio_segment(audio_chunk, audio_sr, clap_model, device)
            results.append({"embedding": embedding})
        except Exception as e:
            print(f"[WARN] Audio embedding falhou para t={t_center:.2f}s: {e}")

    return results


# ==============================================================================
# Gerar embeddings de áudio alinhados com janelas temporais (modo legado)
# ==============================================================================
def generate_audio_embeddings_from_windows(
    video_path: str,
    windows:    list[dict],
    clap_model,
    device:     str,
    sr:         int = 16000,
) -> list[dict]:
    """
    Versão legada: gera embeddings CLAP para cada janela temporal (1 segundo).
    """
    full_audio, audio_sr = extract_audio_from_video(video_path, sr=sr)
    window_duration = 1.0
    results = []

    for w in windows:
        t_center = w["timestamp_sec"]
        t_start  = max(0.0, t_center - window_duration / 2)
        t_end    = t_center + window_duration / 2

        start_sample = int(t_start * audio_sr)
        end_sample   = min(len(full_audio), int(t_end * audio_sr))
        audio_chunk  = full_audio[start_sample:end_sample]

        if len(audio_chunk) == 0:
            continue

        try:
            embedding = embed_audio_segment(audio_chunk, audio_sr, clap_model, device)
            results.append({
                "timestamp_sec": t_center,
                "embedding":     embedding,
            })
        except Exception as e:
            print(f"[WARN] Audio embedding falhou para t={t_center:.2f}s: {e}")

    return results