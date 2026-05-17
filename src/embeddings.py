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
import torchaudio
import librosa

import keyframes as ky

# Compatibilidade com versões antigas do NumPy que usavam np.float_
np.float_ = np.float64

# Cache de modelos (evita reload)
_clip_model = None
_clip_preprocess = None
_clap_model = None
_device = None


# ==============================
# Carregar modelo CLIP
# ==============================
# No início de embeddings.py, após os imports

# Cache de modelos (evita reload)
_clip_model = None
_clip_preprocess = None
_clap_model = None
_device = None


def load_all_models(device: str | None = None):
    """
    Carrega CLIP e CLAP uma única vez. Usa cache global.
    Retorna (clip_model, clip_preprocess, clap_model, device).
    """
    global _clip_model, _clip_preprocess, _clap_model, _device

    # Se já foram carregados, retorna imediatamente
    if _clip_model is not None and _clap_model is not None:
        return _clip_model, _clip_preprocess, _clap_model, _device

    if device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        _device = device

    # CLIP
    _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=_device)
    _clip_model.eval()

    # CLAP
    _clap_model = laion_clap.CLAP_Module(enable_fusion=False)
    _clap_model.load_ckpt()
    _clap_model.to(_device)
    _clap_model.eval()

    return _clip_model, _clip_preprocess, _clap_model, _device

def clear_model_cache():
    global _clip_model, _clip_preprocess, _clap_model, _device
    _clip_model = None
    _clip_preprocess = None
    _clap_model = None
    _device = None
    torch.cuda.empty_cache()  # libera memória GPU, se aplicável

# ==============================
# Converter frame OpenCV → PIL
# ==============================
def frame_to_pil(frame: np.ndarray) -> Image.Image:
    """OpenCV armazena em BGR; PIL espera RGB."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


# ==============================
# Embedding de um único frame
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
# Embedding de uma janela (agregação de frames)
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
# Gerar embeddings para todas as janelas (modo antigo, por keyframes)
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
                "embedding":     embedding,
                "modality":      "video",
            })
        except Exception as e:
            print(f"[WARN] Janela ignorada (center_frame={w.get('center_frame')}): {e}")

    return results


# ==============================
# Gerar embeddings de vídeo a partir de cenas (modo novo)
# ==============================
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
    Para cada cena, extrai N partes usando split_scene_into_parts (com limite de
    max_frames) e gera um embedding agregado da janela correspondente.
    Retorna lista de dicts com metadados e embedding.
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
                emb = embed_window(
                    part["frames"], model, preprocess, device, method=method
                )
                results.append({
                    "scene":        (scene_start, scene_end),
                    "part_index":   part["part_index"],
                    "center_frame": part["center_frame"],
                    "timestamp_sec": part["timestamp_sec"],
                    "embedding":    emb,
                    "modality":     "video",
                })
            except Exception as e:
                print(f"[WARN] Embedding ignorado cena={scene_start:.1f}-{scene_end:.1f}, "
                      f"parte={part['part_index']}: {e}")

    return results


# ==============================
# Salvar embeddings em JSON (suporte a modality)
# ==============================
def save_embeddings_json(embeddings: list[dict], path: str = "../data/embeddings.json") -> None:
    """
    Serializa a lista de embeddings para JSON.
    Cria o diretório pai automaticamente se não existir.
    Cada item pode ter a chave 'modality' (opcional, padrão 'video').
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    serializable = []
    for item in embeddings:
        d = {
            "embedding": item["embedding"].tolist(),
        }
        # Copia metadados que existirem
        for key in ["center_frame", "timestamp_sec", "scene", "part_index", "modality"]:
            if key in item:
                d[key] = item[key]
        serializable.append(d)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f)

    print(f"Embeddings salvos em '{path}' ({len(serializable)} itens).")


# ==============================
# Carregar embeddings do JSON (suporte a modality)
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
        # Garante que 'modality' exista
        if "modality" not in item:
            item["modality"] = "video"

    return data


# ======================================================================
#  NOVAS FUNÇÕES PARA ÁUDIO COM CLAP
# ======================================================================

# Carregar modelo CLAP
def load_clap_model(device: str | None = None):
    """
    Carrega o modelo CLAP (HTSAT-fused) e retorna o objeto modelo.
    O modelo é carregado no device escolhido (CPU/CUDA).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()  # baixa o checkpoint padrão se necessário
    model.to(device)
    model.eval()
    return model, device

# Extrair áudio do vídeo (via ffmpeg + librosa)
def extract_audio_from_video(video_path: str, sr: int = 16000):
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
            tmp_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        audio_np, _ = librosa.load(tmp_path, sr=sr, mono=True)
        return audio_np, sr
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# Embedding de um segmento de áudio com CLAP
def embed_audio_segment(audio: np.ndarray, sr: int, clap_model, device: str) -> np.ndarray:
    import librosa
    # Garante array 1D float32
    audio = np.atleast_1d(np.asarray(audio, dtype=np.float32))
    if audio.ndim != 1:
        audio = audio.flatten()
    # Se ficou muito curto, adiciona silêncio
    if len(audio) < 2:
        audio = np.pad(audio, (0, 2 - len(audio)), mode='constant')
    # Reamostra para 48 kHz (esperado pelo CLAP)
    target_sr = 48000
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    embedding = clap_model.get_audio_embedding_from_data(x=audio)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.astype(np.float32)


# Gerar embeddings de áudio alinhados com os segmentos de vídeo
def generate_audio_embeddings_from_segments(
    video_path: str,
    segments: list[dict],      # cada dict tem 'timestamp_sec' (centro) e duração implícita via segment_duration
    clap_model,
    device: str,
    segment_duration: float = 1.5,
    sr: int = 16000,
) -> list[dict]:
    full_audio, audio_sr = extract_audio_from_video(video_path, sr=sr)
    results = []
    for seg in segments:
        t_center = seg["timestamp_sec"]
        t_start = max(0, t_center - segment_duration / 2)
        t_end   = t_center + segment_duration / 2
        start_sample = int(t_start * audio_sr)
        end_sample   = min(len(full_audio), int(t_end * audio_sr))
        clip = full_audio[start_sample:end_sample]
        if len(clip) < 2:
            continue
        try:
            emb = embed_audio_segment(clip, audio_sr, clap_model, device)
            results.append({
                "embedding": emb,
            })
        except Exception as e:
            print(f"[WARN] Audio embedding failed for segment at {t_center:.2f}s: {e}")
    return results


def embed_audio_segment(audio: np.ndarray, sr: int, clap_model, device: str) -> np.ndarray:
    """
    Salva o segmento de áudio em um arquivo WAV temporário e usa
    get_audio_embedding_from_filelist do CLAP para gerar o embedding.
    """
    import tempfile
    import soundfile as sf

    # Garante que o áudio seja um array 1D float32
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if len(audio) == 0:
        raise ValueError("Segmento de áudio vazio.")

    # Salva em arquivo temporário
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        sf.write(tmp_path, audio, sr)
        embedding = clap_model.get_audio_embedding_from_filelist([tmp_path])
        # get_audio_embedding_from_filelist retorna um array 2D (1, dim)
        if embedding.ndim == 2:
            embedding = embedding.flatten()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.astype(np.float32)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# Gerar embeddings de áudio alinhados com as cenas
def generate_audio_embeddings_from_segments(
    video_path: str,
    segments: list[dict],
    clap_model,
    device: str,
    segment_duration: float = 1.5,
    sr: int = 16000,
) -> list[dict]:
    full_audio, audio_sr = extract_audio_from_video(video_path, sr=sr)
    results = []
    for seg in segments:
        t_center = seg["timestamp_sec"]
        t_start = max(0, t_center - segment_duration / 2)
        t_end   = t_center + segment_duration / 2
        start_sample = int(t_start * audio_sr)
        end_sample   = min(len(full_audio), int(t_end * audio_sr))
        clip = full_audio[start_sample:end_sample]

        # Descarta segmentos muito curtos (menos de 10 ms)
        if len(clip) < int(0.01 * audio_sr):
            continue

        try:
            emb = embed_audio_segment(clip, audio_sr, clap_model, device)
            results.append({"embedding": emb})
        except Exception as e:
            print(f"[WARN] Audio embedding failed for segment at {t_center:.2f}s: {e}")
    return results

# Gera os embeddings de áudio com base numa janela d etempo
def generate_audio_embeddings_from_windows(
    video_path: str,
    windows: list[dict],      # lista de dicts com 'timestamp_sec'
    clap_model,
    device: str,
    sr: int = 16000,
) -> list[dict]:
    """
    Extrai o áudio do vídeo e gera embeddings CLAP para cada janela temporal.
    Cada janela de áudio tem a mesma duração da janela de frames:
    1 segundo (padrão para k_seconds=0.5).
    Retorna lista de dicts com 'timestamp_sec' e 'embedding'.
    """
    full_audio, audio_sr = extract_audio_from_video(video_path, sr=sr)

    results = []
    window_duration = 1.0  # 2 * k_seconds (padrão)

    for w in windows:
        t_center = w["timestamp_sec"]
        t_start = max(0, t_center - window_duration / 2)
        t_end   = t_center + window_duration / 2

        start_sample = int(t_start * audio_sr)
        end_sample   = min(len(full_audio), int(t_end * audio_sr))
        clip = full_audio[start_sample:end_sample]

        if len(clip) == 0:
            continue

        try:
            emb_audio = embed_audio_segment(clip, audio_sr, clap_model, device)
            results.append({
                "timestamp_sec": t_center,
                "embedding": emb_audio,
            })
        except Exception as e:
            print(f"[WARN] Audio embedding falhou para t={t_center:.2f}s: {e}")

    return results