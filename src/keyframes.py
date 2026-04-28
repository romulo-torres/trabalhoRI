import cv2
import numpy as np

# Compatibilidade com versões antigas do NumPy
np.float_ = np.float64


# ==============================
# 1. Extrair TODOS os frames
# Uso: quando o vídeo cabe inteiro na memória (vídeos curtos)
# ==============================
def extract_all_frames(video_path: str) -> tuple[list[np.ndarray], float]:
    """
    Lê todos os frames do vídeo para uma lista em memória.
    Retorna (frames, fps).

    ⚠️  Evite em vídeos longos — prefira generate_windows_stream_centered.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Não foi possível abrir o vídeo: '{video_path}'")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0   # fallback se FPS não reportado
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, fps


# ==============================
# 2. Índices sincronizados (1 por segundo)
# ==============================
def get_sync_indices(fps: float, total_frames: int) -> list[int]:
    """
    Retorna os índices dos frames-chave, um a cada segundo.
    Ex.: fps=30, total=300 → [0, 30, 60, …, 270]
    """
    interval = max(1, int(fps))   # evita divisão por zero se fps=0
    return list(range(0, total_frames, interval))


# ==============================
# 3. Janela temporal centrada num índice
# Uso: somente quando os frames já estão em memória (extract_all_frames)
# ==============================
def get_window(
    frames:     list[np.ndarray],
    center_idx: int,
    k_frames:   int,
) -> list[np.ndarray]:
    """
    Retorna os frames no intervalo [center_idx - k_frames, center_idx + k_frames].
    Índices fora dos limites são ignorados (sem padding).
    """
    start = max(0, center_idx - k_frames)
    end   = min(len(frames), center_idx + k_frames + 1)
    return frames[start:end]


# ==============================
# 4. Geração de janelas em streaming (baixa memória)
#
# k_seconds: contexto temporal de cada janela.
#   Ex.: fps=30, k_seconds=0.5 → k_frames=15
#   Cada janela terá 31 frames (15 antes + centro + 15 depois).
#
# start_time / end_time: intervalo opcional dentro do vídeo (em segundos).
#   Útil para processar apenas uma cena detectada.
# ==============================
def generate_windows_stream_centered(
    video_path:  str,
    k_seconds:   float       = 0.5,
    start_time:  float | None = None,
    end_time:    float | None = None,
) -> list[dict]:
    """
    Percorre o vídeo frame a frame (sem carregar tudo na memória) e
    produz janelas temporais centradas nos frames-chave (1 por segundo).

    Retorna lista de dicts com chaves:
        center_frame  — índice global do frame central
        timestamp_sec — tempo em segundos do frame central
        window        — lista de frames (np.ndarray BGR)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Não foi possível abrir o vídeo: '{video_path}'")

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(fps))               # 1 keyframe por segundo
    k_frames = max(1, int(fps * k_seconds))   # raio da janela em frames
    win_size = 2 * k_frames + 1               # tamanho total da janela

    # Converte start/end para índices de frame
    first_frame = int(start_time * fps) if start_time is not None else 0
    last_frame  = int(end_time   * fps) if end_time   is not None else None

    # Pula até o frame inicial se necessário
    if first_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)

    buffer:    list[np.ndarray] = []
    windows:   list[dict]       = []
    frame_idx: int              = first_frame   # índice global do frame atual

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Para se ultrapassou o fim da cena
        if last_frame is not None and frame_idx >= last_frame:
            break

        buffer.append(frame)

        # Mantém o buffer com exatamente win_size frames
        if len(buffer) > win_size:
            buffer.pop(0)

        # Só processa quando o buffer está completo
        if len(buffer) == win_size:
            # O frame central corresponde ao frame que entrou k_frames atrás
            center_global_idx = frame_idx - k_frames

            # Emite janela apenas nos keyframes sincronizados (1/segundo)
            if (center_global_idx - first_frame) % interval == 0:
                windows.append({
                    "center_frame":  center_global_idx,
                    "timestamp_sec": center_global_idx / fps,
                    "window":        buffer.copy(),   # cópia para isolar do buffer
                })

        frame_idx += 1

    cap.release()
    return windows