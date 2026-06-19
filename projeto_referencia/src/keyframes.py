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

    Evite em vídeos longos — prefira generate_windows_stream_centered.
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
    k_seconds:   float        = 0.5,
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


# ==============================
# 5. Dividir cena em N partes iguais e extrair frames de cada parte
# Uso: chamado por process_video()
# ==============================
def split_scene_into_parts(
    video_path: str,
    start_time: float,
    end_time:   float,
    n_parts:    int = 4,
    max_window_frames: int = 45,          # <-- novo parâmetro
) -> list[dict]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # k_frames original
    k_frames_raw = max(1, int(fps * 0.5))

    # Limita para que a janela tenha no máximo max_window_frames
    max_radius = (max_window_frames - 1) // 2
    k_frames = min(k_frames_raw, max_radius)

    win_size = 2 * k_frames + 1

    duration = end_time - start_time
    part_dur = duration / n_parts

    center_times = [
        start_time + (i + 0.5) * part_dur
        for i in range(n_parts)
    ]
    center_frames_target = [int(t * fps) for t in center_times]

    first_frame = int(start_time * fps)
    last_frame  = int(end_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, first_frame - k_frames))

    buffer: list[np.ndarray] = []
    parts:  list[dict]       = []
    frame_idx = max(0, first_frame - k_frames)
    targets   = set(center_frames_target)

    while frame_idx <= last_frame + k_frames and len(parts) < n_parts:
        ret, frame = cap.read()
        if not ret:
            break

        buffer.append(frame)
        if len(buffer) > win_size:
            buffer.pop(0)

        if len(buffer) == win_size:
            center_global = frame_idx - k_frames
            if center_global in targets:
                i = center_frames_target.index(center_global)
                parts.append({
                    "part_index":    i,
                    "center_frame":  center_global,
                    "timestamp_sec": center_global / fps,
                    "frames":        buffer.copy(),
                })

        frame_idx += 1

    cap.release()
    return parts

def split_scene_into_segments(
    video_path: str,
    start_time: float,
    end_time: float,
    max_frames_per_segment: int = 45,
) -> list[dict]:
    """
    Divide uma cena em segmentos consecutivos de no máximo max_frames_per_segment frames.
    Retorna lista de dicts com:
        - "frames": lista de np.ndarray (BGR)
        - "center_frame": índice do frame central do segmento (global)
        - "timestamp_sec": timestamp central do segmento
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Converte tempos para índices de frame
    start_frame = int(start_time * fps)
    end_frame   = int(end_time * fps) - 1   # último frame inclusive
    if end_frame < start_frame:
        cap.release()
        return []

    segments = []
    current_start = start_frame

    while current_start <= end_frame:
        # Calcula o fim deste segmento (limitado a max_frames_per_segment)
        current_end = min(current_start + max_frames_per_segment - 1, end_frame)

        # Posiciona o vídeo no início do segmento
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_start)
        frames = []
        for _ in range(current_start, current_end + 1):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        if frames:
            # Frame central do segmento (índice global)
            center_idx = (current_start + current_end) // 2
            timestamp = center_idx / fps
            segments.append({
                "frames": frames,
                "center_frame": center_idx,
                "timestamp_sec": timestamp,
            })

        # Avança para o próximo segmento (sem sobreposição)
        current_start = current_end + 1

    cap.release()
    return segments