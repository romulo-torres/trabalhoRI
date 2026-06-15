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

# ==============================
# 6. Fallback chain: torchcodec → subprocess ffmpeg
# ==============================

def _read_frames_fallback(video_path: str, fps: float, step: int) -> list[np.ndarray] | None:
    import logging as _lg
    import torch as _torch
    if _torch.cuda.is_available():
        _lg.getLogger("keyframes").info("CUDA disponivel — tentando torchcodec...")
        try:
            from torchcodec.decoders import VideoDecoder
            decoder = VideoDecoder(video_path)
            n = decoder.metadata.num_frames
            indices = list(range(0, n, step))
            tensor = decoder.get_frames_at(indices)
            frames = [tensor[i].cpu().numpy()[..., ::-1].copy() for i in range(tensor.shape[0])]
            _lg.getLogger("keyframes").info("torchcodec: %d frames", len(frames))
            return frames
        except Exception as e:
            _lg.getLogger("keyframes").warning("torchcodec falhou (CUDA ativo): %s", e)
    else:
        _lg.getLogger("keyframes").info("CUDA indisponivel — pulando torchcodec")

    _lg.getLogger("keyframes").info("Tentando ffmpeg subprocess...")
    try:
        import subprocess, json
        probe = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "json", video_path],
            stderr=subprocess.DEVNULL,
        )
        info = json.loads(probe)
        w, h = info["streams"][0]["width"], info["streams"][0]["height"]
        raw = subprocess.check_output(
            ["ffmpeg", "-i", video_path, "-vf", "fps=1", "-f", "rawvideo",
             "-pix_fmt", "bgr24", "-vcodec", "rawvideo", "-"],
            stderr=subprocess.DEVNULL,
        )
        nbytes = w * h * 3
        frames = [
            np.frombuffer(raw[i:i+nbytes], dtype=np.uint8).reshape(h, w, 3)
            for i in range(0, len(raw), nbytes)
            if i + nbytes <= len(raw)
        ]
        _lg.getLogger("keyframes").info("ffmpeg subprocess: %d frames", len(frames))
        return frames
    except Exception as e:
        _lg.getLogger("keyframes").warning("ffmpeg subprocess falhou: %s", e)

    return None


# ==============================
# 7. Streaming: segmenta + embeda CLIP em 1 passada (baixa memoria)
# Uso: substitui split_scene_into_segments + loop CLIP
# ==============================
def stream_segments(
    video_path: str,
    scenes: list,
    model,
    preprocess,
    device: str,
    max_frames: int = 45,
) -> list[dict]:
    """
    Abre o video 1 vez, percorre cenas, amostra 1 frame por segundo,
    agrupa frames em segmentos de ate max_frames, executa CLIP
    embedding e descarta os frames. Uso de memoria constante:
    ~40 MB (1 segmento de 45 frames).
    """
    import embeddings as emb
    from logger import setup_logger

    _log = setup_logger(name="keyframes")
    _log.propagate = False

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Nao foi possivel abrir o video: '{video_path}'")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    _log.debug("[stream_segments] fps=%.4f total_frames=%d", fps, total_frames)

    # Testa leitura do primeiro frame
    ret_test, _ = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not ret_test:
        _log.warning("OpenCV nao leu o primeiro frame — tentando fallback...")
        cap.release()
        fallback_frames = _read_frames_fallback(video_path, fps, max(1, int(fps)))
        if fallback_frames is None:
            _log.warning("Fallback tambem falhou — retornando vazio")
            return []
        # Processa frames do fallback como se fossem streaming
        results = []
        buf = []
        for idx, frame in enumerate(fallback_frames):
            global_f = idx * max(1, int(fps))
            if len(buf) == 0:
                buf_start = global_f
            buf.append(frame)
            if len(buf) >= max_frames:
                vector = emb.embed_window(buf, model, preprocess, device, method="mean")
                center = buf_start + len(buf) // 2
                results.append({"scene_index": 0, "part_index": len(results),
                                "timestamp_sec": center / fps, "center_frame": center,
                                "embedding": vector})
                buf = []
                buf_start = None
        if buf:
            vector = emb.embed_window(buf, model, preprocess, device, method="mean")
            center = buf_start + len(buf) // 2
            results.append({"scene_index": 0, "part_index": len(results),
                            "timestamp_sec": center / fps, "center_frame": center,
                            "embedding": vector})
        return results

    results = []
    buf = []
    buf_start = None
    frames_read = 0
    frames_expected = 0

    step = max(1, int(fps))

    for sc_idx, (start, end) in enumerate(scenes):
        start_f = int(start * fps)
        end_f = int(end * fps)
        frames_expected += (end_f - start_f + 1)
        if start_f != 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        for global_f in range(start_f, end_f + 1, step):
            ret, frame = cap.read()
            if not ret and global_f == start_f:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                ret, frame = cap.read()
            if not ret:
                _log.warning(
                    "[stream_segments] cap.read() falhou no frame %d (scene %d / %d-%d)",
                    global_f, sc_idx, start_f, end_f,
                )
                break
            frames_read += 1
            if len(buf) == 0:
                buf_start = global_f
            buf.append(frame)

            if len(buf) >= max_frames:
                vector = emb.embed_window(
                    buf, model, preprocess, device, method="mean"
                )
                center = buf_start + len(buf) // 2
                results.append({
                    "scene_index": sc_idx,
                    "part_index": len(results),
                    "timestamp_sec": center / fps,
                    "center_frame": center,
                    "embedding": vector,
                })
                buf = []
                buf_start = None

    if buf:
        vector = emb.embed_window(
            buf, model, preprocess, device, method="mean"
        )
        center = buf_start + len(buf) // 2
        results.append({
            "scene_index": scenes[-1][0] if scenes else 0,
            "part_index": len(results),
            "timestamp_sec": center / fps,
            "center_frame": center,
            "embedding": vector,
        })

    cap.release()
    if frames_read == 0 and frames_expected > 0:
        _log.warning(
            "[stream_segments] 0 frames lidos de %d esperados (total_frames=%.0f) — "
            "codec/backend pode nao suportar decodificacao",
            frames_expected, total_frames,
        )
    return results


# ==============================
# 7. Modo batch: carrega tudo na RAM e processa CLIP em lote
# Uso: quando ha RAM suficiente (cloud, >8 GB disponivel)
# ==============================
def batch_segments(
    video_path: str,
    scenes: list,
    model,
    preprocess,
    device: str,
    max_frames: int = 45,
) -> list[dict]:
    """
    Carrega frames amostrados (1 por segundo) do video na RAM,
    agrupa em segmentos e processa CLIP em um unico batch GPU.
    Mais rapido que stream_segments (1 chamada GPU vs N chamadas).
    """
    import torch
    import embeddings as emb

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Nao foi possivel abrir o video: '{video_path}'")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Testa leitura do primeiro frame
    ret_test, _ = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not ret_test:
        cap.release()
        from logger import setup_logger as _sl
        _sl("keyframes").warning("batch_segments: OpenCV falhou, tentando fallback...")
        fallback_frames = _read_frames_fallback(video_path, fps, max(1, int(fps)))
        if fallback_frames is None:
            return []
        # Processa via batch: preprocessa todos e faz 1 chamada CLIP
        step = max(1, int(fps))
        # Build segments from fallback frames
        buf = []
        all_segments = []
        for idx, frame in enumerate(fallback_frames):
            global_f = idx * step
            buf.append(frame)
            if len(buf) >= max_frames:
                center = global_f - len(buf) + 1 + len(buf) // 2
                all_segments.append({"frames": buf, "center_frame": center,
                                     "timestamp_sec": center / fps, "scene_index": 0})
                buf = []
        if buf:
            center = (len(fallback_frames) - 1) * step - len(buf) + len(buf) // 2
            all_segments.append({"frames": buf, "center_frame": center,
                                 "timestamp_sec": center / fps, "scene_index": 0})
        if not all_segments:
            return []
        frames_tensors = []
        seg_counts = []
        for seg in all_segments:
            for frame in seg["frames"]:
                frames_tensors.append(preprocess(emb.frame_to_pil(frame)))
            seg_counts.append(len(seg["frames"]))
            seg.pop("frames", None)
        big_tensor = torch.stack(frames_tensors).to(device)
        del frames_tensors
        with torch.no_grad():
            all_embeddings = model.encode_image(big_tensor)
        all_embeddings = all_embeddings / all_embeddings.norm(dim=-1, keepdim=True)
        del big_tensor
        results = []
        idx = 0
        for s, seg in enumerate(all_segments):
            n = seg_counts[s]
            seg_emb = all_embeddings[idx:idx + n].mean(dim=0)
            seg_emb = seg_emb / seg_emb.norm()
            results.append({"scene_index": seg["scene_index"], "part_index": s,
                            "timestamp_sec": seg["timestamp_sec"],
                            "center_frame": seg["center_frame"],
                            "embedding": seg_emb.cpu().numpy().flatten()})
            idx += n
        return results

    step = max(1, int(fps))

    # 1. Agrupa frames em segmentos (sem embed)
    all_segments = []
    buf = []

    for sc_idx, (start, end) in enumerate(scenes):
        start_f = int(start * fps)
        end_f = int(end * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        for global_f in range(start_f, end_f + 1, step):
            ret, frame = cap.read()
            if not ret:
                break
            buf.append(frame)

            if len(buf) >= max_frames:
                center = (global_f - len(buf) + 1) + len(buf) // 2
                all_segments.append({
                    "frames": buf,
                    "center_frame": center,
                    "timestamp_sec": center / fps,
                    "scene_index": sc_idx,
                })
                buf = []

    if buf:
        center = (end_f if scenes else 0) - len(buf) + len(buf) // 2
        all_segments.append({
            "frames": buf,
            "center_frame": center,
            "timestamp_sec": center / fps,
            "scene_index": scenes[-1][0] if scenes else 0,
        })

    cap.release()

    if not all_segments:
        return []

    # 2. Preprocessa todos os frames em um tensor unico
    frames_tensors = []
    seg_counts = []
    for seg in all_segments:
        for frame in seg["frames"]:
            frames_tensors.append(preprocess(emb.frame_to_pil(frame)))
        seg_counts.append(len(seg["frames"]))
        seg.pop("frames", None)

    big_tensor = torch.stack(frames_tensors).to(device)
    del frames_tensors

    # 3. CLIP batch: 1 chamada GPU para todos os frames
    with torch.no_grad():
        all_embeddings = model.encode_image(big_tensor)
    all_embeddings = all_embeddings / all_embeddings.norm(dim=-1, keepdim=True)
    del big_tensor

    # 4. Desagrupa embeddings por segmento (media intra-segmento)
    results = []
    idx = 0
    for s, seg in enumerate(all_segments):
        n = seg_counts[s]
        seg_emb = all_embeddings[idx:idx+n].mean(dim=0)
        seg_emb = seg_emb / seg_emb.norm()
        results.append({
            "scene_index": seg["scene_index"],
            "part_index": s,
            "timestamp_sec": seg["timestamp_sec"],
            "center_frame": seg["center_frame"],
            "embedding": seg_emb.cpu().numpy().flatten(),
        })
        idx += n

    return results


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