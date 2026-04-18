import cv2
import numpy as np

np.float_ = np.float64

# ==============================
# 1. Extrair TODOS os frames
# ==============================
def extract_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames, fps


# ==============================
# 2. Pegar frames sincronizados (1 por segundo)
# ==============================
def get_sync_indices(fps, total_frames):
    interval = int(fps)
    return list(range(0, total_frames, interval))


# ==============================
# 3. Janela temporal baseada no índice REAL
# ==============================
def get_window(frames, center_idx, k_frames):
    window = []
    
    for i in range(center_idx - k_frames, center_idx + k_frames + 1):
        if 0 <= i < len(frames):
            window.append(frames[i])
    
    return window


# ==============================
# 4. Pipeline completo
# ==============================

# o k_seconds é quantos k segundos ele pega de contexto para o embedding como contexto
# essa versão faz streaming para evitar o problema de memória
def generate_windows_stream_centered(video_path, k_seconds=0.5):
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS) # fps do video
    interval = int(fps) # intervalo de frames que vai capturar
    k_frames = int(fps * k_seconds) # quantos k_frames vai ter de contexto fps * k_seconds. Ex: fps 30 e k_frames 0.5  30 * 0.5 = 15 frames para frente e 15 frames para trás
    
    buffer = []
    windows = []
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        buffer.append(frame)
        
        if len(buffer) > (2 * k_frames + 1):
            buffer.pop(0)
        
        # só quando buffer está completo
        if len(buffer) == (2 * k_frames + 1):
            
            center_global_idx = frame_idx - k_frames
            
            # verifica se o CENTRO é sincronizado
            if center_global_idx % interval == 0:
                
                windows.append({
                    "center_frame": center_global_idx,
                    "timestamp_sec": center_global_idx / fps,
                    "window": buffer.copy()
                })
        
        frame_idx += 1

    cap.release()
    return windows


# ==============================
# 5. Exemplo de uso
# ==============================
# if __name__ == "__main__":
#     video_path = "video.mp4"
    
#     windows = generate_windows_stream_centered(video_path, k_seconds=0.5)
    
#     print(f"Total de janelas geradas: {len(windows)}")
    
#     # acessar primeira janela
#     first = windows[0]
    
#     print(f"Frame central: {first['center_frame']}")
#     print(f"Timestamp: {first['timestamp_sec']}s")
#     print(f"Número de frames na janela: {len(first['window'])}")