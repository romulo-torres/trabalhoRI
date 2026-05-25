import index_elastic as ind
import embeddings as emb
from logger import setup_logger
import os
import time
import gc 
import torch

logger = setup_logger()


def main_index() -> None:
    MAX_VIDEOS = 2000          # <-- processar no máximo 100 vídeos

    logger.info("Conectando ao Elasticsearch...")
    es = ind.connect_elasticsearch()

    video_dir = "./data/videos"
    os.makedirs(video_dir, exist_ok=True)

    # ⚠️ ATENÇÃO: não apague o índice se ele já existir
    # Se o create_index atual apaga, altere-o para apenas criar se não existir.
    ind.create_index(es, index_name="video_index", dims=512)

    # Carrega CLIP e CLAP
    logger.info("Carregando modelos CLIP e CLAP...")
    clip_model, clip_preprocess, clap_model, device = emb.load_all_models()

    # Caminho do JSON do ActivityNet
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(BASE_DIR, "..", "data", "activity_net.v1-3.min.json")
    ind.ensure_activitynet_json(json_path)

    dataset = ind.load_activitynet(json_path)
    
    # Carrega taxonomia (para feature_categorias)
    taxonomy_lookup = ind.build_taxonomy_lookup(json_path)

    # Filtra vídeos de validação (todos os 1000 disponíveis no dataset)
    all_validation = [
        (video_id, meta)
        for video_id, meta in dataset.items()
        if meta["subset"] == "validation"
    ]
    logger.info(f"Total de vídeos no dataset de validação: {len(all_validation)}")

    # ── Etapa 1: Descobrir quais vídeos já estão baixados ─────────────
    local_mp4 = sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4"))
    local_ids = [f.replace(".mp4", "") for f in local_mp4]
    logger.info(f"Vídeos locais encontrados: {len(local_ids)}")

    # ── Etapa 2: Selecionar até MAX_VIDEOS vídeos para processar ─────
    selected = []   # lista de (video_id, meta)

    # Primeiro, pega os que já estão baixados e pertencem ao dataset
    for vid, meta in all_validation:
        if vid in local_ids:
            selected.append((vid, meta))
        if len(selected) >= MAX_VIDEOS:
            break

    # Se ainda não atingiu MAX_VIDEOS, completa com vídeos que precisam ser baixados
    if len(selected) < MAX_VIDEOS:
        for vid, meta in all_validation:
            if vid not in local_ids:   # não baixado ainda
                selected.append((vid, meta))
            if len(selected) >= MAX_VIDEOS:
                break

    logger.info(f"Vídeos selecionados para processar: {len(selected)} (máx {MAX_VIDEOS})")

    # ── Etapa 3: Processar cada vídeo selecionado ─────────────────────
    for video_id, meta in selected:
        label = ""
        annotations = meta.get("annotations", [])
        if annotations:
            label = annotations[0].get("label", "")
        title = meta.get("url", "")

        # Verifica se o vídeo já está no Elasticsearch
        if ind.already_indexed(es, video_id):
            logger.info(f"{video_id} já está no índice — pulando.")
            continue

        # Obtém o caminho do vídeo (local ou via download)
        local_path = os.path.join(video_dir, f"{video_id}.mp4")
        if os.path.exists(local_path):
            logger.info(f"{video_id} encontrado localmente.")
            video_path = local_path
        else:
            logger.info(f"Baixando {video_id}...")
            video_path = ind.download_video(video_id, video_dir)

        if video_path is None:
            logger.warning(f"Não foi possível obter o vídeo {video_id}.")
            continue

        try:
            ind.process_video(
                video_path=video_path,
                video_id=video_id,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                clap_model=clap_model,
                device=device,
                es=es,
                taxonomy_lookup=taxonomy_lookup,
                label=label,
                title=title,
            )
        except Exception as e:
            logger.error(f"Erro no vídeo {video_id}: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info("Indexação concluída.")


if __name__ == "__main__":
    main_index()