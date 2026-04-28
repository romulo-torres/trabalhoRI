import index_elastic as ind
import embeddings as emb
from logger import setup_logger
import os
import time

logger = setup_logger()


def main_index() -> None:
    logger.info("Conectando ao Elasticsearch...")
    es = ind.connect_elasticsearch()

    video_dir = "./data/videos"
    os.makedirs(video_dir, exist_ok=True)

    ind.create_index(es, index_name="video_index", dims=512)

    logger.info("Carregando modelo CLIP...")
    model, preprocess, device = emb.load_model()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(BASE_DIR, "..", "data", "activity_net.v1-3.min.json")

    ind.ensure_activitynet_json(json_path)

    dataset = ind.load_activitynet(json_path)

    test_videos = [
        (video_id, meta)
        for video_id, meta in dataset.items()
        if meta["subset"] == "validation"
    ]
    test_videos = test_videos[:1000]

    for video_id, meta in test_videos:

        video_path = ind.download_video(video_id, video_dir)
        time.sleep(2)

        if video_path is None:
            logger.warning(f"Falha ao baixar {video_id}")
            continue

        if ind.already_indexed(es, video_id):
            logger.info(f"{video_id} já indexado — pulando.")
            continue

        try:
            ind.process_video(
                video_path,
                video_id,
                model,
                preprocess,
                device,
                es
            )
        except Exception as e:
            logger.error(f"Erro no vídeo {video_id}: {e}")


if __name__ == "__main__":
    main_index()