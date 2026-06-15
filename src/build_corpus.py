import json
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return _PROJECT_ROOT / p if not p.is_absolute() else p


def build_corpus(
    metadata_path: str | Path = "data/metadata/videos_metadata.json",
    corpus_dir: str | Path = "data/corpus",
    corpus_file: str = "corpus.jsonl",
    only_with_video: bool = False,
    video_dir: str | Path = "data/videos",
) -> None:
    meta_path = _resolve(metadata_path)
    corp_dir = _resolve(corpus_dir)
    vid_dir = _resolve(video_dir) if only_with_video else None

    corp_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    video_ids_on_disk = set()
    if only_with_video and vid_dir:
        if vid_dir.exists():
            video_ids_on_disk = {
                f.replace(".mp4", "") for f in os.listdir(vid_dir) if f.endswith(".mp4")
            }
        print(f"Filtrando apenas {len(video_ids_on_disk)} videos com MP4 em disco")

    corpus_path = corp_dir / corpus_file
    count = 0
    skipped = 0

    with open(corpus_path, "w", encoding="utf-8") as out:
        for video_id, meta in metadata.items():
            if only_with_video and video_id not in video_ids_on_disk:
                skipped += 1
                continue

            title = meta.get("title") or ""
            description = meta.get("description") or ""
            tags = meta.get("tags") or []
            tags_text = ", ".join(tags) if tags else ""
            categories = meta.get("categories") or []
            categories_text = ", ".join(categories) if categories else ""
            transcript = meta.get("transcript") or ""
            feature_desc = meta.get("feature_desc") or ""
            keywords = meta.get("keywords") or ""

            text_parts = []
            if title:
                text_parts.append(title)
            if description:
                text_parts.append(description)
            if tags_text:
                text_parts.append(tags_text)
            if categories_text:
                text_parts.append(categories_text)
            if transcript:
                text_parts.append(transcript)
            if feature_desc:
                text_parts.append(feature_desc)
            if keywords:
                text_parts.append(keywords)

            text = ". ".join(text_parts)
            text = text.replace("..", ".").strip()

            record = {
                "_id": video_id,
                "text": text,
                "title": title,
                "metadata": {
                    "duration": meta.get("duration", 0),
                    "channel": meta.get("channel", ""),
                    "upload_date": meta.get("upload_date", ""),
                    "anet_label": meta.get("anet_label", ""),
                    "anet_subset": meta.get("anet_subset", ""),
                    "status": meta.get("status", ""),
                },
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"corpus.jsonl gerado: {corpus_path}")
    print(f"  Documentos escritos: {count}")
    if only_with_video:
        print(f"  Pulados (sem video): {skipped}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Constroi corpus.jsonl no formato BEIR a partir dos metadados"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/metadata/videos_metadata.json",
        help="Caminho para o JSON de metadados",
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="data/corpus",
        help="Diretorio de saida do corpus",
    )
    parser.add_argument(
        "--corpus-file",
        type=str,
        default="corpus.jsonl",
        help="Nome do arquivo corpus",
    )
    parser.add_argument(
        "--only-with-video",
        action="store_true",
        help="Inclui apenas videos que possuem MP4 em disco",
    )
    args = parser.parse_args()

    build_corpus(
        metadata_path=args.metadata,
        corpus_dir=args.corpus_dir,
        corpus_file=args.corpus_file,
        only_with_video=args.only_with_video,
    )
