import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import index_elastic as ind

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path(*parts: str) -> Path:
    return _PROJECT_ROOT.joinpath(*parts)


def format_duration(sec: float) -> str:
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def main():
    parser = argparse.ArgumentParser(
        description="Analisa estatisticas dos videos indexados no Elasticsearch",
    )
    parser.add_argument(
        "--index", type=str, default="video_index",
        help="Nome do indice ES (default: video_index)",
    )
    parser.add_argument(
        "--es-host", type=str, default="",
        help="Host do ES (default: localhost:9200 ou env ES_HOST)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Mostra lista detalhada de video_ids indexados",
    )
    args = parser.parse_args()

    es_host = args.es_host or os.environ.get("ES_HOST", "http://localhost:9200")

    print(f"Conectando ao ES em {es_host}...")
    es = ind.connect_elasticsearch()

    if not es.indices.exists(index=args.index):
        print(f"ERRO: Indice '{args.index}' nao existe.", file=sys.stderr)
        sys.exit(1)

    # Contagem total de documentos
    total = es.count(index=args.index)
    total_docs = total["count"]
    print(f"\nTotal de documentos no indice: {total_docs}")

    # Agregações
    aggs = es.search(
        index=args.index,
        size=0,
        aggs={
            "por_modality": {"terms": {"field": "modality", "size": 10}},
            "por_video": {"terms": {"field": "video_id", "size": 10000}},
            "stats_duration": {"stats": {"field": "duration_sec"}},
        },
    )

    # Por modalidade
    print("\ Documentos por modalidade")
    for bucket in aggs["aggregations"]["por_modality"]["buckets"]:
        print(f"  {bucket['key']:>8}: {bucket['doc_count']:>6}")

    # Por vídeo (únicos)
    video_buckets = aggs["aggregations"]["por_video"]["buckets"]
    unique_videos = len(video_buckets)
    print(f"\ Videos unicos indexados: {unique_videos}")

    # Segmentos (video + audio) por vídeo
    seg_counts = [b["doc_count"] for b in video_buckets]
    if seg_counts:
        print(f"  Segmentos por video — min: {min(seg_counts)}  "
              f"media: {sum(seg_counts)/len(seg_counts):.1f}  "
              f"max: {max(seg_counts)}  "
              f"total: {sum(seg_counts)}")

    # Duração
    dur = aggs["aggregations"]["stats_duration"]
    if dur["count"] > 0:
        print(f"\ Duracao dos videos (segundos)")
        print(f"  Total: {dur['sum']:.0f}s  ({format_duration(dur['sum'])})")
        print(f"  Min:   {dur['min']:.0f}s")
        print(f"  Media: {dur['avg']:.0f}s")
        print(f"  Max:   {dur['max']:.0f}s")

    # Metadados em disco
    meta_files = {
        "videos_filtered": _path("data", "metadata", "videos_filtered.json"),
        "videos_metadata": _path("data", "metadata", "videos_metadata.json"),
    }
    print("\ Metadados em disco")
    for label, p in meta_files.items():
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            print(f"  {label}: {len(data)} entradas")
        else:
            print(f"  {label}: (ausente)")

    # Vídeos MP4 em disco
    video_dir = _path("data", "videos")
    if video_dir.exists():
        mp4s = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
        print(f"\ Videos MP4 em disco: {len(mp4s)}")
        total_size = sum(
            os.path.getsize(os.path.join(video_dir, f)) for f in mp4s
        )
        print(f"  Tamanho total: {total_size / 1024 / 1024:.0f} MB")
    else:
        print("\ Videos MP4 em disco: (pasta ausente)")

    # Embeddings em disco
    emb_dir = _path("data", "embeddings")
    if emb_dir.exists():
        emb_files = [f for f in os.listdir(emb_dir) if f.endswith(".json")]
        video_jsons = [f for f in emb_files if f.endswith("_video.json")]
        audio_jsons = [f for f in emb_files if f.endswith("_audio.json")]
        print(f"\ Embeddings em disco")
        print(f"  Video JSONs: {len(video_jsons)}")
        print(f"  Audio JSONs: {len(audio_jsons)}")

        vid_ids_v = {f.replace("_video.json", "") for f in video_jsons}
        vid_ids_a = {f.replace("_audio.json", "") for f in audio_jsons}
        completos = vid_ids_v & vid_ids_a
        print(f"  Pares completos (video+audio): {len(completos)}")
        print(f"  So video: {len(vid_ids_v - vid_ids_a)}")
        print(f"  So audio: {len(vid_ids_a - vid_ids_v)}")
    else:
        print("\ Embeddings em disco: (pasta ausente)")

    # Comparação ES × disco
    es_video_ids = {b["key"] for b in video_buckets}
    if completos:
        apenas_es = es_video_ids - completos
        apenas_disco = completos - es_video_ids
        if apenas_es:
            print(f"\n  {len(apenas_es)} videos no ES sem embedding em disco")
        if apenas_disco:
            print(f"\n  {len(apenas_disco)} embeddings em disco sem indexacao no ES")

    # Lista detalhada
    if args.verbose and video_buckets:
        print(f"\ Lista de videos indexados ({len(video_buckets)})")
        for b in sorted(video_buckets, key=lambda x: x["key"]):
            print(f"  {b['key']}: {b['doc_count']} segmentos")

    # Resumo
    print(f"\n{'='*50}")
    print(f"  RESUMO")
    print(f"{'='*50}")
    print(f"  Videos unicos no ES:  {unique_videos}")
    print(f"  Documentos totais:    {total_docs}")
    print(f"  Videos MP4 em disco:  {len(mp4s) if video_dir.exists() else 'N/A'}")
    print(f"  Embedding pairs:      {len(completos) if emb_dir.exists() else 'N/A'}")
    completude = f"{unique_videos / len(mp4s) * 100:.1f}%" if video_dir.exists() and mp4s else "N/A"
    print(f"  Completude:           {completude}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
