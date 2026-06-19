import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_corpus(path: Path) -> dict[str, dict]:
    docs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            docs[doc["_id"]] = doc
    return docs


def load_queries(path: Path) -> list[dict]:
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    return queries


def load_existing_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    if not path.exists():
        return qrels
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("query-id"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            qid, doc_id, rel = parts[0], parts[1], int(parts[2])
            qrels.setdefault(qid, {})[doc_id] = rel
    return qrels


def load_judgments(judgment_dir: Path, query_ids: set[str]) -> dict[str, list[dict]]:
    judgments: dict[str, list[dict]] = {}
    if not judgment_dir.exists():
        return judgments
    for fname in os.listdir(judgment_dir):
        if not fname.endswith(".json") or fname == "judgments_progress.json":
            continue
        qid = fname.replace(".json", "")
        if qid not in query_ids:
            continue
        with open(judgment_dir / fname, encoding="utf-8") as f:
            data = json.load(f)
        judgments[qid] = data.get("judgments", [])
    return judgments


def write_qrels(qrels: dict[str, dict[str, int]], path: Path) -> None:
    os.makedirs(path.parent, exist_ok=True)
    rows = []
    for qid in sorted(qrels):
        for doc_id in sorted(qrels[qid]):
            rel = qrels[qid][doc_id]
            rows.append(f"{qid}\t{doc_id}\t{rel}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("query-id\tdocument-id\trelevance\n")
        f.write("\n".join(rows) + "\n")
    print(f"Qrels escrito: {path} ({len(rows)} linhas)")


def main():
    parser = argparse.ArgumentParser(
        description="Monta a colecao BEIR final: mescla julgamentos LLM + auto qrels e extrai hard negatives"
    )
    parser.add_argument("--corpus", type=Path, default=_PROJECT_ROOT / "data" / "corpus" / "corpus.jsonl")
    parser.add_argument("--queries", type=Path, default=_PROJECT_ROOT / "data" / "queries" / "queries.jsonl")
    parser.add_argument("--judgments", type=Path, default=_PROJECT_ROOT / "data" / "judgments")
    parser.add_argument("--pools", type=Path, default=_PROJECT_ROOT / "data" / "pooling")
    parser.add_argument("--qrels", type=Path, default=_PROJECT_ROOT / "data" / "qrels" / "qrels.tsv")
    parser.add_argument("--output-qrels", type=Path, default=_PROJECT_ROOT / "data" / "qrels" / "qrels.tsv")
    parser.add_argument("--hard-negatives", type=Path, default=_PROJECT_ROOT / "data" / "hard_negatives" / "hard_negatives.jsonl")
    parser.add_argument("--hard-top-k", type=int, default=100, help="Candidatos do topo do BM25 a considerar")
    parser.add_argument("--hard-per-query", type=int, default=10, help="Max hard negatives por query")
    parser.add_argument("--validate", action="store_true", help="Validar com beir.datasets")
    args = parser.parse_args()

    print("Carregando corpus...")
    corpus = load_corpus(args.corpus)
    print(f"  {len(corpus)} documentos")

    print("Carregando queries...")
    queries = load_queries(args.queries)
    all_qids = {q["_id"] for q in queries}
    print(f"  {len(queries)} consultas")

    print("Carregando qrels existente...")
    existing_qrels = load_existing_qrels(args.qrels)
    print(f"  {sum(len(v) for v in existing_qrels.values())} pares")

    print("Carregando julgamentos LLM...")
    judgments = load_judgments(args.judgments, all_qids)
    print(f"  {len(judgments)} queries com julgamento")

    # ---- Merge qrels ----
    print("\n=== Merge qrels ===")
    merged = {}
    for qid in sorted(all_qids):
        if qid in judgments:
            jlist = judgments[qid]
            merged[qid] = {}
            for j in jlist:
                doc_id = j.get("doc_id", "")
                rel = j.get("relevance", 0)
                if doc_id and rel >= 0:
                    merged[qid][doc_id] = rel
            src = next((j.get("doc_id", "") for j in jlist if j.get("auto")), "")
        elif qid in existing_qrels:
            merged[qid] = dict(existing_qrels[qid])
            src = list(existing_qrels[qid].keys())[0] if existing_qrels[qid] else ""
        else:
            print(f"  AVISO: {qid} sem julgamento e sem auto qrels, pulando")
            continue

        n_rel = sum(1 for v in merged[qid].values() if v > 0)
        n_irr = sum(1 for v in merged[qid].values() if v == 0)
        fonte = "LLM" if qid in judgments else "auto"
        print(f"  {qid}: {len(merged[qid])} pares ({n_rel} relevantes, {n_irr} irrelevantes) [{fonte}]")

    total_qrels = sum(len(v) for v in merged.values())
    print(f"\nTotal: {total_qrels} pares em {len(merged)} queries")
    write_qrels(merged, args.output_qrels)

    # ---- Hard negatives ----
    print("\n=== Hard Negatives ===")
    os.makedirs(args.hard_negatives.parent, exist_ok=True)
    hn_entries = []
    hn_total = 0
    for qid in sorted(all_qids):
        if qid not in judgments:
            continue

        pool_path = args.pools / f"{qid}.json"
        if not pool_path.exists():
            continue

        with open(pool_path, encoding="utf-8") as f:
            pool = json.load(f)

        candidates = pool.get("merged") or pool.get("pools", {}).get("bm25", [])
        if not candidates:
            continue

        candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        top_k = candidates[:args.hard_top_k]

        q_judgments = {j["doc_id"]: j.get("relevance", 0) for j in judgments.get(qid, [])}

        hard_negatives = []
        for c in top_k:
            doc_id = c["doc_id"]
            rel = q_judgments.get(doc_id)
            if rel == 0:
                hard_negatives.append({
                    "doc_id": doc_id,
                    "bm25_rank": c.get("rank", 0),
                    "bm25_score": c.get("score", 0),
                })
                if len(hard_negatives) >= args.hard_per_query:
                    break

        positive_docs = [doc_id for doc_id, rel in q_judgments.items() if rel > 0]

        if hard_negatives:
            qtext = pool.get("query_text") or next(
                (q.get("text", "") for q in queries if q["_id"] == qid), ""
            )
            entry = {
                "query_id": qid,
                "query_text": qtext,
                "positive_docs": positive_docs,
                "hard_negative_docs": hard_negatives,
            }
            hn_entries.append(entry)
            hn_total += len(hard_negatives)
            print(f"  {qid}: {len(hard_negatives)} hard negatives ({len(positive_docs)} positives)")

    with open(args.hard_negatives, "w", encoding="utf-8") as f:
        for entry in hn_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nTotal: {hn_total} hard negatives em {len(hn_entries)} queries")
    print(f"Salvo em: {args.hard_negatives}")

    # ---- Validate ----
    if args.validate:
        print("\n=== Validacao BEIR ===")
        try:
            from beir.datasets import GenericDataLoader
            from beir.retrieval.evaluation import EvaluateRetrieval
            corpus_dl, queries_dl, qrels_dl = GenericDataLoader(
                data_folder=str(args.output_qrels.parent.parent)
            ).load()
            print("  OK: BEIR carregou os dados sem erros")
        except ImportError:
            print("  AVISO: beir nao instalado, pulando validacao")
        except Exception as e:
            print(f"  ERRO na validacao: {e}")

    print("\nConcluido.")


if __name__ == "__main__":
    main()
