import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── helpers ────────────────────────────────────────────────────────


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
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


def load_queries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    return queries


def load_pool(path: Path, source: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        pool = json.load(f)

    candidates = pool.get(source) or pool.get("pools", {}).get("bm25", [])
    candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
    return [c["doc_id"] for c in candidates]


def get_query_type(qid: str, queries: list[dict]) -> str:
    for q in queries:
        if q["_id"] == qid:
            return q.get("type", "unknown")
    return "unknown"


# ── métricas ───────────────────────────────────────────────────────


def precision_at_k(ranking: list[str], qrels: dict[str, int], k: int) -> float:
    top_k = ranking[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for d in top_k if qrels.get(d, 0) > 0)
    return relevant / k


def recall_at_k(ranking: list[str], qrels: dict[str, int], k: int) -> float:
    top_k = ranking[:k]
    total_relevant = sum(1 for v in qrels.values() if v > 0)
    if total_relevant == 0:
        return 0.0
    relevant = sum(1 for d in top_k if qrels.get(d, 0) > 0)
    return relevant / total_relevant


def mrr(ranking: list[str], qrels: dict[str, int]) -> float:
    for i, doc_id in enumerate(ranking, 1):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / i
    return 0.0


def average_precision(ranking: list[str], qrels: dict[str, int]) -> float:
    total_relevant = sum(1 for v in qrels.values() if v > 0)
    if total_relevant == 0:
        return 0.0
    hits = 0
    sum_precision = 0.0
    for i, doc_id in enumerate(ranking, 1):
        if qrels.get(doc_id, 0) > 0:
            hits += 1
            sum_precision += hits / i
    return sum_precision / total_relevant


def _dcg(rel_scores: list[int], k: int) -> float:
    dcg = 0.0
    for i in range(min(k, len(rel_scores))):
        gain = (2 ** rel_scores[i]) - 1
        discount = math.log2(i + 2)
        dcg += gain / discount
    return dcg


def ndcg_at_k(ranking: list[str], qrels: dict[str, int], k: int) -> float:
    rel_scores = [qrels.get(d, 0) for d in ranking]
    dcg = _dcg(rel_scores, k)
    ideal = sorted([v for v in qrels.values() if v > 0], reverse=True)
    if not ideal:
        return 0.0
    idcg = _dcg(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


# ── main ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Avalia o ranking BM25 contra os qrels, calculando metricas TREC"
    )
    parser.add_argument("--pools", type=Path, default=_PROJECT_ROOT / "data" / "pooling")
    parser.add_argument("--qrels", type=Path, default=_PROJECT_ROOT / "data" / "qrels" / "qrels.tsv")
    parser.add_argument("--queries", type=Path, default=_PROJECT_ROOT / "data" / "queries" / "queries.jsonl")
    parser.add_argument("--output", type=Path, default=_PROJECT_ROOT / "data" / "evaluation")
    parser.add_argument("--pool-source", default="merged", help="Chave do pool: merged, bm25, bm25l, bm25plus")
    parser.add_argument("--k", default="10,100", help="Valores de K separados por virgula")
    args = parser.parse_args()

    ks = [int(k.strip()) for k in args.k.split(",") if k.strip().isdigit()]
    if not ks:
        print("ERRO: --k deve ser uma lista de inteiros separados por virgula", file=sys.stderr)
        sys.exit(1)

    print("Carregando qrels...")
    qrels = load_qrels(args.qrels)
    total_q = len(qrels)
    print(f"  {total_q} queries, {sum(len(v) for v in qrels.values())} pares")

    print("Carregando queries...")
    queries = load_queries(args.queries)
    print(f"  {len(queries)} queries no jsonl")

    print(f"Carregando pools (fonte: {args.pool_source})...")
    pool_ids = set()
    for fname in os.listdir(args.pools):
        if fname.endswith(".json"):
            pool_ids.add(fname.replace(".json", ""))
    print(f"  {len(pool_ids)} pools disponiveis")

    qids = sorted(qrels.keys() & pool_ids)
    print(f"  {len(qids)} queries com qrels + pool")

    # ── calcular métricas ──

    results: dict[str, dict] = {}
    overall: dict[str, list[float]] = defaultdict(list)
    by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for qi, qid in enumerate(qids, 1):
        pool_path = args.pools / f"{qid}.json"
        if not pool_path.exists():
            continue

        ranking = load_pool(pool_path, args.pool_source)
        qr = qrels[qid]
        qtype = get_query_type(qid, queries)

        per_q: dict[str, float] = {}
        for k in ks:
            per_q[f"P@{k}"] = precision_at_k(ranking, qr, k)
            per_q[f"R@{k}"] = recall_at_k(ranking, qr, k)
        per_q["MRR"] = mrr(ranking, qr)
        per_q["MAP"] = average_precision(ranking, qr)
        per_q["nDCG@10"] = ndcg_at_k(ranking, qr, 10)

        results[qid] = per_q

        for metric, value in per_q.items():
            overall[metric].append(value)
            by_type[qtype][metric].append(value)

        if qi % 50 == 0 or qi == len(qids):
            print(f"  [{qi}/{len(qids)}] ...")

    # ── agregar ──

    def avg(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    overall_summary: dict[str, float] = {m: avg(v) for m, v in overall.items()}
    type_summary: dict[str, dict[str, float]] = {}
    for t, metrics in by_type.items():
        type_summary[t] = {m: avg(v) for m, v in metrics.items()}

    # ── salvar JSON ──

    os.makedirs(args.output, exist_ok=True)
    json_path = args.output / "results.json"
    json_output = {
        "overall": overall_summary,
        "by_type": type_summary,
        "per_query": results,
        "config": {
            "pool_source": args.pool_source,
            "qrels": str(args.qrels),
            "k_values": ks,
            "total_queries": len(qids),
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"\nJSON salvo: {json_path}")

    # ── salvar Markdown ──

    md_path = args.output / "report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Relatório de Avaliação\n\n")
        f.write(f"Pool source: `{args.pool_source}` | Queries: {len(qids)} | K: {ks}\n\n")
        f.write("## Métricas Gerais\n\n")
        f.write("| Métrica | Valor |\n|---------|-------|\n")
        for m in ks:
            f.write(f"| P@{m} | {overall_summary.get(f'P@{m}', 0):.4f} |\n")
            f.write(f"| R@{m} | {overall_summary.get(f'R@{m}', 0):.4f} |\n")
        f.write(f"| MRR | {overall_summary.get('MRR', 0):.4f} |\n")
        f.write(f"| MAP | {overall_summary.get('MAP', 0):.4f} |\n")
        f.write(f"| nDCG@10 | {overall_summary.get('nDCG@10', 0):.4f} |\n")

        if type_summary:
            f.write("\n## Por Tipo de Consulta\n\n")
            f.write("| Tipo | Queries |")
            for m in ks:
                f.write(f" P@{m} | R@{m} |")
            f.write(" MRR | MAP | nDCG@10 |\n")
            f.write("|------|---------|")
            for _ in ks:
                f.write("------|------|")
            f.write("------|------|---------|\n")
            for t in sorted(type_summary):
                ts = type_summary[t]
                nq = len(by_type[t].get("MAP", []))
                f.write(f"| {t} | {nq} |")
                for m in ks:
                    f.write(f" {ts.get(f'P@{m}', 0):.4f} | {ts.get(f'R@{m}', 0):.4f} |")
                f.write(f" {ts.get('MRR', 0):.4f} | {ts.get('MAP', 0):.4f} | {ts.get('nDCG@10', 0):.4f} |\n")

        worst = sorted(results.items(), key=lambda x: x[1].get("MAP", 0))[:10]
        best = sorted(results.items(), key=lambda x: x[1].get("MAP", 0), reverse=True)[:10]
        f.write("\n## Piores 10 (por MAP)\n\n")
        f.write("| Query | MAP | MRR | P@10 | nDCG@10 |\n")
        f.write("|-------|-----|-----|------|---------|\n")
        for qid, m in worst:
            f.write(f"| {qid} | {m.get('MAP', 0):.4f} | {m.get('MRR', 0):.4f} | {m.get('P@10', 0):.4f} | {m.get('nDCG@10', 0):.4f} |\n")
        f.write("\n## Melhores 10 (por MAP)\n\n")
        f.write("| Query | MAP | MRR | P@10 | nDCG@10 |\n")
        f.write("|-------|-----|-----|------|---------|\n")
        for qid, m in best:
            f.write(f"| {qid} | {m.get('MAP', 0):.4f} | {m.get('MRR', 0):.4f} | {m.get('P@10', 0):.4f} | {m.get('nDCG@10', 0):.4f} |\n")

    print(f"Relatorio salvo: {md_path}")
    print(f"\n{'='*55}")
    for m in ks:
        print(f"P@{m}: {overall_summary.get(f'P@{m}', 0):.4f}")
        print(f"R@{m}: {overall_summary.get(f'R@{m}', 0):.4f}")
    print(f"MRR:      {overall_summary.get('MRR', 0):.4f}")
    print(f"MAP:      {overall_summary.get('MAP', 0):.4f}")
    print(f"nDCG@10:  {overall_summary.get('nDCG@10', 0):.4f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
