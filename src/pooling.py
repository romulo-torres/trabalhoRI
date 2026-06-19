import argparse
import json
import os
import re
import time
from collections import defaultdict
from multiprocessing import Pool as ProcessPool
from pathlib import Path

from rank_bm25 import BM25L, BM25Okapi, BM25Plus

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

VARIANTS = {
    "bm25": BM25Okapi,
    "bm25l": BM25L,
    "bm25plus": BM25Plus,
}

VARIANT_DEFAULTS = {
    "bm25": {"k1": 1.5, "b": 0.75, "epsilon": 0.25},
    "bm25l": {"k1": 1.5, "b": 0.75, "delta": 0.5},
    "bm25plus": {"k1": 1.5, "b": 0.75, "delta": 1},
}


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


class BM25Index:
    def __init__(self, variant: str, corpus_texts: list[str], doc_ids: list[str], params: dict | None = None):
        self.variant = variant
        self.doc_ids = doc_ids
        tokenized = [_tokenize(t) for t in corpus_texts]
        cls = VARIANTS[variant]
        p = params or VARIANT_DEFAULTS.get(variant, {})
        self.model = cls(tokenized, **p)

    def search(self, query: str, top_k: int) -> list[dict]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.model.get_scores(tokens)
        paired = list(zip(self.doc_ids, scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        return [
            {"doc_id": doc_id, "score": round(float(score), 4), "rank": rank + 1}
            for rank, (doc_id, score) in enumerate(paired[:top_k])
            if score > 0
        ]


def _build_index(args: tuple) -> dict:
    variant, corpus_texts, doc_ids, params = args
    idx = BM25Index(variant, corpus_texts, doc_ids, params)
    return {variant: idx}


def load_corpus(path: Path) -> tuple[list[str], list[str]]:
    doc_ids = []
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_ids.append(doc["_id"])
            texts.append(doc.get("text", ""))
    return doc_ids, texts


def load_queries(path: Path) -> list[dict]:
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            queries.append(q)
    return queries


def main():
    parser = argparse.ArgumentParser(
        description="Pooling BM25: recupera top-k candidatos para cada consulta usando variantes BM25"
    )
    parser.add_argument("--corpus", type=Path, default=_PROJECT_ROOT / "data" / "corpus" / "corpus.jsonl")
    parser.add_argument("--queries", type=Path, default=_PROJECT_ROOT / "data" / "queries" / "queries.jsonl")
    parser.add_argument("--output", type=Path, default=_PROJECT_ROOT / "data" / "pooling")
    parser.add_argument("--variants", type=str, default="bm25,bm25l,bm25plus",
                        help="Variantes separadas por virgula (default: bm25,bm25l,bm25plus)")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0,
                        help="Numero de workers (0 = CPU count)")
    parser.add_argument("--merge", action="store_true",
                        help="Gera tambem um pool mergeado (uniao de todas as variantes)")
    args = parser.parse_args()

    variant_list = [v.strip() for v in args.variants.split(",") if v.strip() in VARIANTS]
    if not variant_list:
        print(f"ERRO: Nenhuma variante valida. Disponiveis: {', '.join(VARIANTS)}")
        return

    print("Carregando corpus...")
    doc_ids, corpus_texts = load_corpus(args.corpus)
    print(f"  {len(doc_ids)} documentos")

    print("Carregando queries...")
    queries = load_queries(args.queries)
    print(f"  {len(queries)} consultas")

    print(f"\nIndexando BM25 ({', '.join(variant_list)})...")
    t0 = time.time()
    indexes = {}
    for variant in variant_list:
        idx = BM25Index(variant, corpus_texts, doc_ids)
        indexes[variant] = idx
        print(f"  {variant}: OK ({time.time() - t0:.1f}s)")
    t_index = time.time() - t0

    os.makedirs(args.output, exist_ok=True)

    print(f"\nExecutando buscas (top-{args.top_k})...")
    t0 = time.time()
    total_pairs = 0
    for qi, q in enumerate(queries, 1):
        qid = q["_id"]
        qtext = q.get("text", "")
        qtype = q.get("type", "")
        pool = {
            "query_id": qid,
            "query_text": qtext,
            "query_type": qtype,
            "pools": {},
            "config": {
                "top_k": args.top_k,
                "corpus_size": len(doc_ids),
                "variants": variant_list,
            },
        }

        for variant in variant_list:
            results = indexes[variant].search(qtext, args.top_k)
            pool["pools"][variant] = results
            total_pairs += len(results)

        if args.merge:
            seen = set()
            merged = []
            for variant in variant_list:
                for r in pool["pools"][variant]:
                    if r["doc_id"] not in seen:
                        seen.add(r["doc_id"])
                        r["source_variant"] = variant
                        merged.append(r)
            pool["merged"] = merged

        out_path = args.output / f"{qid}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pool, f, indent=2, ensure_ascii=True)

        if qi % 50 == 0 or qi == len(queries):
            print(f"  [{qi}/{len(queries)}] salvos ({time.time() - t0:.1f}s)")

    t_search = time.time() - t0

    print(f"\n{'='*55}")
    print(f"Indexacao: {t_index:.1f}s")
    print(f"Buscas:    {t_search:.1f}s")
    print(f"Pools:     {len(queries)} arquivos em {args.output}")
    print(f"Pares:     {total_pairs} (query, doc)")
    print(f"Merge:     {'sim' if args.merge else 'nao'}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
