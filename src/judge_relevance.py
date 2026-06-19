import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROVIDERS = {}

def register_provider(name: str):
    def wrapper(cls):
        PROVIDERS[name] = cls
        return cls
    return wrapper


class ProviderError(Exception):
    pass


def _is_rate_limit(msg: str) -> bool:
    return "429" in msg or "rate limit" in msg.lower() or "quota" in msg.lower() or "too many requests" in msg.lower()


def _call_llm(provider: str, api_key: str, model: str, prompt: str, max_tokens: int = 4096, base_url: str = "") -> str:
    cls = PROVIDERS.get(provider)
    if not cls:
        raise ProviderError(f"Provedor desconhecido: {provider}. Disponiveis: {', '.join(PROVIDERS)}")
    max_retries = 5
    base_wait = 5
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return cls().generate(api_key, model, prompt, max_tokens, base_url)
        except Exception as e:
            msg = str(e)
            if _is_rate_limit(msg) and attempt < max_retries:
                wait = base_wait * (2 ** (attempt - 1)) + random.uniform(0, 3)
                print(f"  Rate limit, tentativa {attempt}/{max_retries}, aguardando {wait:.0f}s...")
                time.sleep(wait)
                last_error = e
            else:
                raise
    raise last_error


@register_provider("openai")
class OpenAIProvider:
    def generate(self, api_key: str, model: str, prompt: str, max_tokens: int = 4096, base_url: str = "") -> str:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url or None)
        resp = client.chat.completions.create(
            model=model or "gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()


@register_provider("anthropic")
class AnthropicProvider:
    def generate(self, api_key: str, model: str, prompt: str, max_tokens: int = 4096, base_url: str = "") -> str:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model or "claude-sonnet-4-20250514",
            max_tokens=max_tokens, temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()


@register_provider("gemini")
class GeminiProvider:
    def generate(self, api_key: str, model: str, prompt: str, max_tokens: int = 4096, base_url: str = "") -> str:
        from google import genai
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model or "gemini-2.0-flash",
            contents=prompt,
            config={"max_output_tokens": max_tokens, "temperature": 0.3},
        )
        return resp.text.strip()


@register_provider("deepseek")
class DeepSeekProvider:
    def generate(self, api_key: str, model: str, prompt: str, max_tokens: int = 4096, base_url: str = "") -> str:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model=model or "deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()


@register_provider("openrouter")
class OpenRouterProvider:
    def generate(self, api_key: str, model: str, prompt: str, max_tokens: int = 4096, base_url: str = "") -> str:
        from openai import OpenAI
        url = base_url or "https://openrouter.ai/api/v1"
        client = OpenAI(api_key=api_key, base_url=url)
        resp = client.chat.completions.create(
            model=model or "openai/gpt-oss-120b:free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()


@register_provider("lm-studio")
class LMStudioProvider:
    def generate(self, api_key: str, model: str, prompt: str, max_tokens: int = 4096, base_url: str = "") -> str:
        from openai import OpenAI
        url = base_url or api_key or os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        client = OpenAI(api_key="not-needed", base_url=url.rstrip("/"))
        resp = client.chat.completions.create(
            model=model or "local-model",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()


@register_provider("ollama")
class OllamaProvider:
    def generate(self, api_key: str, model: str, prompt: str, max_tokens: int = 4096, base_url: str = "") -> str:
        import requests
        url = base_url or api_key or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        resp = requests.post(
            f"{url.rstrip('/')}/api/generate",
            json={"model": model or "llama3", "prompt": prompt, "stream": False, "options": {"num_predict": max_tokens, "temperature": 0.3}},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()


JUDGE_PROMPT = """You are evaluating relevance for a text-to-video search system.
For each candidate document, judge how relevant it is to the query.

Relevance scale:
  3 = highly relevant  — directly answers or matches the query topic
  2 = relevant         — related content, partial match
  1 = partially relevant — tangential connection, weak match
  0 = irrelevant       — unrelated content

IMPORTANT: Be generous and use the full scale. Only use 0 for completely unrelated content.
If there is ANY connection to the query topic, assign 1 or 2.
If the document clearly matches, assign 3.

Query: {query_text}
Query type: {query_type}
Source video: {source_video_id}

Documents:
{doc_list}

Output ONLY a valid JSON array, no other text:
[
  {{"doc_id": "...", "relevance": 0, "reason": "brief reason"}},
  ...
]"""


def _truncate(text: str, max_chars: int = 600) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _build_doc_list(docs: list[dict], corpus_map: dict, metadata_map: dict, source_video_id: str) -> str:
    lines = []
    for doc in docs:
        doc_id = doc["doc_id"]
        meta = corpus_map.get(doc_id, {})
        title = meta.get("title", "") or ""

        # Dados completos do metadado original
        vmeta = metadata_map.get(doc_id, {})
        desc = vmeta.get("description", "")
        tags = vmeta.get("tags", [])
        cats = vmeta.get("categories", [])
        transcript = vmeta.get("transcript", "")
        duration = vmeta.get("duration", 0)
        activity = vmeta.get("anet_label", "")

        marker = " [SOURCE]" if doc_id == source_video_id else ""
        lines.append(f"--- {doc_id}{marker} ---")
        lines.append(f"Title: {title}")
        if activity:
            lines.append(f"Activity: {activity}")
        if duration:
            lines.append(f"Duration: {duration}s")
        if desc:
            lines.append(f"Description: {desc[:500]}")
        if tags:
            lines.append(f"Tags: {', '.join(tags[:20])}")
        if cats:
            lines.append(f"Categories: {', '.join(cats)}")
        if transcript:
            t = transcript[:600]
            lines.append(f"Transcript: {t}")
        lines.append("")
    return "\n".join(lines)


def _parse_judgments(text: str, valid_doc_ids: set) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            raise
    if not isinstance(parsed, list):
        raise ValueError("Resposta nao e um array JSON")
    results = []
    for item in parsed:
        doc_id = item.get("doc_id", "")
        if doc_id not in valid_doc_ids:
            continue
        rel = item.get("relevance", 0)
        if rel not in (0, 1, 2, 3):
            rel = 0
        results.append({
            "doc_id": doc_id,
            "relevance": rel,
            "reason": item.get("reason", ""),
        })
    return results


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


def load_query_to_video(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_progress(path: Path) -> dict:
    if path.exists() and path.stat().st_size > 0:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"completed_queries": [], "in_progress": None, "total_docs_judged": 0, "total_calls": 0}


def save_progress(progress: dict, path: Path) -> None:
    os.makedirs(path.parent, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def get_pool_ids(pool_path: Path) -> list[str]:
    ids = []
    for f in pool_path.iterdir():
        if f.suffix == ".json" and f.stem.startswith("q_"):
            ids.append(f.stem)
    return sorted(ids)


def load_pool(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Atribui relevancia (0-3) para cada par (query, doc) nos pools via LLM"
    )
    parser.add_argument("--pool-dir", type=Path, default=_PROJECT_ROOT / "data" / "pooling")
    parser.add_argument("--corpus", type=Path, default=_PROJECT_ROOT / "data" / "corpus" / "corpus.jsonl")
    parser.add_argument("--metadata", type=Path, default=_PROJECT_ROOT / "data" / "filtered" / "metadata" / "videos_metadata.json")
    parser.add_argument("--query-to-video", type=Path, default=_PROJECT_ROOT / "data" / "queries" / "query_to_video.json")
    parser.add_argument("--output", type=Path, default=_PROJECT_ROOT / "data" / "judgments")
    parser.add_argument("--progress", type=Path, default=_PROJECT_ROOT / "data" / "judgments" / "judgments_progress.json")
    parser.add_argument("--query-ids", type=str, default="", help="Virgula separada: q_000001,q_000005 (default: todos)")
    parser.add_argument("--provider", default="openai", choices=list(PROVIDERS), help="Provedor LLM")
    parser.add_argument("--model", default="", help="Modelo (default por provedor)")
    parser.add_argument("--api-key", "-k", default="", help="Chave de API ou base URL para locais")
    parser.add_argument("--base-url", default="", help="Base URL para o provedor (opcional, sobrescreve o padrao)")
    parser.add_argument("--batch-size", type=int, default=0, help="Docs por chamada (0 = query inteira, default 20 se auto-fallback ativar)")
    parser.add_argument("--max-doc-chars", type=int, default=300, help="Max caracteres por doc no prompt")
    parser.add_argument("--sleep", type=float, default=0.3, help="Pausa entre chamadas")
    parser.add_argument("--window", "-w", type=int, default=0, help="Quantidade de queries a processar (0 = todas)")
    parser.add_argument("--offset", "-o", type=int, default=0, help="Pula as primeiras N queries")
    parser.add_argument("--force", action="store_true", help="Rejulgar queries ja existentes")
    parser.add_argument("--no-progress", action="store_true", help="Nao usar arquivo de progresso compartilhado (para multiplos terminais)")
    args = parser.parse_args()

    env_var_map = {
        "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY", "deepseek": "DEEPSEEK_API_KEY",
        "ollama": "OLLAMA_BASE_URL", "lm-studio": "LMSTUDIO_BASE_URL",
        "openrouter": "OPENROUTER_API_KEY",
    }
    api_key = args.api_key or os.environ.get(env_var_map[args.provider], "")
    if not api_key and args.provider not in ("ollama", "lm-studio", "openrouter"):
        print(f"ERRO: Chave nao informada. Use --api-key ou defina {env_var_map[args.provider]}.", file=sys.stderr)
        sys.exit(1)

    print("Carregando corpus...")
    corpus_map = load_corpus(args.corpus)
    print(f"  {len(corpus_map)} documentos")

    print("Carregando metadados originais...")
    metadata_map = {}
    if args.metadata.exists():
        with open(args.metadata, encoding="utf-8") as f:
            metadata_map = json.load(f)
        print(f"  {len(metadata_map)} videos")
    else:
        print("  arquivo nao encontrado, usando apenas corpus")

    print("Carregando query_to_video...")
    query_to_video = load_query_to_video(args.query_to_video)
    print(f"  {len(query_to_video)} mapeamentos")

    print("Carregando pools...")
    all_pool_ids = get_pool_ids(args.pool_dir)
    print(f"  {len(all_pool_ids)} pools disponiveis")

    if args.query_ids:
        query_ids = [q.strip() for q in args.query_ids.split(",") if q.strip()]
        query_ids = [q for q in query_ids if q in all_pool_ids]
        print(f"  Filtrando para {len(query_ids)} queries especificadas")
    else:
        query_ids = all_pool_ids

    if args.offset > 0:
        query_ids = query_ids[args.offset:]
        print(f"  Offset: pulando {args.offset}")

    if args.no_progress:
        progress = {"completed_queries": [], "in_progress": None, "total_docs_judged": 0, "total_calls": 0}
    else:
        progress = load_progress(args.progress)
        if not args.force:
            query_ids = [qid for qid in query_ids if qid not in progress["completed_queries"]]
            skipped = len(all_pool_ids) - len(query_ids)
            print(f"  Pulando {skipped} ja julgadas")

    if args.window > 0:
        query_ids = query_ids[:args.window]

    os.makedirs(args.output, exist_ok=True)

    print(f"\nConfiguracao:")
    print(f"  Provedor: {args.provider}  Modelo: {args.model or '(padrao)'}")
    print(f"  Batch size: {args.batch_size or 'query inteira'}")
    print(f"  Max doc chars: {args.max_doc_chars}")
    print(f"  Offset: {args.offset}")
    print(f"  Window: {args.window or 'todas'}")
    print(f"  Queries a processar: {len(query_ids)}")
    print()

    total_judged = progress.get("total_docs_judged", 0)
    total_calls = progress.get("total_calls", 0)
    start_time = time.time()

    for qi, qid in enumerate(query_ids, 1):
        pool_path = args.pool_dir / f"{qid}.json"
        if not pool_path.exists():
            print(f"[{qi}/{len(query_ids)}] {qid}: pool nao encontrado, pulando")
            continue

        pool = load_pool(pool_path)
        query_text = pool.get("query_text", "")
        query_type = pool.get("query_type", "")
        source_video_id = query_to_video.get(qid, "")

        candidates = pool.get("merged") or pool.get("pools", {}).get("bm25", [])
        if not candidates:
            print(f"[{qi}/{len(query_ids)}] {qid}: nenhum candidato, pulando")
            progress["completed_queries"].append(qid)
            if not args.no_progress:
                save_progress(progress, args.progress)
            continue

        all_doc_ids = set(c["doc_id"] for c in candidates)
        doc_ids_in_corpus = all_doc_ids & set(corpus_map.keys())

        if not doc_ids_in_corpus:
            print(f"[{qi}/{len(query_ids)}] {qid}: nenhum doc encontrado no corpus, pulando")
            progress["completed_queries"].append(qid)
            if not args.no_progress:
                save_progress(progress, args.progress)
            continue

        judgments_map = {}

        # Auto relevance 3 para documento fonte
        if source_video_id:
            judgments_map[source_video_id] = {
                "doc_id": source_video_id,
                "relevance": 3,
                "reason": "Documento fonte da consulta",
                "auto": True,
            }

        # Filtra candidatos que existem no corpus + remove o fonte (ja julgado)
        to_judge = [c for c in candidates if c["doc_id"] in doc_ids_in_corpus and c["doc_id"] != source_video_id]

        algum_sucesso = False
        if to_judge:
            batch_size = args.batch_size or len(to_judge)
            batch_start = 0
            while batch_start < len(to_judge):
                batch = to_judge[batch_start:batch_start + batch_size]
                doc_list_str = _build_doc_list(batch, corpus_map, metadata_map, source_video_id)
                prompt = JUDGE_PROMPT.format(
                    query_text=query_text,
                    query_type=query_type or "unknown",
                    source_video_id=source_video_id or "none",
                    doc_list=doc_list_str,
                )

                valid_ids = set(d["doc_id"] for d in batch)
                batch_num = batch_start // max(batch_size, 1) + 1
                total_batches = (len(to_judge) - 1) // max(batch_size, 1) + 1
                print(f"[{qi}/{len(query_ids)}] {qid} (batch {batch_num}/{total_batches}, {len(batch)} docs)...", end=" ", flush=True)

                try:
                    raw = _call_llm(args.provider, api_key, args.model, prompt, base_url=args.base_url)
                    parsed = _parse_judgments(raw, valid_ids)
                    total_calls += 1

                    for j in parsed:
                        j["auto"] = False
                        judgments_map[j["doc_id"]] = j

                    algum_sucesso = True
                    n_parsed = len(parsed)
                    print(f"OK ({n_parsed}/{len(batch)})")
                    batch_start += batch_size
                except Exception as e:
                    msg = str(e)
                    if "context length" in msg.lower() or "n_ctx" in msg.lower() or "too many tokens" in msg.lower():
                        if batch_size <= 5:
                            print(f"CONTEXTO INSUFICIENTE mesmo com batch=5. Pulando restante.")
                            break
                        batch_size = max(batch_size // 2, 5)
                        print(f"CONTEXTO EXCEDIDO. Reduzindo batch para {batch_size} docs...")
                        continue
                    else:
                        print(f"ERRO: {msg}")
                        break

                if batch_start < len(to_judge):
                    time.sleep(args.sleep)

        judgments_list = list(judgments_map.values())
        nao_julgados = len(candidates) - len(judgments_list)
        if nao_julgados > 0:
            print(f"  ({nao_julgados} docs sem julgamento, ignorados)")

        output_entry = {
            "query_id": qid,
            "query_text": query_text,
            "query_type": query_type or "",
            "source_video_id": source_video_id or "",
            "judgments": judgments_list,
            "config": {
                "provider": args.provider,
                "model": args.model or "(padrao)",
                "batch_size": args.batch_size or len(to_judge),
            },
        }

        out_path = args.output / f"{qid}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_entry, f, indent=2, ensure_ascii=True)

        if algum_sucesso or not to_judge:
            progress["completed_queries"].append(qid)
            if not algum_sucesso and not to_judge:
                print(f"  (sem candidatos para julgar, marcando como concluido)")
        elif not args.force:
            print(f"  (LLM falhou em todos os batches, query mantida como pendente)")

        progress["in_progress"] = None
        progress["total_docs_judged"] = total_judged + len(judgments_list)
        progress["total_calls"] = total_calls
        progress["last_query"] = qid
        progress["timestamp"] = datetime.now(timezone.utc).isoformat()
        if not args.no_progress:
            save_progress(progress, args.progress)

        total_judged += len(judgments_list)

        elapsed = time.time() - start_time
        rate = (qi) / (elapsed / 60) if elapsed > 0 else 0
        print(f"  -> {len(judgments_list)} julgamentos salvos ({elapsed:.0f}s decorridos, {rate:.1f} queries/min)")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Queries processadas: {len(query_ids)}")
    print(f"Total julgamentos: {total_judged}")
    print(f"Chamadas LLM: {total_calls}")
    print(f"Tempo: {elapsed/60:.1f} min")
    print(f"Julgamentos salvos em: {args.output}")


if __name__ == "__main__":
    main()
