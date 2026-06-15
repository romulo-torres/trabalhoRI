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

def _path(*parts: str) -> Path:
    return _PROJECT_ROOT.joinpath(*parts)

# ---------------------------------------------------------------------------
# Providers (mesmo código anterior)
# ---------------------------------------------------------------------------

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


def _call_llm(provider: str, api_key: str, model: str, prompt: str, base_url: str = "") -> str:
    cls = PROVIDERS.get(provider)
    if not cls:
        raise ProviderError(f"Provedor desconhecido: {provider}. Disponiveis: {', '.join(PROVIDERS)}")
    max_retries = 5
    base_wait = 5
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return cls().generate(api_key, model, prompt, base_url=base_url)
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
    def generate(self, api_key: str, model: str, prompt: str, base_url: str = "") -> str:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url or None)
        resp = client.chat.completions.create(
            model=model or "gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=512,
        )
        return resp.choices[0].message.content.strip()


@register_provider("anthropic")
class AnthropicProvider:
    def generate(self, api_key: str, model: str, prompt: str, base_url: str = "") -> str:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model or "claude-sonnet-4-20250514",
            max_tokens=512, temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()


@register_provider("gemini")
class GeminiProvider:
    def generate(self, api_key: str, model: str, prompt: str, base_url: str = "") -> str:
        from google import genai
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model or "gemini-2.0-flash",
            contents=prompt,
        )
        return resp.text.strip()


@register_provider("deepseek")
class DeepSeekProvider:
    def generate(self, api_key: str, model: str, prompt: str, base_url: str = "") -> str:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model=model or "deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=512,
        )
        return resp.choices[0].message.content.strip()


@register_provider("openrouter")
class OpenRouterProvider:
    def generate(self, api_key: str, model: str, prompt: str, base_url: str = "") -> str:
        from openai import OpenAI
        url = base_url or "https://openrouter.ai/api/v1"
        client = OpenAI(api_key=api_key, base_url=url)
        resp = client.chat.completions.create(
            model=model or "openai/gpt-oss-120b:free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=512,
        )
        return resp.choices[0].message.content.strip()


@register_provider("lm-studio")
class LMStudioProvider:
    def generate(self, api_key: str, model: str, prompt: str, base_url: str = "") -> str:
        from openai import OpenAI
        url = base_url or api_key or os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        client = OpenAI(api_key="not-needed", base_url=url.rstrip("/"))
        resp = client.chat.completions.create(
            model=model or "local-model",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=512,
        )
        return resp.choices[0].message.content.strip()


@register_provider("ollama")
class OllamaProvider:
    def generate(self, api_key: str, model: str, prompt: str, base_url: str = "") -> str:
        import requests
        url = base_url or api_key or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        resp = requests.post(
            f"{url.rstrip('/')}/api/generate",
            json={"model": model or "llama3", "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()


# ---------------------------------------------------------------------------
# Prompt — 4 tipos de consulta por vídeo
# ---------------------------------------------------------------------------

MULTI_QUERY_PROMPT = """You are an expert at creating text-to-video search queries. Given the metadata of a video, you MUST generate exactly 2 search queries: one factoid and one keyword.

REQUIREMENTS (do not skip any):
1. **factoid** — A short fact-based question (max 10 words). Example: "What is BM25?"
2. **keyword** — A concise keyword query (max 8 words). Example: "bm25 parameters tutorial"

Rules:
- You MUST output BOTH types, always. Never omit one.
- Each query must be different from the other.
- Focus on visual activity: action, people, setting, objects.
- Do NOT include video IDs, metadata field names, or JSON syntax.
- Do NOT copy phrases directly from the metadata.

Output ONLY a valid JSON array with exactly 2 objects, no other text:
[
  {"type": "factoid", "query": "..."},
  {"type": "keyword", "query": "..."}
]"""


def build_prompt(entry: dict) -> str:
    parts = []
    if entry.get("title"):
        parts.append(f"Title: {entry['title']}")
    if entry.get("description"):
        parts.append(f"Description: {entry['description'][:500]}")
    if entry.get("tags"):
        parts.append(f"Tags: {', '.join(entry['tags'][:15])}")
    if entry.get("categories"):
        parts.append(f"Categories: {', '.join(entry['categories'])}")
    if entry.get("anet_label"):
        parts.append(f"Activity label: {entry['anet_label']}")
    transcript = entry.get("transcript", "")
    if transcript:
        clean = transcript[:600]
        for prefix in ["Kind: captions", "Language: en"]:
            if clean.lower().startswith(prefix.lower()):
                clean = clean[len(prefix):].strip()
        for sep in ["Kind: captions", "Language: en"]:
            if sep.lower() in clean.lower():
                clean = clean.split(sep, 1)[-1]
        clean = clean.strip().lstrip(",").strip()
        if clean:
            parts.append(f"Transcript excerpt: {clean[:400]}")
    metadata_block = "\n".join(parts)
    return f"{MULTI_QUERY_PROMPT}\n\nVideo metadata:\n{metadata_block}\n\nJSON output:"


# ---------------------------------------------------------------------------
# Parse LLM response
# ---------------------------------------------------------------------------

def _parse_queries_response(text: str, video_id: str) -> list[dict]:
    """Extrai array JSON da resposta do LLM."""
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
        qtype = item.get("type", "natural")
        query = item.get("query", "").strip()
        if query:
            results.append({"type": qtype, "query": query})
    return results


# ---------------------------------------------------------------------------
# Filtragem pós-geração
# ---------------------------------------------------------------------------

def _is_generic(query: str) -> bool:
    """Detecta consultas muito genéricas."""
    q = query.lower().strip()
    generic = {
        "video", "watch", "see", "look", "show", "find", "search",
        "how to", "what is", "funny", "amazing", "cool", "best",
    }
    tokens = q.split()
    if len(tokens) < 3:
        return True
    # Se todo token é stopword-like, é genérico
    specific = sum(1 for t in tokens if t not in generic and len(t) > 2)
    return specific < 2


def _copied_from_text(query: str, meta_text: str) -> bool:
    """Detecta se a query copiou frase literal do texto."""
    q = query.lower().strip()
    t = meta_text.lower()
    # Se mais de 60% da query aparece literalmente no texto
    words = q.split()
    if len(words) < 4:
        return False
    match_len = 0
    for w in words:
        if len(w) > 3 and w in t:
            match_len += len(w)
    return match_len / len(q) > 0.6


def _contains_video_id(query: str, video_id: str) -> bool:
    return video_id.lower() in query.lower()


def _deduplicate(queries: list[dict], threshold: float = 0.85) -> list[dict]:
    """Remove consultas muito similares via TF-IDF + cosseno."""
    texts = [q["query"] for q in queries]
    if len(texts) <= 1:
        return queries
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(stop_words="english").fit_transform(texts)
        sim = cosine_similarity(vec)
        keep = [True] * len(texts)
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if keep[j] and sim[i, j] > threshold:
                    keep[j] = False
        return [q for i, q in enumerate(queries) if keep[i]]
    except ImportError:
        return queries


def filter_queries(queries: list[dict], video_id: str, meta_text: str) -> list[dict]:
    """Aplica todos os filtros pós-geração."""
    filtered = []
    for q in queries:
        if _contains_video_id(q["query"], video_id):
            continue
        if _copied_from_text(q["query"], meta_text):
            continue
        if _is_generic(q["query"]):
            continue
        filtered.append(q)
    return _deduplicate(filtered)


# ---------------------------------------------------------------------------
# Estado / checkpointing
# ---------------------------------------------------------------------------

def load_state(path: Path) -> dict:
    if path.exists() and path.stat().st_size > 0:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict, path: Path) -> None:
    os.makedirs(path.parent, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Export BEIR
# ---------------------------------------------------------------------------

def export_beir(state: dict, queries_jsonl: Path, qrels_tsv: Path, query_to_video: Path) -> None:
    """Exporta consultas no formato BEIR (queries.jsonl + qrels.tsv)."""
    os.makedirs(queries_jsonl.parent, exist_ok=True)
    os.makedirs(qrels_tsv.parent, exist_ok=True)

    qid_map = {}  # video_id -> qid list

    total_q_before = sum(len(e.get("queries", [])) for e in state.values())
    print(f"\nBEIR export: {total_q_before} queries no estado, escrevendo para {queries_jsonl}...")

    with open(queries_jsonl, "w", encoding="utf-8") as fq:
        for vid, entry in sorted(state.items()):
            qlist = entry.get("queries", [])
            for q in qlist:
                if "id" not in q:
                    print(f"  AVISO: query sem 'id' no video {vid}: {q}")
                    continue
                qid = q["id"]
                fq.write(json.dumps({"_id": qid, "text": q["query"], "type": q["type"]}, ensure_ascii=False) + "\n")
                qid_map.setdefault(vid, []).append(qid)

    # qrels.tsv: documento fonte = relevance 3
    with open(qrels_tsv, "w", encoding="utf-8") as f:
        f.write("query-id\tdocument-id\trelevance\n")
        for vid, qids in qid_map.items():
            for qid in qids:
                f.write(f"{qid}\t{vid}\t3\n")

    # query_to_video mapping
    mapping = {}
    for vid, qids in qid_map.items():
        for qid in qids:
            mapping[qid] = vid
    with open(query_to_video, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    total_q = sum(len(e.get("queries", [])) for e in state.values())
    print(f"\nBEIR export:")
    print(f"  queries.jsonl: {queries_jsonl} ({total_q} queries)")
    print(f"  qrels.tsv:     {qrels_tsv} ({sum(len(v) for v in qid_map.values())} linhas)")
    print(f"  query_to_video: {query_to_video}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gera 4 consultas textuais (factoid, keyword, natural, verbose) por video via LLM",
    )
    parser.add_argument("--input", type=Path, default=_path("data", "metadata", "videos_metadata.json"),
                        help="JSON de entrada (default: videos_metadata.json)")
    parser.add_argument("--output", type=Path, default=_path("data", "queries", "queries.json"),
                        help="JSON de saida com estado completo (default: data/queries/queries.json)")
    parser.add_argument("--queries-jsonl", type=Path, default=_path("data", "queries", "queries.jsonl"),
                        help="Saida BEIR queries.jsonl")
    parser.add_argument("--qrels", type=Path, default=_path("data", "qrels", "qrels.tsv"),
                        help="Saida BEIR qrels.tsv")
    parser.add_argument("--query-to-video", type=Path, default=_path("data", "queries", "query_to_video.json"),
                        help="Mapeamento query_id -> video_id")
    parser.add_argument("--provider", default="openai",
                        choices=["openai", "anthropic", "gemini", "deepseek", "ollama", "lm-studio", "openrouter"],
                        help="Provedor de LLM (default: openai)")
    parser.add_argument("--model", default="",
                        help="Modelo (default por provedor)")
    parser.add_argument("--api-key", "-k", default="",
                        help="Chave de API ou base URL para locais")
    parser.add_argument("--base-url", default="",
                        help="Base URL para o provedor (opcional, sobrescreve o padrao)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Forca regeneracao mesmo se ja existir")
    parser.add_argument("--window", "-w", type=int, default=0,
                        help="Janela: quantidade de videos a processar (0 = todos pendentes)")
    parser.add_argument("--offset", "-o", type=int, default=0,
                        help="Pula os primeiros N videos")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Pausa entre chamadas (default: 0.5s)")
    parser.add_argument("--no-export", action="store_true",
                        help="Nao exporta BEIR ao final")

    args = parser.parse_args()

    # API key
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

    # Load input
    if not args.input.exists():
        print(f"ERRO: {args.input} nao encontrado", file=sys.stderr)
        sys.exit(1)
    print(f"Carregando {args.input}...")
    with open(args.input, encoding="utf-8") as f:
        videos = json.load(f)

    # Load state anterior
    state = load_state(args.output)
    next_qid = 1
    for entry in state.values():
        for q in entry.get("queries", []):
            mid = int(q.get("id", "0").lstrip("q_"))
            if mid >= next_qid:
                next_qid = mid + 1

    # Prepara lista de videos
    video_ids = sorted(videos.keys())
    if args.offset > 0:
        video_ids = video_ids[args.offset:]

    if not args.force:
        pendentes = [vid for vid in video_ids if vid not in state]
        ja_existem = len(video_ids) - len(pendentes)
        if ja_existem:
            print(f"Pulando {ja_existem} videos ja processados (use --force para regenerar)")
        video_ids = pendentes

    window_size = args.window
    if window_size > 0:
        video_ids = video_ids[:window_size]

    # Conjunto de queries existentes para dedup
    existing_queries = set()
    for entry in state.values():
        for q in entry.get("queries", []):
            existing_queries.add(q["query"].lower().strip())

    print(f"\nTotal videos: {len(videos)}")
    print(f"Janela: offset={args.offset}  window={window_size or 'todos pendentes'}")
    print(f"A processar: {len(video_ids)}")
    print(f"Provedor: {args.provider}  Modelo: {args.model or '(padrao)'}")
    print(f"Proximo qid: q_{next_qid:06d}\n")

    ok = 0
    errors = 0
    start_time = time.time()

    for idx, vid in enumerate(video_ids, 1):
        entry = videos[vid]
        prompt = build_prompt(entry)

        # Monta texto completo dos metadados para filtro anti-cópia
        meta_text = " ".join(filter(None, [
            entry.get("title", ""), entry.get("description", ""),
            " ".join(entry.get("tags", [])),
            " ".join(entry.get("categories", [])),
            entry.get("transcript", ""),
        ]))

        print(f"[{idx}/{len(video_ids)}] {vid}...", end=" ", flush=True)

        try:
            raw = _call_llm(args.provider, api_key, args.model, prompt, base_url=args.base_url)
            parsed = _parse_queries_response(raw, vid)

            if not parsed:
                raise ProviderError("Nenhuma consulta extraida da resposta")

            # Filtragem
            parsed = filter_queries(parsed, vid, meta_text)

            if not parsed:
                raise ProviderError("Todas as consultas foram filtradas")

            # Atribui IDs
            queries_out = []
            for q in parsed:
                qtext = q["query"].strip()
                if qtext.lower() in existing_queries:
                    print(f"  [dedup: {qtext[:50]}]", end=" ")
                    continue
                qid_str = f"q_{next_qid:06d}"
                queries_out.append({"id": qid_str, "type": q["type"], "query": qtext})
                next_qid += 1
                existing_queries.add(qtext.lower())

            state[vid] = {
                "queries": queries_out,
                "model": args.model or "(padrao)",
                "provider": args.provider,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            ok += 1
            types = ", ".join(q["type"] for q in queries_out)
            print(f"[{types}] \"{queries_out[0]['query'][:60]}...\"")

        except Exception as e:
            errors += 1
            print(f"ERRO: {e}")

        if (idx % 10) == 0 or idx == len(video_ids):
            save_state(state, args.output)
            if not args.no_export:
                try:
                    export_beir(state, args.queries_jsonl, args.qrels, args.query_to_video)
                except Exception as e:
                    print(f"\nERRO no BEIR export: {e}")
                    import traceback
                    traceback.print_exc()

        if idx < len(video_ids):
            time.sleep(args.sleep)

    save_state(state, args.output)
    if not args.no_export:
        try:
            export_beir(state, args.queries_jsonl, args.qrels, args.query_to_video)
        except Exception as e:
            print(f"\nERRO no BEIR export final: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time
    total_q = sum(len(e.get("queries", [])) for e in state.values())
    print(f"\n{'='*60}")
    print(f"Videos processados: {ok}  Erros: {errors}")
    print(f"Total consultas: {total_q}")
    print(f"Tempo: {elapsed/60:.1f} min  ({elapsed/max(ok,1):.1f}s/video)")
    print(f"Estado salvo em: {args.output}")


if __name__ == "__main__":
    main()
