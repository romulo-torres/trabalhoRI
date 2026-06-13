import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _path(*parts: str) -> Path:
    return _PROJECT_ROOT.joinpath(*parts)

# ---------------------------------------------------------------------------
# Providers
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


def _call_llm(provider: str, api_key: str, model: str, prompt: str) -> str:
    cls = PROVIDERS.get(provider)
    if not cls:
        raise ProviderError(f"Provedor desconhecido: {provider}. Disponiveis: {', '.join(PROVIDERS)}")

    max_retries = 5
    base_wait = 5
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return cls().generate(api_key, model, prompt)
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


# OpenAI

@register_provider("openai")
class OpenAIProvider:
    def generate(self, api_key: str, model: str, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ProviderError("openai nao instalado. rode: pip install openai")

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model or "gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=256,
        )
        return resp.choices[0].message.content.strip()


# Anthropic

@register_provider("anthropic")
class AnthropicProvider:
    def generate(self, api_key: str, model: str, prompt: str) -> str:
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ProviderError("anthropic nao instalado. rode: pip install anthropic")

        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model or "claude-sonnet-4-20250514",
            max_tokens=256,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()


# Google Gemini

@register_provider("gemini")
class GeminiProvider:
    def generate(self, api_key: str, model: str, prompt: str) -> str:
        try:
            from google import genai
        except ImportError:
            raise ProviderError("google-genai nao instalado. rode: pip install google-genai")

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model or "gemini-2.0-flash",
            contents=prompt,
        )
        return resp.text.strip()


# DeepSeek (API compatível com OpenAI)

@register_provider("deepseek")
class DeepSeekProvider:
    def generate(self, api_key: str, model: str, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ProviderError("openai nao instalado. rode: pip install openai")

        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model=model or "deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=256,
        )
        return resp.choices[0].message.content.strip()


# LM Studio (local)

@register_provider("lm-studio")
class LMStudioProvider:
    def generate(self, api_key: str, model: str, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ProviderError("openai nao instalado. rode: pip install openai")

        base_url = api_key or os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        base_url = base_url.rstrip("/")

        client = OpenAI(api_key="not-needed", base_url=base_url)
        resp = client.chat.completions.create(
            model=model or "local-model",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=256,
        )
        return resp.choices[0].message.content.strip()


# Ollama (local)

@register_provider("ollama")
class OllamaProvider:
    def generate(self, api_key: str, model: str, prompt: str) -> str:
        try:
            import requests
        except ImportError:
            raise ProviderError("requests nao instalado. rode: pip install requests")

        base_url = api_key or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        base_url = base_url.rstrip("/")

        resp = requests.post(
            f"{base_url}/api/generate",
            json={"model": model or "llama3", "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert at creating text-to-video search queries. Given the metadata of a video, generate a natural language query in English that would retrieve this exact video in a search engine.

Rules:
- Write a single concise query (5-9 words).
- Focus on the visual activity: what action is happening, who is doing it, and the setting.
- Use keywords from the title, description, tags, and transcript.
- Do NOT include metadata field names, labels, or JSON syntax.
- Output ONLY the query, nothing else."""


def build_prompt(entry: dict) -> str:
    parts = []
    if entry.get("title"):
        parts.append(f"Title: {entry['title']}")
    if entry.get("description"):
        desc = entry["description"][:500]
        parts.append(f"Description: {desc}")
    if entry.get("tags"):
        parts.append(f"Tags: {', '.join(entry['tags'][:15])}")
    if entry.get("categories"):
        parts.append(f"Categories: {', '.join(entry['categories'])}")
    if entry.get("anet_label"):
        parts.append(f"Activity label: {entry['anet_label']}")
    if entry.get("feature_desc"):
        parts.append(f"Feature description: {entry['feature_desc']}")
    if entry.get("keywords"):
        parts.append(f"Keywords: {entry['keywords']}")
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
    return f"{SYSTEM_PROMPT}\n\nVideo metadata:\n{metadata_block}\n\nQuery:"


# ---------------------------------------------------------------------------
# Progress / state
# ---------------------------------------------------------------------------

def load_queries(path: Path) -> dict:
    if path.exists() and path.stat().st_size > 0:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_queries(queries: dict, path: Path) -> None:
    os.makedirs(path.parent, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gera consultas de busca textual para cada video da colecao usando LLM",
    )
    parser.add_argument(
        "--input", type=Path,
        default=_path("data", "metadata", "videos_metadata.json"),
        help="JSON de entrada (default: videos_metadata.json)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=_path("data", "queries", "queries.json"),
        help="JSON de saida das consultas geradas",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "anthropic", "gemini", "deepseek", "ollama", "lm-studio"],
        help="Provedor de LLM (default: openai)",
    )
    parser.add_argument(
        "--model",
        default="",
        help=(
            "Modelo a usar. Se omitido, usa o padrao do provedor:\n"
            "  openai:    gpt-4o\n"
            "  anthropic: claude-sonnet-4-20250514\n"
            "  gemini:    gemini-2.0-flash\n"
            "  deepseek:  deepseek-chat\n"
            "  ollama:    llama3\n"
            "  lm-studio: local-model (modelo carregado no LM Studio)"
        ),
    )
    parser.add_argument(
        "--api-key", "-k",
        default="",
        help="Chave de API. Se omitida, usa a variavel de ambiente correspondente.",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Forca regeneracao mesmo se a consulta ja existir",
    )
    parser.add_argument(
        "--window", "-w", type=int, default=0,
        help="Janela: quantidade de NOVAS consultas a gerar (0 = todas as pendentes)",
    )
    parser.add_argument(
        "--offset", "-o", type=int, default=0,
        help="Pula os primeiros N videos na lista completa (para retomar de um ponto)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="(deprecated) Usar --window no lugar",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.5,
        help="Pausa entre chamadas para evitar rate limits (default: 0.5s)",
    )

    args = parser.parse_args()

    # API key
    env_var_map = {
        "openai":   "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini":    "GOOGLE_API_KEY",
        "deepseek":  "DEEPSEEK_API_KEY",
        "ollama":    "OLLAMA_BASE_URL",
        "lm-studio": "LMSTUDIO_BASE_URL",
    }
    api_key = args.api_key or os.environ.get(env_var_map[args.provider], "")
    if not api_key and args.provider not in ("ollama", "lm-studio"):
        print(
            f"ERRO: Chave nao informada. Use --api-key ou defina {env_var_map[args.provider]}.\n"
            f"Para modelos locais: --provider ollama ou lm-studio.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load data
    if not args.input.exists():
        print(f"ERRO: Arquivo de entrada nao encontrado: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Carregando {args.input}...")
    with open(args.input, encoding="utf-8") as f:
        videos = json.load(f)

    queries = load_queries(args.output)

    # Lista completa ordenada
    video_ids = sorted(videos.keys())
    total_disponivel = len(video_ids)

    # Aplica offset na lista completa (para retomar de um ponto)
    if args.offset > 0:
        video_ids = video_ids[args.offset:]

    # Filtra videos que ja tem consulta (a menos que --force)
    if not args.force:
        pendentes = [vid for vid in video_ids if vid not in queries]
        ja_existem = len(video_ids) - len(pendentes)
        if ja_existem:
            print(f"Pulando {ja_existem} videos com consulta ja existente (use --force para regenerar)")
        video_ids = pendentes

    # Aplica --window sobre os videos PENDENTES
    window_size = args.window if args.window > 0 else args.limit
    if window_size > 0:
        video_ids = video_ids[:window_size]

    print(f"Total de videos no arquivo: {total_disponivel}")
    print(f"Janela (novas consultas): offset={args.offset}  window={window_size or 'ilimitado'}")
    print(f"Novas consultas a gerar:  {len(video_ids)}")
    print(f"Provedor: {args.provider}  Modelo: {args.model or '(padrao)'}")

    ok = 0
    errors = 0
    start_time = time.time()

    for idx, vid in enumerate(video_ids, 1):
        entry = videos[vid]
        prompt = build_prompt(entry)

        print(f"[{idx}/{len(video_ids)}] {vid}...", end=" ", flush=True)

        try:
            query = _call_llm(args.provider, api_key, args.model, prompt)

            if not query:
                raise ProviderError("LLM retornou resposta vazia")

            queries[vid] = {
                "query": query,
                "model": args.model or "(padrao)",
                "provider": args.provider,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            ok += 1
            print(f'"{query[:70]}{"..." if len(query) > 70 else ""}"')

        except Exception as e:
            errors += 1
            print(f"ERRO: {e}")

        if (idx % 10) == 0 or idx == len(video_ids):
            save_queries(queries, args.output)

        if idx < len(video_ids):
            time.sleep(args.sleep)

    save_queries(queries, args.output)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Geradas: {ok}  Erros: {errors}")
    print(f"Tempo: {elapsed/60:.1f} min  ({elapsed/max(ok,1):.1f}s/consulta)")
    print(f"Salvo em: {args.output}")


if __name__ == "__main__":
    main()
