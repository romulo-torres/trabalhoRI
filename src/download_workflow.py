import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

from logger import setup_logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _proj_path(*parts: str) -> Path:
    return _PROJECT_ROOT.joinpath(*parts)


logger = setup_logger("download_workflow", str(_proj_path("logs", "pipeline.log")))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEAD_VIDEOS_PATH = _proj_path("data", "state", "dead_videos.txt")
ANET_JSON_PATH   = _proj_path("data", "activity_net.v1-3.min.json")
STATE_PATH       = _proj_path("data", "state", "download_state.json")
OUTPUT_DIR       = _proj_path("data", "videos")
METADATA_DIR     = _proj_path("data", "metadata")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_dead_videos(path: Path = DEAD_VIDEOS_PATH) -> set[str]:
    if not path.exists():
        logger.warning("dead_videos.txt nao encontrado - nenhum video sera filtrado.")
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def load_activitynet_database(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data["database"]


def _default_state() -> dict:
    return {"downloaded": [], "failed": [], "skipped": [], "permanently_dead": []}


def load_state(path: Path = STATE_PATH) -> dict:
    if not path.exists():
        return _default_state()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"State corrompido: {e}. Resetando.")
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return _default_state()


def _append_dead_video(video_id: str, path: Path = DEAD_VIDEOS_PATH) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = set()
        if path.exists():
            with open(path) as f:
                existing = {line.strip() for line in f if line.strip()}
        if video_id not in existing:
            with open(path, "a") as f:
                f.write(f"{video_id}\n")
            logger.info(f"{video_id}: adicionado a {path}")
    except Exception as e:
        logger.warning(f"{video_id}: erro ao registrar dead: {e}")


def save_state(state: dict, path: Path = STATE_PATH) -> None:
    os.makedirs(path.parent, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        tmp.replace(path)
    except OSError as e:
        logger.warning(f"Nao foi possivel salvar state: {e}")


def load_existing_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(f"Metadados corrompidos em {path}: {e}. Iniciando do zero.")
        backup = path.with_suffix(".json.bak")
        try:
            path.rename(backup)
            logger.info(f"Backup salvo em {backup}")
        except Exception:
            pass
        return {}


# ---------------------------------------------------------------------------
# Detecção de deadvideos
# ---------------------------------------------------------------------------
_DEAD_PATTERNS = [
    "video unavailable",
    "this video is private",
    "this video has been removed",
    "this video is not available",
    "this content is age-restricted and",
    "sign in to confirm your age",
    "this video has been flagged",
    "content is not available on this platform",
    "http error 403",
    "http error 404",
    "http error 410",
    "unable to extract video data",
]


def _is_permanently_dead(stderr: str) -> bool:
    err_lower = stderr.lower()
    return any(p in err_lower for p in _DEAD_PATTERNS)


# ---------------------------------------------------------------------------
# yt-dlp helpers
# ---------------------------------------------------------------------------
def video_exists(video_id: str, output_dir: Path) -> bool:
    path = output_dir / f"{video_id}.mp4"
    return path.exists() and path.stat().st_size > 10_000


def is_valid_mp4(path: Path) -> bool:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(path)],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0 and result.stdout.strip()
    except Exception:
        return False


def download_video(video_id: str, output_dir: Path, browser: str = "chrome") -> str:
    output_path = output_dir / f"{video_id}.mp4"

    if output_path.exists():
        if is_valid_mp4(output_path):
            logger.info(f"{video_id}: ja existe e valido")
            return "ok"
        else:
            logger.warning(f"{video_id}: corrompido - removendo")
            output_path.unlink(missing_ok=True)

    url = f"https://www.youtube.com/watch?v={video_id}"

    def _run(cookies_from_browser: str | None) -> str | None:
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]",
            "--merge-output-format", "mp4",
            "--no-write-info-json",
            "--no-write-thumbnail",
            "--no-playlist",
            "--retries", "5",
            "--fragment-retries", "5",
            "-o", str(output_path),
            url,
        ]
        if cookies_from_browser:
            cmd.insert(5, "--cookies-from-browser")
            cmd.insert(6, cookies_from_browser)
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
            if output_path.exists() and is_valid_mp4(output_path):
                return "ok"
            logger.warning(f"{video_id}: download concluido mas arquivo invalido")
            return "fail"
        except subprocess.TimeoutExpired:
            logger.warning(f"{video_id}: timeout ao baixar")
            return "fail"
        except subprocess.CalledProcessError as e:
            err = (e.stderr or "")[:400]
            if cookies_from_browser and "could not find" in err.lower():
                logger.info(f"{video_id}: cookies nao disponiveis - tentando sem cookies")
                return None
            if _is_permanently_dead(err):
                logger.warning(f"{video_id}: video permanentemente indisponivel: {err[:120]}")
                return "dead"
            logger.warning(f"{video_id}: erro yt-dlp: {err[:200]}")
            return "fail"
        except FileNotFoundError:
            logger.error(f"{video_id}: yt-dlp nao encontrado. Instale com: pip install yt-dlp")
            return "fail"
        except Exception as e:
            logger.warning(f"{video_id}: erro inesperado: {e}")
            return "fail"

    if browser:
        result = _run(browser)
        if result is not None:
            return result

    return _run(None) or "fail"


def fetch_metadata(video_id: str) -> dict:
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-playlist", url],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return {}
        data = json.loads(result.stdout)
        return {
            "video_id":    video_id,
            "title":       data.get("title", ""),
            "description": data.get("description", ""),
            "upload_date": data.get("upload_date", ""),
            "duration":    data.get("duration", 0),
            "view_count":  data.get("view_count", 0),
            "like_count":  data.get("like_count", 0),
            "channel":     data.get("uploader", ""),
            "tags":        data.get("tags", []),
            "categories":  data.get("categories", []),
            "url":         url,
            "transcript":  fetch_transcript(video_id),
        }
    except Exception as e:
        logger.warning(f"{video_id}: falha ao obter metadados: {e}")
        return {}


def fetch_transcript(video_id: str, output_dir: str | Path = "") -> str:
    """Baixa legenda automática EN via yt-dlp e retorna o texto limpo.
    Retorna string vazia se indisponível."""
    subs_dir = Path(output_dir) if output_dir else METADATA_DIR / "subs"
    subs_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        subprocess.run([
            "yt-dlp",
            "--write-auto-sub",
            "--sub-lang", "en",
            "--sub-format", "vtt",
            "--skip-download",
            "--no-playlist",
            "-o", str(subs_dir / f"{video_id}.%(ext)s"),
            url,
        ], check=True, capture_output=True, timeout=30)

        for suffix in ["en", "en-orig"]:
            vtt_path = subs_dir / f"{video_id}.{suffix}.vtt"
            if vtt_path.exists():
                text = _parse_vtt(vtt_path)
                vtt_path.unlink(missing_ok=True)
                return text

    except subprocess.CalledProcessError as e:
        err = (e.stderr or "")[:200].strip()
        if err:
            logger.debug(f"{video_id}: transcript indisponivel: {err}")
    except Exception as e:
        logger.warning(f"{video_id}: erro ao buscar transcript: {e}")
    finally:
        # limpa qualquer .vtt que possa ter sobrado
        for suffix in ["en", "en-orig"]:
            vtt_path = subs_dir / f"{video_id}.{suffix}.vtt"
            vtt_path.unlink(missing_ok=True)
    return ""


def _parse_vtt(path: Path) -> str:
    """Extrai texto limpo de VTT, removendo timestamps, tags e duplicatas."""
    content = path.read_text(encoding="utf-8")

    content = re.sub(r"\d{2}:\d{2}:\d{2}\.\d+ --> .*\n", "", content)
    content = re.sub(r"<[^>]+>", "", content)
    content = re.sub(r"^WEBVTT.*\n", "", content, flags=re.MULTILINE)

    lines   = [l.strip() for l in content.splitlines() if l.strip()]
    deduped = [lines[0]] if lines else []
    for line in lines[1:]:
        if line != deduped[-1]:
            deduped.append(line)

    return " ".join(deduped)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _save_metadata_safe(all_metadata: dict, path: Path) -> None:
    os.makedirs(path.parent, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False)
        tmp.replace(path)
    except OSError as e:
        logger.warning(f"Nao foi possivel salvar metadados: {e}")


def _enrich_metadata(meta_info: dict, anet_entry: dict, video_id: str, status: str = "downloaded") -> dict:
    anet_label = ""
    if anet_entry.get("annotations"):
        anet_label = anet_entry["annotations"][0].get("label", "")
    meta_info["anet_label"] = anet_label
    meta_info["anet_subset"] = anet_entry.get("subset", "")
    meta_info["anet_duration"] = anet_entry.get("duration", 0)
    meta_info["status"] = status
    meta_info["video_id"] = video_id
    return meta_info


def _handle_download_result(
    vid: str,
    result: str,
    state: dict,
    failed_ids: set,
    permanently_dead: set,
    metadata_dir: Path,
) -> tuple[int, int]:
    if result == "ok":
        state["failed"] = [v for v in state["failed"] if v != vid]
        state["permanently_dead"] = [v for v in state["permanently_dead"] if v != vid]
        save_state(state)
        return 1, 0

    if result == "dead":
        state["failed"] = [v for v in state["failed"] if v != vid]
        dead_set = set(state.get("permanently_dead", [])) | {vid}
        state["permanently_dead"] = sorted(dead_set)
        save_state(state)
        _append_dead_video(vid)
        return 0, 1

    state["failed"] = sorted(set(state.get("failed", [])) | {vid})
    save_state(state)
    return 0, 1


def run(
    n_videos: int,
    subset: str = "validation",
    json_path: Path = ANET_JSON_PATH,
    output_dir: Path = OUTPUT_DIR,
    metadata_dir: Path = METADATA_DIR,
    browser: str = "chrome",
    force: bool = False,
    window: bool = False,
    parity: bool = False,
    transcript: bool = False,
) -> None:
    """Download N videos from ActivityNet, resume from state."""
    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    logger.info("Carregando ActivityNet...")
    database = load_activitynet_database(json_path)
    dead = load_dead_videos()

    candidates = [(vid, meta) for vid, meta in database.items() if meta.get("subset") == subset]
    logger.info(f"Total {subset}: {len(candidates)} videos")

    candidates = [(vid, meta) for vid, meta in candidates if vid not in dead]
    logger.info(f"Apos remover dead.txt: {len(candidates)} videos")

    state = load_state()
    failed_ids = set(state.get("failed", []))
    permanently_dead_set = set(state.get("permanently_dead", []))

    candidates = [(vid, meta) for vid, meta in candidates if vid not in permanently_dead_set]
    logger.info(f"Apos remover permanentemente mortos: {len(candidates)} candidatos")

    if not force:
        candidates = [(vid, meta) for vid, meta in candidates if vid not in failed_ids]
        logger.info(f"Apos remover falhas anteriores: {len(candidates)} candidatos")

    metadata_path = metadata_dir / "videos_metadata.json"
    all_metadata = load_existing_metadata(metadata_path)

    if transcript:
        _run_backfill_transcript(metadata_path=metadata_path, all_metadata=all_metadata, metadata_dir=metadata_dir, start_time=start_time)
        return

    if parity:
        _run_parity(
            output_dir=output_dir,
            metadata_path=metadata_path,
            all_metadata=all_metadata,
            state=state,
            failed_ids=failed_ids,
            permanently_dead_set=permanently_dead_set,
            browser=browser,
            metadata_dir=metadata_dir,
            database=database,
            start_time=start_time,
        )
        return

    if window:
        _run_window(
            n_videos=n_videos,
            candidates=candidates,
            output_dir=output_dir,
            metadata_path=metadata_path,
            all_metadata=all_metadata,
            state=state,
            failed_ids=failed_ids,
            permanently_dead_set=permanently_dead_set,
            browser=browser,
            metadata_dir=metadata_dir,
            start_time=start_time,
        )
        return

    selected = candidates[:n_videos]
    logger.info(f"Selecionados {len(selected)} videos para processar")

    if not selected:
        logger.info("Nada para processar.")
        return

    total = len(selected)
    video_downloads = 0
    meta_fetches    = 0
    skips           = 0
    fails           = 0

    for idx, (vid, anet_entry) in enumerate(selected, 1):
        logger.info(f"[{idx}/{total}] Processando {vid}...")
        has_video = video_exists(vid, output_dir)
        has_meta  = vid in all_metadata

        # Caso 1: tudo ok
        if has_video and has_meta:
            logger.info(f"{vid}: video + metadata ok, pulando")
            skips += 1
            continue

        # Caso 2: so falta o video
        if has_meta and not has_video:
            logger.info(f"{vid}: metadata existe, baixando apenas o video")
            result = download_video(vid, output_dir, browser=browser)
            d, f = _handle_download_result(vid, result, state, failed_ids, permanently_dead_set, metadata_dir)
            video_downloads += d
            fails += f
            if result == "ok":
                all_metadata[vid]["status"] = "downloaded"
                _save_metadata_safe(all_metadata, metadata_path)
            continue

        # Caso 3: video existe, mas metadata nao
        if has_video and not has_meta:
            logger.info(f"{vid}: video existe, buscando apenas metadata")
            meta_info = fetch_metadata(vid)
            if meta_info:
                _enrich_metadata(meta_info, anet_entry, vid)
                all_metadata[vid] = meta_info
                _save_metadata_safe(all_metadata, metadata_path)
                meta_fetches += 1
                if vid in failed_ids:
                    state["failed"] = [v for v in state["failed"] if v != vid]
                    save_state(state)
            else:
                logger.warning(f"{vid}: falha ao obter metadata")
                all_metadata[vid] = _enrich_metadata({"video_id": vid}, anet_entry, vid, status="no_metadata")
                _save_metadata_safe(all_metadata, metadata_path)
            continue

        # Caso 4: falta ambos
        logger.info(f"{vid}: baixando video + metadata")
        result = download_video(vid, output_dir, browser=browser)
        d, f = _handle_download_result(vid, result, state, failed_ids, permanently_dead_set, metadata_dir)
        video_downloads += d
        fails += f
        if result == "ok":
            meta_info = fetch_metadata(vid)
            if meta_info:
                _enrich_metadata(meta_info, anet_entry, vid)
                all_metadata[vid] = meta_info
                meta_fetches += 1
            else:
                logger.warning(f"{vid}: video baixado, mas metadata indisponivel")
                all_metadata[vid] = _enrich_metadata({"video_id": vid}, anet_entry, vid, status="downloaded_no_metadata")
            _save_metadata_safe(all_metadata, metadata_path)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Resumo: {video_downloads} videos baixados, {meta_fetches} metadados obtidos, {skips} pulados, {fails} falhas")
    logger.info(f"Tempo total: {elapsed / 60:.1f} min")
    logger.info(f"State salvo em: {STATE_PATH}")
    logger.info(f"Metadados salvos em: {metadata_path}")
    logger.info("=" * 60)


def _run_window(
    n_videos: int,
    candidates: list,
    output_dir: Path,
    metadata_path: Path,
    all_metadata: dict,
    state: dict,
    failed_ids: set,
    permanently_dead_set: set,
    browser: str,
    metadata_dir: Path,
    start_time: float,
) -> None:
    logger.info(f"Modo janela movel: alvo de {n_videos} novos downloads")

    new_downloads = 0
    meta_fetches  = 0
    processed     = 0

    for vid, anet_entry in candidates:
        if new_downloads >= n_videos:
            break

        processed += 1
        has_video = video_exists(vid, output_dir)
        has_meta  = vid in all_metadata

        if has_video and has_meta:
            continue

        if has_meta and not has_video:
            logger.info(f"[{new_downloads+1}/{n_videos}] {vid}: baixando video")
            result = download_video(vid, output_dir, browser=browser)
            d, f = _handle_download_result(vid, result, state, failed_ids, permanently_dead_set, metadata_dir)
            new_downloads += d
            if result == "ok":
                all_metadata[vid]["status"] = "downloaded"
                _save_metadata_safe(all_metadata, metadata_path)
                logger.info(f"[{new_downloads}/{n_videos}] {vid}: baixado")
            elif result == "dead":
                logger.info(f"{vid}: morto, avancando (alvo: {n_videos - new_downloads} restantes)")
            else:
                logger.info(f"{vid}: falha temporaria, avancando (alvo: {n_videos - new_downloads} restantes)")
            continue

        if has_video and not has_meta:
            logger.info(f"{vid}: buscando metadata (nao conta para o alvo)")
            meta_info = fetch_metadata(vid)
            if meta_info:
                _enrich_metadata(meta_info, anet_entry, vid)
                all_metadata[vid] = meta_info
                _save_metadata_safe(all_metadata, metadata_path)
                meta_fetches += 1
            else:
                all_metadata[vid] = _enrich_metadata({"video_id": vid}, anet_entry, vid, status="no_metadata")
                _save_metadata_safe(all_metadata, metadata_path)
            continue

        # Falta ambos
        logger.info(f"[{new_downloads+1}/{n_videos}] {vid}: baixando video + metadata")
        result = download_video(vid, output_dir, browser=browser)
        d, f = _handle_download_result(vid, result, state, failed_ids, permanently_dead_set, metadata_dir)
        new_downloads += d
        if result == "ok":
            meta_info = fetch_metadata(vid)
            if meta_info:
                _enrich_metadata(meta_info, anet_entry, vid)
                all_metadata[vid] = meta_info
                meta_fetches += 1
            else:
                all_metadata[vid] = _enrich_metadata({"video_id": vid}, anet_entry, vid, status="downloaded_no_metadata")
            _save_metadata_safe(all_metadata, metadata_path)
            logger.info(f"[{new_downloads}/{n_videos}] {vid}: completo")
        elif result == "dead":
            logger.info(f"{vid}: morto, avancando (alvo: {n_videos - new_downloads} restantes)")
        else:
            logger.info(f"{vid}: falha temporaria, avancando (alvo: {n_videos - new_downloads} restantes)")

    elapsed = time.time() - start_time
    status = "ALCANCADO" if new_downloads >= n_videos else "esgotou candidatos"
    logger.info("=" * 60)
    logger.info(f"Janela movel: {new_downloads}/{n_videos} novos downloads ({status})")
    logger.info(f"Metadados obtidos (paridade): {meta_fetches}")
    logger.info(f"Candidatos examinados: {processed}")
    logger.info(f"Tempo total: {elapsed / 60:.1f} min")
    logger.info(f"State salvo em: {STATE_PATH}")
    logger.info(f"Metadados salvos em: {metadata_path}")
    logger.info("=" * 60)


def _run_parity(
    output_dir: Path,
    metadata_path: Path,
    all_metadata: dict,
    state: dict,
    failed_ids: set,
    permanently_dead_set: set,
    browser: str,
    metadata_dir: Path,
    database: dict,
    start_time: float,
) -> None:
    """Check and fix parity between video files and metadata. Also retry failed downloads."""
    logger.info("Modo paridade: verificando consistencia entre videos e metadados")

    #  Discover all video files on disk ─
    video_ids_on_disk = set()
    if output_dir.exists():
        for f in output_dir.glob("*.mp4"):
            if video_exists(f.stem, output_dir):
                video_ids_on_disk.add(f.stem)
    logger.info(f"Videos em disco: {len(video_ids_on_disk)}")

    #  IDs from metadata 
    meta_ids = set(all_metadata.keys())
    logger.info(f"Entradas no metadata: {len(meta_ids)}")

    video_only = video_ids_on_disk - meta_ids
    meta_only  = meta_ids - video_ids_on_disk

    logger.info(f"So video (sem metadata): {len(video_only)}")
    logger.info(f"So metadata (sem video): {len(meta_only)}")
    logger.info(f"Falhas pendentes no state: {len(failed_ids)}")

    video_fixes = 0
    meta_fixes  = 0
    fails       = 0

    #  Fix: video exists, no metadata ─
    for vid in sorted(video_only):
        anet_entry = database.get(vid, {})
        logger.info(f"{vid}: video existe, buscando metadata")
        meta_info = fetch_metadata(vid)
        if meta_info:
            _enrich_metadata(meta_info, anet_entry, vid)
            all_metadata[vid] = meta_info
            meta_fixes += 1
        else:
            all_metadata[vid] = _enrich_metadata({"video_id": vid}, anet_entry, vid, status="no_metadata")
        _save_metadata_safe(all_metadata, metadata_path)

    #  Fix: metadata exists, no video ─
    for vid in sorted(meta_only):
        if vid in permanently_dead_set:
            logger.info(f"{vid}: metadata existe mas video e permanentemente morto, pulando")
            continue
        logger.info(f"{vid}: metadata existe, baixando video")
        result = download_video(vid, output_dir, browser=browser)
        d, f = _handle_download_result(vid, result, state, failed_ids, permanently_dead_set, metadata_dir)
        video_fixes += d
        fails += f
        if result == "ok":
            all_metadata[vid]["status"] = "downloaded"
            _save_metadata_safe(all_metadata, metadata_path)

    #  Retry failed downloads ─
    if failed_ids:
        logger.info(f"Tentando baixar {len(failed_ids)} videos com falha anterior...")
        for vid in sorted(failed_ids):
            if vid in permanently_dead_set or vid in video_ids_on_disk | meta_ids:
                continue
            if vid not in database:
                continue
            anet_entry = database[vid]
            logger.info(f"{vid}: retentativa de download")
            result = download_video(vid, output_dir, browser=browser)
            d, f = _handle_download_result(vid, result, state, failed_ids, permanently_dead_set, metadata_dir)
            video_fixes += d
            fails += f
            if result == "ok":
                meta_info = fetch_metadata(vid)
                if meta_info:
                    _enrich_metadata(meta_info, anet_entry, vid)
                    all_metadata[vid] = meta_info
                    meta_fixes += 1
                else:
                    all_metadata[vid] = _enrich_metadata({"video_id": vid}, anet_entry, vid, status="downloaded_no_metadata")
                _save_metadata_safe(all_metadata, metadata_path)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Paridade: {video_fixes} videos baixados, {meta_fixes} metadados obtidos, {fails} falhas")
    logger.info(f"Tempo total: {elapsed / 60:.1f} min")
    logger.info(f"State salvo em: {STATE_PATH}")
    logger.info(f"Metadados salvos em: {metadata_path}")
    logger.info("=" * 60)


def _run_backfill_transcript(
    metadata_path: Path,
    all_metadata: dict,
    metadata_dir: Path,
    start_time: float,
) -> None:
    """Backfill transcripts for all entries in videos_metadata.json."""
    logger.info("Modo transcript: preenchendo transcripts para todos os videos existentes")

    if not all_metadata:
        logger.warning("Nenhuma entrada de metadados encontrada.")
        return

    total = len(all_metadata)
    ok = 0
    empty = 0
    errors = 0

    for idx, (vid, meta) in enumerate(all_metadata.items(), 1):
        if meta.get("status") != "downloaded":
            logger.info(f"[{idx}/{total}] {vid}: pulando (status={meta.get('status')})")
            continue

        logger.info(f"[{idx}/{total}] {vid}: buscando transcript...")
        transcript = fetch_transcript(vid, output_dir=metadata_dir)

        if transcript:
            all_metadata[vid]["transcript"] = transcript
            ok += 1
            logger.info(f"[{idx}/{total}] {vid}: transcript obtido ({len(transcript)} chars)")
        else:
            all_metadata[vid]["transcript"] = ""
            empty += 1
            logger.info(f"[{idx}/{total}] {vid}: sem transcript (campo vazio)")

        if (idx % 10) == 0:
            _save_metadata_safe(all_metadata, metadata_path)

    _save_metadata_safe(all_metadata, metadata_path)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Transcript: {ok} obtidos, {empty} vazios, {errors} erros")
    logger.info(f"Tempo total: {elapsed / 60:.1f} min")
    logger.info(f"Metadados salvos em: {metadata_path}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Workflow de download de videos do ActivityNet para a colecao de referencia",
    )
    parser.add_argument(
        "-n", "--n-videos",
        type=int, default=10,
        help="Numero de videos para baixar (default: 10)",
    )
    parser.add_argument(
        "--subset",
        type=str, default="validation",
        choices=["training", "validation", "testing"],
        help="Subset do ActivityNet (default: validation)",
    )
    parser.add_argument(
        "--json-path",
        type=Path, default=ANET_JSON_PATH,
        help="Caminho para o JSON do ActivityNet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path, default=OUTPUT_DIR,
        help="Diretorio de saida dos videos",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path, default=METADATA_DIR,
        help="Diretorio de saida dos metadados",
    )
    parser.add_argument(
        "--browser",
        type=str, default="chrome", choices=["chrome", "firefox", "edge", "none"],
        help="Navegador para cookies (default: chrome)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forca redownload mesmo se ja existir",
    )
    parser.add_argument(
        "-w", "--window",
        action="store_true",
        help="Modo janela movel: avanca pelos candidatos ate atingir N novos downloads",
    )
    parser.add_argument(
        "-p", "--parity",
        action="store_true",
        help="Modo paridade: corrige inconsistencias entre video e metadados e retenta falhas",
    )
    parser.add_argument(
        "-t", "--transcript",
        action="store_true",
        help="Modo transcript: preenche campo transcript para todos os videos existentes (backfill)",
    )
    args = parser.parse_args()

    browser_val = args.browser if args.browser != "none" else ""
    run(
        n_videos=args.n_videos,
        subset=args.subset,
        json_path=args.json_path,
        output_dir=args.output_dir,
        metadata_dir=args.metadata_dir,
        browser=browser_val,
        force=args.force,
        window=args.window,
        parity=args.parity,
        transcript=args.transcript,
    )


if __name__ == "__main__":
    main()
