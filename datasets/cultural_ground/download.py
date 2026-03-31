"""
Download and filter the CulturalGround dataset for Southeast Asian languages.

Strategy
--------
1. For each SEA country, stream the LLM-refined open-ended VQA JSONL file
   from HuggingFace (`CulturalGround-Recipes/CulturalGround-Refined-OE/`).
2. Filter records whose `language` field is a SEA language code.
3. Apply reservoir sampling to keep at most *max_samples* per language.
4. Batch-resolve Wikidata QIDs (embedded in each image filename as `Q\d+`)
   to Wikimedia Commons image URLs via the Wikidata API (P18 property).
   Results are cached to disk so re-runs skip the API calls.
5. Save one metadata.parquet per language.

Checkpointing
-------------
- A language whose metadata.parquet already exists is skipped.
- The Wikidata URL cache persists across runs at <cache_dir>/.cg_url_cache.json.
"""

import json
import logging
import random
import re
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from tqdm import tqdm

from config import DEFAULT_MAX_SAMPLES, DEFAULT_OUTPUT_DIR, DEFAULT_CACHE_DIR, REQUEST_TIMEOUT
from datasets.cultural_ground.config import (
    LANGUAGES,
    OE_REFINED_JSONL_TEMPLATE,
    SEA_COUNTRIES,
    WIKIDATA_API_URL,
    WIKIDATA_BATCH_SIZE,
    WIKIDATA_REQUEST_DELAY,
    WIKIMEDIA_FILEPATH_URL,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_URL_CACHE_FILE = ".cg_url_cache.json"


# ---------------------------------------------------------------------------
# Reservoir sampling (same pattern as WIT)
# ---------------------------------------------------------------------------

class ReservoirSampler:
    """Uniform random sample of at most *k* items seen so far."""

    def __init__(self, k: int, seed: int = 42):
        self._k = k
        self._rng = random.Random(seed)
        self._reservoir: list[dict] = []
        self._n = 0

    def add(self, item: dict) -> None:
        self._n += 1
        if len(self._reservoir) < self._k:
            self._reservoir.append(item)
        else:
            j = self._rng.randint(0, self._n - 1)
            if j < self._k:
                self._reservoir[j] = item

    @property
    def samples(self) -> list[dict]:
        return list(self._reservoir)

    @property
    def total_seen(self) -> int:
        return self._n


# ---------------------------------------------------------------------------
# JSONL streaming helpers
# ---------------------------------------------------------------------------

def _extract_qid(image_path: str) -> str | None:
    """Extract the Wikidata QID from a path like 'images/indonesia/Q12345_foo.jpg'."""
    m = re.search(r"(Q\d+)", Path(image_path).name)
    return m.group(1) if m else None


def _parse_conversation(record: dict) -> tuple[str, str]:
    """Return (question, answer) from a LLaVA-format conversations list."""
    question = ""
    answer = ""
    for turn in record.get("conversations", []):
        role = turn.get("from", "")
        value = turn.get("value", "")
        if role == "human":
            question = value.replace("<image>\n", "").replace("<image>", "").strip()
        elif role == "gpt":
            answer = value.strip()
    return question, answer


def _stream_country_jsonl(
    country: str,
    session: requests.Session,
    cache_dir: Path | None,
) -> list[dict]:
    """Download (optionally caching) and parse the JSONL for one SEA country.

    Returns a list of raw record dicts.  Returns [] and logs a warning if the
    file is missing (HTTP 404) or a network error occurs.
    """
    url = OE_REFINED_JSONL_TEMPLATE.format(country=country)
    headers = {"User-Agent": "seacrowd-cg-downloader/1.0 (SEACrowd dataset pipeline)"}

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / f"cg_{country}_refined.jsonl"
        if not cached.exists():
            logger.info("Downloading %s → %s", country, cached)
            part = cached.with_suffix(".part")
            part.unlink(missing_ok=True)
            try:
                with session.get(url, stream=True, timeout=REQUEST_TIMEOUT, headers=headers) as resp:
                    if resp.status_code == 404:
                        logger.warning("No data for country '%s' (HTTP 404) — skipping", country)
                        return []
                    resp.raise_for_status()
                    total = int(resp.headers.get("content-length", 0)) or None
                    with part.open("wb") as f:
                        with tqdm(total=total, unit="B", unit_scale=True,
                                  desc=f"cg/{country}", leave=False) as pbar:
                            for chunk in resp.iter_content(chunk_size=1 << 20):
                                f.write(chunk)
                                pbar.update(len(chunk))
                part.rename(cached)
                logger.info("Cached → %s", cached)
            except Exception as exc:
                part.unlink(missing_ok=True)
                logger.error("Failed to download %s: %s", country, exc)
                return []
        source_iter = cached.open("rb")
    else:
        try:
            resp = session.get(url, stream=True, timeout=REQUEST_TIMEOUT, headers=headers)
            if resp.status_code == 404:
                logger.warning("No data for country '%s' (HTTP 404) — skipping", country)
                return []
            resp.raise_for_status()
            source_iter = resp.iter_lines()
        except Exception as exc:
            logger.error("Failed to stream %s: %s", country, exc)
            return []

    records: list[dict] = []
    try:
        for line in source_iter:
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    finally:
        if hasattr(source_iter, "close"):
            source_iter.close()

    logger.info("  Parsed %d records from %s", len(records), country)
    return records


# ---------------------------------------------------------------------------
# Wikidata image URL resolution
# ---------------------------------------------------------------------------

def _load_url_cache(cache_path: Path) -> dict[str, str]:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_url_cache(cache_path: Path, cache: dict[str, str]) -> None:
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_qids_to_urls(
    qids: list[str],
    cache_path: Path,
    session: requests.Session,
) -> dict[str, str]:
    """Batch-resolve Wikidata QIDs to Wikimedia Commons Special:FilePath URLs.

    Caches results to *cache_path* (JSON).  Only missing QIDs hit the API.
    Returns {qid: image_url} for resolved entities (P18 claim present).
    """
    cache = _load_url_cache(cache_path)
    missing = [q for q in qids if q not in cache]
    if not missing:
        logger.info("All %d QIDs resolved from cache.", len(qids))
        return {q: cache[q] for q in qids if q in cache}

    logger.info("Resolving %d/%d QIDs via Wikidata API…", len(missing), len(qids))
    headers = {"User-Agent": "seacrowd-cg-downloader/1.0 (SEACrowd; Wikidata API)"}

    n_batches = (len(missing) + WIKIDATA_BATCH_SIZE - 1) // WIKIDATA_BATCH_SIZE
    for i in tqdm(range(0, len(missing), WIKIDATA_BATCH_SIZE),
                  total=n_batches, desc="Wikidata API", unit="batch"):
        batch = missing[i : i + WIKIDATA_BATCH_SIZE]
        params = {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": "claims",
            "format": "json",
        }
        try:
            resp = session.get(
                WIKIDATA_API_URL, params=params, headers=headers, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            for qid, entity in data.get("entities", {}).items():
                p18_claims = entity.get("claims", {}).get("P18", [])
                if p18_claims:
                    try:
                        filename = p18_claims[0]["mainsnak"]["datavalue"]["value"]
                        # Wikimedia uses underscores in filenames, not spaces
                        filename = filename.replace(" ", "_")
                        # URL-encode special characters, preserving normal filename chars
                        encoded = quote(filename, safe="!()*/:@_.-~")
                        cache[qid] = WIKIMEDIA_FILEPATH_URL.format(filename=encoded)
                    except (KeyError, IndexError, TypeError):
                        pass
        except Exception as exc:
            logger.warning("Wikidata API error (batch %d): %s", i // WIKIDATA_BATCH_SIZE, exc)

        time.sleep(WIKIDATA_REQUEST_DELAY)

    _save_url_cache(cache_path, cache)
    resolved = {q: cache[q] for q in qids if q in cache}
    logger.info(
        "Resolved %d/%d QIDs to image URLs (%.1f%%).",
        len(resolved), len(qids), 100 * len(resolved) / max(len(qids), 1),
    )
    return resolved


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def download_all(
    output_dir: Path,
    languages: list[str],
    max_samples: int,
    seed: int = 42,
    cache_dir: Path | None = None,
    countries: list[str] | None = None,
) -> dict[str, Path]:
    """Download and filter CulturalGround VQA data for the given SEA language codes.

    Parameters
    ----------
    output_dir:   Root directory; per-language metadata.parquet files go here.
    languages:    Language codes to collect (subset of LANGUAGES.keys()).
    max_samples:  Maximum samples per language (reservoir-sampled).
    seed:         RNG seed for reservoir sampling.
    cache_dir:    If set, JSONL files are cached here so re-runs skip re-downloading.
    countries:    SEA country names to scan (default: all SEA_COUNTRIES).

    Returns
    -------
    dict mapping language code -> Path of saved metadata.parquet.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Skip languages that are already done ---
    pending: list[str] = []
    results: dict[str, Path] = {}
    for lang in languages:
        out_path = output_dir / lang / "metadata.parquet"
        if out_path.exists():
            logger.info("Skipping %s — metadata.parquet already exists", lang)
            results[lang] = out_path
        else:
            pending.append(lang)

    if not pending:
        return results

    lang_set = set(pending)
    samplers: dict[str, ReservoirSampler] = {
        lang: ReservoirSampler(max_samples, seed) for lang in pending
    }

    scan_countries = countries if countries is not None else SEA_COUNTRIES
    session = requests.Session()

    # Phase 1: Stream per-country JSONLs and reservoir-sample by language
    for country in scan_countries:
        logger.info("=== Scanning country: %s ===", country)
        records = _stream_country_jsonl(country, session, cache_dir)

        for record in records:
            lang = record.get("language", "")
            if lang not in lang_set:
                continue
            qid = _extract_qid(record.get("image", ""))
            if not qid:
                continue
            question, answer = _parse_conversation(record)
            if not answer:
                continue
            samplers[lang].add({
                "language": lang,
                "entity_id": qid,
                "image_path": record.get("image", ""),
                "country": country,
                "question": question,
                "caption": answer,
                "page_url": f"https://www.wikidata.org/wiki/{qid}",
                "page_title": qid,
            })

        for lang in pending:
            s = samplers[lang]
            logger.info("  [%s] seen: %d  reservoir: %d", lang, s.total_seen, len(s.samples))

    # Phase 2: Batch-resolve Wikidata QIDs → Wikimedia Commons URLs
    all_qids = list({
        s["entity_id"]
        for lang in pending
        for s in samplers[lang].samples
    })
    url_cache_path = (cache_dir or DEFAULT_CACHE_DIR) / _URL_CACHE_FILE
    url_map = _resolve_qids_to_urls(all_qids, url_cache_path, session)

    # Phase 3: Write per-language parquets (dropping samples with no image URL)
    for lang in pending:
        sampler = samplers[lang]
        raw_samples = sampler.samples
        logger.info(
            "[%s] %s — total seen: %d  sampled: %d",
            lang, LANGUAGES.get(lang, lang), sampler.total_seen, len(raw_samples),
        )
        if not raw_samples:
            logger.warning("[%s] No rows found — skipping", lang)
            continue

        rows = []
        for s in raw_samples:
            url = url_map.get(s["entity_id"])
            if url:
                rows.append({**s, "image_url": url})

        skipped = len(raw_samples) - len(rows)
        if skipped:
            logger.warning("[%s] %d rows skipped (no Wikimedia image URL)", lang, skipped)
        if not rows:
            logger.warning("[%s] No image URLs resolved — skipping", lang)
            continue

        logger.info("[%s] Saving %d rows", lang, len(rows))
        out_path = output_dir / lang / "metadata.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        logger.info("Saved → %s", out_path)
        results[lang] = out_path

    return results


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Download CulturalGround VQA data for SEA languages."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--languages", nargs="+", default=list(LANGUAGES.keys()), metavar="LANG"
    )
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
        help="Cache directory for downloaded JSONL files (empty string = no cache).",
    )
    parser.add_argument(
        "--countries", nargs="+", default=None, choices=SEA_COUNTRIES, metavar="COUNTRY"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cache_dir = args.cache_dir if str(args.cache_dir) else None
    results = download_all(
        output_dir=args.output_dir,
        languages=args.languages,
        max_samples=args.max_samples,
        seed=args.seed,
        cache_dir=cache_dir,
        countries=args.countries,
    )
    logger.info(
        "Done. Saved metadata for %d language(s): %s",
        len(results),
        ", ".join(results.keys()),
    )
