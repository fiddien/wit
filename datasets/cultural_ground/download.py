"""
Download and filter the CulturalGround dataset for Southeast Asian languages.

Strategy
--------
1. For each SEA country, stream the LLM-refined open-ended VQA JSONL file
   from HuggingFace and reservoir-sample records per language.
2. Download the pre-packaged per-country image tarball from HuggingFace
   (CultureGroundImages/{country}.tar.gz) and selectively extract only the
   images needed by the sampled records — no Wikidata API calls required.
3. Match sampled records to extracted images by Wikidata QID.
4. Save one metadata.parquet per language.

Checkpointing
-------------
- A language whose metadata.parquet already exists is skipped.
- Downloaded JSONL files are cached under <cache_dir>/.
- Downloaded tarballs are cached as <cache_dir>/{country}.tar.gz.
- Extracted images live under <cache_dir>/images/{country}/ and are reused
  on re-runs (only missing QIDs trigger a new tarball scan).
"""

import json
import logging
import random
import re
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd
import requests
from tqdm import tqdm

from config import DEFAULT_MAX_SAMPLES, DEFAULT_OUTPUT_DIR, DEFAULT_CACHE_DIR, REQUEST_TIMEOUT
from datasets.cultural_ground.config import (
    IMAGE_TARBALL_TEMPLATE,
    LANGUAGES,
    OE_REFINED_JSONL_TEMPLATE,
    SEA_COUNTRIES,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
    """Return (question, answer) from a CulturalGround-Refined record."""
    question = (
        record.get("reformulated_question")
        or record.get("original_question")
        or ""
    ).strip()
    answer = (
        record.get("reformulated_answer")
        or record.get("original_answer")
        or ""
    ).strip()
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
# Tarball download + selective extraction
# ---------------------------------------------------------------------------

def _download_country_tarball(
    country: str,
    session: requests.Session,
    cache_dir: Path,
) -> Path | None:
    """Download {country}.tar.gz to *cache_dir*. Returns local path or None on error.

    Uses atomic rename via a .part file so an interrupted download does not
    leave a corrupt archive on disk.
    """
    url = IMAGE_TARBALL_TEMPLATE.format(country=country)
    dest = cache_dir / f"{country}.tar.gz"
    if dest.exists():
        logger.info("Tarball already cached: %s", dest)
        return dest

    cache_dir.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".gz.part")
    part.unlink(missing_ok=True)
    headers = {"User-Agent": "seacrowd-cg-downloader/1.0 (SEACrowd dataset pipeline)"}
    try:
        with session.get(url, stream=True, timeout=REQUEST_TIMEOUT, headers=headers) as resp:
            if resp.status_code == 404:
                logger.warning("Tarball not found for '%s' (HTTP 404)", country)
                return None
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0)) or None
            with part.open("wb") as f:
                with tqdm(total=total, unit="B", unit_scale=True,
                          desc=f"cg/{country}.tar.gz", leave=False) as pbar:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        pbar.update(len(chunk))
        part.rename(dest)
        logger.info("Downloaded → %s", dest)
        return dest
    except Exception as exc:
        part.unlink(missing_ok=True)
        logger.error("Failed to download tarball for '%s': %s", country, exc)
        return None


def _extract_needed_images(
    tarball_path: Path,
    needed_qids: set[str],
    needed_filenames: set[str],
    extract_dir: Path,
) -> dict[str, Path]:
    """Selectively extract images for *needed_qids* from *tarball_path*.

    Images are extracted flat into *extract_dir* (directory components stripped).
    Matching is done by QID (``Q\\d+`` in stem) and, as a fallback, by exact
    filename.  Already-extracted files are reused without re-scanning the
    archive.

    Returns ``{qid: local_path}`` for every successfully located image.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    qid_to_path: dict[str, Path] = {}
    filename_to_qid: dict[str, str] = {f: _extract_qid(f) for f in needed_filenames
                                        if _extract_qid(f)}

    # Discover already-extracted images
    for existing in extract_dir.iterdir():
        if not existing.is_file():
            continue
        qid = _extract_qid(existing.name)
        if qid and qid in needed_qids:
            qid_to_path[qid] = existing
        elif existing.name in filename_to_qid:
            qid_to_path[filename_to_qid[existing.name]] = existing

    missing_qids = needed_qids - set(qid_to_path)
    missing_filenames = {f for f, q in filename_to_qid.items() if q in missing_qids}

    if not missing_qids:
        logger.info("All %d images already extracted for %s", len(needed_qids), tarball_path.name)
        return qid_to_path

    logger.info(
        "Extracting %d/%d missing images from %s…",
        len(missing_qids), len(needed_qids), tarball_path.name,
    )
    extracted = 0
    with tarfile.open(tarball_path, "r:gz") as tf:
        for member in tqdm(tf, desc=f"scanning {tarball_path.name}", unit="entry", leave=False):
            if not member.isfile():
                continue
            basename = Path(member.name).name
            qid = _extract_qid(basename)

            match = (qid and qid in missing_qids) or (basename in missing_filenames)
            if not match:
                continue

            # Strip directory prefix to extract flat
            member_copy = member.__class__.frombuf(member.tobuf(), tarfile.ENCODING, "surrogateescape")
            member_copy.name = basename
            dest = extract_dir / basename

            try:
                fileobj = tf.extractfile(member)
                if fileobj is None:
                    continue
                dest.write_bytes(fileobj.read())
            except Exception as exc:
                logger.warning("Failed to extract %s: %s", basename, exc)
                continue

            resolved_qid = qid or filename_to_qid.get(basename)
            if resolved_qid:
                qid_to_path[resolved_qid] = dest
                missing_qids.discard(resolved_qid)
                extracted += 1

            if not missing_qids:
                break  # all found, no need to continue scanning

    logger.info("Extracted %d/%d images from %s", extracted, len(needed_qids), tarball_path.name)
    return qid_to_path


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
    cache_dir:    Directory for JSONL and tarball caches (default: DEFAULT_CACHE_DIR).
    countries:    SEA country names to scan (default: all SEA_COUNTRIES).

    Returns
    -------
    dict mapping language code -> Path of saved metadata.parquet.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _cache_dir = cache_dir or DEFAULT_CACHE_DIR

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

    # Phase 1: Stream per-country JSONLs in parallel and reservoir-sample by language
    samplers_lock = Lock()

    def _scan_country_jsonl(country: str) -> tuple[str, list[dict]]:
        logger.info("=== Scanning JSONL: %s ===", country)
        return country, _stream_country_jsonl(country, session, cache_dir)

    max_workers = min(4, len(scan_countries))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_scan_country_jsonl, c): c for c in scan_countries}
        for future in as_completed(futures):
            country, records = future.result()
            for record in records:
                lang = record.get("language", "")
                if lang not in lang_set:
                    continue
                if not record.get("image"):
                    continue
                qid = record.get("id") or _extract_qid(record.get("image", ""))
                if not qid:
                    continue
                question, answer = _parse_conversation(record)
                if not answer:
                    continue
                with samplers_lock:
                    samplers[lang].add({
                        "language": lang,
                        "entity_id": qid,
                        "image_filename": Path(record["image"]).name,
                        "country": country,
                        "question": question,
                        "caption": answer,
                        "page_url": f"https://www.wikidata.org/wiki/{qid}",
                        "page_title": qid,
                    })
            for lang in pending:
                s = samplers[lang]
                logger.info("  [%s] seen: %d  reservoir: %d", lang, s.total_seen, len(s.samples))

    # Phase 2: Download tarballs and selectively extract needed images per country
    country_qids: dict[str, set[str]] = {c: set() for c in scan_countries}
    country_filenames: dict[str, set[str]] = {c: set() for c in scan_countries}
    for lang in pending:
        for s in samplers[lang].samples:
            country_qids[s["country"]].add(s["entity_id"])
            country_filenames[s["country"]].add(s["image_filename"])

    qid_to_local: dict[str, Path] = {}
    qid_to_local_lock = Lock()

    def _fetch_and_extract(country: str) -> None:
        needed_qids = country_qids[country]
        if not needed_qids:
            return
        tarball = _download_country_tarball(country, session, _cache_dir)
        if tarball is None:
            return
        extract_dir = _cache_dir / "images" / country
        local_map = _extract_needed_images(
            tarball, needed_qids, country_filenames[country], extract_dir
        )
        with qid_to_local_lock:
            qid_to_local.update(local_map)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_fetch_and_extract, scan_countries))

    # Phase 3: Write per-language parquets (dropping samples with no local image)
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
            local_path = qid_to_local.get(s["entity_id"])
            if local_path and local_path.exists():
                rows.append({**s, "image_path": str(local_path)})

        skipped = len(raw_samples) - len(rows)
        if skipped:
            logger.warning("[%s] %d rows skipped (image not found in tarball)", lang, skipped)
        if not rows:
            logger.warning("[%s] No matched images — skipping", lang)
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
