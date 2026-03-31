"""
Download and filter the WIT dataset for Southeast Asian languages.

Streams the 10 gzipped TSV files directly from Google Cloud Storage
(full 37M-row training set, ~25 GB total) and saves one parquet file
per language under <output_dir>/<lang>/metadata.parquet.

Checkpointing:
  - A language whose metadata.parquet already exists is skipped.
  - Per-shard progress is tracked in <output_dir>/.download_progress.json so
    an interrupted run resumes from the next unfinished GCS shard.
  - Reservoir sampling keeps memory bounded while guaranteeing a uniform
    random sample when the dataset exceeds max_samples.
"""

import argparse
import gzip
import json
import logging
import random
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from config import (
    DEFAULT_MAX_SAMPLES,
    DEFAULT_OUTPUT_DIR,
    REQUEST_TIMEOUT,
    SEA_LANGUAGES,
    WIT_TRAIN_FILES,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_CAPTION_COL = "caption_reference_description"
_ALT_TEXT_COL = "caption_alt_text_description"
_LANG_COL = "language"
_IMG_COL = "image_url"
_TITLE_COL = "page_title"
_PAGE_COL = "page_url"

_PROGRESS_FILE = ".download_progress.json"


# ---------------------------------------------------------------------------
# Reservoir sampling
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
# Progress helpers
# ---------------------------------------------------------------------------

def _load_progress(output_dir: Path) -> dict:
    p = output_dir / _PROGRESS_FILE
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _save_progress(output_dir: Path, progress: dict) -> None:
    p = output_dir / _PROGRESS_FILE
    p.write_text(json.dumps(progress, indent=2))


# ---------------------------------------------------------------------------
# Core streaming scan
# ---------------------------------------------------------------------------

def _stream_tsv_gz(url: str):
    """Yield dicts for each data row in a gzipped WIT TSV file."""
    headers = {"User-Agent": "wit-sea-downloader/1.0"}
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT, headers=headers) as resp:
        resp.raise_for_status()
        gz_stream = gzip.GzipFile(fileobj=resp.raw)
        header_line = gz_stream.readline().decode("utf-8").rstrip("\n")
        col_names = header_line.split("\t")

        for raw_line in gz_stream:
            line = raw_line.decode("utf-8").rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != len(col_names):
                continue
            yield dict(zip(col_names, parts))


def _best_caption(row: dict) -> str:
    caption = (row.get(_CAPTION_COL) or "").strip()
    if not caption:
        caption = (row.get(_ALT_TEXT_COL) or "").strip()
    return caption


def download_all(
    output_dir: Path,
    languages: list[str],
    max_samples: int,
    seed: int = 42,
) -> dict[str, Path]:
    """
    Scan all 10 GCS shards once, collecting rows for each requested language
    with reservoir sampling. Returns a mapping of language code -> parquet path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = _load_progress(output_dir)

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

    completed_shards: list[int] = progress.get("completed_shards", [])

    for shard_idx, url in enumerate(WIT_TRAIN_FILES):
        if shard_idx in completed_shards:
            logger.info("Shard %d/%d already done — skipping", shard_idx, len(WIT_TRAIN_FILES))
            continue

        logger.info("Scanning shard %d/%d: %s", shard_idx + 1, len(WIT_TRAIN_FILES), url)
        try:
            pbar = tqdm(desc=f"Shard {shard_idx:02d}", unit="rows", leave=False)
            for row in _stream_tsv_gz(url):
                pbar.update(1)
                lang = row.get(_LANG_COL, "")
                if lang not in lang_set:
                    continue
                caption = _best_caption(row)
                if not caption or not row.get(_IMG_COL):
                    continue
                samplers[lang].add({
                    "language": lang,
                    "image_url": row[_IMG_COL],
                    "caption": caption,
                    "page_title": row.get(_TITLE_COL, ""),
                    "page_url": row.get(_PAGE_COL, ""),
                })
            pbar.close()
        except Exception:
            logger.exception("Error on shard %d — will retry on next run", shard_idx)
            _save_progress(output_dir, {"completed_shards": completed_shards})
            continue

        completed_shards.append(shard_idx)
        _save_progress(output_dir, {"completed_shards": completed_shards})
        for lang, sampler in samplers.items():
            logger.info(
                "  [%s] rows seen: %d  reservoir: %d",
                lang, sampler.total_seen, len(sampler.samples),
            )

    # Save parquet files
    for lang in pending:
        sampler = samplers[lang]
        samples = sampler.samples
        logger.info(
            "[%s] %s — total seen: %d  saving %d rows",
            lang, SEA_LANGUAGES.get(lang, lang), sampler.total_seen, len(samples),
        )
        if not samples:
            logger.warning("[%s] No rows found — skipping", lang)
            continue
        out_path = output_dir / lang / "metadata.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(samples).to_parquet(out_path, index=False)
        logger.info("Saved -> %s", out_path)
        results[lang] = out_path

    progress_file = output_dir / _PROGRESS_FILE
    if progress_file.exists():
        progress_file.unlink()

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and filter WIT (full GCS dataset) for SEA languages."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for output files (default: %(default)s)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(SEA_LANGUAGES.keys()),
        choices=list(SEA_LANGUAGES.keys()),
        metavar="LANG",
        help="Language codes to include (default: all SEA languages)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Maximum samples per language (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reservoir sampling (default: %(default)s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = download_all(
        output_dir=args.output_dir,
        languages=args.languages,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    logger.info(
        "Done. Saved metadata for %d language(s): %s",
        len(results),
        ", ".join(results.keys()),
    )


if __name__ == "__main__":
    args = parse_args()
    results = download_all(
        output_dir=args.output_dir,
        languages=args.languages,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    logger.info(
        "Done. Saved metadata for %d language(s): %s",
        len(results),
        ", ".join(results.keys()),
    )
