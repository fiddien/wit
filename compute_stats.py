"""
Compute and display statistics for the WIT Southeast Asian languages dataset.

Reads WebDataset tar shards and the metadata parquet files under <data_dir>.
Outputs:
  - Per-language stats.json
  - Cross-language summary_stats.json
  - Human-readable table printed to stdout
"""

import argparse
import json
import logging
import tarfile
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from config import DEFAULT_OUTPUT_DIR

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _caption_stats(captions: list[str]) -> dict:
    if not captions:
        return {"count": 0, "mean_chars": 0, "median_chars": 0, "min_chars": 0, "max_chars": 0,
                "mean_words": 0, "median_words": 0}
    lengths = [len(c) for c in captions]
    word_counts = [len(c.split()) for c in captions]
    s = pd.Series(lengths)
    w = pd.Series(word_counts)
    return {
        "count": len(captions),
        "mean_chars": round(s.mean(), 1),
        "median_chars": round(s.median(), 1),
        "min_chars": int(s.min()),
        "max_chars": int(s.max()),
        "mean_words": round(w.mean(), 1),
        "median_words": round(w.median(), 1),
    }


def stats_from_parquet(lang_dir: Path) -> dict | None:
    """Compute stats from the metadata parquet (pre-download)."""
    parquet = lang_dir / "metadata.parquet"
    if not parquet.exists():
        return None
    df = pd.read_parquet(parquet)
    captions = df["caption"].dropna().tolist()
    image_col = "image_url" if "image_url" in df.columns else "image_path"
    return {
        "source": "parquet",
        "total_rows": len(df),
        "unique_images": int(df[image_col].nunique()) if image_col in df.columns else None,
        "caption_stats": _caption_stats(captions),
    }


def stats_from_shards(lang_dir: Path) -> dict | None:
    """Compute stats by scanning WebDataset tar shards."""
    shards = sorted(lang_dir.glob("shard-*.tar"))
    if not shards:
        return None

    total_samples = 0
    captions: list[str] = []
    failed_reads = 0

    for shard_path in shards:
        try:
            with tarfile.open(shard_path, "r") as tf:
                members = tf.getmembers()
                txt_members = [m for m in members if m.name.endswith(".txt")]
                for m in txt_members:
                    f = tf.extractfile(m)
                    if f:
                        captions.append(f.read().decode("utf-8", errors="replace").strip())
                        total_samples += 1
        except Exception as exc:
            logger.warning("Could not read shard %s: %s", shard_path, exc)
            failed_reads += 1

    error_log = lang_dir / "errors.log"
    failed_downloads = 0
    if error_log.exists():
        failed_downloads = sum(1 for _ in error_log.open())

    return {
        "source": "shards",
        "total_shards": len(shards),
        "total_samples": total_samples,
        "failed_downloads": failed_downloads,
        "failed_shard_reads": failed_reads,
        "caption_stats": _caption_stats(captions),
    }


def compute_language_stats(
    lang: str,
    data_dir: Path,
    language_names: dict[str, str] | None = None,
) -> dict:
    lang_dir = data_dir / lang
    names = language_names or {}
    stats: dict = {
        "language": lang,
        "language_name": names.get(lang, lang),
    }

    parquet_stats = stats_from_parquet(lang_dir)
    if parquet_stats:
        stats["metadata"] = parquet_stats

    shard_stats = stats_from_shards(lang_dir)
    if shard_stats:
        stats["webdataset"] = shard_stats

    return stats


def _build_summary_table(all_stats: list[dict]) -> list[list]:
    rows = []
    for s in all_stats:
        lang = s["language"]
        lang_name = s["language_name"]
        meta = s.get("metadata", {})
        wds = s.get("webdataset", {})

        total_rows = meta.get("total_rows", "—")
        total_samples = wds.get("total_samples", "—")
        failed = wds.get("failed_downloads", "—")
        shards = wds.get("total_shards", "—")

        cap_stats = wds.get("caption_stats") or meta.get("caption_stats") or {}
        mean_words = cap_stats.get("mean_words", "—")
        mean_chars = cap_stats.get("mean_chars", "—")

        rows.append([lang, lang_name, total_rows, total_samples, failed, shards, mean_words, mean_chars])
    return rows


def display_stats(all_stats: list[dict]) -> None:
    """Print a formatted statistics table to stdout."""
    table = _build_summary_table(all_stats)
    headers = [
        "Code", "Language", "Metadata\nRows", "Downloaded\nSamples",
        "Failed\nDownloads", "Shards", "Avg Caption\nWords", "Avg Caption\nChars",
    ]
    print("\n" + "=" * 80)
    print("  Dataset Statistics")
    print("=" * 80)
    print(tabulate(table, headers=headers, tablefmt="rounded_outline", numalign="right"))

    # Per-language caption detail
    print("\n--- Caption Length Detail (characters) ---")
    cap_rows = []
    for s in all_stats:
        wds = s.get("webdataset", {})
        meta = s.get("metadata", {})
        cs = wds.get("caption_stats") or meta.get("caption_stats") or {}
        if cs.get("count", 0):
            cap_rows.append([
                s["language"],
                s["language_name"],
                cs.get("min_chars", "—"),
                cs.get("median_chars", "—"),
                cs.get("mean_chars", "—"),
                cs.get("max_chars", "—"),
            ])
    if cap_rows:
        print(tabulate(
            cap_rows,
            headers=["Code", "Language", "Min", "Median", "Mean", "Max"],
            tablefmt="rounded_outline",
            numalign="right",
        ))
    print()


def compute_all_stats(
    data_dir: Path,
    languages: list[str],
    language_names: dict[str, str] | None = None,
) -> list[dict]:
    all_stats = []
    for lang in languages:
        logger.info("Computing stats for %s…", lang)
        stats = compute_language_stats(lang, data_dir, language_names)
        all_stats.append(stats)

        # Save per-language stats
        lang_stats_path = data_dir / lang / "stats.json"
        lang_stats_path.parent.mkdir(parents=True, exist_ok=True)
        with lang_stats_path.open("w") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    # Save cross-language summary
    summary_path = data_dir / "summary_stats.json"
    with summary_path.open("w") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    logger.info("Saved summary stats -> %s", summary_path)

    return all_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and display dataset statistics."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root data directory (default: %(default)s)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        metavar="LANG",
        help="Language codes to compute stats for (default: all found in data-dir)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    languages = args.languages or [
        p.parent.name for p in args.data_dir.glob("*/metadata.parquet")
    ]
    all_stats = compute_all_stats(args.data_dir, languages)
    display_stats(all_stats)
