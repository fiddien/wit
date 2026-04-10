"""
Compute and display statistics for all datasets under <data_dir>.

Expected layout:
  <data_dir>/<source>/<split>/<lang>/shard-*.tar
  <data_dir>/<source>/train/<lang>/metadata.parquet   (optional)
  <data_dir>/<source>/<split>/<lang>/errors.log        (optional)

Outputs:
  - <data_dir>/<source>/<lang>/stats.json   per-(source, language) stats
  - <data_dir>/summary_stats.json           cross-language summary
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

SPLITS = ("train", "val", "test")


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
    """Compute stats from the metadata parquet (pre-download). Parquet lives in the train split dir."""
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


def stats_from_shards(dirs_by_split: dict[str, Path]) -> dict | None:
    """Compute stats by scanning shard dirs across all splits."""
    splits_data: dict[str, dict] = {}
    all_captions: list[str] = []

    for split in SPLITS:
        lang_dir = dirs_by_split.get(split)
        if lang_dir is None:
            continue
        shards = sorted(lang_dir.glob("shard-*.tar"))
        if not shards:
            continue

        captions: list[str] = []
        failed_reads = 0
        for shard_path in shards:
            try:
                with tarfile.open(shard_path, "r") as tf:
                    for m in tf.getmembers():
                        if m.name.endswith(".txt"):
                            f = tf.extractfile(m)
                            if f:
                                captions.append(f.read().decode("utf-8", errors="replace").strip())
            except Exception as exc:
                logger.warning("Could not read shard %s: %s", shard_path, exc)
                failed_reads += 1

        error_log = lang_dir / "errors.log"
        failed_downloads = sum(1 for _ in error_log.open()) if error_log.exists() else 0

        splits_data[split] = {
            "shards": len(shards),
            "samples": len(captions),
            "failed_downloads": failed_downloads,
            "failed_shard_reads": failed_reads,
        }
        all_captions.extend(captions)

    if not splits_data:
        return None

    total_shards = sum(s["shards"] for s in splits_data.values())
    total_samples = sum(s["samples"] for s in splits_data.values())
    return {
        "source": "shards",
        "total_shards": total_shards,
        "total_samples": total_samples,
        "splits": splits_data,
        "caption_stats": _caption_stats(all_captions),
    }


def compute_language_stats(
    lang: str,
    source_dir: Path,
    language_names: dict[str, str] | None = None,
) -> dict:
    """Compute stats for a (source, language) pair, aggregating across splits."""
    names = language_names or {}
    stats: dict = {
        "language": lang,
        "language_name": names.get(lang, lang),
        "source": source_dir.name,
    }

    # Parquet lives in the train split directory
    train_lang_dir = source_dir / "train" / lang
    parquet_stats = stats_from_parquet(train_lang_dir)
    if parquet_stats:
        stats["metadata"] = parquet_stats

    dirs_by_split = {
        split_dir.name: split_dir / lang
        for split_dir in source_dir.iterdir()
        if split_dir.is_dir() and split_dir.name in SPLITS and (split_dir / lang).is_dir()
    }
    shard_stats = stats_from_shards(dirs_by_split)
    if shard_stats:
        stats["webdataset"] = shard_stats

    return stats


def discover_source_lang_pairs(data_dir: Path) -> list[tuple[Path, str]]:
    """Return sorted (source_dir, lang) pairs that have at least one shard."""
    pairs: set[tuple[Path, str]] = set()
    for shard in data_dir.glob("*/*/*/shard-*.tar"):
        parts = shard.relative_to(data_dir).parts
        # expected: source / split / lang / shard-*.tar
        if len(parts) == 4 and parts[1] in SPLITS:
            pairs.add((data_dir / parts[0], parts[2]))
    return sorted(pairs, key=lambda t: (t[0].name, t[1]))


def _build_summary_table(all_stats: list[dict]) -> list[list]:
    rows = []
    for s in all_stats:
        lang = s["language"]
        lang_name = s["language_name"]
        source = s.get("source", "?")
        meta = s.get("metadata", {})
        wds = s.get("webdataset", {})

        total_rows = meta.get("total_rows", "—")
        total_samples = wds.get("total_samples", "—")
        shards = wds.get("total_shards", "—")

        # Per-split breakdown
        splits = wds.get("splits", {})
        train_s = splits.get("train", {}).get("samples", "—")
        val_s = splits.get("val", {}).get("samples", "—")
        test_s = splits.get("test", {}).get("samples", "—")
        failed = sum(sp.get("failed_downloads", 0) for sp in splits.values()) or "—"

        cap_stats = wds.get("caption_stats") or meta.get("caption_stats") or {}
        mean_words = cap_stats.get("mean_words", "—")
        mean_chars = cap_stats.get("mean_chars", "—")

        rows.append([source, lang, lang_name,
                    #  total_rows, total_samples,
                     train_s, val_s, test_s,
                    #  failed,
                     shards, mean_words, mean_chars])
    return rows


def _int_or_zero(v) -> int:
    return v if isinstance(v, int) else 0


def _build_by_language_table(all_stats: list[dict]) -> list[list]:
    """Aggregate stats across all sources, grouped by language code."""
    from collections import defaultdict

    agg: dict[str, dict] = defaultdict(lambda: {
        "language_name": "",
        "train": 0, "val": 0, "test": 0, "total": 0, "shards": 0,
    })
    for s in all_stats:
        lang = s["language"]
        agg[lang]["language_name"] = s["language_name"]
        wds = s.get("webdataset", {})
        splits = wds.get("splits", {})
        agg[lang]["train"] += _int_or_zero(splits.get("train", {}).get("samples", 0))
        agg[lang]["val"] += _int_or_zero(splits.get("val", {}).get("samples", 0))
        agg[lang]["test"] += _int_or_zero(splits.get("test", {}).get("samples", 0))
        agg[lang]["total"] += _int_or_zero(wds.get("total_samples", 0))
        agg[lang]["shards"] += _int_or_zero(wds.get("total_shards", 0))

    rows = []
    for lang in sorted(agg):
        a = agg[lang]
        rows.append([lang, a["language_name"], a["train"], a["val"], a["test"], a["total"], a["shards"]])
    return rows


def _build_total_table(all_stats: list[dict]) -> list[list]:
    """Single-row grand totals and per-split source-level breakdown."""
    from collections import defaultdict

    # per-source totals
    src_agg: dict[str, dict] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0, "total": 0, "shards": 0})
    grand = {"train": 0, "val": 0, "test": 0, "total": 0, "shards": 0}

    for s in all_stats:
        source = s.get("source", "?")
        wds = s.get("webdataset", {})
        splits = wds.get("splits", {})
        for split in SPLITS:
            n = _int_or_zero(splits.get(split, {}).get("samples", 0))
            src_agg[source][split] += n
            grand[split] += n
        total = _int_or_zero(wds.get("total_samples", 0))
        shards = _int_or_zero(wds.get("total_shards", 0))
        src_agg[source]["total"] += total
        src_agg[source]["shards"] += shards
        grand["total"] += total
        grand["shards"] += shards

    rows = []
    for src in sorted(src_agg):
        a = src_agg[src]
        rows.append([src, a["train"], a["val"], a["test"], a["total"], a["shards"]])
    rows.append(["TOTAL", grand["train"], grand["val"], grand["test"], grand["total"], grand["shards"]])
    return rows


def display_stats(all_stats: list[dict]) -> None:
    """Print a formatted statistics table to stdout."""
    table = _build_summary_table(all_stats)
    headers = [
        "Source", "Code", "Language",
        # "Metadata\nRows", "Total\nSamples",
        "Train", "Val", "Test",
        # "Failed\nDownloads",
        "Shards",
        "Avg Caption\nWords", "Avg Caption\nChars",
    ]
    print("\n" + "=" * 80)
    print("  Dataset Statistics")
    print("=" * 80)
    print(tabulate(table, headers=headers, tablefmt="rounded_outline", numalign="right"))

    # By-language table
    by_lang = _build_by_language_table(all_stats)
    if by_lang:
        print("\n--- By Language (all sources combined) ---")
        print(tabulate(
            by_lang,
            headers=["Code", "Language", "Train", "Val", "Test", "Total", "Shards"],
            tablefmt="rounded_outline",
            numalign="right",
        ))

    # Totals table
    totals = _build_total_table(all_stats)
    if totals:
        print("\n--- Totals by Source ---")
        print(tabulate(
            totals,
            headers=["Source", "Train", "Val", "Test", "Total", "Shards"],
            tablefmt="rounded_outline",
            numalign="right",
        ))

    # Per-language caption detail
    print("\n--- Caption Length Detail (characters) ---")
    cap_rows = []
    for s in all_stats:
        wds = s.get("webdataset", {})
        meta = s.get("metadata", {})
        cs = wds.get("caption_stats") or meta.get("caption_stats") or {}
        if cs.get("count", 0):
            cap_rows.append([
                s.get("source", "?"),
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
            headers=["Source", "Code", "Language", "Min", "Median", "Mean", "Max"],
            tablefmt="rounded_outline",
            numalign="right",
        ))
    print()


def compute_all_stats(
    data_dir: Path,
    languages: list[str] | None = None,
    language_names: dict[str, str] | None = None,
) -> list[dict]:
    pairs = discover_source_lang_pairs(data_dir)
    if languages:
        pairs = [(src, lang) for src, lang in pairs if lang in languages]

    if not pairs:
        logger.warning("No (source, lang) pairs with shards found under %s", data_dir)

    all_stats = []
    for source_dir, lang in pairs:
        logger.info("Computing stats for %s/%s…", source_dir.name, lang)
        stats = compute_language_stats(lang, source_dir, language_names)
        all_stats.append(stats)

        # Save per-(source, lang) stats
        lang_stats_path = source_dir / lang / "stats.json"
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
    all_stats = compute_all_stats(args.data_dir, languages=args.languages)
    display_stats(all_stats)
