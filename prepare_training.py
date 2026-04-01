"""
Prepare WebDataset shards for training.

Scans all data/<source>/<split>/<lang>/shard-*.tar files and produces:
  <data-dir>/manifest.json       — shard registry with splits and language weights
  <data-dir>/train_shards.txt    — newline-separated training shard paths
  <data-dir>/val_shards.txt      — newline-separated validation shard paths
  <data-dir>/test_shards.txt     — newline-separated test shard paths

All sources use a uniform layout:
  <source>/train/<lang>/shard-*.tar
  <source>/val/<lang>/shard-*.tar
  <source>/test/<lang>/shard-*.tar

Split strategy: for official-split sources (e.g. WIT) all splits are
pre-assigned. For other sources (e.g. CulturalGround) shards under train/
are carved: the last N shards become val where
N = max(min_val_shards, ceil(n_shards * val_fraction)).
Groups with only one shard are kept entirely in train (no val).

Language sampling weights use temperature-scaled frequency so that low-resource
languages are upsampled relative to dominant ones:
  w_l ∝ N_l^(1/temperature)
  temperature=1.0  → proportional to corpus size (no correction)
  temperature→0.0  → uniform across languages

Usage:
  python prepare_training.py
  python prepare_training.py --val-fraction 0.1 --weight-temp 0.7
  python prepare_training.py --data-dir /abs/path/to/data
"""

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

from config import DEFAULT_OUTPUT_DIR, DEFAULT_SHARD_SIZE

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Known dataset source directories (in order of discovery)
KNOWN_SOURCES = ["wit", "cultural-ground"]

# Sources that provide their own official train/val/test splits.
# Shards from these sources are NOT subject to the artificial split heuristic.
OFFICIAL_SPLIT_SOURCES = {"wit"}

# Subdirectory names that hold per-split shards (same for all sources)
_SPLIT_DIRS = frozenset({"train", "val", "test"})

# Language names (union of all dataset configs)
ALL_LANGUAGE_NAMES: dict[str, str] = {
    "id": "Indonesian",
    "vi": "Vietnamese",
    "th": "Thai",
    "tl": "Tagalog",
    "ms": "Malay",
    "my": "Burmese",
    "lo": "Lao",
    "jv": "Javanese",
    "su": "Sundanese",
}


# ---------------------------------------------------------------------------
# Shard discovery
# ---------------------------------------------------------------------------

def _read_total_samples(lang_dir: Path) -> "int | None":
    """Return total_samples from stats.json in *lang_dir*, or None."""
    stats_path = lang_dir / "stats.json"
    if not stats_path.exists():
        return None
    try:
        with stats_path.open() as f:
            stats = json.load(f)
        return stats.get("webdataset", {}).get("total_samples")
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", stats_path, exc)
        return None


def discover_shards(data_dir: Path) -> list[dict]:
    """
    Scan *data_dir* for all shard-*.tar files under recognised source directories.

    All sources use a uniform three-tier layout:

      <source>/train/<lang>/shard-*.tar  → split pre-assigned for OFFICIAL_SPLIT_SOURCES;
                                           split=None for others (carved by assign_splits)
      <source>/val/<lang>/shard-*.tar    → split="val"  (always pre-assigned)
      <source>/test/<lang>/shard-*.tar   → split="test" (always pre-assigned)

    Returns a list of shard records (dicts).  Each record has:
      path            — path relative to the workspace root (str)
      source          — dataset name (str)
      language        — BCP-47 language code (str)
      split           — "train"/"val"/"test" or None (see above)
      n_samples       — estimated sample count (int or None)
      _abs_path       — resolved absolute path for I/O (str, stripped before writing)
      _lang_total     — total samples for this (source, lang) group (int or None)
      _n_shards_group — number of shards in this group (int)
    """
    shards: list[dict] = []
    workspace_root = data_dir.parent

    for source in KNOWN_SOURCES:
        source_dir = data_dir / source
        if not source_dir.is_dir():
            continue

        for split_name in ("train", "val", "test"):
            split_dir = source_dir / split_name
            if not split_dir.is_dir():
                continue

            # Official-split sources: all splits are pre-assigned as-is.
            # Other sources: val/test are pre-assigned; train shards are left as
            # split=None so that assign_splits can carve a val subset.
            if source in OFFICIAL_SPLIT_SOURCES or split_name in ("val", "test"):
                pre_split: str | None = split_name
            else:
                pre_split = None

            for lang_dir in sorted(split_dir.iterdir()):
                if not lang_dir.is_dir():
                    continue
                lang = lang_dir.name
                shard_files = sorted(lang_dir.glob("shard-*.tar"))
                if not shard_files:
                    continue
                total_samples = _read_total_samples(lang_dir)
                n = len(shard_files)
                for shard_path in shard_files:
                    shards.append({
                        "path": str(shard_path.relative_to(workspace_root)),
                        "source": source,
                        "language": lang,
                        "split": pre_split,
                        "n_samples": None,
                        "_abs_path": str(shard_path.resolve()),
                        "_lang_total": total_samples,
                        "_n_shards_group": n,
                    })

    logger.info("Discovered %d shard(s) across %d source(s)", len(shards), len(KNOWN_SOURCES))
    return shards


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def assign_splits(
    shards: list[dict],
    val_fraction: float,
    min_val_shards: int,
) -> None:
    """
    Assign 'train' or 'val' to each shard's ``split`` field in-place.

    Shards within each (source, language) group are ordered by filename;
    the last *n_val* shards become 'val', the rest become 'train'.
    Groups with only one shard are assigned entirely to 'train'.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for s in shards:
        if s["split"] is not None:
            continue  # pre-assigned (official split) — leave untouched
        key = (s["source"], s["language"])
        groups.setdefault(key, []).append(s)

    for (source, lang), group in groups.items():
        n = len(group)
        if n < 2:
            for s in group:
                s["split"] = "train"
            logger.debug("[%s/%s] 1 shard → train only", source, lang)
            continue

        n_val = max(min_val_shards, math.ceil(n * val_fraction))
        n_val = min(n_val, n - 1)  # always keep ≥1 shard for train

        for i, s in enumerate(group):
            s["split"] = "val" if i >= n - n_val else "train"

        logger.debug(
            "[%s/%s] %d shards → %d train, %d val",
            source, lang, n, n - n_val, n_val,
        )


# ---------------------------------------------------------------------------
# Sample count estimation
# ---------------------------------------------------------------------------

def estimate_shard_samples(shards: list[dict], shard_size: int) -> None:
    """
    Estimate the per-shard sample count and store it in ``n_samples``.

    For a group of *n* shards with *total* samples:
      - shards 0 … n-2 each hold ``shard_size`` samples (full)
      - shard n-1 holds the remainder

    When the group total is unknown, every shard is assumed full.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for s in shards:
        key = (s["source"], s["language"])
        groups.setdefault(key, []).append(s)

    for group in groups.values():
        total = group[0]["_lang_total"]
        n = len(group)

        if total is None:
            for s in group:
                s["n_samples"] = shard_size
            continue

        for i, s in enumerate(group):
            if n == 1:
                s["n_samples"] = total
            elif i < n - 1:
                s["n_samples"] = shard_size
            else:
                remainder = total - (n - 1) * shard_size
                s["n_samples"] = max(remainder, 1)


# ---------------------------------------------------------------------------
# Language sampling weights
# ---------------------------------------------------------------------------

def compute_weights(shards: list[dict], temperature: float) -> dict[str, float]:
    """
    Compute per-language sampling weights over the train split using
    temperature-scaled frequency.

    w_l ∝ N_l^(1/T), then normalised so sum(weights) == 1.

    Args:
        shards:      list of shard records with ``split`` and ``n_samples`` set
        temperature: controls upsampling of low-resource languages (0 < T ≤ 1)
    """
    if temperature <= 0:
        raise ValueError(f"weight_temperature must be > 0, got {temperature}")

    train_by_lang: dict[str, int] = {}
    for s in shards:
        if s["split"] == "train":
            train_by_lang[s["language"]] = (
                train_by_lang.get(s["language"], 0) + (s["n_samples"] or 0)
            )

    if not train_by_lang:
        return {}

    scaled = {lang: n ** (1.0 / temperature) for lang, n in train_by_lang.items()}
    total = sum(scaled.values())
    return {lang: round(v / total, 6) for lang, v in sorted(scaled.items())}


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _summary(shard_list: list[dict]) -> dict:
    total_samples = sum(s["n_samples"] or 0 for s in shard_list)
    by_lang: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for s in shard_list:
        by_lang[s["language"]] = by_lang.get(s["language"], 0) + (s["n_samples"] or 0)
        by_source[s["source"]] = by_source.get(s["source"], 0) + (s["n_samples"] or 0)
    return {
        "total_shards": len(shard_list),
        "total_samples": total_samples,
        "by_language": dict(sorted(by_lang.items())),
        "by_source": dict(sorted(by_source.items())),
    }


def write_manifest(
    data_dir: Path,
    shards: list[dict],
    weights: dict[str, float],
    val_fraction: float,
    weight_temperature: float,
) -> Path:
    """Write data/manifest.json and return the path."""
    clean_shards = [
        {k: v for k, v in s.items() if not k.startswith("_")}
        for s in shards
    ]
    train_shards = [s for s in shards if s["split"] == "train"]
    val_shards = [s for s in shards if s["split"] == "val"]
    test_shards = [s for s in shards if s["split"] == "test"]

    manifest = {
        "version": 1,
        "created": datetime.now(timezone.utc).isoformat(),
        "val_fraction": val_fraction,
        "weight_temperature": weight_temperature,
        "sources": sorted({s["source"] for s in shards}),
        "languages": {
            lang: ALL_LANGUAGE_NAMES.get(lang, lang)
            for lang in sorted({s["language"] for s in shards})
        },
        "sampling_weights": weights,
        "summary": {
            "train": _summary(train_shards),
            "val": _summary(val_shards),
            "test": _summary(test_shards),
        },
        "shards": clean_shards,
    }

    out = data_dir / "manifest.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("Wrote manifest → %s  (%d shards)", out, len(clean_shards))
    return out


def write_shard_lists(data_dir: Path, shards: list[dict]) -> None:
    """Write train_shards.txt, val_shards.txt, and test_shards.txt."""
    for split in ("train", "val", "test"):
        paths = [s["_abs_path"] for s in shards if s["split"] == split]
        out = data_dir / f"{split}_shards.txt"
        with out.open("w", encoding="utf-8") as f:
            f.write("\n".join(paths))
            if paths:
                f.write("\n")
        logger.info("Wrote %d %s shard path(s) → %s", len(paths), split, out)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(shards: list[dict], weights: dict[str, float]) -> None:
    """Print a human-readable training prep summary."""
    train_by_lang: dict[str, int] = {}
    val_by_lang: dict[str, int] = {}
    test_by_lang: dict[str, int] = {}
    for s in shards:
        lang = s["language"]
        n = s["n_samples"] or 0
        if s["split"] == "train":
            train_by_lang[lang] = train_by_lang.get(lang, 0) + n
        elif s["split"] == "val":
            val_by_lang[lang] = val_by_lang.get(lang, 0) + n
        elif s["split"] == "test":
            test_by_lang[lang] = test_by_lang.get(lang, 0) + n

    all_langs = sorted({s["language"] for s in shards})

    header = f"{'Lang':<6}  {'Name':<15}  {'Train':>10}  {'Val':>8}  {'Test':>8}  {'Weight':>8}"
    print()
    print("Training preparation summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for lang in all_langs:
        name = ALL_LANGUAGE_NAMES.get(lang, lang)
        tr = train_by_lang.get(lang, 0)
        vl = val_by_lang.get(lang, 0)
        ts = test_by_lang.get(lang, 0)
        wt = weights.get(lang, 0.0)
        print(f"{lang:<6}  {name:<15}  {tr:>10,}  {vl:>8,}  {ts:>8,}  {wt:>8.4f}")
    print("-" * len(header))
    print(
        f"{'TOTAL':<6}  {'':<15}  {sum(train_by_lang.values()):>10,}  "
        f"{sum(val_by_lang.values()):>8,}  {sum(test_by_lang.values()):>8,}  {'1.0000':>8}"
    )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate training manifest and shard lists from WebDataset shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root data directory containing <source>/<lang>/shard-*.tar files",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of shards per (source, language) group to reserve for validation",
    )
    parser.add_argument(
        "--min-val-shards",
        type=int,
        default=1,
        help="Minimum number of val shards per group (only applied when the group has ≥2 shards)",
    )
    parser.add_argument(
        "--weight-temp",
        type=float,
        default=0.7,
        help=(
            "Temperature for language sampling weights. "
            "1.0 = proportional to corpus size; closer to 0 = more uniform across languages."
        ),
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help="Expected samples per full shard (used to estimate per-shard counts)",
    )
    return parser.parse_args()


def prepare(
    data_dir: Path,
    val_fraction: float = 0.1,
    min_val_shards: int = 1,
    weight_temperature: float = 0.7,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> dict:
    """
    Run the full preparation pipeline and return the manifest dict.

    This function is importable for programmatic use (e.g. from main.py).
    """
    shards = discover_shards(data_dir)
    if not shards:
        logger.warning("No shards found under %s — nothing to do.", data_dir)
        return {}

    assign_splits(shards, val_fraction, min_val_shards)
    estimate_shard_samples(shards, shard_size)
    weights = compute_weights(shards, weight_temperature)

    write_manifest(data_dir, shards, weights, val_fraction, weight_temperature)
    write_shard_lists(data_dir, shards)
    print_summary(shards, weights)

    return {
        "n_shards": len(shards),
        "n_train": sum(1 for s in shards if s["split"] == "train"),
        "n_val": sum(1 for s in shards if s["split"] == "val"),
        "n_test": sum(1 for s in shards if s["split"] == "test"),
        "weights": weights,
    }


def main() -> None:
    args = parse_args()
    prepare(
        data_dir=args.data_dir,
        val_fraction=args.val_fraction,
        min_val_shards=args.min_val_shards,
        weight_temperature=args.weight_temp,
        shard_size=args.shard_size,
    )


if __name__ == "__main__":
    main()
