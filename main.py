"""
Main orchestration script for the WIT Southeast Asian dataset pipeline.

Steps:
  1. download  — fetch WIT from Hugging Face and save per-language parquet files
  2. convert   — download images and write WebDataset tar shards
  3. stats     — compute and display dataset statistics

Each step is idempotent; re-running skips already-completed work.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SHARD_SIZE,
    DEFAULT_WORKERS,
    SEA_LANGUAGES,
    WIT_QUICKSTART_FILE,
    WIT_TRAIN_FILES,
)
from compute_stats import compute_all_stats, display_stats
from convert_to_webdataset import convert_all
from download_wit import download_all

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

STEPS = ("download", "convert", "stats")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end pipeline: download WIT for Southeast Asian languages, "
            "convert to OpenCLIP WebDataset format, and display statistics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=list(STEPS),
        choices=list(STEPS),
        metavar="STEP",
        help="Pipeline steps to run: download, convert, stats",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(SEA_LANGUAGES.keys()),
        choices=list(SEA_LANGUAGES.keys()),
        metavar="LANG",
        help="Language codes to include",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Maximum samples per language for the download step",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help="Samples per WebDataset tar shard",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel image download workers for the convert step",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--quick-start",
        action="store_true",
        help=(
            "Use the 1%% sample TSV (~370K rows, ~250 MB) instead of the full "
            "10-shard dataset. Useful for testing the pipeline end-to-end."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory to cache downloaded TSV.gz files (default: %(default)s). "
             "Pass an empty string to stream directly without caching.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", args.output_dir.resolve())
    logger.info("Languages: %s", ", ".join(args.languages))
    logger.info("Steps: %s", " -> ".join(args.steps))

    if "download" in args.steps:
        logger.info("=== Step 1/3: Download WIT metadata ===")
        if args.quick_start:
            logger.info("Quick-start mode: using 1%% sample file (%s)", WIT_QUICKSTART_FILE)
            source_files = [WIT_QUICKSTART_FILE]
        else:
            source_files = WIT_TRAIN_FILES
        cache_dir = args.cache_dir if str(args.cache_dir) else None
        results = download_all(
            output_dir=args.output_dir,
            languages=args.languages,
            max_samples=args.max_samples,
            seed=args.seed,
            source_files=source_files,
            cache_dir=cache_dir,
        )
        logger.info(
            "Download complete: %d language(s) saved.",
            len(results),
        )

    if "convert" in args.steps:
        logger.info("=== Step 2/3: Convert to WebDataset ===")
        summaries = asyncio.run(
            convert_all(
                input_dir=args.output_dir,
                output_dir=args.output_dir,
                languages=args.languages,
                shard_size=args.shard_size,
                workers=args.workers,
            )
        )
        logger.info("Conversion complete.")
        for s in summaries:
            logger.info(
                "  [%s] %s — downloaded=%d  failed=%d  shards=%d",
                s["language"],
                s["language_name"],
                s.get("downloaded", 0),
                s.get("failed", 0),
                s.get("shards", 0),
            )

    if "stats" in args.steps:
        logger.info("=== Step 3/3: Compute statistics ===")
        all_stats = compute_all_stats(args.output_dir, args.languages)
        display_stats(all_stats)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(1)
