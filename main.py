"""
Main orchestration script for the SEACrowd image-text dataset pipeline.

Steps:
  1. download  — fetch dataset from its source and save per-language parquet files
  2. convert   — download images and write WebDataset tar shards
  3. stats     — compute and display dataset statistics

Each step is idempotent; re-running skips already-completed work.

Supported datasets (--dataset):
  wit               Wikipedia-based Image Text (WIT) — SEA languages via GCS
  cultural-ground   CulturalGround VQA — SEA languages via HuggingFace
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_IMAGE_CACHE_DIR,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SHARD_SIZE,
    DEFAULT_WORKERS,
)
from compute_stats import compute_all_stats, display_stats
from convert_to_webdataset import convert_all

# Registry of supported datasets.  Each entry must expose:
#   LANGUAGES     dict[str, str]  language code -> human name
#   download_all  callable        downloads metadata; see each datasets/<name>/download.py
import datasets.wit as _wit
import datasets.cultural_ground as _cg

DATASET_REGISTRY = {
    "wit": _wit,
    "cultural-ground": _cg,
}

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
            "End-to-end pipeline: download a dataset for Southeast Asian languages, "
            "convert to OpenCLIP WebDataset format, and display statistics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="wit",
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset to process",
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
        default=None,
        metavar="LANG",
        help="Language codes to include (default: all languages for the chosen dataset)",
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
    parser.add_argument(
        "--image-cache-dir",
        type=Path,
        default=DEFAULT_IMAGE_CACHE_DIR,
        help="Directory to cache downloaded images to avoid re-fetching on re-runs "
             "(default: %(default)s). Pass an empty string to disable image caching.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = DATASET_REGISTRY[args.dataset]
    language_names: dict[str, str] = dataset.LANGUAGES
    languages: list[str] = args.languages or list(language_names.keys())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Dataset: %s", args.dataset)
    logger.info("Output directory: %s", args.output_dir.resolve())
    logger.info("Languages: %s", ", ".join(languages))
    logger.info("Steps: %s", " -> ".join(args.steps))

    if "download" in args.steps:
        logger.info("=== Step 1/3: Download %s metadata ===", args.dataset.upper())

        # WIT supports an optional quick-start sample and caching; pass kwargs
        # that the download_all function accepts.
        download_kwargs: dict = dict(
            output_dir=args.output_dir,
            languages=languages,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        cache_dir = args.cache_dir if str(args.cache_dir) else None
        if args.dataset == "wit":
            if args.quick_start:
                logger.info("Quick-start mode: using 1%% sample file")
                download_kwargs["source_files"] = [dataset.WIT_QUICKSTART_FILE]
            else:
                download_kwargs["source_files"] = dataset.WIT_TRAIN_FILES
            download_kwargs["cache_dir"] = cache_dir
        elif args.dataset == "cultural-ground":
            download_kwargs["cache_dir"] = cache_dir

        results = dataset.download_all(**download_kwargs)
        logger.info(
            "Download complete: %d language(s) saved.",
            len(results),
        )

    if "convert" in args.steps:
        logger.info("=== Step 2/3: Convert to WebDataset ===")
        asyncio.run(
            convert_all(
                input_dir=args.output_dir,
                output_dir=args.output_dir,
                languages=languages,
                shard_size=args.shard_size,
                workers=args.workers,
                language_names=language_names,
                image_cache_dir=args.image_cache_dir or None,
            )
        )
        logger.info("Conversion complete.")

    if "stats" in args.steps:
        logger.info("=== Step 3/3: Compute statistics ===")
        all_stats = compute_all_stats(args.output_dir, languages, language_names)
        display_stats(all_stats)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(1)
