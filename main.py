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
from convert_to_webdataset import convert_all, convert_all_img2dataset
from prepare_training import prepare as prepare_training

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

STEPS = ("download", "convert", "stats", "prepare")


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
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of shards per (source, language) group to reserve for validation (prepare step)",
    )
    parser.add_argument(
        "--weight-temp",
        type=float,
        default=0.7,
        help="Temperature for language sampling weights in the prepare step (1.0 = proportional, <1 = upsample low-resource)",
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
    parser.add_argument(
        "--use-img2dataset",
        action="store_true",
        help="Use img2dataset for the convert step instead of the built-in aiohttp downloader. "
             "Faster for large datasets; requires `pip install img2dataset`.",
    )
    parser.add_argument(
        "--img2dataset-threads",
        type=int,
        default=64,
        help="Number of download threads per process when --use-img2dataset is set (default: %(default)s)",
    )
    parser.add_argument(
        "--img2dataset-image-size",
        type=int,
        default=512,
        help="Resize images to this size (longest edge) when --use-img2dataset is set (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = DATASET_REGISTRY[args.dataset]
    language_names: dict[str, str] = dataset.LANGUAGES
    languages: list[str] = args.languages or list(language_names.keys())

    # Namespace output by dataset when the user hasn't overridden the default,
    # so that `--dataset wit` and `--dataset cultural-ground` don't collide.
    if args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = DEFAULT_OUTPUT_DIR / args.dataset

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
            output_dir=args.output_dir / "train",
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

        # For WIT (full run only), also download official val and test splits.
        if args.dataset == "wit" and not args.quick_start:
            for split_name, split_files in [
                ("val", dataset.WIT_VALIDATION_FILES),
                ("test", dataset.WIT_TEST_FILES),
            ]:
                split_out = args.output_dir / split_name
                logger.info("=== Downloading WIT official %s split ===", split_name)
                dataset.download_all(
                    output_dir=split_out,
                    languages=languages,
                    max_samples=args.max_samples,
                    seed=args.seed,
                    source_files=split_files,
                    cache_dir=cache_dir,
                )

    if "convert" in args.steps:
        logger.info("=== Step 2/3: Convert to WebDataset ===")

        def _run_convert(input_dir: Path, output_dir: Path) -> None:
            if args.use_img2dataset:
                convert_all_img2dataset(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    languages=languages,
                    shard_size=args.shard_size,
                    processes_count=args.workers,
                    thread_count=args.img2dataset_threads,
                    image_size=args.img2dataset_image_size,
                    language_names=language_names,
                    max_samples=args.max_samples,
                )
            else:
                asyncio.run(
                    convert_all(
                        input_dir=input_dir,
                        output_dir=output_dir,
                        languages=languages,
                        shard_size=args.shard_size,
                        workers=args.workers,
                        language_names=language_names,
                        image_cache_dir=args.image_cache_dir or None,
                        max_samples=args.max_samples,
                    )
                )

        _run_convert(args.output_dir / "train", args.output_dir / "train")

        # For WIT (full run only), also convert official val and test shards.
        if args.dataset == "wit" and not args.quick_start:
            for split_name in ("val", "test"):
                split_dir = args.output_dir / split_name
                if not split_dir.exists():
                    continue
                logger.info("=== Converting WIT official %s shards ===", split_name)
                _run_convert(split_dir, split_dir)
        logger.info("Conversion complete.")

    if "stats" in args.steps:
        logger.info("=== Step 3/3: Compute statistics ===")
        all_stats = compute_all_stats(args.output_dir / "train", languages, language_names)
        display_stats(all_stats)

    if "prepare" in args.steps:
        logger.info("=== Step 4/4: Prepare training manifest ===")
        # When running the full pipeline for a single dataset, the data-dir scanned
        # for shards should be the multi-dataset root (DEFAULT_OUTPUT_DIR), not the
        # dataset-specific subfolder, so that shards from different sources are
        # combined in one manifest.
        from config import DEFAULT_OUTPUT_DIR as _ROOT
        prepare_training(
            data_dir=_ROOT,
            val_fraction=args.val_fraction,
            weight_temperature=args.weight_temp,
            shard_size=args.shard_size,
        )
        logger.info("Prepare complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(1)
