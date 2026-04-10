"""
Export WebDataset shards to flat image files + CSV for the training team.

Reads  : data/<source>/<split>/<lang>/shard-*.tar
Writes :
  <images-dir>/<source>/<split>/<lang>/<idx>.jpg   — extracted images
  <data-dir>/train.csv                              — combined train split
  <data-dir>/val.csv
  <data-dir>/test.csv

CSV format (no BOM, UTF-8):
  img_path,caption
  /abs/path/to/img.jpg,caption text

Each sample produces one row; multiple captions per image are not collapsed.

Usage (local, for testing):
  python export_to_csv.py

Usage (on cluster):
  python export_to_csv.py \\
      --data-dir /home/ifiddien/multilingual-datasets/data \\
      --images-dir /home/ifiddien/multilingual-datasets/data/images

  # or as a SLURM job:
  sbatch --mem=16G --time=02:00:00 --wrap \\
      "python export_to_csv.py \\
           --data-dir /home/ifiddien/multilingual-datasets/data \\
           --images-dir /home/ifiddien/multilingual-datasets/data/images"
"""

import argparse
import csv
import logging
import tarfile
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

KNOWN_SOURCES = ["wit", "cultural-ground"]
SPLITS = ("train", "val", "test")


def _discover_shards(data_dir: Path) -> dict[str, list[tuple[str, str, Path]]]:
    """
    Returns {split: [(source, lang, shard_path), ...]} for all found shards.
    """
    result: dict[str, list] = defaultdict(list)

    for source in KNOWN_SOURCES:
        source_dir = data_dir / source
        if not source_dir.is_dir():
            continue
        for split in SPLITS:
            split_dir = source_dir / split
            if not split_dir.is_dir():
                continue
            for lang_dir in sorted(split_dir.iterdir()):
                if not lang_dir.is_dir():
                    continue
                lang = lang_dir.name
                for shard in sorted(lang_dir.glob("shard-*.tar")):
                    result[split].append((source, lang, shard))

    return result


def _extract_shard(
    shard_path: Path,
    source: str,
    lang: str,
    split: str,
    images_dir: Path,
    writer: csv.writer,
) -> int:
    """
    Extract all samples from one shard tar into *images_dir* and append rows
    to *writer*.  Returns the number of samples written.
    """
    out_dir = images_dir / source / split / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group tar members by sample key (strip extension)
    members: dict[str, dict[str, bytes]] = defaultdict(dict)
    try:
        with tarfile.open(shard_path, "r") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                name = member.name  # e.g. "00000000.jpg"
                stem, ext = name.rsplit(".", 1) if "." in name else (name, "")
                ext = ext.lower()
                if ext in ("jpg", "jpeg", "txt", "json"):
                    f = tf.extractfile(member)
                    if f is not None:
                        members[stem][ext] = f.read()
    except (tarfile.TarError, OSError) as exc:
        logger.warning("Skipping corrupt shard %s: %s", shard_path, exc)
        return 0

    written = 0
    for stem in sorted(members):
        exts = members[stem]
        jpg_bytes = exts.get("jpg") or exts.get("jpeg")
        caption_bytes = exts.get("txt")

        if jpg_bytes is None or caption_bytes is None:
            continue  # incomplete sample

        img_path = out_dir / f"{stem}.jpg"
        if not img_path.exists():
            img_path.write_bytes(jpg_bytes)

        caption = caption_bytes.decode("utf-8", errors="replace").strip()
        if caption:
            writer.writerow([str(img_path.resolve()), caption])
            written += 1

    return written


def export(data_dir: Path, images_dir: Path) -> None:
    shards_by_split = _discover_shards(data_dir)
    if not any(shards_by_split.values()):
        logger.error("No shards found under %s. Nothing to export.", data_dir)
        return

    total_written = 0

    for split in SPLITS:
        entries = shards_by_split.get(split, [])
        if not entries:
            logger.info("No shards for split=%s, skipping.", split)
            continue

        csv_path = data_dir / f"{split}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["img_path", "caption"])

            split_written = 0
            for source, lang, shard in entries:
                n = _extract_shard(shard, source, lang, split, images_dir, writer)
                split_written += n
                logger.info(
                    "[%s/%s/%s] %s → %d sample(s)",
                    source,
                    split,
                    lang,
                    shard.name,
                    n,
                )

        logger.info("Wrote %d row(s) → %s", split_written, csv_path)
        total_written += split_written

    logger.info("Export complete. Total rows written: %d", total_written)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export WebDataset shards to flat images + CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory containing shard tars (also where CSVs are written)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help=(
            "Directory to extract images into. "
            "Defaults to <data-dir>/images. "
            "Image paths in the CSV will be absolute."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir.resolve()
    images_dir: Path = (args.images_dir or data_dir / "images").resolve()

    logger.info("data-dir   : %s", data_dir)
    logger.info("images-dir : %s", images_dir)

    export(data_dir, images_dir)


if __name__ == "__main__":
    main()
