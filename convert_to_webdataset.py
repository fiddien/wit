"""
Convert WIT metadata parquet files to OpenCLIP-compatible WebDataset tar shards.

For each language the output layout is:
  <output_dir>/<lang>/shard-{n:06d}.tar

Each shard contains up to SHARD_SIZE samples; each sample consists of:
  {idx:08d}.jpg   — downloaded image (JPEG)
  {idx:08d}.txt   — caption text
  {idx:08d}.json  — metadata (url, title, language)

Failed downloads are logged to <output_dir>/<lang>/errors.log and skipped.
Existing shards are detected by a sentinel file and skipped on re-run.
"""

import argparse
import asyncio
import io
import json
import logging
import tarfile
from pathlib import Path

import aiofiles
import aiohttp
import cairosvg
import pandas as pd
from PIL import Image
from tqdm.asyncio import tqdm as atqdm

from config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SHARD_SIZE,
    DEFAULT_WORKERS,
    IMAGE_EXT,
    MAX_RETRIES,
    META_EXT,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF,
    SEA_LANGUAGES,
    TEXT_EXT,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image download helpers
# ---------------------------------------------------------------------------

async def _fetch_image_bytes(
    session: aiohttp.ClientSession,
    url: str,
) -> bytes | None:
    """Download *url* and return JPEG bytes, or None on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
                if resp.status != 200:
                    return None
                content_type = resp.headers.get("Content-Type", "")
                raw = await resp.read()
            # Convert SVG to PNG
            if "svg" in content_type or url.lower().endswith(".svg"):
                raw = cairosvg.svg2png(bytestring=raw)
            # Validate and normalise to JPEG
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return buf.getvalue()
        except asyncio.TimeoutError:
            logger.warning("Timeout on attempt %d for %s", attempt, url)
        except Exception as exc:
            logger.warning("Error on attempt %d for %s: %s", attempt, url, exc)
        if attempt < MAX_RETRIES:
            await asyncio.sleep(RETRY_BACKOFF * attempt)
    logger.debug("All retries exhausted for %s", url)
    return None


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------

def _write_shard(shard_path: Path, samples: list[dict]) -> int:
    """Write *samples* to a WebDataset tar file. Returns number written."""
    written = 0
    with tarfile.open(shard_path, "w") as tf:
        for idx, sample in enumerate(samples):
            stem = f"{idx:08d}"

            def _add(name: str, data: bytes) -> None:
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

            _add(f"{stem}.{IMAGE_EXT}", sample["image_bytes"])
            _add(f"{stem}.{TEXT_EXT}", sample["caption"].encode())
            _add(
                f"{stem}.{META_EXT}",
                json.dumps(
                    {
                        "url": sample["image_url"],
                        "title": sample["page_title"],
                        "language": sample["language"],
                        "page_url": sample.get("page_url", ""),
                    },
                    ensure_ascii=False,
                ).encode(),
            )
            written += 1
    return written


# ---------------------------------------------------------------------------
# Language converter
# ---------------------------------------------------------------------------

async def convert_language(
    lang: str,
    input_dir: Path,
    output_dir: Path,
    shard_size: int,
    workers: int,
) -> dict:
    """
    Convert the metadata parquet for *lang* to WebDataset shards.
    Returns a summary dict with counts.
    """
    parquet_path = input_dir / lang / "metadata.parquet"
    if not parquet_path.exists():
        logger.warning("No metadata found for %s at %s — skipping", lang, parquet_path)
        return {"language": lang, "total": 0, "downloaded": 0, "shards": 0}

    lang_out = output_dir / lang
    lang_out.mkdir(parents=True, exist_ok=True)
    error_log = lang_out / "errors.log"

    df = pd.read_parquet(parquet_path)
    total = len(df)
    logger.info("[%s] Converting %d rows to WebDataset shards…", lang, total)

    # Determine which shards already exist
    existing_shards = set(int(p.stem.split("-")[1]) for p in lang_out.glob("shard-*.tar"))

    semaphore = asyncio.Semaphore(workers)

    async def _bounded_fetch(session, url):
        async with semaphore:
            await asyncio.sleep(0.05)  # to avoid rate-limiting
            return await _fetch_image_bytes(session, url)

    downloaded = 0
    failed = 0
    shard_num = 0
    shard_buffer: list[dict] = []
    total_shards = 0

    connector = aiohttp.TCPConnector(limit=workers * 2)
    headers = {"User-Agent": "wit-sea-downloader/1.0 (https://github.com/fiddien/wit)"}
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        pbar = atqdm(total=total, desc=f"[{lang}] downloading", unit="img")

        async def _process_row(row):
            nonlocal downloaded, failed
            img_bytes = await _bounded_fetch(session, row["image_url"])
            pbar.update(1)
            if img_bytes is None:
                async with aiofiles.open(error_log, "a") as ef:
                    await ef.write(row["image_url"] + "\n")
                failed += 1
                return None
            downloaded += 1
            return {
                "image_bytes": img_bytes,
                "caption": row["caption"],
                "image_url": row["image_url"],
                "page_title": row.get("page_title", ""),
                "page_url": row.get("page_url", ""),
                "language": lang,
            }

        tasks = [_process_row(row) for row in df.to_dict("records")]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result is None:
                continue
            shard_buffer.append(result)
            if len(shard_buffer) >= shard_size:
                if shard_num not in existing_shards:
                    shard_path = lang_out / f"shard-{shard_num:06d}.tar"
                    _write_shard(shard_path, shard_buffer)
                    logger.info("[%s] Wrote %s", lang, shard_path.name)
                else:
                    logger.info("[%s] Shard %06d already exists — skipping", lang, shard_num)
                total_shards += 1
                shard_num += 1
                shard_buffer = []

        # Write remaining samples as a partial shard
        if shard_buffer:
            if shard_num not in existing_shards:
                shard_path = lang_out / f"shard-{shard_num:06d}.tar"
                _write_shard(shard_path, shard_buffer)
                logger.info("[%s] Wrote final shard %s (%d samples)", lang, shard_path.name, len(shard_buffer))
            total_shards += 1

        pbar.close()

    summary = {
        "language": lang,
        "language_name": SEA_LANGUAGES.get(lang, lang),
        "total_rows": total,
        "downloaded": downloaded,
        "failed": failed,
        "shards": total_shards,
    }
    logger.info(
        "[%s] Done — %d downloaded, %d failed, %d shards",
        lang, downloaded, failed, total_shards,
    )
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def convert_all(
    input_dir: Path,
    output_dir: Path,
    languages: list[str],
    shard_size: int,
    workers: int,
) -> list[dict]:
    summaries = []
    for lang in languages:
        summary = await convert_language(lang, input_dir, output_dir, shard_size, workers)
        summaries.append(summary)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert WIT parquet metadata to OpenCLIP WebDataset tar shards."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing per-language metadata.parquet files (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory for WebDataset shards (default: %(default)s)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(SEA_LANGUAGES.keys()),
        choices=list(SEA_LANGUAGES.keys()),
        metavar="LANG",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help="Samples per tar shard (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel image download workers (default: %(default)s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summaries = asyncio.run(
        convert_all(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            languages=args.languages,
            shard_size=args.shard_size,
            workers=args.workers,
        )
    )
    for s in summaries:
        print(
            f"[{s['language']:>4}] {s['language_name']:<12} "
            f"rows={s['total_rows']:>7}  downloaded={s['downloaded']:>7}  "
            f"failed={s['failed']:>5}  shards={s['shards']:>4}"
        )
