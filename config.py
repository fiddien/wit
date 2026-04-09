"""
Shared pipeline configuration (dataset-agnostic defaults).

Dataset-specific settings (languages, source URLs, field mappings, etc.)
live in their respective datasets/<name>/config.py.
"""

from pathlib import Path

# Pipeline defaults
DEFAULT_MAX_SAMPLES = 10_000      # per language
DEFAULT_SHARD_SIZE = 10_000
DEFAULT_WORKERS = 4
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_CACHE_DIR = Path("cache")        # local cache for downloaded TSV.gz files
DEFAULT_IMAGE_CACHE_DIR = Path("cache") / "images"  # local cache for downloaded images

# HTTP settings for image downloads
REQUEST_TIMEOUT = 300              # seconds
REQUEST_DELAY = 0.2                # seconds between requests per worker (throttle, legacy)
WIKIMEDIA_RATE_LIMIT = 15          # max requests per second (Wikimedia policy)
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0                # seconds between retries

# WebDataset file extensions
IMAGE_EXT = "jpg"
TEXT_EXT = "txt"
META_EXT = "json"
