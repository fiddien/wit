"""
Configuration for WIT Southeast Asian languages dataset pipeline.
"""

from pathlib import Path

# Southeast Asian language codes present in WIT
SEA_LANGUAGES = {
    "id": "Indonesian",
    "vi": "Vietnamese",
    "th": "Thai",
    "tl": "Tagalog",
    "ms": "Malay",
    "km": "Khmer",
    "my": "Burmese",
    "lo": "Lao",
    "jv": "Javanese",
    "su": "Sundanese",
    "ceb": "Cebuano",
}

# Full WIT training set: 10 TSV.gz shards on Google Cloud Storage (~2.5 GB each, ~25 GB total)
WIT_GCS_BASE = "https://storage.googleapis.com/gresearch/wit"
WIT_QUICKSTART_FILE = f"{WIT_GCS_BASE}/wit_v1.train.all-1percent_sample.tsv.gz"
WIT_TRAIN_FILES = [
    f"{WIT_GCS_BASE}/wit_v1.train.all-{i:05d}-of-00010.tsv.gz"
    for i in range(10)
]

# TSV column order
WIT_TSV_COLUMNS = [
    "language",
    "page_url",
    "image_url",
    "page_title",
    "section_title",
    "hierarchical_section_title",
    "caption_reference_description",
    "caption_attribution_description",
    "caption_alt_text_description",
    "mime_type",
    "original_height",
    "original_width",
    "is_main_image",
    "attribution_passes_lang_id",
    "page_changed_recently",
    "context_page_description",
    "context_section_description",
]

# Pipeline defaults
DEFAULT_MAX_SAMPLES = 100_000      # per language
DEFAULT_SHARD_SIZE = 10_000
DEFAULT_WORKERS = 10
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")

# HTTP settings for image downloads
REQUEST_TIMEOUT = 30               # seconds
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0                # seconds between retries

# WebDataset file extensions
IMAGE_EXT = "jpg"
TEXT_EXT = "txt"
META_EXT = "json"

# WIT dataset field names
WIT_FIELDS = {
    "language": "language",
    "image_url": "image_url",
    "caption": "caption_reference_description",
    "alt_text": "caption_alt_text_description",
    "title": "page_title",
    "page_url": "page_url",
    "attribution": "attribution_passes_lang_id",
}
