"""
WIT (Wikipedia-based Image Text) dataset package.

Public interface consumed by the shared pipeline:
  LANGUAGES   — dict[str, str] mapping language code -> language name
  download_all — download and filter WIT metadata into per-language parquets
"""

from datasets.wit.config import LANGUAGES, WIT_QUICKSTART_FILE, WIT_TRAIN_FILES, WIT_VALIDATION_FILES, WIT_TEST_FILES
from datasets.wit.download import download_all

__all__ = ["LANGUAGES", "WIT_QUICKSTART_FILE", "WIT_TRAIN_FILES", "WIT_VALIDATION_FILES", "WIT_TEST_FILES", "download_all"]
