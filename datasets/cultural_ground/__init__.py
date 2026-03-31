"""
CulturalGround dataset package.

Public interface consumed by the shared pipeline:
  LANGUAGES     — dict[str, str] mapping language code -> language name
  download_all  — download and filter CulturalGround VQA data into per-language parquets
"""

from datasets.cultural_ground.config import LANGUAGES, SEA_COUNTRIES
from datasets.cultural_ground.download import download_all

__all__ = ["LANGUAGES", "SEA_COUNTRIES", "download_all"]
