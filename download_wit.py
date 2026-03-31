"""
Backward-compatibility shim.  WIT download logic has moved to datasets/wit/download.py.
Import from there directly; this file will be removed in a future cleanup.
"""
from datasets.wit.download import (  # noqa: F401
    ReservoirSampler,
    download_all,
)
from datasets.wit.config import LANGUAGES as SEA_LANGUAGES  # noqa: F401
