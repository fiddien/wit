"""
WIT (Wikipedia-based Image Text) dataset configuration.
"""

# Southeast Asian language codes present in WIT
LANGUAGES = {
    "id": "Indonesian",
    "vi": "Vietnamese",
    "th": "Thai",
    "ms": "Malay",
    "my": "Burmese",
    "jv": "Javanese",
    # not present in WIT shards
    # "tl": "Tagalog",
    # "km": "Khmer",
    # "lo": "Lao",
    # "su": "Sundanese",
    # "ceb": "Cebuano",
}

# Full WIT training set: 10 TSV.gz shards on Google Cloud Storage (~2.5 GB each, ~25 GB total)
WIT_GCS_BASE = "https://storage.googleapis.com/gresearch/wit"
WIT_QUICKSTART_FILE = f"{WIT_GCS_BASE}/wit_v1.train.all-1percent_sample.tsv.gz"
WIT_TRAIN_FILES = [
    f"{WIT_GCS_BASE}/wit_v1.train.all-{i:05d}-of-00010.tsv.gz"
    for i in range(10)
]
WIT_VALIDATION_FILES = [
    f"{WIT_GCS_BASE}/wit_v1.val.all-{i:05d}-of-00005.tsv.gz"
    for i in range(5)
]
WIT_TEST_FILES = [
    f"{WIT_GCS_BASE}/wit_v1.test.all-{i:05d}-of-00005.tsv.gz"
    for i in range(5)
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
