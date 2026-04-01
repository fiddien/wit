"""
CulturalGround dataset configuration.

CulturalGround is a large-scale culturally-rich VQA dataset spanning 42 countries
and 39 languages, curated from Wikidata and Wikimedia Commons.

Paper: https://arxiv.org/abs/2508.07414
HuggingFace: https://huggingface.co/datasets/neulab/CulturalGround
"""

# Southeast Asian language codes present in CulturalGround
LANGUAGES = {
    "id": "Indonesian",
    "vi": "Vietnamese",
    "th": "Thai",
    "ms": "Malay",
    "jv": "Javanese",
    "su": "Sundanese",
    # Note: tl (Tagalog), my (Burmese), lo (Lao) are not in CulturalGround.
}

# SEA countries covered by CulturalGround.
# Used to decide which per-country JSONL files are scanned.
SEA_COUNTRIES = [
    "indonesia",
    "vietnam",
    "thailand",
    "malaysia",
    "singapore",
]

HF_BASE_URL = (
    "https://huggingface.co/datasets/neulab/CulturalGround/resolve/main"
)

# Per-country LLM-refined open-ended VQA (JSONL, one record per line)
OE_REFINED_JSONL_TEMPLATE = (
    f"{HF_BASE_URL}/CulturalGround-Recipes/CulturalGround-Refined-OE/{{country}}_refined.jsonl"
)

# Pre-packaged per-country image tarballs (one .tar.gz per country)
IMAGE_TARBALL_TEMPLATE = (
    f"{HF_BASE_URL}/CultureGroundImages/{{country}}.tar.gz"
)
