# WIT Southeast Asia Dataset

Download and process the [Wikipedia-based Image Text (WIT)](https://github.com/google-research-datasets/wit) dataset for Southeast Asian languages, formatted as [OpenCLIP](https://github.com/mlfoundations/open_clip)-compatible WebDataset archives.

## Languages Covered

| Code | Language   |
|------|------------|
| id   | Indonesian |
| vi   | Vietnamese |
| th   | Thai       |
| tl   | Tagalog    |
| ms   | Malay      |
| km   | Khmer      |
| my   | Burmese    |
| lo   | Lao        |
| jv   | Javanese   |
| su   | Sundanese  |
| ceb  | Cebuano    |

## Output Format

WebDataset `.tar` shards compatible with OpenCLIP's training pipeline:

```
data/
  {language_code}/
    shard-000000.tar   # 10K samples per shard
    shard-000001.tar
    ...
    stats.json
```

Each tar shard contains paired files:
- `{id:08d}.jpg` — downloaded image
- `{id:08d}.txt` — caption text
- `{id:08d}.json` — metadata (image URL, Wikipedia page title, language)

## Usage

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
# Download, convert, and compute stats (default: up to 100K samples per language)
python main.py

# Limit sample size
python main.py --max-samples 10000

# Specific languages only
python main.py --languages id vi th

# Custom output directory
python main.py --output-dir /path/to/data

# Adjust parallelism for image downloads
python main.py --workers 20
```

### Run Individual Steps

```bash
# 1. Download and filter WIT metadata
python download_wit.py --output-dir data --max-samples 100000

# 2. Convert to WebDataset (downloads images)
python convert_to_webdataset.py --input-dir data --output-dir data --workers 10

# 3. Compute and display statistics
python compute_stats.py --data-dir data
```

## Dataset Source

- **WIT Dataset**: [google-research-datasets/wit](https://github.com/google-research-datasets/wit)
- **Hugging Face**: [google/wit](https://huggingface.co/datasets/google/wit)

## Notes

- The pipeline is resumable: re-running will skip already-downloaded shards.
- Failed image downloads are logged to `data/{language}/errors.log` and skipped.
- Statistics are saved to `data/{language}/stats.json` and `data/summary_stats.json`.
