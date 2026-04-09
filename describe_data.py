"""
Describe the state of all datasets under the data/ directory.

Reads manifest.json, stats.json files, and scans for tar shards on disk.
Prints a human-readable summary to stdout.
"""

import json
import os
from pathlib import Path

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

DATA_DIR = Path(__file__).parent / "data"


def fmt_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def print_table(rows, headers):
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="simple_outline"))
    else:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
                      for i, h in enumerate(headers)]
        fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
        print(fmt.format(*headers))
        print("  ".join("-" * w for w in col_widths))
        for row in rows:
            print(fmt.format(*row))


# ── 1. Manifest overview ──────────────────────────────────────────────────────

manifest_path = DATA_DIR / "manifest.json"
if not manifest_path.exists():
    print(f"[ERROR] manifest.json not found in {DATA_DIR}")
    raise SystemExit(1)

manifest = json.loads(manifest_path.read_text())

print("=" * 60)
print("DATASET MANIFEST")
print("=" * 60)
print(f"  Version   : {manifest.get('version')}")
print(f"  Created   : {manifest.get('created')}")
print(f"  Sources   : {', '.join(manifest.get('sources', []))}")
lang_map = manifest.get("languages", {})
print(f"  Languages : {', '.join(f'{k} ({v})' for k, v in lang_map.items())}")

summary = manifest.get("summary", {})
print()
print("  Split summary:")
split_rows = []
for split, info in summary.items():
    split_rows.append([
        split,
        info.get("total_shards", "?"),
        f"{info.get('total_samples', 0):,}",
    ])
print_table(split_rows, ["Split", "Shards", "Samples"])

# ── 2. Sampling weights ───────────────────────────────────────────────────────

weights = manifest.get("sampling_weights", {})
if weights:
    print()
    print("  Sampling weights (trained):")
    w_rows = [[lang, lang_map.get(lang, lang), f"{w:.4f}"] for lang, w in weights.items()]
    print_table(w_rows, ["Lang", "Name", "Weight"])

# ── 3. Per-source / per-split breakdown from manifest shards ─────────────────

shards = manifest.get("shards", [])
# Aggregate: source -> split -> language -> samples
from collections import defaultdict
agg = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
shard_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for s in shards:
    agg[s["source"]][s["split"]][s["language"]] += s["n_samples"]
    shard_count[s["source"]][s["split"]][s["language"]] += 1

print()
print("=" * 60)
print("SHARD BREAKDOWN (from manifest)")
print("=" * 60)
shard_rows = []
for source in sorted(agg):
    for split in sorted(agg[source]):
        for lang in sorted(agg[source][split]):
            shard_rows.append([
                source, split, lang,
                shard_count[source][split][lang],
                f"{agg[source][split][lang]:,}",
            ])
print_table(shard_rows, ["Source", "Split", "Lang", "Shards", "Samples"])

# ── 4. Disk state: which shards actually exist ────────────────────────────────

print()
print("=" * 60)
print("DISK STATE")
print("=" * 60)

disk_rows = []
total_size = 0
for s in shards:
    p = DATA_DIR.parent / s["path"]
    exists = p.exists()
    size = p.stat().st_size if exists else 0
    total_size += size
    disk_rows.append([
        s["source"], s["split"], s["language"],
        "✓" if exists else "✗",
        fmt_size(size) if exists else "-",
    ])

print_table(disk_rows, ["Source", "Split", "Lang", "On disk", "Size"])
print(f"\n  Total shard data on disk: {fmt_size(total_size)}")

# ── 5. Per-language stats from stats.json files ───────────────────────────────

print()
print("=" * 60)
print("PER-LANGUAGE STATS (from stats.json)")
print("=" * 60)

stats_files = sorted(DATA_DIR.rglob("stats.json"))
if not stats_files:
    print("  No stats.json files found.")
else:
    stats_rows = []
    for sf in stats_files:
        rel = sf.relative_to(DATA_DIR)
        data = json.loads(sf.read_text())
        if isinstance(data, list):
            entries = data
        else:
            entries = [data]
        for entry in entries:
            lang = entry.get("language", "?")
            wd = entry.get("webdataset", {})
            meta = entry.get("metadata", {})
            samples = wd.get("total_samples", meta.get("total_rows", "?"))
            shards_n = wd.get("total_shards", "?")
            failed = wd.get("failed_downloads", "-")
            cap = wd.get("caption_stats") or meta.get("caption_stats") or {}
            mean_words = cap.get("mean_words", "-")
            stats_rows.append([
                str(rel.parent), lang,
                f"{samples:,}" if isinstance(samples, int) else samples,
                shards_n, failed, mean_words,
            ])
    print_table(stats_rows, ["Path", "Lang", "Samples", "Shards", "Failed DL", "Avg words"])

# ── 6. Error logs ─────────────────────────────────────────────────────────────

error_logs = sorted(DATA_DIR.rglob("errors.log"))
if error_logs:
    print()
    print("=" * 60)
    print("ERROR LOGS")
    print("=" * 60)
    for el in error_logs:
        lines = el.read_text().strip().splitlines()
        print(f"\n  {el.relative_to(DATA_DIR)}  ({len(lines)} lines)")
        for line in lines[:5]:
            print(f"    {line}")
        if len(lines) > 5:
            print(f"    ... ({len(lines) - 5} more lines)")

print()
print("=" * 60)
print("DONE")
print("=" * 60)
