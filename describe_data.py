"""
Describe the state of all datasets under the data/ directory.

Directly scans data/<source>/<split>/<lang>/shard-*.tar on disk.
Also reads stats.json files when present.
Prints a human-readable summary to stdout.
"""

import json
from collections import defaultdict
from pathlib import Path

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

DATA_DIR = Path(__file__).parent / "data"
SPLITS = ("train", "val", "test")


def fmt_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


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


# ── 1. Scan shards on disk ────────────────────────────────────────────────────
# Expected layout: data/<source>/<split>/<lang>/shard-*.tar

# source -> split -> lang -> [Path, ...]
found: dict[str, dict[str, dict[str, list[Path]]]] = defaultdict(
    lambda: defaultdict(lambda: defaultdict(list))
)

for source_dir in sorted(DATA_DIR.iterdir()):
    if not source_dir.is_dir():
        continue
    for split_dir in sorted(source_dir.iterdir()):
        if not split_dir.is_dir() or split_dir.name not in SPLITS:
            continue
        for lang_dir in sorted(split_dir.iterdir()):
            if not lang_dir.is_dir():
                continue
            shards = sorted(lang_dir.glob("shard-*.tar"))
            if shards:
                found[source_dir.name][split_dir.name][lang_dir.name] = shards

# ── 2. Shard breakdown ────────────────────────────────────────────────────────

print("=" * 60)
print("SHARD BREAKDOWN (scanned from disk)")
print("=" * 60)

if not found:
    print("  No shards found. Expected: data/<source>/<split>/<lang>/shard-*.tar")
else:
    split_totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))  # split -> {shards, size}
    shard_rows = []
    total_size = 0

    for source in sorted(found):
        for split in SPLITS:
            if split not in found[source]:
                continue
            for lang in sorted(found[source][split]):
                shard_paths = found[source][split][lang]
                n = len(shard_paths)
                size = sum(p.stat().st_size for p in shard_paths)
                total_size += size
                split_totals[split]["shards"] += n
                split_totals[split]["size"] += size
                shard_rows.append([source, split, lang, n, fmt_size(size)])

    print_table(shard_rows, ["Source", "Split", "Lang", "Shards", "Size"])

    print()
    print("  Split totals:")
    tot_rows = [
        [sp, split_totals[sp]["shards"], fmt_size(split_totals[sp]["size"])]
        for sp in SPLITS if sp in split_totals
    ]
    print_table(tot_rows, ["Split", "Shards", "Size"])
    print(f"\n  Total shard data on disk: {fmt_size(total_size)}")

# ── 3. Per-language stats from stats.json files ───────────────────────────────

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
        entries = data if isinstance(data, list) else [data]
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

# ── 4. Manifest (optional) ────────────────────────────────────────────────────

manifest_path = DATA_DIR / "manifest.json"
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    print()
    print("=" * 60)
    print("MANIFEST METADATA")
    print("=" * 60)
    print(f"  Version : {manifest.get('version')}")
    print(f"  Created : {manifest.get('created')}")
    weights = manifest.get("sampling_weights", {})
    if weights:
        lang_map = manifest.get("languages", {})
        print()
        print("  Sampling weights:")
        w_rows = [[lang, lang_map.get(lang, lang), f"{w:.4f}"] for lang, w in weights.items()]
        print_table(w_rows, ["Lang", "Name", "Weight"])

# ── 5. Error logs ─────────────────────────────────────────────────────────────

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
