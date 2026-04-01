"""
DatasetMixer — weighted multi-source WebDataset pipeline for training.

Reads the manifest produced by prepare_training.py and constructs a
WebDataset pipeline that mixes languages according to temperature-scaled
sampling weights.  One wds.WebDataset is built per language and they are
combined via wds.RandomMix.

Basic usage
-----------
    from mixer import DatasetMixer

    mixer = DatasetMixer("data/manifest.json")

    # Infinite, resampled training stream (shuffled)
    train_dataset = mixer.build(split="train")

    # Finite validation stream (no resampling)
    val_dataset = mixer.build(split="val", resampled=False)

    # Each sample is a dict with keys: "jpg", "txt", "json"
    for sample in train_dataset:
        image_bytes = sample["jpg"]
        caption     = sample["txt"]
        meta        = sample["json"]

Advanced usage
--------------
    # Override language weights (must sum to 1)
    mixer = DatasetMixer("data/manifest.json", weight_overrides={"id": 0.4, "vi": 0.4, "th": 0.2})

    # Restrict to specific languages or sources
    ds = mixer.build(split="train", languages=["id", "vi"], sources=["wit"])

    # Get the OpenCLIP --train-data glob string (single-source, unweighted)
    print(mixer.openclip_args(split="train"))
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DatasetMixer:
    """
    Weighted WebDataset mixer for multi-source, multi-language datasets.

    Parameters
    ----------
    manifest_path:
        Path to ``manifest.json`` produced by ``prepare_training.py``.
    weight_overrides:
        Optional dict mapping language code → weight (must sum to 1.0).
        Overrides the pre-computed weights stored in the manifest.
    shardshuffle:
        Number of shards to buffer when shuffling the shard list.
        Passed to ``wds.WebDataset``.  Set to 0 to disable shard shuffling.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        weight_overrides: dict[str, float] | None = None,
        shardshuffle: int = 100,
    ) -> None:
        manifest_path = Path(manifest_path)
        with manifest_path.open(encoding="utf-8") as f:
            self._manifest: dict = json.load(f)

        self._shardshuffle = shardshuffle
        self._weights: dict[str, float] = (
            weight_overrides if weight_overrides is not None
            else self._manifest.get("sampling_weights", {})
        )
        self._shards: list[dict] = self._manifest["shards"]

        # Resolve paths relative to the manifest file's directory
        self._manifest_dir = manifest_path.parent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def languages(self) -> list[str]:
        """Sorted list of language codes present in the manifest."""
        return sorted(self._manifest.get("languages", {}).keys())

    @property
    def sources(self) -> list[str]:
        """List of source dataset names present in the manifest."""
        return self._manifest.get("sources", [])

    @property
    def sampling_weights(self) -> dict[str, float]:
        """Per-language sampling weights (may differ from manifest if overridden)."""
        return dict(self._weights)

    def build(
        self,
        split: str = "train",
        languages: list[str] | None = None,
        sources: list[str] | None = None,
        resampled: bool | None = None,
        sample_shuffle: int = 1000,
    ):
        """
        Build a WebDataset pipeline for the requested split.

        Parameters
        ----------
        split:
            ``"train"`` or ``"val"``.
        languages:
            Restrict to these language codes (default: all in manifest).
        sources:
            Restrict to these source names (default: all in manifest).
        resampled:
            Whether to use infinite resampling (``wds.ResampledShards``).
            Defaults to ``True`` for training, ``False`` for validation.
        sample_shuffle:
            Buffer size for within-shard sample shuffling.  Set to 0 to
            disable.  Only applied when *resampled* is True.

        Returns
        -------
        wds.WebDataset
            A pipeline yielding dicts with keys ``"jpg"``, ``"txt"``,
            ``"json"`` (raw bytes for each).
        """
        import webdataset as wds

        if resampled is None:
            resampled = split == "train"

        lang_filter = set(languages) if languages else None
        src_filter = set(sources) if sources else None

        # Group shard paths by language
        shards_by_lang: dict[str, list[str]] = {}
        for shard in self._shards:
            if shard.get("split") != split:
                continue
            lang = shard["language"]
            if lang_filter and lang not in lang_filter:
                continue
            if src_filter and shard.get("source") not in src_filter:
                continue
            path = self._resolve_path(shard["path"])
            shards_by_lang.setdefault(lang, []).append(path)

        if not shards_by_lang:
            raise ValueError(
                f"No shards found for split={split!r}, "
                f"languages={languages!r}, sources={sources!r}"
            )

        active_langs = sorted(shards_by_lang.keys())
        weights = self._normalise_weights(active_langs)

        logger.info(
            "Building %s pipeline: %d language(s), resampled=%s",
            split, len(active_langs), resampled,
        )
        for lang in active_langs:
            logger.debug(
                "  %s: %d shard(s), weight=%.4f",
                lang, len(shards_by_lang[lang]), weights[lang],
            )

        # Build one dataset per language, then mix
        datasets = []
        for lang in active_langs:
            urls = shards_by_lang[lang]
            ds = wds.WebDataset(
                urls,
                resampled=resampled,
                shardshuffle=self._shardshuffle if resampled else False,
                nodesplitter=wds.split_by_node,
            )
            if resampled and sample_shuffle > 0:
                ds = ds.shuffle(sample_shuffle)
            datasets.append(ds)

        if len(datasets) == 1:
            return datasets[0]

        probs = [weights[lang] for lang in active_langs]
        return wds.RandomMix(datasets, probs)

    def shard_paths(
        self,
        split: str = "train",
        languages: list[str] | None = None,
        sources: list[str] | None = None,
    ) -> list[str]:
        """
        Return a flat list of resolved shard paths for the given filters.

        Useful for passing directly to ``wds.WebDataset(urls)`` or for
        generating OpenCLIP-compatible shard lists.
        """
        lang_filter = set(languages) if languages else None
        src_filter = set(sources) if sources else None
        paths = []
        for shard in self._shards:
            if shard.get("split") != split:
                continue
            if lang_filter and shard["language"] not in lang_filter:
                continue
            if src_filter and shard.get("source") not in src_filter:
                continue
            paths.append(self._resolve_path(shard["path"]))
        return paths

    def openclip_args(self, split: str = "train") -> dict[str, object]:
        """
        Return a dict of suggested OpenCLIP training arguments.

        Usage::

            args = mixer.openclip_args("train")
            # then pass to open_clip_train as e.g.:
            #   --train-data  "{args['train_data']}"
            #   --train-num-samples {args['train_num_samples']}
        """
        paths = self.shard_paths(split)
        n_samples = sum(
            s.get("n_samples") or 0
            for s in self._shards
            if s.get("split") == split
        )
        # WebDataset accepts a "::" joined list of URLs
        train_data = "::".join(paths)
        return {
            "train_data" if split == "train" else "val_data": train_data,
            f"{split}_num_samples": n_samples,
            "dataset_type": "webdataset",
        }

    def num_samples(self, split: str = "train") -> int:
        """Return the estimated total sample count for the given split."""
        return sum(
            s.get("n_samples") or 0
            for s in self._shards
            if s.get("split") == split
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, rel_path: str) -> str:
        """
        Resolve a manifest path to an absolute path.

        Manifest paths are relative to the workspace root (one level above
        the data/ directory).  We resolve them relative to the manifest's
        parent directory if the path starts with the data dir name, otherwise
        relative to the workspace root inferred from the manifest location.
        """
        p = Path(rel_path)
        if p.is_absolute():
            return str(p)
        # manifest lives at data/manifest.json → workspace root is one up
        workspace_root = self._manifest_dir.parent
        resolved = workspace_root / p
        return str(resolved)

    def _normalise_weights(self, active_langs: list[str]) -> dict[str, float]:
        """
        Return normalised weights for *active_langs* only.

        If a language has no entry in the manifest weights (e.g. due to
        filters reducing the active set), it falls back to uniform weighting.
        """
        raw = {lang: self._weights.get(lang, 1.0) for lang in active_langs}
        total = sum(raw.values())
        if total == 0:
            total = len(active_langs)
            return {lang: 1.0 / total for lang in active_langs}
        return {lang: v / total for lang, v in raw.items()}
