"""FRAMES dataset loader."""

import ast
from typing import Any

from inspect_ai.dataset import Dataset, MemoryDataset, Sample

HF_DATASET_PATH = "google/frames-benchmark"
HF_DATASET_REVISION = "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef"


def frames_dataset() -> Dataset:
    """Load the FRAMES benchmark from HuggingFace."""
    from datasets import load_dataset  # type: ignore[import-untyped]

    hf = load_dataset(HF_DATASET_PATH, split="test", revision=HF_DATASET_REVISION)
    samples: list[Sample] = []

    for row in hf:
        wiki_links = _parse_links(row.get("wiki_links") or [])
        if not wiki_links:
            continue
        samples.append(
            Sample(
                input=row["Prompt"],
                target=row["Answer"],
                metadata={
                    "wiki_links": wiki_links,
                    "reasoning_types": row.get("reasoning_types", ""),
                },
            )
        )

    return MemoryDataset(samples=samples, name="frames")


def _parse_links(raw: Any) -> list[str]:
    """Parse wiki_links which is stored as a stringified Python list."""
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    return [l for l in raw if l and l != "None"]
