# frames-eval

FRAMES multi-hop reasoning evaluation for [inspect_ai](https://inspect.aisi.org.uk/).

Paper: [FRAMES: Factual, Retrieval-Augmented Multi-hop Evaluation Set](https://arxiv.org/abs/2409.12941) (Mialon et al., 2024)

## Overview

FRAMES tests whether models can answer questions that require chaining information across multiple Wikipedia documents. Each question in the dataset has 5–15 ground-truth Wikipedia sources.

Two task variants are provided:

- **frames_baseline** — all source documents are injected upfront. Tests whether a model can reason across a full document set without retrieval.
- **frames_socrates** — documents are withheld. The model must iteratively request them via a `request_document` tool. Retrieval is constrained to a per-sample allowlist of the ground-truth Wikipedia sources, preventing eval-awareness contamination.

## Dataset

[google/frames-benchmark](https://huggingface.co/datasets/google/frames-benchmark) on HuggingFace, `test` split (824 samples).

Pinned revision: `58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef`

Wikipedia content is fetched live at eval time via the MediaWiki API (intro section only).

## Scorer

### frames_baseline

Uses inspect_ai's built-in `includes()` scorer — correct if the target answer appears anywhere in the model's response (case-insensitive).

### frames_socrates

Decaying reward that penalises unnecessary hops and hallucinated document requests:

```
score = accuracy × min(optimal_hops / actual_hops, 1) − 0.2 × hallucinations
```

- `optimal_hops = 5`
- `accuracy = 1` if the target string appears in the final response, else `0`
- `actual_hops` = number of `generate()` calls made (each tool-call round counts as one hop)
- `hallucinations` = number of `request_document` calls for titles not in the allowlist

Score is clamped to `[0, 1]`.

### Allowlist design

The `request_document` tool only serves documents from that sample's ground-truth Wikipedia source pool. Requests for any other title return an error and increment the hallucination counter. This prevents models from gaming the eval by browsing the open web or referencing eval-specific knowledge.

## Installation

```bash
git clone https://github.com/sahil350/frames-eval
cd frames-eval
uv sync
```

Requires Python ≥ 3.11.

## Running

```bash
# Baseline (all docs upfront)
uv run inspect eval src/frames_eval/frames.py@frames_baseline --model openai/gpt-4o --limit 50

# Socrates (iterative retrieval)
uv run inspect eval src/frames_eval/frames.py@frames_socrates --model openai/gpt-4o --limit 50

# With a local Ollama model
uv run inspect eval src/frames_eval/frames.py@frames_baseline --model ollama/llama3.1:8b

# View results
uv run inspect view
```

Task parameters:

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `max_hops` | `10` | Maximum retrieval turns for `frames_socrates` before forcing a final answer |

## Validation

Evals were run on 40–50 samples with local Ollama models to verify the pipeline end-to-end. Results are included in the inspect_evals registry entry.

| Model | Task | Accuracy | Mean Score |
| ----- | ---- | -------- | ---------- |
| ollama/llama3.2 | frames_baseline | 0.060 | — |
| ollama/llama3.2 | frames_socrates | 0.120 | 0.108 |
| ollama/llama3.1:8b | frames_baseline | 0.080 | — |
| ollama/llama3.1:8b | frames_socrates | 0.275 | 0.270 |

Low accuracy for small models is expected — FRAMES is designed to be challenging (the paper reports GPT-4o at ~0.72).

## Structure

```
src/frames_eval/
    frames.py     # Task definitions (frames_baseline, frames_socrates)
    dataset.py    # HuggingFace dataset loader
```
