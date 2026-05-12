"""
FRAMES: Factual, Retrieval-Augmented Multi-hop Evaluation Set

Mialon et al., 2024 — https://arxiv.org/abs/2409.12941

Two task variants:

    frames_baseline  — all source documents provided upfront (no retrieval)
    frames_socrates  — iterative retrieval via a request_document tool
                       scored with a decaying reward for extra hops taken

# baseline
inspect eval frames_eval/frames.py@frames_baseline --limit 50

# socrates (iterative retrieval)
inspect eval frames_eval/frames.py@frames_socrates --limit 50
"""

import ast
import re
from typing import Any

import httpx
from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import Score, Scorer, Target, includes, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, tool
from inspect_ai.util import store

from frames_eval.dataset import frames_dataset

OPTIMAL_HOPS = 5
HALLUCINATION_PENALTY = 0.2
USER_AGENT = "frames-eval/1.0 (inspect_ai evaluation)"

# Store keys for per-sample state in the Socrates eval
_STORE_DOCS = "frames_docs"
_STORE_REQUESTED = "frames_requested"
_STORE_HALLUCINATIONS = "frames_hallucinations"


# ---------------------------------------------------------------------------
# Wikipedia fetch
# ---------------------------------------------------------------------------

async def _fetch_wikipedia(url: str) -> dict[str, str] | None:
    """Fetch the intro section of a Wikipedia article via the MediaWiki API."""
    clean_url = url.split("#")[0].rstrip("/")
    title_slug = clean_url.split("/wiki/")[-1]
    api_url = "https://en.wikipedia.org/w/api.php"
    params: dict[str, Any] = {
        "action": "query",
        "titles": title_slug,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "format": "json",
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            r = await client.get(api_url, params=params, timeout=15)
            data = r.json()
            pages = data["query"]["pages"]
            page = next(iter(pages.values()))
            if "missing" in page:
                return None
            content = page.get("extract", "").strip()
            if not content:
                return None
            return {"url": url, "title": page["title"], "content": content}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Top-level request_document tool (uses store for per-sample state)
# ---------------------------------------------------------------------------

@tool
def request_document() -> Tool:  # type: ignore[return]
    """Request a Wikipedia document by title to gather information for answering the question."""

    async def execute(title: str) -> str:
        """Retrieve a Wikipedia document from the allowlisted source pool.

        Args:
            title: The title of the Wikipedia article to retrieve.
        """
        s = store()
        docs: list[dict[str, str]] = s.get(_STORE_DOCS, [])
        requested: list[str] = s.get(_STORE_REQUESTED, [])
        hallucinations: list[str] = s.get(_STORE_HALLUCINATIONS, [])

        requested.append(title)
        s.set(_STORE_REQUESTED, requested)

        key = title.lower().strip()
        # Strip wiki-link formatting if the model includes it
        key = re.sub(r"[\[\]{}|]", "", key).strip()

        for doc in docs:
            avail = doc["title"].lower()
            if key in avail or avail in key:
                return f"### {doc['title']}\n{doc['content']}"

        hallucinations.append(title)
        s.set(_STORE_HALLUCINATIONS, hallucinations)
        return (
            f"Document '{title}' is not in the source pool for this question. "
            "Check the available document titles and request one of those."
        )

    return execute  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Socrates scorer
# ---------------------------------------------------------------------------

@scorer(metrics=[])
def socrates_scorer() -> Scorer:
    """Score = accuracy × min(optimal_hops / actual_hops, 1) − hallucination_penalty."""

    async def score(state: TaskState, target: Target) -> Score:
        answer = state.output.completion.strip().lower()
        is_correct = target.text.strip().lower() in answer

        hops: int = state.metadata.get("hops_taken", 1)
        hallucination_count: int = state.metadata.get("hallucination_count", 0)

        efficiency = min(OPTIMAL_HOPS / max(hops, 1), 1.0)
        base = float(is_correct) * efficiency
        final = max(0.0, base - HALLUCINATION_PENALTY * hallucination_count)

        return Score(
            value=final,
            explanation=(
                f"correct={is_correct}, hops={hops}, "
                f"hallucinations={hallucination_count}, score={final:.3f}"
            ),
        )

    return score


# ---------------------------------------------------------------------------
# Baseline task — all documents provided upfront
# ---------------------------------------------------------------------------

@task
def frames_baseline() -> Task:
    """FRAMES baseline: all source documents provided to the model at once."""

    @solver
    def inject_all_docs() -> Solver:
        async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
            urls: list[str] = state.metadata.get("wiki_links", [])
            docs = [d for d in [await _fetch_wikipedia(u) for u in urls] if d]
            doc_text = "\n\n".join(
                f"### {d['title']}\n{d['content']}" for d in docs
            )
            state.messages.append(
                ChatMessageUser(
                    content=(
                        "Use the following documents to answer the question. "
                        "Provide only the final answer — no explanation needed.\n\n"
                        f"{doc_text}\n\n"
                        f"Question: {state.input_text}"
                    )
                )
            )
            return await generate_fn(state)

        return solve

    return Task(
        dataset=frames_dataset(),
        solver=[inject_all_docs()],
        scorer=includes(),
    )


# ---------------------------------------------------------------------------
# Socrates task — iterative retrieval from an allowlisted document pool
# ---------------------------------------------------------------------------

@task
def frames_socrates(max_hops: int = 10) -> Task:
    """FRAMES Socrates: model retrieves documents iteratively via a tool.

    The document pool is restricted to the ground-truth source documents for
    each question (allowlist), preventing eval-awareness contamination.
    Score decays for hops beyond the optimal (5) and penalises hallucinated
    document requests.

    Args:
        max_hops: Maximum retrieval turns before forcing a final answer.
    """

    @solver
    def socrates_solver() -> Solver:
        async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
            urls: list[str] = state.metadata.get("wiki_links", [])

            # Pre-fetch all allowlisted docs upfront so the tool is fast
            docs = [d for d in [await _fetch_wikipedia(u) for u in urls] if d]

            # Seed store with per-sample state
            state.store.set(_STORE_DOCS, docs)
            state.store.set(_STORE_REQUESTED, [])
            state.store.set(_STORE_HALLUCINATIONS, [])

            # Seed 3 initial documents: first, middle, last
            seeded = _seed_docs(docs)
            seed_text = "\n\n".join(
                f"### {d['title']}\n{d['content']}" for d in seeded
            )

            available_titles = ", ".join(f'"{d["title"]}"' for d in docs)

            state.messages.append(
                ChatMessageUser(
                    content=(
                        "You are solving a multi-hop research question. "
                        "You have 3 initial documents below. "
                        "Use the request_document tool to retrieve additional "
                        f"documents by title. Available titles: {available_titles}. "
                        "Once you have enough information, state your final answer.\n\n"
                        f"Initial documents:\n{seed_text}\n\n"
                        f"Question: {state.input_text}"
                    )
                )
            )

            state.tools = [request_document()]
            state.tool_choice = "auto"

            for hop in range(max_hops):
                state = await generate_fn(state)
                if not _has_pending_tool_call(state) or hop == max_hops - 1:
                    state.metadata["hops_taken"] = hop + 1
                    break

            state.metadata["hallucination_count"] = len(
                state.store.get(_STORE_HALLUCINATIONS, [])
            )
            state.metadata["requested_titles"] = state.store.get(_STORE_REQUESTED, [])
            return state

        return solve

    return Task(
        dataset=frames_dataset(),
        solver=[socrates_solver()],
        scorer=socrates_scorer(),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_docs(docs: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return 3 seed docs: first (random context), middle (hint), last (bridge)."""
    if len(docs) <= 3:
        return docs
    return [docs[0], docs[len(docs) // 2], docs[-1]]


def _has_pending_tool_call(state: TaskState) -> bool:
    """Return True if the last assistant message contains a tool call."""
    for msg in reversed(state.messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            return True
        if hasattr(msg, "role") and msg.role == "assistant":
            break
    return False
