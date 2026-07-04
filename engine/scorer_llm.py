"""
Optional semantic re-ranking layer on top of the cheap v1 scorer.

The v1 scorer (scorer.py) is fast and free but only measures "loud/energetic".
This module asks Claude to actually read the transcript windows and judge which
ones are genuinely funny, surprising, or quotable — better precision, at the
cost of an API call per job. Good candidate for the paid tier, or as a
secondary pass over just the top ~15 candidates from v1 (keeps API cost low).
"""
import os
import json
import logging
from typing import List

import anthropic

from .models import HighlightCandidate

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are helping select the best short-form highlight clips from a \
stream/video transcript. You will be given several candidate time windows with their \
transcript text. Rate each candidate from 0-10 on how likely it is to work as a \
standalone short clip (funny, surprising, high-energy, or a satisfying self-contained \
moment). Respond ONLY with a JSON array of objects: [{"index": 0, "score": 7.5, "reason": "..."}]. \
No other text."""


def rerank_with_llm(candidates: List[HighlightCandidate], top_k: int = 15) -> List[HighlightCandidate]:
    """
    Re-scores the top_k candidates (by existing score) via Claude, and returns
    the full candidate list with .score updated for the re-ranked subset,
    re-sorted by the blended result.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set; skipping LLM re-ranking")
        return candidates

    subset = candidates[:top_k]
    if not subset:
        return candidates

    client = anthropic.Anthropic(api_key=api_key)

    payload = [
        {"index": i, "start": c.start, "end": c.end, "transcript": c.transcript_text}
        for i, c in enumerate(subset)
    ]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload)}],
    )

    text = "".join(block.text for block in response.content if block.type == "text")
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        ratings = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM re-ranking response, skipping")
        return candidates

    for r in ratings:
        idx = r.get("index")
        if idx is None or idx >= len(subset):
            continue
        llm_score = float(r.get("score", 0)) / 10.0
        # blend: LLM judgment weighted higher than the cheap heuristic score
        subset[idx].score = 0.7 * llm_score + 0.3 * subset[idx].score
        if r.get("reason"):
            subset[idx].reason += f" | llm: {r['reason']}"

    merged = subset + candidates[top_k:]
    merged.sort(key=lambda c: c.score, reverse=True)
    return merged
