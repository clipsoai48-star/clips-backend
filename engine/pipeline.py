"""
Full pipeline orchestration. This is what a job-queue worker (e.g. a BullMQ/RQ/Celery
consumer) would import and call per job. Kept framework-agnostic on purpose — wire
this function up to whatever queue you choose in the web app.
"""
import os
import logging
from typing import List

from .models import ClipJob, RenderOptions, HighlightCandidate
from .transcriber import transcribe
from .scorer import score_segments, select_non_overlapping
from .renderer import render_clip

logger = logging.getLogger(__name__)


def run_pipeline(
    job: ClipJob,
    use_llm_rerank: bool = False,
    caption_style_override: str = None,
    min_score: float = None,
    speaker_colors: bool = False,
) -> List[str]:
    """
    Runs the end-to-end pipeline for a single source video and returns a list
    of output filepaths for the generated clips.

    min_score: quality floor for highlight candidates (0-1). If set, only
    genuinely clip-worthy moments are rendered — the job may return fewer
    clips than job.target_clip_count (even zero) if the source video doesn't
    have that many strong moments, rather than padding out with filler.
    A sensible default is applied automatically when use_llm_rerank=True
    (LLM scores are much better calibrated for "is this actually good" than
    the raw heuristic score).
    """
    logger.info("Job %s: transcribing %s", job.job_id, job.source_path)
    segments = transcribe(job.source_path)

    logger.info("Job %s: scoring highlight candidates", job.job_id)
    candidates = score_segments(job.source_path, segments, window_seconds=job.max_clip_seconds)

    if use_llm_rerank:
        from .scorer_llm import rerank_with_llm
        candidates = rerank_with_llm(candidates)
        if min_score is None:
            min_score = 0.55  # LLM-blended score above ~5.5/10 — filters out "meh" moments

    speaker_turns = []
    if speaker_colors:
        from .diarization import diarize
        logger.info("Job %s: running speaker diarization", job.job_id)
        speaker_turns = diarize(job.source_path)
        if not speaker_turns:
            logger.info("Job %s: no diarization data available, captions will use a single color", job.job_id)

    selected = select_non_overlapping(candidates, count=job.target_clip_count, min_score=min_score)
    logger.info(
        "Job %s: selected %d clip-worthy moment(s) (requested up to %d, min_score=%s)",
        job.job_id, len(selected), job.target_clip_count, min_score,
    )
    if not selected:
        logger.warning(
            "Job %s: no candidates met the quality bar (min_score=%s) — "
            "returning 0 clips rather than forcing out low-quality filler",
            job.job_id, min_score,
        )
        return []

    options = _render_options_for_tier(job.is_paid_tier)
    options.speaker_colors = speaker_colors and job.is_paid_tier  # paid-only feature
    if caption_style_override:
        # Free tier is only allowed "basic" — enforce that here, server-side,
        # regardless of what a client requests.
        if job.is_paid_tier or caption_style_override == "basic":
            options.caption_style = caption_style_override
        else:
            logger.warning(
                "Ignoring caption_style_override=%s for free-tier job %s (paid feature)",
                caption_style_override, job.job_id,
            )

    output_paths = []
    for i, candidate in enumerate(selected):
        out_path = os.path.join(job.output_dir, f"{job.job_id}_clip{i+1}.mp4")
        render_clip(job.source_path, candidate, segments, out_path, options, speaker_turns=speaker_turns)
        output_paths.append(out_path)
        logger.info("Job %s: rendered %s (score=%.2f, %s)", job.job_id, out_path, candidate.score, candidate.reason)

    return output_paths


def _render_options_for_tier(is_paid: bool) -> RenderOptions:
    """Central place that maps subscription tier -> feature flags. Enforce this
    server-side only — never trust a client-supplied 'is_paid' flag without
    checking the user's actual subscription status in your DB first."""
    if is_paid:
        return RenderOptions(
            vertical_crop=True,
            burn_captions=True,
            caption_style="bold_yellow",
            words_per_caption=3,
            caption_position="middle",
            add_music=True,
            music_track_path=None,  # set to a licensed track path chosen by the user
            add_zoom_punch=True,
        )
    return RenderOptions(
        vertical_crop=True,
        burn_captions=True,
        caption_style="basic",
        words_per_caption=3,
        caption_position="middle",
        add_music=False,
        add_zoom_punch=False,
    )
