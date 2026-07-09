"""Transcribe a video's audio track into timestamped segments/words using faster-whisper."""
import logging
from typing import List

from faster_whisper import WhisperModel, BatchedInferencePipeline

from .models import TranscriptSegment, TranscriptWord

logger = logging.getLogger(__name__)

_MODELS = {}
_BATCHED_PIPELINES = {}


def _get_model(model_size: str = "base", device: str = "auto", compute_type: str = "int8") -> WhisperModel:
    """
    Lazily load the whisper model once per process per size (workers are
    long-lived and handle both free and paid jobs, which now use different
    model sizes — so the cache must be keyed by size, not a single global).

    model_size: "tiny"/"base"/"small"/"medium"/"large-v3" — tradeoff speed vs accuracy.
    "tiny" is used for free tier, "base" for paid — both are plenty accurate
    for highlight *timing*, which is all we actually need this for.
    """
    if model_size not in _MODELS:
        logger.info("Loading whisper model: %s", model_size)
        _MODELS[model_size] = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _MODELS[model_size]


def _get_batched_pipeline(model_size: str) -> BatchedInferencePipeline:
    """
    Wraps the base model in a batched pipeline, which processes multiple audio
    chunks in parallel instead of strictly sequentially — meaningfully faster
    on long videos even without a GPU.
    """
    if model_size not in _BATCHED_PIPELINES:
        model = _get_model(model_size)
        _BATCHED_PIPELINES[model_size] = BatchedInferencePipeline(model=model)
    return _BATCHED_PIPELINES[model_size]


def transcribe(video_path: str, model_size: str = "base", batch_size: int = 16) -> List[TranscriptSegment]:
    """
    Run transcription and return a list of TranscriptSegment with word-level timestamps.
    Word-level timestamps are required later for accurate caption burn-in.

    Tuned for speed over transcript-perfection: greedy decoding (beam_size=1,
    best_of=1) and no cross-segment context conditioning both cut compute
    substantially, and only affect exact wording — not the segment/word
    timing this pipeline actually relies on for clip selection and captions.
    """
    pipeline = _get_batched_pipeline(model_size)

    segments, info = pipeline.transcribe(
        video_path,
        word_timestamps=True,
        vad_filter=True,  # skip silence, improves segment boundaries
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
        batch_size=batch_size,
    )

    result: List[TranscriptSegment] = []
    for seg in segments:
        words = [
            TranscriptWord(word=w.word.strip(), start=w.start, end=w.end)
            for w in (seg.words or [])
        ]
        result.append(
            TranscriptSegment(
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
                words=words,
                avg_logprob=seg.avg_logprob,
            )
        )

    logger.info("Transcribed %d segments, detected language=%s", len(result), info.language)
    return result
