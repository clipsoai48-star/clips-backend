"""Transcribe a video's audio track into timestamped segments/words using faster-whisper."""
import logging
from typing import List

from faster_whisper import WhisperModel

from .models import TranscriptSegment, TranscriptWord

logger = logging.getLogger(__name__)

_MODELS = {}


def _get_model(model_size: str = "small", device: str = "auto", compute_type: str = "int8") -> WhisperModel:
    """
    Lazily load the whisper model once per process per size (workers are
    long-lived and handle both free and paid jobs, which now use different
    model sizes — so the cache must be keyed by size, not a single global).

    model_size: "tiny"/"base"/"small"/"medium"/"large-v3" — tradeoff speed vs accuracy.
    "tiny"/"base" are noticeably faster with slightly less accurate word timing;
    fine for free-tier turnaround time. "small" is used for paid tier.
    """
    if model_size not in _MODELS:
        logger.info("Loading whisper model: %s", model_size)
        _MODELS[model_size] = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _MODELS[model_size]


def transcribe(video_path: str, model_size: str = "small") -> List[TranscriptSegment]:
    """
    Run transcription and return a list of TranscriptSegment with word-level timestamps.
    Word-level timestamps are required later for accurate caption burn-in.
    """
    model = _get_model(model_size)

    segments, info = model.transcribe(
        video_path,
        word_timestamps=True,
        vad_filter=True,  # skip silence, improves segment boundaries
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
