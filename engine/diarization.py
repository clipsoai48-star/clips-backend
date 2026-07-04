"""
Speaker diarization: detects *who* is speaking and when, separate from *what*
they said (that's transcriber.py's job). Needed so captions can switch color
per speaker.

Uses pyannote.audio, which requires a free Hugging Face account:
  1. Create an account at https://huggingface.co/join
  2. Accept the model terms at:
       https://huggingface.co/pyannote/speaker-diarization-3.1
       https://huggingface.co/pyannote/segmentation-3.0
     (both are gated — just click "Agree" on each page, it's instant)
  3. Create a read-access token at https://huggingface.co/settings/tokens
  4. Put it in a `.env` file in this project as: HF_TOKEN=hf_xxxxxxxxxxxx

If no token is configured, diarization is skipped gracefully and every clip
just falls back to a single caption color (speaker_colors effectively off).
"""
import os
import logging
from typing import List, Optional

from .models import SpeakerTurn

logger = logging.getLogger(__name__)

_PIPELINE = None
_DIARIZATION_UNAVAILABLE_REASON: Optional[str] = None


def _get_pipeline():
    global _PIPELINE, _DIARIZATION_UNAVAILABLE_REASON
    if _PIPELINE is not None:
        return _PIPELINE
    if _DIARIZATION_UNAVAILABLE_REASON is not None:
        return None  # already tried and failed this process, don't retry every clip

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        _DIARIZATION_UNAVAILABLE_REASON = (
            "HF_TOKEN not set — speaker-color captions disabled. "
            "See diarization.py module docstring for setup steps."
        )
        logger.warning(_DIARIZATION_UNAVAILABLE_REASON)
        return None

    try:
        from pyannote.audio import Pipeline
        _PIPELINE = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
        )
        logger.info("Loaded speaker diarization model")
        return _PIPELINE
    except Exception as e:
        _DIARIZATION_UNAVAILABLE_REASON = f"Failed to load diarization model: {e}"
        logger.warning(_DIARIZATION_UNAVAILABLE_REASON)
        return None


def diarize(audio_path: str) -> List[SpeakerTurn]:
    """
    Runs diarization on the full source audio once (not per-clip — much cheaper).
    Returns [] if diarization isn't available/configured, in which case callers
    should fall back to single-speaker (no color switching) captions.
    """
    pipeline = _get_pipeline()
    if pipeline is None:
        return []

    diarization = pipeline(audio_path)
    turns = [
        SpeakerTurn(start=turn.start, end=turn.end, speaker=speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    logger.info("Diarization found %d speaker turns", len(turns))
    return turns


def speaker_at(turns: List[SpeakerTurn], t: float, default: str = "SPEAKER_00") -> str:
    """Returns which speaker was talking at time t (seconds into the source video)."""
    for turn in turns:
        if turn.start <= t <= turn.end:
            return turn.speaker
    return default
