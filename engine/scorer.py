"""
Highlight scoring.

v1 approach (cheap, fast, no external API dependency):
  - audio energy / loudness spikes (laughter, shouting, crowd reaction)
  - lexical cues in the transcript (exclamations, laughter markers, keyword hits)
  - speech density (rapid back-and-forth reads as more "energetic")

These combine into a single score per sliding window. Optional LLM-based
re-ranking (scorer_llm.py) can be layered on top later for semantic quality
("this is actually funny" vs just "this is loud") — kept separate so v1 has
zero external API cost/latency.
"""
import logging
import subprocess
import numpy as np
from typing import List

from .models import TranscriptSegment, HighlightCandidate

logger = logging.getLogger(__name__)

# Words/phrases that tend to correlate with clip-worthy moments in streams/commentary
_HYPE_MARKERS = [
    "oh my god", "no way", "what", "let's go", "insane", "crazy",
    "wow", "yo", "holy", "actually", "wait wait", "!",
]


def _get_audio_energy(video_path: str, sample_rate: int = 100) -> np.ndarray:
    """
    Extract a coarse loudness/energy curve from the video's audio track using ffmpeg's
    astats filter, sampled at `sample_rate` windows per second-equivalent resolution.
    Returns an array of RMS energy values over time.
    """
    # Decode to raw mono PCM via ffmpeg, then compute short-time RMS energy in numpy.
    # This avoids needing librosa as a heavy dependency.
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0

    window = 16000 // sample_rate  # samples per analysis window
    if window <= 0:
        window = 160
    n_windows = len(audio) // window
    trimmed = audio[: n_windows * window].reshape(n_windows, window)
    rms = np.sqrt(np.mean(trimmed ** 2, axis=1) + 1e-9)
    return rms  # index i corresponds to time i / sample_rate seconds


def _lexical_score(text: str) -> float:
    text_lower = text.lower()
    hits = sum(1 for marker in _HYPE_MARKERS if marker in text_lower)
    return min(hits / 3.0, 1.0)  # cap contribution


def score_segments(
    video_path: str,
    segments: List[TranscriptSegment],
    window_seconds: float = 30.0,
) -> List[HighlightCandidate]:
    """
    Slide a window across the transcript, scoring each window using a blend of
    audio energy and lexical hype markers. Returns candidates sorted by score desc.
    """
    if not segments:
        return []

    try:
        energy = _get_audio_energy(video_path)
        energy_rate = 100  # matches sample_rate default in _get_audio_energy
        # normalize energy to 0..1 so it combines cleanly with lexical score
        if energy.max() > 0:
            energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-9)
        else:
            energy_norm = energy
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("Audio energy extraction failed (%s); falling back to lexical-only scoring", e)
        energy_norm, energy_rate = None, None

    total_duration = segments[-1].end
    candidates: List[HighlightCandidate] = []

    step = window_seconds / 2  # 50% overlap between windows
    t = 0.0
    while t < total_duration:
        window_start, window_end = t, min(t + window_seconds, total_duration)
        window_segs = [s for s in segments if s.start < window_end and s.end > window_start]

        if not window_segs:
            t += step
            continue

        text = " ".join(s.text for s in window_segs)
        lex_score = _lexical_score(text)

        if energy_norm is not None:
            start_idx = int(window_start * energy_rate)
            end_idx = int(window_end * energy_rate)
            energy_slice = energy_norm[start_idx:end_idx]
            energy_score = float(np.percentile(energy_slice, 90)) if len(energy_slice) else 0.0
        else:
            energy_score = 0.0

        combined = 0.6 * energy_score + 0.4 * lex_score

        candidates.append(
            HighlightCandidate(
                start=window_start,
                end=window_end,
                score=combined,
                reason=f"energy={energy_score:.2f} lexical={lex_score:.2f}",
                transcript_text=text,
            )
        )
        t += step

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def select_non_overlapping(
    candidates: List[HighlightCandidate],
    count: int,
    min_gap_seconds: float = 20.0,
    min_score: float = None,
) -> List[HighlightCandidate]:
    """
    Greedily pick top-scoring candidates while avoiding clips that overlap/sit
    too close together. If min_score is set, candidates below that score are
    excluded entirely — meaning this can return fewer than `count` (or zero)
    clips if the source video just doesn't have that many strong moments,
    rather than padding out with mediocre filler clips.
    """
    selected: List[HighlightCandidate] = []
    for c in candidates:
        if len(selected) >= count:
            break
        if min_score is not None and c.score < min_score:
            continue
        overlaps = any(
            not (c.end + min_gap_seconds < s.start or c.start - min_gap_seconds > s.end)
            for s in selected
        )
        if not overlaps:
            selected.append(c)
    selected.sort(key=lambda c: c.start)
    return selected
