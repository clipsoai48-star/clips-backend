"""Shared data models for the clip pipeline."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TranscriptWord:
    word: str
    start: float
    end: float


@dataclass
class TranscriptSegment:
    text: str
    start: float
    end: float
    words: List[TranscriptWord] = field(default_factory=list)
    avg_logprob: float = 0.0  # confidence signal from whisper, can hint at emphasis


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str  # e.g. "SPEAKER_00", "SPEAKER_01"


@dataclass
class HighlightCandidate:
    start: float
    end: float
    score: float
    reason: str
    transcript_text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class RenderOptions:
    """Feature flags gated by subscription tier."""
    vertical_crop: bool = True
    burn_captions: bool = True
    caption_style: str = "basic"  # one of: basic (free), bold_yellow, minimal_white, boxed, creator_pink (paid)
    words_per_caption: int = 3    # short flashy word bursts instead of full sentences
    caption_position: str = "middle"  # "middle" (flashy, TikTok-style) or "bottom" (traditional)
    speaker_colors: bool = False  # paid only — different color per speaker (needs diarization)
    add_music: bool = False       # paid only
    music_track_path: Optional[str] = None
    add_transitions: bool = False  # paid only
    add_zoom_punch: bool = False   # paid only ("effects")
    output_width: int = 1080
    output_height: int = 1920


@dataclass
class ClipJob:
    job_id: str
    source_path: str          # local path to downloaded/uploaded source video
    output_dir: str
    target_clip_count: int = 5
    min_clip_seconds: float = 15.0
    max_clip_seconds: float = 30.0
    is_paid_tier: bool = False
    job_type: str = "standard"  # "standard" or "football"
    sfx_choice: Optional[str] = None  # e.g. "clip_01.mp3", used when job_type="football"
