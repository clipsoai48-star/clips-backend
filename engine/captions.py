"""
Builds captions as an .ass subtitle file (richer than .srt — supports per-line
color, position, and styling, which we need for flashy word-burst captions and
speaker-based color switching).

Instead of burning full sentences on screen (the old approach), this groups
word-level timestamps into short bursts (default: 3 words) so captions punch
in and out quickly — the "TikTok auto-caption" look.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional

from .models import TranscriptSegment, TranscriptWord, SpeakerTurn
from .diarization import speaker_at

logger = logging.getLogger(__name__)

# ASS colors are &HAABBGGRR (alpha, blue, green, red — reversed vs normal hex).
# Cycled through in the order speakers are first detected.
SPEAKER_COLOR_PALETTE = [
    "&H0000FFFF",  # yellow
    "&H00FF6EC7",  # pink/magenta
    "&H0000FF7F",  # spring green
    "&H00FFA500",  # orange-ish
    "&H00FFFFFF",  # white (fallback / 5th+ speaker)
]

CAPTION_POSITIONS = {
    "middle": 5,   # ASS alignment code: 5 = middle-center — flashy, TikTok-style
    "bottom": 2,   # 2 = bottom-center — traditional subtitle position
}


@dataclass
class CaptionChunk:
    words: List[TranscriptWord]
    speaker: str = "SPEAKER_00"

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(w.word for w in self.words).strip().upper()


def _format_ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    cs_total = int(round(seconds * 100))
    h, rem = divmod(cs_total, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def build_word_chunks(
    segments: List[TranscriptSegment],
    clip_start: float,
    clip_end: float,
    words_per_chunk: int = 3,
    speaker_turns: Optional[List[SpeakerTurn]] = None,
) -> List[CaptionChunk]:
    """
    Flattens all words within [clip_start, clip_end) across segments, then
    groups them into short chunks of `words_per_chunk` words each, tagging
    each chunk with whichever speaker was talking during it (if diarization
    data is available).
    """
    all_words = [
        w for seg in segments for w in seg.words
        if w.end > clip_start and w.start < clip_end and w.word.strip()
    ]
    all_words.sort(key=lambda w: w.start)

    chunks: List[CaptionChunk] = []
    current: List[TranscriptWord] = []
    for w in all_words:
        current.append(w)
        if len(current) >= words_per_chunk:
            chunks.append(_finalize_chunk(current, speaker_turns))
            current = []
    if current:
        chunks.append(_finalize_chunk(current, speaker_turns))

    return chunks


def _finalize_chunk(words: List[TranscriptWord], speaker_turns: Optional[List[SpeakerTurn]]) -> CaptionChunk:
    speaker = "SPEAKER_00"
    if speaker_turns:
        # tag the chunk with whoever was speaking at its midpoint
        midpoint = (words[0].start + words[-1].end) / 2
        speaker = speaker_at(speaker_turns, midpoint)
    return CaptionChunk(words=words, speaker=speaker)


def build_ass_file(
    chunks: List[CaptionChunk],
    clip_start: float,
    output_path: str,
    video_width: int = 1080,
    video_height: int = 1920,
    font_name: str = "Arial Black",
    font_size: int = 68,
    position: str = "middle",
    use_speaker_colors: bool = False,
    default_color: str = "&H0000FFFF",
    outline_color: str = "&H00000000",
) -> str:
    """
    Writes an .ass subtitle file for the given caption chunks. Timestamps are
    rebased to 0 (relative to clip_start) since the source clip gets cut before
    captions are burned in.
    """
    alignment = CAPTION_POSITIONS.get(position, 5)

    # assign a color per speaker, in order of first appearance
    speaker_color_map = {}
    color_idx = 0

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,{outline_color},&H00000000,1,0,0,0,100,100,0,0,1,5,2,{alignment},60,60,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = [header]
    for chunk in chunks:
        start = _format_ass_time(chunk.start - clip_start)
        end = _format_ass_time(chunk.end - clip_start)
        if end <= start:
            continue

        if use_speaker_colors:
            if chunk.speaker not in speaker_color_map:
                speaker_color_map[chunk.speaker] = SPEAKER_COLOR_PALETTE[
                    color_idx % len(SPEAKER_COLOR_PALETTE)
                ]
                color_idx += 1
            color = speaker_color_map[chunk.speaker]
        else:
            color = default_color

        # \c override tag sets color per-line; small pop-in scale for flashiness
        text = f"{{\\c{color}&\\fscx105\\fscy105}}{chunk.text}"
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(
        "Wrote %d caption lines to %s (speakers=%d)",
        len(chunks), output_path, len(speaker_color_map) or 1,
    )
    return output_path
