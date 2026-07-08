"""
Render a HighlightCandidate into a final output clip using ffmpeg.

Free tier: cut + vertical crop + burned-in captions.
Paid tier: adds transitions (intro/outro), background music bed, and a zoom-punch
effect keyed to caption emphasis words.
"""
import os
import subprocess
import logging
from typing import List

from .models import HighlightCandidate, RenderOptions, TranscriptSegment, SpeakerTurn
from .captions import build_word_chunks, build_ass_file

logger = logging.getLogger(__name__)

# Caption style presets — structured so they plug directly into captions.build_ass_file.
# Colors are &HAABBGGRR (alpha, blue, green, red — reversed vs normal hex).
# Free tier gets "basic" only; paid tier can pick any of these.
CAPTION_STYLES = {
    "basic": {
        "font_name": "Arial", "font_size": 54,
        "default_color": "&H00FFFFFF", "outline_color": "&H00000000",
    },
    "bold_yellow": {
        "font_name": "Arial Black", "font_size": 68,
        "default_color": "&H0000FFFF", "outline_color": "&H00000000",
    },
    "minimal_white": {
        "font_name": "Helvetica", "font_size": 56,
        "default_color": "&H00FFFFFF", "outline_color": "&H80000000",
    },
    "boxed": {
        "font_name": "Arial", "font_size": 56,
        "default_color": "&H00FFFFFF", "outline_color": "&H00000000",
    },
    "creator_pink": {
        "font_name": "Arial Black", "font_size": 64,
        "default_color": "&H00D355FF", "outline_color": "&H00000000",
    },
}





def render_clip(
    source_path: str,
    candidate: HighlightCandidate,
    segments: List[TranscriptSegment],
    output_path: str,
    options: RenderOptions,
    speaker_turns: List[SpeakerTurn] = None,
) -> str:
    """
    Cuts and renders a single clip. Returns the output filepath.

    Builds an ffmpeg filter graph in stages so free vs. paid features are just
    filters that get included or skipped based on `options`.
    """
    work_dir = os.path.dirname(output_path)
    os.makedirs(work_dir, exist_ok=True)
    ass_path = os.path.join(work_dir, f"{os.path.splitext(os.path.basename(output_path))[0]}.ass")

    video_filters = []

    # 1. Crop to vertical (center crop; a production system would run a
    #    face/motion tracker to choose the crop region per-frame — start simple).
    if options.vertical_crop:
        # Commas inside min(...) must be escaped with a backslash — ffmpeg's
        # filtergraph parser otherwise reads an unescaped comma as the
        # separator between filters, corrupting the whole chain.
        video_filters.append(
            f"crop=w='min(iw\\,ih*{options.output_width}/{options.output_height})'"
            f":h='min(ih\\,iw*{options.output_height}/{options.output_width})'"
        )
        video_filters.append(f"scale={options.output_width}:{options.output_height}")

    # 2. Paid-tier zoom punch — a subtle scale pulse for emphasis.
    if options.add_zoom_punch:
        video_filters.append(
            "zoompan=z='min(zoom+0.0007\\,1.08)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":s={options.output_width}x{options.output_height}"
        )

    # 3. Captions — short flashy word-burst chunks (not full sentences), burned
    #    in via an .ass file so we get per-line color + flexible positioning.
    if options.burn_captions:
        style = CAPTION_STYLES.get(options.caption_style, CAPTION_STYLES["basic"])
        chunks = build_word_chunks(
            segments, candidate.start, candidate.end,
            words_per_chunk=options.words_per_caption,
            speaker_turns=speaker_turns if options.speaker_colors else None,
        )
        build_ass_file(
            chunks, candidate.start, ass_path,
            video_width=options.output_width, video_height=options.output_height,
            font_name=style["font_name"], font_size=style["font_size"],
            position=options.caption_position,
            use_speaker_colors=options.speaker_colors and bool(speaker_turns),
            default_color=style["default_color"], outline_color=style["outline_color"],
        )
        # Colons in the path must be escaped for ffmpeg's filtergraph parser.
        escaped_ass = ass_path.replace(":", "\\:")
        video_filters.append(f"subtitles={escaped_ass}")

    vf_chain = ",".join(video_filters) if video_filters else "null"

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(candidate.start),
        "-to", str(candidate.end),
        "-i", source_path,
    ]

    # 4. Paid-tier background music bed, mixed under the original audio.
    #    Mutually exclusive with the football SFX overlay below — football jobs
    #    never set add_music, so only one of these branches fires.
    if options.add_music and options.music_track_path:
        cmd += ["-i", options.music_track_path]
        filter_complex = (
            f"[0:v]{vf_chain}[v];"
            f"[1:a]volume=0.15,aloop=loop=-1:size=2e9[music];"
            f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[a]"
        )
        cmd += [
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "[a]",
        ]
    elif options.sfx_path and candidate.sfx_offset_seconds is not None:
        # Football SFX: a one-shot sound effect layered on top of the original
        # audio at the moment the highlight peak occurs, no background music.
        cmd += ["-i", options.sfx_path]
        delay_ms = max(0, int(candidate.sfx_offset_seconds * 1000))
        filter_complex = (
            f"[0:v]{vf_chain}[v];"
            f"[1:a]adelay={delay_ms}|{delay_ms}[sfx];"
            f"[0:a][sfx]amix=inputs=2:duration=first:dropout_transition=2[a]"
        )
        cmd += [
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "[a]",
        ]
    else:
        cmd += ["-vf", vf_chain, "-map", "0:v", "-map", "0:a"]

    cmd += [
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]

    logger.info("Rendering clip %s: %s", output_path, " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def apply_transition_stitch(clip_paths: List[str], output_path: str, transition: str = "fade") -> str:
    """
    Paid-tier: stitch an intro/outro sting or crossfade between clips if the
    user is exporting a compilation rather than individual clips.
    Uses ffmpeg's xfade filter for a smooth transition between concatenated segments.
    """
    if len(clip_paths) < 2:
        raise ValueError("Need at least 2 clips to apply transitions between them")

    inputs = []
    for p in clip_paths:
        inputs += ["-i", p]

    # Simple pairwise xfade chain; duration/offset would be computed per-clip length
    # in a full implementation — this shows the ffmpeg pattern to build on.
    filter_parts = []
    prev = "0:v"
    for i in range(1, len(clip_paths)):
        out_label = f"v{i}"
        filter_parts.append(f"[{prev}][{i}:v]xfade=transition={transition}:duration=0.5:offset=0[{out_label}]")
        prev = out_label

    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filter_parts), "-map", f"[{prev}]", output_path]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path
