"""
Lightweight active-speaker face tracking for the vertical-crop pipeline.

WHAT THIS DOES
Samples frames a few times per second across a clip, detects faces with
OpenCV's built-in Haar cascade (fast, ships with opencv-python, no model
download needed), and picks the "likely speaking" face using a simple
mouth-motion heuristic: the lower third of each detected face is compared
frame-to-frame, and whichever face has the most movement there is assumed
to be talking. That face's horizontal center becomes a waypoint; waypoints
are smoothed over time so the crop pans instead of jump-cutting.

WHAT THIS DELIBERATELY DOES NOT DO
- No face recognition / identity tracking across clips.
- No audio-to-face speaker attribution (diarization tells you *who*, not
  *where* — this module only ever answers "where", using vision alone).
- No GPU/deep-learning model — Haar cascades are CPU-fast, which matters
  since this runs inline in the render path per clip.

FALLBACK BEHAVIOR
If no faces are detected in a clip (screen recording, no webcam, etc.),
this returns an empty waypoint list and the caller should fall back to a
static center crop — exactly the previous behavior.
"""
import logging
from typing import List, Tuple

import cv2

logger = logging.getLogger(__name__)

# Sample this many times per second. Higher = smoother tracking but slower
# to compute; 2/sec is enough to follow natural speaker turn-taking without
# meaningfully slowing down rendering.
SAMPLE_FPS = 2.0

# How much a face's mouth-region has to move (0-255 avg pixel diff) before
# we consider it "possibly speaking" rather than just camera/encoding noise.
MOTION_THRESHOLD = 6.0

# Smoothing factor for the exponential moving average applied to the chosen
# crop-center-x across samples. Lower = smoother/slower to react, higher =
# snappier but more jittery. 0.35 favors stability over responsiveness,
# since jittery pans look worse than a slightly-late pan.
SMOOTHING_ALPHA = 0.35

_face_cascade = None


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def track_active_speaker_crop(
    source_path: str,
    start: float,
    end: float,
    sample_fps: float = SAMPLE_FPS,
) -> List[Tuple[float, float]]:
    """
    Returns a list of (relative_time_seconds, crop_center_x_fraction)
    waypoints for the clip window [start, end) in the source video.

    relative_time_seconds is relative to `start` (i.e. starts near 0),
    matching how ffmpeg's `t` behaves when we input-seek with -ss before -i.

    crop_center_x_fraction is 0.0-1.0, the horizontal center of the chosen
    face as a fraction of the original frame width. Caller converts this to
    an absolute pixel x-offset once it knows the crop window width.

    Returns [] if no faces were ever detected — caller should fall back to
    a static center crop in that case.
    """
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        logger.warning("face_tracker: could not open %s, falling back to center crop", source_path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    start_frame = int(start * fps)
    end_frame = int(end * fps)
    step_frames = max(1, int(fps / sample_fps))

    cascade = _get_face_cascade()
    waypoints: List[Tuple[float, float]] = []
    prev_gray_by_face = {}  # rough face slot index -> previous mouth-region grayscale crop

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

        if len(faces) > 0:
            best_face = None
            best_motion = -1.0

            for i, (x, y, w, h) in enumerate(faces):
                # Lower third of the face box ~= mouth/jaw region.
                mouth_y = y + int(h * 0.65)
                mouth_h = max(1, int(h * 0.30))
                mouth_region = gray[mouth_y:mouth_y + mouth_h, x:x + w]

                motion_score = 0.0
                prev_region = prev_gray_by_face.get(i)
                if prev_region is not None and prev_region.shape == mouth_region.shape:
                    diff = cv2.absdiff(prev_region, mouth_region)
                    motion_score = float(diff.mean())
                prev_gray_by_face[i] = mouth_region

                # Prefer whichever face is moving most; if nothing clears the
                # noise threshold, fall back to the largest face in frame
                # (usually the closest/most prominent speaker on camera).
                effective_score = motion_score if motion_score >= MOTION_THRESHOLD else (w * h) / 1_000_000

                if effective_score > best_motion:
                    best_motion = effective_score
                    best_face = (x, y, w, h)

            if best_face is not None:
                x, y, w, h = best_face
                center_x_fraction = (x + w / 2) / frame_width
                relative_time = (frame_idx - start_frame) / fps
                waypoints.append((relative_time, center_x_fraction))

        frame_idx += step_frames
        if step_frames > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    cap.release()

    if not waypoints:
        logger.info("face_tracker: no faces detected in %.1f-%.1f, falling back to center crop", start, end)
        return []

    return _smooth_waypoints(waypoints, alpha=SMOOTHING_ALPHA)


def _smooth_waypoints(
    waypoints: List[Tuple[float, float]],
    alpha: float,
) -> List[Tuple[float, float]]:
    """Exponential moving average over the x-center values so the crop pans
    smoothly instead of snapping between faces on every sample."""
    smoothed = []
    ema = waypoints[0][1]
    for t, x in waypoints:
        ema = alpha * x + (1 - alpha) * ema
        smoothed.append((t, ema))
    return smoothed


def build_ffmpeg_x_expression(
    waypoints: List[Tuple[float, float]],
    source_frame_width: int,
    crop_width: int,
) -> str:
    """
    Turns a list of (time, center_x_fraction) waypoints into an ffmpeg
    filtergraph expression for the crop filter's `x` parameter, so the crop
    window's left edge pans over time to keep the chosen face centered.

    Builds a piecewise `if(lt(t,T1),X1,if(lt(t,T2),X2,...))` expression,
    clamping so the crop window never goes outside the source frame.
    """
    if not waypoints:
        # Static center crop — same as the previous default behavior.
        return f"(iw-{crop_width})/2"

    max_x = source_frame_width - crop_width

    def clamp_px(center_fraction: float) -> int:
        px = int(center_fraction * source_frame_width - crop_width / 2)
        return max(0, min(max_x, px))

    # Build nested if() expression, most-recent-first so lt(t, T) short-circuits
    # correctly as ffmpeg evaluates left-to-right.
    expr = str(clamp_px(waypoints[-1][1]))
    for t, x_fraction in reversed(waypoints[:-1]):
        expr = f"if(lt(t\\,{t:.3f})\\,{clamp_px(x_fraction)}\\,{expr})"

    return expr