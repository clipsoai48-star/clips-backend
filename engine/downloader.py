"""Download source video from a YouTube/Twitch URL, or accept a local upload as-is."""
import os
import uuid
import logging

import yt_dlp

logger = logging.getLogger(__name__)


def download_source(url: str, out_dir: str) -> str:
    """
    Download a video from YouTube/Twitch (or any yt-dlp-supported site) into out_dir.
    Returns the local filepath of the downloaded video.
    """
    os.makedirs(out_dir, exist_ok=True)
    job_id = uuid.uuid4().hex[:12]
    out_template = os.path.join(out_dir, f"{job_id}.%(ext)s")

    ydl_opts = {
        "outtmpl": out_template,
        # cap resolution — we don't need 4k source for short-form clip generation,
        # this keeps download + ffmpeg processing time/cost down
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = ydl.prepare_filename(info)
        # merge_output_format may change the extension after postprocessing
        base, _ = os.path.splitext(filepath)
        mp4_path = base + ".mp4"
        if os.path.exists(mp4_path):
            return mp4_path
        return filepath


def register_local_upload(local_path: str) -> str:
    """
    For direct file uploads (not a URL), the caller already has the file on disk
    (e.g. saved from a multipart upload). This just validates and returns the path.
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Uploaded file not found: {local_path}")
    return local_path
