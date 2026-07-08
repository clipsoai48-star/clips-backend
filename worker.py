"""
The actual background job. Enqueued by the API (see main.py's POST /jobs),
picked up by an RQ worker process (run via `rq worker` — see README).

Runs entirely separately from the web server process, which is why video
processing doesn't block HTTP requests or time out.
"""
import os
import logging
import datetime

from database import SessionLocal
from models_db import ClipJobRecord, ClipRecord, User
from engine.models import ClipJob
from engine.downloader import download_source, register_local_upload
from engine.pipeline import run_pipeline

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.environ.get("CLIPSO_OUTPUT_DIR", "./storage/outputs")


def process_clip_job(job_record_id: str) -> None:
    """
    This is the function RQ actually calls. Takes just the job ID (not the
    full object) since RQ serializes arguments — always re-fetch fresh state
    from the DB inside the task rather than passing ORM objects across the
    queue boundary.
    """
    db = SessionLocal()
    try:
        job_record = db.query(ClipJobRecord).filter(ClipJobRecord.id == job_record_id).first()
        if job_record is None:
            logger.error("Job record %s not found", job_record_id)
            return

        user = db.query(User).filter(User.id == job_record.user_id).first()

        job_record.status = "processing"
        job_record.progress = 0
        db.commit()

        def _update_progress(pct: int):
            # Re-fetch inside the callback so we always write to a live session
            # row rather than holding a stale reference across the whole job.
            job_record.progress = pct
            db.commit()

        job_output_dir = os.path.join(OUTPUT_DIR, job_record.id)
        os.makedirs(job_output_dir, exist_ok=True)

        # Resolve the source video — either download it fresh from a URL, or
        # point at the already-saved upload.
        if job_record.source_url:
            source_path = download_source(job_record.source_url, out_dir=os.path.join(job_output_dir, "_source"))
        elif job_record.source_filename:
            source_path = register_local_upload(job_record.source_filename)
        else:
            raise ValueError("Job has neither source_url nor source_filename set")

        engine_job = ClipJob(
            job_id=job_record.id,
            source_path=source_path,
            output_dir=job_output_dir,
            target_clip_count=job_record.target_clip_count,
            max_clip_seconds=job_record.clip_length_seconds,
            min_clip_seconds=min(15.0, job_record.clip_length_seconds),
            # Re-check the user's real subscription status from the DB right
            # now, at execution time — never trust a stale value stored on
            # the job record, in case they cancelled between submitting and
            # the worker picking it up.
            is_paid_tier=bool(user and user.is_paid_tier),
            job_type=job_record.job_type,
            sfx_choice=job_record.sfx_choice,
        )

        output_paths = run_pipeline(
            engine_job,
            use_llm_rerank=job_record.use_llm_rerank,
            caption_style_override=job_record.caption_style,
            speaker_colors=job_record.speaker_colors,
            progress_callback=_update_progress,
        )

        for path in output_paths:
            db.add(ClipRecord(job_id=job_record.id, file_path=path))

        job_record.status = "done"
        job_record.completed_at = datetime.datetime.utcnow()
        db.commit()
        logger.info("Job %s completed with %d clip(s)", job_record.id, len(output_paths))

    except Exception as e:
        logger.exception("Job %s failed", job_record_id)
        job_record = db.query(ClipJobRecord).filter(ClipJobRecord.id == job_record_id).first()
        if job_record:
            job_record.status = "failed"
            job_record.error_message = str(e)
            db.commit()
    finally:
        db.close()
