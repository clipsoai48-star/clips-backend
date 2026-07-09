"""
Clipso AI backend — API server.

Run with: uvicorn main:app --reload
(Requires a separate `rq worker clipso_jobs` process running too — see README.)
"""
import os
import shutil
import logging

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from redis import Redis
from rq import Queue

from database import get_db, init_db
from models_db import User, ClipJobRecord
from schemas import (
    SignupRequest, LoginRequest, TokenResponse, UserResponse,
    CreateJobRequest, JobResponse,
)
from auth import hash_password, verify_password, create_access_token, get_current_user
from worker import process_clip_job

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

UPLOAD_DIR = os.environ.get("CLIPSO_UPLOAD_DIR", "./storage/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("./storage/outputs", exist_ok=True)

app = FastAPI(title="Clipso AI API")

# Loosened for local dev so the frontend (running on a different port) can
# call this API. Tighten this to your real frontend domain before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves rendered clip files directly (e.g. /storage/outputs/<job>/<clip>.mp4)
# so the frontend's <video> tags can load them. Local-disk only — swap for
# S3/R2 signed URLs before real users (see README).
app.mount("/storage", StaticFiles(directory="storage"), name="storage")

redis_conn = Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
job_queue = Queue("clipso_jobs", connection=redis_conn)
priority_job_queue = Queue("clipso_jobs_priority", connection=redis_conn)  # pro users


@app.on_event("startup")
def on_startup():
    init_db()


# ---------- Auth ----------

@app.post("/auth/signup", response_model=TokenResponse)
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="An account with this email already exists")

    user = User(email=payload.email, hashed_password=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id)
    return TokenResponse(access_token=token)


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    token = create_access_token(user.id)
    return TokenResponse(access_token=token)


@app.get("/auth/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user


# ---------- Jobs ----------

@app.post("/jobs/upload", response_model=JobResponse)
def create_job_from_upload(
    file: UploadFile = File(...),
    target_clip_count: int = Form(5),
    clip_length_seconds: float = Form(30.0),
    caption_style: str = Form("basic"),
    speaker_colors: bool = Form(False),
    use_llm_rerank: bool = Form(False),
    job_type: str = Form("standard"),
    sfx_choice: str = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a job from a directly uploaded video file (drag-and-drop upload)."""
    safe_filename = os.path.basename(file.filename)  # strip any directory components
    saved_path = os.path.join(UPLOAD_DIR, f"{current_user.id}_{safe_filename}")
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job_record = _create_job_record(
        db, current_user, source_url=None, source_filename=saved_path,
        target_clip_count=target_clip_count, clip_length_seconds=clip_length_seconds,
        caption_style=caption_style, speaker_colors=speaker_colors, use_llm_rerank=use_llm_rerank,
        job_type=job_type, sfx_choice=sfx_choice,
    )
    return job_record


@app.post("/jobs", response_model=JobResponse)
def create_job_from_url(
    payload: CreateJobRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a job from a YouTube/Twitch URL."""
    if not payload.source_url:
        raise HTTPException(status_code=400, detail="source_url is required for this endpoint")

    job_record = _create_job_record(
        db, current_user, source_url=payload.source_url, source_filename=None,
        target_clip_count=payload.target_clip_count, clip_length_seconds=payload.clip_length_seconds,
        caption_style=payload.caption_style, speaker_colors=payload.speaker_colors,
        use_llm_rerank=payload.use_llm_rerank,
        job_type=payload.job_type, sfx_choice=payload.sfx_choice,
    )
    return job_record


def _create_job_record(db, user, source_url, source_filename, target_clip_count,
                        clip_length_seconds, caption_style, speaker_colors, use_llm_rerank,
                        job_type="standard", sfx_choice=None) -> ClipJobRecord:
    # Enforce tier limits server-side — a free-tier user requesting a paid
    # caption style or speaker colors silently gets downgraded to what
    # they're actually entitled to, rather than trusting the request body.
    max_clips = 20 if user.is_paid_tier else 10
    if target_clip_count < 1:
        target_clip_count = 1
    elif target_clip_count > max_clips:
        logger.info(
            "Clamping target_clip_count from %d to %d for user %s (is_paid_tier=%s)",
            target_clip_count, max_clips, user.id, user.is_paid_tier,
        )
        target_clip_count = max_clips

    if not user.is_paid_tier:
        if caption_style != "basic":
            logger.info("Downgrading caption_style to 'basic' for free-tier user %s", user.id)
            caption_style = "basic"
        if speaker_colors:
            speaker_colors = False

    job_record = ClipJobRecord(
        user_id=user.id,
        source_url=source_url,
        source_filename=source_filename,
        target_clip_count=target_clip_count,
        clip_length_seconds=clip_length_seconds,
        caption_style=caption_style,
        speaker_colors=speaker_colors,
        use_llm_rerank=use_llm_rerank,
        job_type=job_type,
        sfx_choice=sfx_choice,
        status="queued",
    )
    db.add(job_record)
    db.commit()
    db.refresh(job_record)

    # Pro users' jobs go to the priority queue, which the worker drains first.
    queue = priority_job_queue if user.is_paid_tier else job_queue
    queue.enqueue(process_clip_job, job_record.id, job_timeout=1800)  # 30 min ceiling
    return job_record


@app.get("/jobs", response_model=list[JobResponse])
def list_jobs(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return (
        db.query(ClipJobRecord)
        .filter(ClipJobRecord.user_id == current_user.id)
        .order_by(ClipJobRecord.created_at.desc())
        .all()
    )


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job_record = db.query(ClipJobRecord).filter(ClipJobRecord.id == job_id).first()
    if job_record is None or job_record.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_record


@app.get("/health")
def health():
    return {"status": "ok"}
