"""
Database models. Uses SQLite for local development — swap the DATABASE_URL
in database.py for a Postgres connection string when you deploy (e.g. Railway
or Supabase Postgres), no model changes needed since SQLAlchemy abstracts that.
"""
import uuid
import datetime
from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Float, Integer, Text
from sqlalchemy.orm import relationship
from database import Base


def _uuid() -> str:
    return uuid.uuid4().hex


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=_uuid)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_paid_tier = Column(Boolean, default=False, nullable=False)
    # Stripe fields — populated once billing is wired up. Left nullable so the
    # app works before Stripe is connected; is_paid_tier can be toggled
    # manually during development.
    stripe_customer_id = Column(String, nullable=True)
    stripe_subscription_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    jobs = relationship("ClipJobRecord", back_populates="user")


class ClipJobRecord(Base):
    __tablename__ = "clip_jobs"

    id = Column(String, primary_key=True, default=_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)

    # Input
    source_url = Column(String, nullable=True)   # set if submitted as a URL
    source_filename = Column(String, nullable=True)  # set if submitted as an upload

    # Requested options (mirrors engine.models.RenderOptions / ClipJob knobs)
    target_clip_count = Column(Integer, default=5)
    clip_length_seconds = Column(Float, default=30.0)
    caption_style = Column(String, default="basic")
    speaker_colors = Column(Boolean, default=False)
    use_llm_rerank = Column(Boolean, default=False)

    # Football feature fields
    job_type = Column(String, default="standard")  # "standard" or "football"
    sfx_choice = Column(String, nullable=True)  # e.g. "clip_01.mp3"

    # Status tracking
    status = Column(String, default="queued")  # queued -> processing -> done | failed
    progress = Column(Integer, default=0)  # 0-100, updated during processing
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="jobs")
    clips = relationship("ClipRecord", back_populates="job", cascade="all, delete-orphan")


class ClipRecord(Base):
    __tablename__ = "clips"

    id = Column(String, primary_key=True, default=_uuid)
    job_id = Column(String, ForeignKey("clip_jobs.id"), nullable=False)
    file_path = Column(String, nullable=False)   # local path (dev) or storage URL (prod)
    score = Column(Float, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    job = relationship("ClipJobRecord", back_populates="clips")
