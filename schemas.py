"""Pydantic schemas — define the shape of API requests/responses, separate from the DB models."""
import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str
    is_paid_tier: bool

    class Config:
        from_attributes = True


class CreateJobRequest(BaseModel):
    source_url: Optional[str] = None
    # source_filename is set separately via the upload endpoint, not here
    target_clip_count: int = 5
    clip_length_seconds: float = 30.0
    caption_style: str = "basic"
    speaker_colors: bool = False
    use_llm_rerank: bool = False


class ClipResponse(BaseModel):
    id: str
    file_path: str
    score: Optional[float]
    duration_seconds: Optional[float]

    class Config:
        from_attributes = True


class JobResponse(BaseModel):
    id: str
    status: str
    error_message: Optional[str]
    source_url: Optional[str]
    caption_style: str
    created_at: datetime.datetime
    completed_at: Optional[datetime.datetime]
    clips: List[ClipResponse] = []

    class Config:
        from_attributes = True
