"""
Database connection setup. SQLite by default for local dev — a single file,
no separate database server to install. To move to Postgres later (recommended
before real users), just change DATABASE_URL and add `psycopg2-binary` to
requirements.txt; nothing else in the app needs to change.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./clipso.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a DB session and ensures it's closed after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Creates all tables if they don't exist yet. Called once on app startup."""
    import models_db  # noqa: F401 — ensures models are registered with Base before create_all
    Base.metadata.create_all(bind=engine)
