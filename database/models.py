from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./explainly.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

Base = declarative_base()

# ── Sessions table ────────────────────────────────────────────────────────
class Session(Base):
    __tablename__ = "sessions"

    id                      = Column(Integer, primary_key=True)
    question                = Column(Text, nullable=False)
    student_link            = Column(String, unique=True)
    question_image_filename = Column(String, nullable=True)
    model_answer            = Column(Text, nullable=True)
    misconceptions          = Column(Text, nullable=True)
    common_errors           = Column(Text, nullable=True)
    created_at              = Column(DateTime, default=datetime.utcnow)

    # ── Question Bank columns ─────────────────────────────────────────────
    status                  = Column(String, default='draft')   # draft / published / closed
    published_at            = Column(DateTime, nullable=True)
    question_source         = Column(String, default='text')  # text / image / both

# ── Responses table ───────────────────────────────────────────────────────
class Response(Base):
    __tablename__ = "responses"

    id                      = Column(Integer, primary_key=True)
    session_id              = Column(Integer, nullable=False)
    student_name            = Column(String, nullable=False)

    # Submission mode
    submission_mode         = Column(String, nullable=True)  # voice_only / written_only / both

    # Files
    audio_filename          = Column(String, nullable=True)
    canvas_image_filename   = Column(String, nullable=True)
    uploaded_image_filename = Column(String, nullable=True)

    # Transcription
    transcript              = Column(Text, nullable=True)

    # AI Feedback — shown to student instantly
    score                   = Column(Integer)
    what_was_right          = Column(Text)
    what_to_improve         = Column(Text)
    ai_teacher_note         = Column(Text)
    language_detected       = Column(String)

    # Deeper AI analysis
    representations_used    = Column(Text, nullable=True)
    representation_strength = Column(String, nullable=True)
    misconception_flag      = Column(Boolean, default=False)
    careless_error_flag     = Column(Boolean, default=False)
    error_type              = Column(String, nullable=True)
    notation_errors         = Column(Text, nullable=True)

    # Teacher actions
    teacher_private_note    = Column(Text, nullable=True)
    follow_up_flag          = Column(Boolean, default=False)
    teacher_annotated       = Column(Boolean, default=False)

    submitted_at            = Column(DateTime, default=datetime.utcnow)

# ── Database setup ────────────────────────────────────────────────────────
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

Base.metadata.create_all(engine)

print("✅ Database tables created successfully!")