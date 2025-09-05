# db.py
# SQLAlchemy 連線/模型與 init_db()

from __future__ import annotations
import os
from datetime import datetime
from typing import Generator, Optional, Dict, Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, DateTime, BigInteger
from sqlalchemy.dialects.mysql import JSON as MySQLJSON  # MySQL 8 原生 JSON

# 讀 .env
BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL 未設定。請在 api/.env 放入例如：\n"
        "DATABASE_URL=mysql+pymysql://root:你的密碼@203.64.84.39:33067/test?charset=utf8mb4"
    )

# 建 engine / Session
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,     # 自動偵測失效連線
    pool_recycle=1800,      # 30 分鐘 recycle
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False, future=True)


class Base(DeclarativeBase):
    pass


class PostureRecord(Base):
    """
    Flattened posture record:
    - Direct columns instead of nested payload JSON
    - Includes metadata and image paths
    """
    __tablename__ = "posture_records"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )

    # metadata fields
    timestamp_iso: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, index=True)
    sacrum_class: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    sacrum_side: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    sacrum_protection: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    sacrum_confidence: Mapped[Optional[float]] = mapped_column(nullable=True)
    heel_class: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    heel_confidence: Mapped[Optional[float]] = mapped_column(nullable=True)
    camera_serial: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    camera_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    # folder info
    folder: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # image paths (flattened)
    rgb_masked_raw: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    rgb_masked_annotated: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    sacrum_depth_proc: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    heel_depth_proc: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)



class GateEvent(Base):
    """
    到場守門（yolo_arrival_gate_only）可以寫入的事件表（可選）
    例如 event='arrival'、'timeout_no_nurse' 等，info 為附加 JSON
    """
    __tablename__ = "gate_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )
    event: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    info: Mapped[Optional[Dict[str, Any]]] = mapped_column(MySQLJSON, nullable=True)


def get_db() -> Generator:
    """FastAPI 依賴：每次請求提供一個 DB session。"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """啟動時建表（若不存在）。"""
    Base.metadata.create_all(bind=engine)

def save_posture_record(metadata, folder, image_paths):
    db = SessionLocal()
    try:
        record = PostureRecord(
            timestamp_iso=datetime.fromisoformat(metadata["timestamp_iso"]),
            sacrum_class=metadata.get("sacrum_class"),
            sacrum_side=metadata.get("sacrum_side"),
            sacrum_protection=metadata.get("sacrum_protection"),
            sacrum_confidence=metadata.get("sacrum_confidence"),
            heel_class=metadata.get("heel_class"),
            heel_confidence=metadata.get("heel_confidence"),
            camera_serial=metadata.get("camera_serial"),
            camera_name=metadata.get("camera_name"),
            folder=folder,
            rgb_masked_raw=image_paths.get("rgb_masked_raw"),
            rgb_masked_annotated=image_paths.get("rgb_masked_annotated"),
            sacrum_depth_proc=image_paths.get("sacrum_depth_proc"),
            heel_depth_proc=image_paths.get("heel_depth_proc"),
        )
        db.add(record)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
