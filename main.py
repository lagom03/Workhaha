# main.py
# FastAPI：/records 寫入你的資料到 MySQL

from __future__ import annotations
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

from sqlalchemy.orm import Session
from sqlalchemy import select, desc

from db import (
    get_db,
    init_db,
    PostureRecord,
    GateEvent,
)

app = FastAPI(title="Posture API", version="1.0.0")

# 若你要給手機 App 直接打，先開放 CORS（視需求限制網域）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ⚠️ 正式環境請改成指定網域
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Pydantic 模型 ----------

class PostureIn(BaseModel):
    """
    你模型要 POST 的 payload（就是 9 個欄位）
    """
    timestamp_iso: str
    sacrum_class: Optional[str] = None
    sacrum_side: Optional[str] = None
    sacrum_protection: Optional[str] = None
    sacrum_confidence: Optional[float] = None
    heel_class: Optional[str] = None
    heel_confidence: Optional[float] = None
    camera_serial: Optional[str] = None
    camera_name: Optional[str] = None

    # 允許額外欄位（例如統計）
    model_config = ConfigDict(extra="allow")


class RecordCreate(BaseModel):
    """
    外層請求模型：
    - data: 上面 PostureIn（必填）
    - folder / image_paths: 可選
    """
    data: PostureIn
    folder: Optional[str] = None
    image_paths: Optional[Dict[str, str]] = Field(
        default=None,
        description="四張圖的相對路徑，可用以下鍵：rgb_masked_raw / rgb_masked_annotated / sacrum_depth_proc / heel_depth_proc",
    )


class RecordOut(BaseModel):
    id: int
    created_at: datetime
    timestamp_iso: Optional[datetime]
    sacrum_class: Optional[str]
    sacrum_side: Optional[str]
    sacrum_protection: Optional[str]
    sacrum_confidence: Optional[float]
    heel_class: Optional[str]
    heel_confidence: Optional[float]
    camera_serial: Optional[str]
    camera_name: Optional[str]
    folder: Optional[str]
    rgb_masked_raw: Optional[str]
    rgb_masked_annotated: Optional[str]
    sacrum_depth_proc: Optional[str]
    heel_depth_proc: Optional[str]


class GateEventIn(BaseModel):
    event: str
    info: Optional[Dict[str, Any]] = None


# ---------- 工具 ----------

def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    s = ts.strip()
    try:
        # 支援 '2025-09-02T16:46:32' 或 '...Z'
        if s.endswith("Z"):
            s = s[:-1]
        return datetime.fromisoformat(s)
    except Exception:
        return None


# ---------- 路由 ----------

@app.on_event("startup")
def on_startup():
    # 啟動就建表
    init_db()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/records", response_model=RecordOut)
def create_record(payload: RecordCreate, db: Session = Depends(get_db)):
    data_dict = payload.data.model_dump()
    event_dt = _parse_ts(data_dict.get("timestamp_iso"))

    rec = PostureRecord(
        timestamp_iso=event_dt,
        sacrum_class=data_dict.get("sacrum_class"),
        sacrum_side=data_dict.get("sacrum_side"),
        sacrum_protection=data_dict.get("sacrum_protection"),
        sacrum_confidence=data_dict.get("sacrum_confidence"),
        heel_class=data_dict.get("heel_class"),
        heel_confidence=data_dict.get("heel_confidence"),
        camera_serial=data_dict.get("camera_serial"),
        camera_name=data_dict.get("camera_name"),
        folder=payload.folder,
        rgb_masked_raw=(payload.image_paths or {}).get("rgb_masked_raw") if payload.image_paths else None,
        rgb_masked_annotated=(payload.image_paths or {}).get("rgb_masked_annotated") if payload.image_paths else None,
        sacrum_depth_proc=(payload.image_paths or {}).get("sacrum_depth_proc") if payload.image_paths else None,
        heel_depth_proc=(payload.image_paths or {}).get("heel_depth_proc") if payload.image_paths else None,
    )

    db.add(rec)
    db.flush()
    db.refresh(rec)

    return RecordOut.model_validate(rec)


@app.get("/records", response_model=List[RecordOut])
def list_records(limit: int = 50, db: Session = Depends(get_db)):
    stmt = (
        select(PostureRecord)
        .order_by(desc(PostureRecord.id))
        .limit(min(max(limit, 1), 200))
    )
    rows = db.execute(stmt).scalars().all()
    return [RecordOut.model_validate(r) for r in rows]


@app.post("/gate-events")
def create_gate_event(ev: GateEventIn, db: Session = Depends(get_db)):
    row = GateEvent(event=ev.event, info=ev.info)
    db.add(row)
    db.flush()
    return {"ok": True, "id": row.id}
