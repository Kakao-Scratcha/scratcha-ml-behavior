#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, io
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ===== env =====
try:
    from dotenv import load_dotenv
    load_dotenv(os.getenv("BACKEND_ENV_FILE", ".env"))
except Exception:
    pass

def getenv_any(names, default=None):
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip() != "":
            return v
    return default

try:
    from zoneinfo import ZoneInfo
    KST = ZoneInfo("Asia/Seoul")
except Exception:
    KST = timezone(timedelta(hours=9))

ENV_NAME = getenv_any(["KS3_ENV", "ENV"], "dev")

# ===== KS3/S3 =====
KS3_ENDPOINT = getenv_any(["KS3_ENDPOINT", "S3_ENDPOINT_URL"], "")
KS3_REGION   = getenv_any(["KS3_REGION", "S3_REGION"], "ap-northeast-2")
KS3_BUCKET   = getenv_any(["KS3_BUCKET", "S3_BUCKET"], "")
KS3_ACCESS   = getenv_any(["KS3_ACCESS_KEY", "S3_ACCESS_KEY"], "")
KS3_SECRET   = getenv_any(["KS3_SECRET_KEY", "S3_SECRET_KEY"], "")
KS3_PREFIX   = getenv_any(["KS3_PREFIX", "S3_PREFIX"], "")
KS3_FORCE_PATH_STYLE = getenv_any(["KS3_FORCE_PATH_STYLE", "S3_FORCE_PATH_STYLE"], "1")

_ENABLE = getenv_any(["KS3_ENABLE"], None)
if _ENABLE is None:
    ENABLE_KS3 = all([KS3_BUCKET, KS3_ENDPOINT, KS3_ACCESS, KS3_SECRET])
else:
    ENABLE_KS3 = (_ENABLE == "1")

CORS_ORIGINS = [
    o.strip() for o in getenv_any(
        ["CORS_ORIGINS"],
        "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",") if o.strip()
]

def model_dump_compat(obj, *, exclude_none: bool = True):
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=exclude_none)
    if hasattr(obj, "dict"):
        return obj.dict(exclude_none=exclude_none)
    return obj

try:
    from pydantic import ConfigDict
    class BaseModelX(BaseModel):
        model_config = ConfigDict(extra="allow")
except Exception:
    class BaseModelX(BaseModel):
        class Config:
            extra = "allow"

# ====== Schemas ======
class Rect(BaseModelX):
    left: float; top: float; w: float; h: float
    version: Optional[int] = None

class SessionMeta(BaseModelX):
    device: Optional[str] = None
    viewport: Dict[str, float]
    dpr: Optional[float] = None
    roi: Optional[Rect] = None
    roi_map: Optional[Dict[str, Any]] = None
    ts_resolution_ms: Optional[int] = None
    session_id: str
    widget_version: Optional[str] = None

class PackedMoves(BaseModelX):
    base_t: int
    dts: List[int]
    xrs: List[float]
    yrs: List[float]
    xs: Optional[List[float]] = None
    ys: Optional[List[float]] = None
    oobs: Optional[List[int]] = None
    on_canvas: Optional[List[int]] = None

class EventItem(BaseModelX):
    t: Optional[int] = None
    type: str
    x: Optional[float] = None; y: Optional[float] = None
    x_raw: Optional[float] = None; y_raw: Optional[float] = None
    on_canvas: Optional[int] = None; oob: Optional[int] = None
    pointerType: Optional[str] = None; pointerId: Optional[int] = None
    is_trusted: Optional[int] = None
    target_role: Optional[str] = None; target_answer: Optional[str] = None
    payload: Optional[PackedMoves] = None
    free: Optional[int] = None

class CollectRequest(BaseModelX):
    meta: SessionMeta
    events: List[EventItem]
    label: Optional[Dict[str, Any]] = None

class LabelPatch(BaseModelX):
    session_id: str
    label: Dict[str, Any]

# ====== KS3 helpers ======
def _ks3_client():
    import boto3
    from botocore.config import Config
    cfg = Config(
        s3={"addressing_style": "path" if KS3_FORCE_PATH_STYLE == "1" else "virtual"},
        signature_version="s3v4",
        retries={"max_attempts": 3, "mode": "standard"},
    )
    session = boto3.session.Session(
        aws_access_key_id=KS3_ACCESS,
        aws_secret_access_key=KS3_SECRET,
        region_name=KS3_REGION or "ap-northeast-2",
    )
    return session.client("s3", endpoint_url=KS3_ENDPOINT, config=cfg)

def _make_session_key(session_id: str, gz: bool = True) -> str:
    ts = datetime.now(KST).strftime("%Y%m%d-%H%M%S")
    fname = f"{ts}_{session_id}.json" + (".gz" if gz else "")
    return f"{KS3_PREFIX.strip('/')}/{fname}".strip("/")

def _serialize_jsonl_bytes(payload: CollectRequest) -> bytes:
    meta = model_dump_compat(payload.meta, exclude_none=True)
    events = [model_dump_compat(e, exclude_none=True) for e in payload.events]
    lines = [json.dumps({"type": "meta", **meta}, ensure_ascii=False)]
    for ev in events:
        lines.append(json.dumps({"type": "event", **ev}, ensure_ascii=False))
    if payload.label:
        lines.append(json.dumps({"type": "label", **payload.label}, ensure_ascii=False))
    return ("\n".join(lines) + "\n").encode("utf-8")

def _gzip_bytes(raw: bytes) -> bytes:
    buf = io.BytesIO()
    import gzip as _gz
    with _gz.GzipFile(fileobj=buf, mode="wb", compresslevel=6, mtime=0) as gz:
        gz.write(raw)
    return buf.getvalue()

def upload_ks3_session(payload: CollectRequest, session_id: str):
    if not ENABLE_KS3 or not KS3_BUCKET or not KS3_ACCESS or not KS3_SECRET or not KS3_ENDPOINT:
        missing = []
        if not ENABLE_KS3:   missing.append("KS3_ENABLE(auto)==False")
        if not KS3_BUCKET:   missing.append("KS3_BUCKET")
        if not KS3_ACCESS:   missing.append("KS3_ACCESS_KEY")
        if not KS3_SECRET:   missing.append("KS3_SECRET_KEY")
        if not KS3_ENDPOINT: missing.append("KS3_ENDPOINT")
        return (None, None, f"Missing: {', '.join(missing)}")
    try:
        body = _serialize_jsonl_bytes(payload)
        gz   = _gzip_bytes(body)
        key  = _make_session_key(session_id, gz=True)
        s3 = _ks3_client()
        s3.put_object(
            Bucket=KS3_BUCKET, Key=key, Body=gz,
            ContentType="application/json", ContentEncoding="gzip",
        )
        return (f"s3://{KS3_BUCKET}/{key}", key, len(gz))
    except Exception as e:
        return (None, None, f"upload error: {e}")

# ====== FastAPI ======
app = FastAPI(title="Scratcha Collector + Inline Verify", version="3.4.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Model (lazy loader + shared entrypoints) ======
import numpy as np
import torch
import torch.nn as nn
from threading import Lock

class CNN1D(nn.Module):
    def __init__(self, in_ch=7, c1=96, c2=192, dropout=0.2, input_bn=True):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_ch) if input_bn else nn.Identity()
        self.feat = nn.Sequential(
            nn.Conv1d(in_ch, c1, kernel_size=7, padding=3),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(c2, 1)
    def forward(self, x):  # (B,7,T)
        x = self.input_bn(x)
        h = self.feat(x)
        h = h.mean(dim=-1)
        return self.head(h).squeeze(1)

BASE_DIR = Path(__file__).resolve().parent
ART_DIR  = (BASE_DIR / ".." / "artifacts" / "cnn").resolve()
BEST_PT  = ART_DIR / "best.pt"
THR_JSON = ART_DIR / "thresholds.json"
CALIB_JSON = ART_DIR / "calibration.json"

_MODEL: Optional[nn.Module] = None
_THRESHOLD: Optional[float] = None
_DEVICE = "cpu"
_MODEL_LOCK = Lock()

def _load_threshold_once() -> float:
    global _THRESHOLD
    if _THRESHOLD is not None:
        return _THRESHOLD
    try:
        with open(THR_JSON, "r", encoding="utf-8") as f:
            _THRESHOLD = float(json.load(f).get("val_threshold", 0.5))
    except Exception as e:
        print(f"[WARN] thresholds.json load failed: {e}")
        _THRESHOLD = 0.5
    return _THRESHOLD

def get_threshold() -> float:
    return _load_threshold_once()

def get_model() -> Optional[nn.Module]:
    """필요 시점에만 안전하게 로딩. 실패해도 다음 요청에서 재시도."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            m = CNN1D(in_ch=7, c1=96, c2=192, dropout=0.2, input_bn=True)
            state = torch.load(BEST_PT, map_location=_DEVICE)
            m.load_state_dict(state, strict=True)
            m.eval()
            _MODEL = m
            print(f"[OK] Loaded best.pt from {BEST_PT}")
        except Exception as e:
            print(f"[WARN] best.pt load failed: {e}")
            _MODEL = None
        return _MODEL

# ====== Calibration (temperature / platt) ======
_CALIB = None
_CALIB_MTIME = None

def _load_calibration():
    """
    calibration.json mtime을 보고 자동 리로드.
    지원: {"type":"temperature","T":...}  |  {"type":"platt","a":...,"b":...}
    """
    global _CALIB, _CALIB_MTIME
    try:
        st = CALIB_JSON.stat()
        if _CALIB is not None and _CALIB_MTIME == st.st_mtime:
            return _CALIB  # 캐시 유효
        with open(CALIB_JSON, "r", encoding="utf-8") as f:
            obj = json.load(f)
        t = str(obj.get("type", "")).lower()
        if t == "temperature":
            _CALIB = ("temperature", float(obj["T"]))
        elif t == "platt":
            _CALIB = ("platt", float(obj["a"]), float(obj["b"]))
        else:
            _CALIB = None
        _CALIB_MTIME = st.st_mtime
    except Exception:
        _CALIB = None
        _CALIB_MTIME = None
    return _CALIB

@app.post("/reload_calibration")
async def reload_calibration():
    """수동 리로드 엔드포인트(운영 중 교체 시 사용)"""
    global _CALIB, _CALIB_MTIME
    _CALIB = None
    _CALIB_MTIME = None
    cal = _load_calibration()
    return {"ok": True, "loaded": (cal is not None)}

# ====== 전처리: 외부 모듈 ======
from behavior_features import build_window_7ch, seq_stats

def run_inference(meta: SessionMeta, events: List[EventItem]):
    model = get_model()
    if model is None:
        return {"ok": False, "error": "model not loaded"}

    # (전처리) canvas 기준 OOB만 모델 입력으로 사용
    X, raw_len, has_track, has_wrap, oob_c, oob_w = build_window_7ch(meta, events, T=300)
    if X is None:
        return {"ok": False, "error": "empty or invalid events/roi"}

    # (추론: 원시 logit)
    xt = torch.from_numpy(np.transpose(X, (1,0))).unsqueeze(0).float()  # (1,7,300)
    with torch.no_grad():
        logit = model(xt).item()

    # (후처리) Calibration 우선순위: temperature → platt
    calib = _load_calibration()
    if calib and calib[0] == "temperature":
        T = max(1.0, float(calib[1]))   # ← 안전: 최소 1.0로 고정
        z = logit / T
    elif calib and calib[0] == "platt":
        a, b = float(calib[1]), float(calib[2])
        z = a * logit + b
    else:
        T = 2.0  # 기본 Temperature
        z = logit / T

    z_raw = float(z)
    z_clip = float(np.clip(z_raw, -3.0, 3.0))  # 시그모이드 전 클립
    prob = float(1.0 / (1.0 + np.exp(-z_clip)))

    thr = float(get_threshold())
    verdict = "bot" if prob >= thr else "human"
    return {
        "ok": True,
        "model": "cnn",
        "bot_prob": prob,
        "threshold": thr,
        "verdict": verdict,
        "stats": seq_stats(X, raw_len, has_track, has_wrap, oob_c, oob_w),
        "debug": {"logit": float(logit), "calib": calib, "z_raw": z_raw, "z_clip": z_clip},
    }

# ====== Endpoints ======
@app.post("/collect")
async def collect(payload: CollectRequest):
    session_id = payload.meta.session_id
    infer = run_inference(payload.meta, payload.events)
    lbl = dict(payload.label or {})
    lbl["verify_ok"] = 1 if infer.get("ok") else 0
    lbl["model"] = infer.get("model", "cnn")

    patched = CollectRequest(meta=payload.meta, events=payload.events, label=lbl)
    ks3_uri, key, size_or_err = upload_ks3_session(patched, session_id)

    if ks3_uri:
        return {"ok": True, "ks3": ks3_uri, "key": key, "size": size_or_err, "verification": infer}
    return {"ok": False, "error": size_or_err, "verification": infer}

_CHUNK_BUF: Dict[str, Dict[str, Any]] = {}

@app.post("/collect_chunk")
async def collect_chunk(payload: CollectRequest):
    sid = payload.meta.session_id
    rec = _CHUNK_BUF.get(sid)
    if rec is None:
        rec = {"meta": model_dump_compat(payload.meta, exclude_none=True), "events": [], "parts": 0}
        _CHUNK_BUF[sid] = rec
    for ev in payload.events:
        rec["events"].append(model_dump_compat(ev, exclude_none=True))
    rec["parts"] += 1
    return {"ok": True, "session_id": sid, "parts": rec["parts"]}

@app.post("/collect_finalize")
async def collect_finalize(payload: CollectRequest):
    sid = payload.meta.session_id
    rec = _CHUNK_BUF.get(sid)
    if rec is None:
        infer = run_inference(payload.meta, payload.events)
        lbl = dict(payload.label or {})
        lbl["verify_ok"] = 1 if infer.get("ok") else 0
        lbl["model"] = infer.get("model", "cnn")
        patched = CollectRequest(meta=payload.meta, events=payload.events, label=lbl)
        ks3_uri, key, size_or_err = upload_ks3_session(patched, sid)
        if ks3_uri:
            return {"ok": True, "ks3": ks3_uri, "key": key, "verification": infer}
        return {"ok": False, "error": size_or_err, "verification": infer}

    meta = rec["meta"]
    events = rec["events"]
    full = CollectRequest(
        meta=SessionMeta(**meta),
        events=[EventItem(**e) for e in events],
        label=model_dump_compat(payload.label, exclude_none=True) if payload.label else None
    )

    infer = run_inference(full.meta, full.events)
    lbl = dict(full.label or {})
    lbl["verify_ok"] = 1 if infer.get("ok") else 0
    lbl["model"] = infer.get("model", "cnn")
    full = CollectRequest(meta=full.meta, events=full.events, label=lbl)

    ks3_uri, key, size_or_err = upload_ks3_session(full, sid)
    try: del _CHUNK_BUF[sid]
    except Exception: pass

    if ks3_uri:
        return {"ok": True, "ks3": ks3_uri, "key": key, "verification": infer}
    return {"ok": False, "error": size_or_err, "verification": infer}

@app.get("/healthz")
async def healthz():
    cal = _load_calibration()  # 강제 로드하여 상태 반영
    return {
        "ok": True,
        "env": ENV_NAME,
        "model_loaded": (get_model() is not None),
        "thr": get_threshold(),
        "calibration_json": str(CALIB_JSON),
        "calib": cal,
        "best_pt": str(BEST_PT),
    }

@app.post("/collect_raw")
async def collect_raw(req: Request):
    body = await req.json()
    return {"ok": True, "echo": body}
