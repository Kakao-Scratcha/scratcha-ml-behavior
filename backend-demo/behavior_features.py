# behavior_features.py
# 전처리 전담 모듈 (canvas 기준 OOB를 강제)

from typing import Any, List, Optional, Tuple
import numpy as np

# ---------- ROI 유틸 ----------
def _to_rect(d):
    try:
        L, T, W, H = float(d["left"]), float(d["top"]), float(d["w"]), float(d["h"])
        if W <= 0 or H <= 0:
            return None
        return (L, T, W, H)
    except Exception:
        return None

def _roi_rects(meta: Any) -> Tuple[Optional[Tuple[float,float,float,float]], Optional[Tuple[float,float,float,float]]]:
    """
    rect_track:  정규화 및 모델 입력 OOB(=canvas 기준)   -> 폴백 금지
    rect_oob:    통계용 wrapper 기준 (없으면 None)
    """
    rmap = (getattr(meta, "roi_map", None) or {})
    rect_canvas = _to_rect(rmap.get("canvas-container")) if rmap.get("canvas-container") else None
    rect_wrap   = _to_rect(rmap.get("scratcha-container")) if rmap.get("scratcha-container") else None
    # 🔒 canvas 기준을 강제: canvas가 없으면 track 없음으로 간주
    rect_track = rect_canvas
    rect_oob   = rect_wrap
    return rect_track, rect_oob

# ---------- 이벤트 평탄화 ----------
def _flatten_events(meta: Any, events: List[Any]):
    out = []
    for ev in events:
        et = getattr(ev, "type", None)
        if et in ("moves", "moves_free"):
            p = getattr(ev, "payload", None)
            if not p:
                continue
            base = int(getattr(p, "base_t", 0) or 0)
            dts  = list(getattr(p, "dts", []) or [])
            xs   = list(getattr(p, "xrs", []) or [])
            ys   = list(getattr(p, "yrs", []) or [])
            t = base
            n = min(len(dts), len(xs), len(ys))
            for i in range(n):
                out.append((t, float(xs[i]), float(ys[i])))
                dt = int(dts[i]) if int(dts[i]) > 0 else 1
                t += dt
        elif et in ("pointerdown", "pointerup", "click"):
            t  = getattr(ev, "t", None)
            xr = getattr(ev, "x_raw", None)
            yr = getattr(ev, "y_raw", None)
            if t is None or xr is None or yr is None:
                continue
            out.append((int(t), float(xr), float(yr)))
    out.sort(key=lambda x: x[0])
    return out

# ---------- 시간 단위 보정 (sec/ms/us → ms) ----------
def _fix_time_units_to_ms(ts_ms_like: np.ndarray) -> np.ndarray:
    ts = np.asarray(ts_ms_like, dtype=np.float64)
    if ts.size < 2:
        return ts
    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return ts
    med = float(np.median(diffs))
    if med <= 0.01:     # 초 단위로 보임 → ms로 승격
        return ts * 1000.0
    if med >= 1000.0:   # us 단위로 보임 → ms로 강등
        return ts / 1000.0
    return ts

def _norm_xy(x_raw: float, y_raw: float, rect: Tuple[float,float,float,float]):
    L, T, W, H = rect
    xr = (x_raw - L) / max(1.0, W)
    yr = (y_raw - T) / max(1.0, H)
    oob = 1 if (xr < 0 or xr > 1 or yr < 0 or yr > 1) else 0
    x = min(1.0, max(0.0, xr))
    y = min(1.0, max(0.0, yr))
    return x, y, oob

# ---------- 특징 구성 (dt 기반, 모델 입력 oob=canvas 기준) ----------
def build_window_7ch(meta: Any, events: List[Any], T: int = 300):
    """
    반환: (X, raw_len, has_track, has_wrap, oob_canvas_rate, oob_wrapper_rate)
      - X: (T,7) float32, 채널=[x,y,vx,vy,speed,accel,oob_canvas]
    """
    rect_track, rect_oob = _roi_rects(meta)
    if rect_track is None:
        # 🔒 canvas가 없으면 모델 입력을 만들지 않음(폴백 금지)
        return None, 0, False, (rect_oob is not None), 0.0, 0.0

    pts = _flatten_events(meta, events)
    if not pts:
        return None, 0, True, (rect_oob is not None), 0.0, 0.0

    # 1) 정규화 + OOB (canvas 기준)
    xs, ys, oobs_canvas, oobs_wrap, ts = [], [], [], [], []
    for t, xr, yr in pts:
        x1, y1, oob_canvas = _norm_xy(xr, yr, rect_track)  # ← 항상 canvas 기준
        xs.append(x1); ys.append(y1); oobs_canvas.append(oob_canvas); ts.append(float(t))

        if rect_oob is not None:
            _, _, oob_wrap = _norm_xy(xr, yr, rect_oob)    # 통계용 wrapper
        else:
            oob_wrap = 0  # wrapper가 없으면 0으로
        oobs_wrap.append(oob_wrap)

    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    oobs_canvas = np.asarray(oobs_canvas, dtype=np.float32)
    oobs_wrap   = np.asarray(oobs_wrap, dtype=np.float32)
    ts = _fix_time_units_to_ms(np.asarray(ts, dtype=np.float64))  # 2) 시간 보정(ms)

    # 3) dt (sec)
    dt_ms = np.diff(ts, prepend=ts[0])
    dt_ms = np.clip(dt_ms, 1e-3, None)
    dt_s  = dt_ms / 1000.0

    # 4) vx, vy, speed, accel
    if len(xs) < 2:
        vx = np.zeros_like(xs); vy = np.zeros_like(ys)
        speed = np.zeros_like(xs); accel = np.zeros_like(xs)
    else:
        dx = np.diff(xs, prepend=xs[0]); dy = np.diff(ys, prepend=ys[0])
        vx = dx / dt_s; vy = dy / dt_s
        speed  = np.sqrt(vx*vx + vy*vy)
        accel  = np.diff(speed, prepend=speed[0]) / dt_s

    # 5) 모델 입력: [x, y, vx, vy, speed, accel, oob_canvas]
    X = np.stack([xs, ys, vx, vy, speed, accel, oobs_canvas], axis=1).astype(np.float32)
    raw_len = X.shape[0]

    # 6) 길이 정규화
    if raw_len < T:
        X = np.concatenate([X, np.zeros((T - raw_len, X.shape[1]), np.float32)], axis=0)
    elif raw_len > T:
        X = X[-T:, :]

    # 7) 통계
    oob_canvas_rate  = float(np.mean(oobs_canvas > 0.5)) if oobs_canvas.size else 0.0
    oob_wrapper_rate = float(np.mean(oobs_wrap   > 0.5)) if oobs_wrap.size   else 0.0
    return X, raw_len, True, (rect_oob is not None), oob_canvas_rate, oob_wrapper_rate

def seq_stats(X, raw_len: int, has_track: bool, has_wrap: bool, oob_canvas_rate: float, oob_wrap_rate: float):
    if X is None or X.size == 0:
        return {
            "oob_rate_canvas": 0.0,
            "oob_rate_wrapper": 0.0,
            "speed_mean": 0.0,
            "n_events": 0,
            "roi_has_canvas": has_track,
            "roi_has_wrapper": has_wrap,
        }
    return {
        "oob_rate_canvas": float(oob_canvas_rate),
        "oob_rate_wrapper": float(oob_wrap_rate),
        "speed_mean": float(np.mean(X[:, 4])),
        "n_events": int(raw_len),
        "roi_has_canvas": has_track,
        "roi_has_wrapper": has_wrap,
    }
