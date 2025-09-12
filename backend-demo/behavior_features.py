# behavior_features.py
# 모델 입력 윈도우 7채널: [x, y, vx, vy, speed, accel, oob_canvas]
# 학습 파이프라인(build_dataset.py)과 **동일한** 시간 스케일/미분 규칙 적용

from typing import Any, List, Optional, Tuple
import numpy as np

# ---------- ROI ----------
def _to_rect(d):
    try:
        L, T, W, H = float(d["left"]), float(d["top"]), float(d["w"]), float(d["h"])
        if W <= 0 or H <= 0:
            return None
        return (L, T, W, H)
    except Exception:
        return None

def _roi_rects(meta: Any):
    rmap = (getattr(meta, "roi_map", None) or {})
    rect_canvas = _to_rect(rmap.get("canvas-container")) if rmap.get("canvas-container") else None
    rect_wrap   = _to_rect(rmap.get("scratcha-container")) if rmap.get("scratcha-container") else None
    return rect_canvas, rect_wrap

# ---------- 이벤트 평탄화 ----------
def _flatten_events(meta: Any, events: List[Any]):
    out = []
    for ev in events:
        et = getattr(ev, "type", None) or (ev.get("type") if isinstance(ev, dict) else None)
        if et in ("moves", "moves_free"):
            p = getattr(ev, "payload", None) or (ev.get("payload") if isinstance(ev, dict) else None)
            if not p: continue
            base = int(getattr(p, "base_t", 0) or (p.get("base_t") if isinstance(p, dict) else 0) or 0)
            dts  = list(getattr(p, "dts", []) or (p.get("dts") if isinstance(p, dict) else []) or [])
            xs   = list(getattr(p, "xrs", []) or (p.get("xrs") if isinstance(p, dict) else []) or [])
            ys   = list(getattr(p, "yrs", []) or (p.get("yrs") if isinstance(p, dict) else []) or [])
            t = base
            n = min(len(dts), len(xs), len(ys))
            for i in range(n):
                out.append((t, float(xs[i]), float(ys[i])))
                dt = int(dts[i]) if int(dts[i]) > 0 else 1
                t += dt
        elif et in ("pointerdown", "pointerup", "click"):
            t  = (getattr(ev, "t", None) if not isinstance(ev, dict) else ev.get("t"))
            xr = (getattr(ev, "x_raw", None) if not isinstance(ev, dict) else ev.get("x_raw"))
            yr = (getattr(ev, "y_raw", None) if not isinstance(ev, dict) else ev.get("y_raw"))
            if t is None or xr is None or yr is None: continue
            out.append((int(t), float(xr), float(yr)))
    out.sort(key=lambda x: x[0])
    return out

# ---------- 시간 스케일 보정 (학습 규칙과 동일) ----------
def _time_scale_to_ms(t: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    규칙 (build_dataset.py와 동일):
    - 이미 ms: rng>=1000ms 또는 med_dt>=5ms → 그대로
    - 초 단위: 0.2<=med_dt<=5.0 and rng<=600 → *1000
    - 프레임 인덱스(60/30Hz): 0.8<=med_dt<=1.2 → *16  (보수적 16ms)
    - 범위가 너무 작음(rng<100ms): 인덱스 재생성 → idx*16
    - 그 외: ms로 간주 (fallback)
    """
    if t.size < 2:
        return t, "time_ok_len1"
    t = t.astype(np.float64)
    rng = float(t[-1] - t[0])
    dt = np.diff(t)
    med_dt = float(np.median(dt)) if dt.size else 0.0

    if rng >= 1000.0 or med_dt >= 5.0:
        return t, "time_ms"
    if 0.2 <= med_dt <= 5.0 and rng <= 600.0:
        return t * 1000.0, "time_seconds_scaled_ms"
    if 0.8 <= med_dt <= 1.2:
        return t * 16.0, "time_frames_scaled_ms"
    if rng < 100.0:
        idx = np.arange(len(t), dtype=np.float64)
        return idx * 16.0, "time_reindexed_16ms"
    return t, "time_ms_fallback"

def _norm_xy(x_raw: float, y_raw: float, rect):
    L, T, W, H = rect
    xr = (x_raw - L) / max(1.0, W)
    yr = (y_raw - T) / max(1.0, H)
    oob = 1 if (xr < 0 or xr > 1 or yr < 0 or yr > 1) else 0
    x = min(1.0, max(0.0, xr))
    y = min(1.0, max(0.0, yr))
    return x, y, oob

# ---------- 특징 구성 (학습과 동일한 미분/하한) ----------
def build_window_7ch(meta: Any, events: List[Any], T: int = 300):
    """
    반환: (X, raw_len, has_track, has_wrap, oob_canvas_rate, oob_wrapper_rate)
    X: (T,7), 채널=[x, y, vx, vy, speed, accel, oob_canvas]
    """
    rect_track, rect_wrap = _roi_rects(meta)
    has_wrap = rect_wrap is not None
    if rect_track is None:
        return None, 0, False, has_wrap, 0.0, 0.0

    pts = _flatten_events(meta, events)
    if not pts:
        return None, 0, True, has_wrap, 0.0, 0.0

    xs, ys, oob_c, oob_w, ts = [], [], [], [], []
    for t, xr, yr in pts:
        x1, y1, oc = _norm_xy(xr, yr, rect_track)          # 입력/oob는 **항상 canvas 기준**
        xs.append(x1); ys.append(y1); oob_c.append(oc); ts.append(float(t))
        if rect_wrap is not None:
            _, _, ow = _norm_xy(xr, yr, rect_wrap)
        else:
            ow = 0
        oob_w.append(ow)

    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    oob_c = np.asarray(oob_c, dtype=np.float32)
    oob_w = np.asarray(oob_w, dtype=np.float32)

    # 시간 스케일 보정 → ms
    ts = np.asarray(ts, dtype=np.float64)
    ts, _ = _time_scale_to_ms(ts)

    # === 학습과 동일한 미분/하한 ===
    # dt_s 하한 = 1e-6 sec (0.001 ms)
    if len(ts) < 2:
        vx = np.zeros_like(xs); vy = np.zeros_like(ys)
        speed = np.zeros_like(xs); acc = np.zeros_like(xs)
    else:
        dt = np.diff(ts, prepend=ts[0]) / 1000.0
        dt[dt <= 1e-6] = 1e-6
        dx = np.diff(xs, prepend=xs[0]); dy = np.diff(ys, prepend=ys[0])
        vx = dx / dt; vy = dy / dt
        speed = np.sqrt(vx * vx + vy * vy)
        acc = np.diff(speed, prepend=speed[0]) / dt

    # ⚠️ 어떤 추가적 클리핑/스케일링도 하지 않음 — 학습 분포와 동일 유지
    X = np.stack([xs, ys, vx, vy, speed, acc, oob_c], axis=1).astype(np.float32)
    raw_len = X.shape[0]

    # 길이 정규화
    if raw_len < T:
        X = np.concatenate([X, np.zeros((T - raw_len, X.shape[1]), np.float32)], axis=0)
    elif raw_len > T:
        X = X[-T:, :]

    oob_canvas_rate  = float(np.mean(oob_c > 0.5)) if oob_c.size else 0.0
    oob_wrapper_rate = float(np.mean(oob_w > 0.5)) if oob_w.size else 0.0
    return X, raw_len, True, has_wrap, oob_canvas_rate, oob_wrapper_rate

def seq_stats(X, raw_len, has_track, has_wrap, oob_canvas_rate, oob_wrapper_rate):
    if X is None or X.size == 0:
        return {
            "oob_rate_canvas": 0.0, "oob_rate_wrapper": 0.0,
            "speed_mean": 0.0, "n_events": 0,
            "roi_has_canvas": has_track, "roi_has_wrapper": has_wrap,
        }
    return {
        "oob_rate_canvas": float(oob_canvas_rate),
        "oob_rate_wrapper": float(oob_wrapper_rate),
        "speed_mean": float(np.mean(X[:, 4])),
        "n_events": int(raw_len),
        "roi_has_canvas": has_track,
        "roi_has_wrapper": has_wrap,
    }
