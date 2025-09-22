#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_dataset.py — robust events + payload 파서 & ROI 사후추정
- 입력: --root 아래 human/, bot/ 폴더의 JSON/JSON.GZ (표준 JSON/NDJSON/Concatenated 지원)
- 처리: 시계열 복원 → ROI 추정/정규화 → 특징 7채널(x,y,vx,vy,speed,accel,oob) → 슬라이딩 윈도우 → 패딩/트림
- 출력:
  1) <out>/dataset_windows.npz  (X: (N,300,7), y: (N,), feature_names 포함)
  2) <out>/sessions_summary.csv
  3) <out>/skipped_sessions.csv
  4) <out>/window_map.csv        (file, window_index, local_index, label)
  5) <out>/roi_validation.json   (x/y 범위 위반, oob 비이진 비율)
"""

import os, io, gzip, json, glob, argparse, csv
import numpy as np
from typing import Any, List, Tuple, Optional, Dict

HUMAN_DIRNAME = "human"
BOT_DIRNAME   = "bot"

# 좌표/시간 키 후보
X_KEYS = ["x","clientX","pageX","screenX","cx","posX","left","cursorX"]
Y_KEYS = ["y","clientY","pageY","screenY","cy","posY","top","cursorY"]
T_KEYS = ["t","ts","time","timestamp","timeStamp","eventTime","evtTime"]

# 패킹 배열 키 후보 (루트/페이로드 모두 지원)
PACKED_X = ["xrs","xs","x_list"]
PACKED_Y = ["yrs","ys","y_list"]
PACKED_DT = ["dts","dt_list","deltas"]

# 좌표 보관 컨테이너 후보
COORD_HOLDERS = ["pos","point","position","coords","cursor"]

# ROI 추출 경로
ROI_PATHS = [
    ["meta","roi_map","canvas-container"],
    ["meta","roi"],
]
VIEWPORT_PATH = ["meta","viewport"]
VW_KEYS = ["vw","viewportW","viewportWidth","innerWidth","cw","clientWidth","width"]
VH_KEYS = ["vh","viewportH","viewportHeight","innerHeight","ch","clientHeight","height"]

TARGET_T = 300  # 윈도우 길이(프레임)

# ---------------- JSON 로딩 ----------------
def _read_text(path:str)->str:
    if path.endswith(".gz"):
        with gzip.open(path,"rb") as f:
            return io.TextIOWrapper(f,encoding="utf-8").read()
    with open(path,"r",encoding="utf-8") as f:
        return f.read()

def _looks_like_event(obj: Any) -> bool:
    return isinstance(obj, dict) and (
        "kind" in obj or "type" in obj or "payload" in obj or
        any(k in obj for k in X_KEYS+Y_KEYS+T_KEYS)
    )

def _wrap_events_if_needed(objs: List[Any]) -> Any:
    """objs가 이벤트 리스트/블록들이 섞여 있을 때 events 컨테이너로 합친다."""
    if not objs:
        return {}
    has_events_container = any(isinstance(o, dict) and "events" in o for o in objs)
    if has_events_container:
        metas, all_events = [], []
        for o in objs:
            if not isinstance(o, dict): continue
            if "events" in o and isinstance(o["events"], list):
                all_events.extend(o["events"])
            if "meta" in o and isinstance(o["meta"], dict):
                metas.append(o["meta"])
        meta_final = {}
        for m in metas:
            for k,v in m.items():
                if k not in meta_final: meta_final[k]=v
        return {"meta": meta_final, "events": all_events}
    # 라인들이 전부 이벤트처럼 보이면 감싸기
    if all(_looks_like_event(o) for o in objs if o is not None):
        return {"events": objs}
    # 그렇지 않으면 첫 객체 반환
    return objs[0]

def load_json_any(path: str) -> Any:
    text = _read_text(path).strip()
    # 1) 표준 JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2) NDJSON
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) > 1:
        objs = []
        ok = True
        for ln in lines:
            try:
                objs.append(json.loads(ln))
            except json.JSONDecodeError:
                ok = False; break
        if ok:
            return _wrap_events_if_needed(objs)
    # 3) Concatenated JSON
    decoder = json.JSONDecoder()
    idx, N = 0, len(text)
    objs = []
    try:
        while idx < N:
            while idx < N and text[idx].isspace(): idx += 1
            if idx >= N: break
            o, end = decoder.raw_decode(text, idx)
            objs.append(o); idx = end
    except json.JSONDecodeError:
        pass
    if objs:
        return _wrap_events_if_needed(objs)
    raise json.JSONDecodeError("cannot parse", text, 0)

# ---------------- 유틸 ----------------
def get_in(d: dict, path: List[str]) -> Optional[Any]:
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def guess_label_from_path(path: str) -> int:
    p = path.replace("\\","/").lower()
    if f"/{BOT_DIRNAME}/" in p or p.endswith(f"/{BOT_DIRNAME}") or p.startswith(f"{BOT_DIRNAME}/"):
        return 1
    if f"/{HUMAN_DIRNAME}/" in p or p.endswith(f"/{HUMAN_DIRNAME}") or p.startswith(f"{HUMAN_DIRNAME}/"):
        return 0
    return 0

# -------- device/meta 유틸 --------
def _get_device_from_meta(meta: dict) -> str:
    """meta 안의 device/device_type를 대소문자/표기 변형 무시하고 안전 추출"""
    if not isinstance(meta, dict):
        return ""
    try:
        lower = {(str(k).lower() if k is not None else ""): v for k, v in meta.items()}
    except Exception:
        return ""
    val = lower.get("device") or lower.get("device_type") or ""
    try:
        return str(val).strip().lower()
    except Exception:
        return ""

def _merge_meta_from_events(data: dict) -> dict:
    """
    NDJSON처럼 meta가 'type':'meta' 이벤트로 들어온 경우를 위해,
    data['meta']와 events 안의 meta 이벤트를 병합하여 meta dict를 반환.
    """
    base_meta = {}
    if isinstance(data, dict) and isinstance(data.get("meta"), dict):
        base_meta = dict(data["meta"])  # shallow copy

    events = data.get("events", [])
    if isinstance(events, list):
        for ev in events:
            if isinstance(ev, dict) and str(ev.get("type","")).lower() == "meta":
                # meta 이벤트의 루트 키들(예: device, viewport, dpr 등)을 병합
                for k, v in ev.items():
                    if k == "type":  # 'type' 키는 건너뜀
                        continue
                    # payload 안에 있다면 그 또한 병합
                    if k == "payload" and isinstance(v, dict):
                        for pk, pv in v.items():
                            if pk not in base_meta:
                                base_meta[pk] = pv
                        continue
                    if k not in base_meta:
                        base_meta[k] = v
                break
    return base_meta

# ---------------- 시계열 복원 ----------------
def _extract_xy_from_mapping(m: dict) -> Tuple[Optional[float], Optional[float]]:
    # 1) 바로 x,y 키
    xx = yy = None
    for kx in X_KEYS:
        if kx in m: xx = m[kx]; break
    for ky in Y_KEYS:
        if ky in m: yy = m[ky]; break
    if xx is not None and yy is not None:
        try: return float(xx), float(yy)
        except: pass
    # 2) 좌표 컨테이너
    for holder in COORD_HOLDERS:
        if holder in m and isinstance(m[holder], dict):
            return _extract_xy_from_mapping(m[holder])
    # 3) 튜플/리스트
    if any(isinstance(m.get(h), (list,tuple)) for h in COORD_HOLDERS):
        for h in COORD_HOLDERS:
            v = m.get(h)
            if isinstance(v, (list,tuple)) and len(v)>=2:
                try: return float(v[0]), float(v[1])
                except: pass
    return None, None

def _extract_t_from_mapping(m: dict) -> Optional[float]:
    for kt in T_KEYS:
        if kt in m:
            try: return float(m[kt])
            except: pass
    return None

def _emit_packed(px, py, pdt, ts_list, xs, ys):
    cur_t = ts_list[-1] if ts_list else 0.0
    for dx, dy, dt in zip(px, py, pdt):
        try:
            cur_t += float(dt)
            xs.append(float(dx)); ys.append(float(dy)); ts_list.append(cur_t)
        except:
            continue

def fix_time_units_to_ms(t: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    t가 초(1,2,3...) 또는 프레임 인덱스(0,1,2..)로 들어왔을 때 ms로 스케일.
    반환: (ms로 보정된 t, reason 문자열)
    """
    if len(t) < 2:
        return t, "time_ok_len1"

    t = t.astype(np.float64)
    rng = float(t[-1] - t[0])
    dt = np.diff(t)
    med_dt = float(np.median(dt)) if len(dt) else 0.0

    # 이미 ms로 보이는 경우
    if rng >= 1000.0 or med_dt >= 5.0:
        return t, "time_ms"

    # 초 단위로 의심
    if 0.2 <= med_dt <= 5.0 and rng <= 600.0:
        return t * 1000.0, "time_seconds_scaled_ms"

    # 프레임 인덱스(60/30Hz)로 의심
    if 0.8 <= med_dt <= 1.2:
        return t * 16.0, "time_frames_scaled_ms"  # 보수적 16ms

    # 범위가 너무 작으면 인덱스로 재생성
    if rng < 100.0:
        idx = np.arange(len(t), dtype=np.float64)
        return idx * 16.0, "time_reindexed_16ms"

    return t, "time_ms_fallback"

def reconstruct_timeseries(data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    events = data.get("events", [])
    if not isinstance(events, list):
        if _looks_like_event(events):
            events = [events]
        else:
            raise ValueError("no_timeseries")

    xs, ys, ts, used = [], [], [], []

    for ev in events:
        if not isinstance(ev, dict): continue
        payload = ev.get("payload", None)

        # 0) 패킹 배열 (루트/페이로드)
        for container in (ev, payload if isinstance(payload, dict) else {}):
            if not isinstance(container, dict): continue
            px = py = pdt = None
            for k in PACKED_X:
                if isinstance(container.get(k), list): px = container[k]; break
            for k in PACKED_Y:
                if isinstance(container.get(k), list): py = container[k]; break
            for k in PACKED_DT:
                if isinstance(container.get(k), list): pdt = container[k]; break
            if px is not None and py is not None and pdt is not None:
                _emit_packed(px, py, pdt, ts, xs, ys)
                used.append(ev)
                break
        else:
            # 1) 단일 포인트 (루트/페이로드)
            x = y = t = None
            x, y = _extract_xy_from_mapping(ev)
            if (x is None or y is None) and isinstance(payload, dict):
                x, y = _extract_xy_from_mapping(payload)
            if x is None or y is None:
                continue

            t = _extract_t_from_mapping(ev)
            if t is None and isinstance(payload, dict):
                t = _extract_t_from_mapping(payload)
            if t is None:
                t = (ts[-1] + 1.0) if ts else 0.0

            try:
                xs.append(float(x)); ys.append(float(y)); ts.append(float(t))
                used.append(ev)
            except:
                continue

    if len(ts) == 0:
        raise ValueError("no_timeseries")

    arr = np.array(list(zip(ts,xs,ys)), dtype=np.float64)
    arr = arr[np.argsort(arr[:,0])]
    t,x,y = arr[:,0], arr[:,1], arr[:,2]

    # 비단조 시간 보정
    for i in range(1,len(t)):
        if t[i] <= t[i-1]:
            t[i] = t[i-1] + 1.0

    t, _ = fix_time_units_to_ms(t)
    return t,x,y,used

# ---------------- ROI 추정 ----------------
def roi_from_meta(meta: dict) -> Optional[Tuple[float,float,float,float]]:
    if not isinstance(meta, dict): return None
    # 명시 ROI
    for p in ROI_PATHS:
        cur = meta
        ok = True
        for k in p:
            if not isinstance(cur, dict) or k not in cur:
                ok=False; break
            cur = cur[k]
        if ok and isinstance(cur, dict):
            left = cur.get("left", cur.get("x", 0.0))
            top  = cur.get("top",  cur.get("y", 0.0))
            w    = cur.get("w",    cur.get("width",  None))
            h    = cur.get("h",    cur.get("height", None))
            if w and h and w>0 and h>0:
                return float(left), float(top), float(w), float(h)
    # viewport
    vp = get_in(meta, VIEWPORT_PATH)
    if isinstance(vp, dict):
        left = vp.get("left", vp.get("x", 0.0))
        top  = vp.get("top",  vp.get("y", 0.0))
        w    = vp.get("w",    vp.get("width",  None))
        h    = vp.get("h",    vp.get("height", None))
        if w and h and w>0 and h>0:
            return float(left), float(top), float(w), float(h)
    return None

def viewport_from_payload(events: List[dict]) -> Optional[Tuple[float,float,float,float]]:
    for ev in events:
        payload = ev.get("payload", {})
        if not isinstance(payload, dict): continue
        w=h=None
        for k in VW_KEYS:
            if isinstance(payload.get(k),(int,float)) and float(payload[k])>10:
                w=float(payload[k]); break
        for k in VH_KEYS:
            if isinstance(payload.get(k),(int,float)) and float(payload[k])>10:
                h=float(payload[k]); break
        if w and h: return (0.0,0.0,w,h)
    return None

def roi_from_points(x: np.ndarray, y: np.ndarray) -> Tuple[Tuple[float,float,float,float], str]:
    x_min,x_max = float(np.min(x)), float(np.max(x))
    y_min,y_max = float(np.min(y)), float(np.max(y))
    sx,sy = max(1e-6, x_max-x_min), max(1e-6, y_max-y_min)
    # 이미 [0,1] 범위면 단위 ROI
    if x_min >= -0.05 and y_min >= -0.05 and x_max <= 1.05 and y_max <= 1.05:
        return (0.0,0.0,1.0,1.0), "roi_inferred_unit"
    left = x_min - 0.05*sx; top = y_min - 0.05*sy
    w = 1.10*sx; h = 1.10*sy
    if w<10 and h<10:
        return (0.0,0.0,1.0,1.0), "roi_fallback_unit_smallspan"
    return (left,top,max(w,1e-3),max(h,1e-3)), "roi_inferred_bbox"

def get_or_infer_roi(data: dict, x: np.ndarray, y: np.ndarray, used_events: List[dict]):
    meta = data.get("meta", {})
    roi = roi_from_meta(meta)
    if roi is not None: return roi, "roi_meta"
    vp = viewport_from_payload(used_events)
    if vp is not None: return vp, "roi_payload_viewport"
    return roi_from_points(x,y)

# ---------------- 정규화/특성/윈도우 ----------------
def norm_xy(x,y,roi):
    ox,oy,w,h = roi
    x_n = (x-ox)/w; y_n = (y-oy)/h
    oob = (x_n<0)|(x_n>1)|(y_n<0)|(y_n>1)
    return x_n.astype(np.float32), y_n.astype(np.float32), oob.astype(np.float32)

def compute_features(t,x,y,oob):
    T = len(t)
    feats = np.zeros((T,7), dtype=np.float32)
    feats[:,0]=x; feats[:,1]=y; feats[:,6]=oob
    if T<2: return feats
    dt = np.diff(t)/1000.0; dt[dt<=1e-6]=1e-6
    dx = np.diff(x); dy = np.diff(y)
    vx = np.concatenate([[0.0], dx/dt]); vy = np.concatenate([[0.0], dy/dt])
    speed = np.sqrt(vx**2 + vy**2)
    acc = np.concatenate([[0.0], np.diff(speed)/dt])
    feats[:,2]=vx; feats[:,3]=vy; feats[:,4]=speed; feats[:,5]=acc
    return feats

def sliding_windows(features, t, window_ms, stride_ms):
    """세그먼트 배열과 (start_ms,end_ms) 리스트를 함께 반환."""
    if len(t)==0: return [], []
    win = float(window_ms); stride=float(stride_ms)
    out=[]; ranges=[]
    cur=t[0]; end=t[-1]
    while cur+win <= end+1e-6:
        idx = (t>=cur) & (t<cur+win)
        seg = features[idx]
        if len(seg)>0:
            out.append(seg)
            ranges.append((float(cur), float(cur+win)))
        cur += stride
    return out, ranges

def pad_or_trim(wins, target_len):
    if not wins: return np.zeros((0,target_len,7), dtype=np.float32)
    C = wins[0].shape[1]
    arr=[]
    for w in wins:
        if len(w)>=target_len: arr.append(w[:target_len])
        else:
            pad = np.zeros((target_len-len(w), C), dtype=w.dtype)
            arr.append(np.vstack([w,pad]))
    return np.stack(arr, axis=0)

# ---------------- 한 파일 처리 ----------------
def process_one(path, window_ms, stride_ms, short_policy="pad", min_session_ms=200):
    try:
        data = load_json_any(path)
    except Exception as e:
        return np.zeros((0,1,7),np.float32), guess_label_from_path(path), {}, f"read_error:{e}", []

    label = guess_label_from_path(path)
    try:
        t,x_raw,y_raw,used_events = reconstruct_timeseries(data)
    except Exception as e:
        return np.zeros((0,1,7),np.float32), label, {}, f"no_timeseries:{e}", []

    # --- 메타 병합: data['meta'] + events의 'type':'meta' 한 줄을 합쳐서 사용 ---
    merged_meta = _merge_meta_from_events(data)

    # device_type 추출
    device_type = _get_device_from_meta(merged_meta)

    # ROI 추정 (병합된 meta를 활용하려면 data를 살짝 치환)
    data_for_roi = dict(data)
    data_for_roi["meta"] = merged_meta
    roi, roi_mode = get_or_infer_roi(data_for_roi, x_raw, y_raw, used_events)

    x_n,y_n,oob = norm_xy(x_raw,y_raw,roi)
    feats = compute_features(t,x_n,y_n,oob)

    wins, ranges = sliding_windows(feats,t,window_ms,stride_ms)
    span_ms = float(t[-1] - t[0])

    if not wins:
        if short_policy == "pad" and span_ms >= min_session_ms:
            wins = [feats]; ranges=[(float(t[0]), float(t[-1]))]
            X = pad_or_trim(wins, TARGET_T)
            summary = {
                "len_ms": span_ms,
                "mean_speed": float(np.mean(feats[:,4])) if len(feats) else 0.0,
                "oob_rate": float(np.mean(feats[:,6])) if len(feats) else 0.0,
                "roi_mode": roi_mode,
                "note": "short_session_padded",
                "device_type": device_type,
            }
            return X, label, summary, "", ranges
        else:
            return np.zeros((0,1,7),np.float32), label, {
                "len_ms": span_ms,
                "mean_speed": float(np.mean(feats[:,4])) if len(feats) else 0.0,
                "oob_rate": float(np.mean(feats[:,6])) if len(feats) else 0.0,
                "roi_mode": roi_mode,
                "device_type": device_type,
            }, "no_active_windows", []

    X = pad_or_trim(wins, TARGET_T)
    summary = {
        "len_ms": float(t[-1]-t[0]),
        "mean_speed": float(np.mean(feats[:,4])),
        "oob_rate": float(np.mean(feats[:,6])),
        "roi_mode": roi_mode,
        "device_type": device_type,
    }
    return X, label, summary, "", ranges

# ---------------- 전체 빌드 ----------------
def build_dataset(files: List[str], window_ms:int, stride_ms:int, out_dir:str, short_policy="pad", min_session_ms=200):
    os.makedirs(out_dir, exist_ok=True)
    sum_rows=[]; skip_rows=[]; X_all=[]; y_all=[]
    wm_rows=[]; global_widx=0

    for fp in files:
        Xw,lbl,summary,skip,ranges = process_one(fp,window_ms,stride_ms, short_policy=short_policy, min_session_ms=min_session_ms)
        base = os.path.basename(fp)
        if skip:
            skip_rows.append({"file":fp,"reason":skip})
        else:
            X_all.append(Xw)
            y_all.append(np.full((Xw.shape[0],), lbl, dtype=np.int64))
            sum_rows.append({
                "file": base,
                "len_ms": summary.get("len_ms",0.0),
                "mean_speed": summary.get("mean_speed",0.0),
                "oob_rate": summary.get("oob_rate",0.0),
                "label": lbl,
                "device_type": summary["device_type"],
                "roi_mode": summary.get("roi_mode","n/a"),
            })
            # window_map 축적 (세션=파일명, 시간구간 포함)
            for li, (s_ms, e_ms) in enumerate(ranges):
                wm_rows.append({
                    "file": base,
                    "window_index": int(global_widx),
                    "local_index": li,
                    "start_ms": float(s_ms),
                    "end_ms": float(e_ms),
                    "label": int(lbl),
                })
                global_widx += 1

    # 저장물
    sum_path = os.path.join(out_dir,"sessions_summary.csv")
    with open(sum_path,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["file","len_ms","mean_speed","oob_rate","label","device_type","roi_mode"])
        for r in sum_rows:
            w.writerow([r["file"],r["len_ms"],r["mean_speed"],r["oob_rate"],r["label"],r["device_type"],r["roi_mode"]])

    skip_path = os.path.join(out_dir,"skipped_sessions.csv")
    with open(skip_path,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["file","reason"])
        for r in skip_rows: w.writerow([r["file"],r["reason"]])

    if not X_all:
        raise RuntimeError("수집된 윈도우가 없습니다.")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    # npz 저장 (feature_names 포함)
    out_npz = os.path.join(out_dir,"dataset_windows.npz")
    feature_names = np.array(["x","y","vx","vy","speed","accel","oob"], dtype=object)
    np.savez_compressed(out_npz, X=X, y=y, feature_names=feature_names)

    # window_map 저장
    wm_path = os.path.join(out_dir,"window_map.csv")
    with open(wm_path,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["file","window_index","local_index","start_ms","end_ms","label"])
        for r in wm_rows:
            w.writerow([r["file"], r["window_index"], r["local_index"], r["start_ms"], r["end_ms"], r["label"]])

    # ROI 정규화 품질 리포트
    flat = X.reshape(-1, X.shape[-1])  # (N*T, 7)
    x_bad = float(np.mean((flat[:,0] < 0) | (flat[:,0] > 1)) * 100.0)
    y_bad = float(np.mean((flat[:,1] < 0) | (flat[:,1] > 1)) * 100.0)
    o_bad = float(np.mean(~np.isin(flat[:,6], [0.0,1.0])) * 100.0)
    roi_report = {
        "x_out_of_range_pct": round(x_bad,4),
        "y_out_of_range_pct": round(y_bad,4),
        "oob_not_binary_pct": round(o_bad,4)
    }
    import json as _json
    with open(os.path.join(out_dir,"roi_validation.json"),"w",encoding="utf-8") as f:
        _json.dump(roi_report, f, ensure_ascii=False, indent=2)

    # 로그
    print(f"X shape: {X.shape} (N,T,C)")
    print(f"y shape: {y.shape}")
    print(f"Saved: {sum_path}")
    print(f"Saved: {skip_path}")
    print(f"Saved: {wm_path}")
    print(f"Saved: {out_npz}")
    print(f"[ROI Validation] {roi_report}")
    return X,y

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out",  type=str, required=True)
    ap.add_argument("--window-ms", type=int, default=3000)
    ap.add_argument("--stride-ms", type=int, default=500)
    ap.add_argument("--short-policy", type=str, choices=["pad","skip"], default="pad",
                    help="세션 길이가 window보다 짧을 때 pad(포함) 또는 skip(제외)")
    ap.add_argument("--min-session-ms", type=int, default=200,
                    help="짧은 세션 패딩으로 살릴 최소 길이(ms)")
    args = ap.parse_args()

    human = glob.glob(os.path.join(args.root,HUMAN_DIRNAME,"**","*.json"), recursive=True) + \
            glob.glob(os.path.join(args.root,HUMAN_DIRNAME,"**","*.json.gz"), recursive=True)
    bot   = glob.glob(os.path.join(args.root,BOT_DIRNAME,"**","*.json"), recursive=True) + \
            glob.glob(os.path.join(args.root,BOT_DIRNAME,"**","*.json.gz"), recursive=True)
    files = human + bot
    if not files:
        raise RuntimeError("입력 파일을 찾지 못했습니다.")

    build_dataset(files, args.window_ms, args.stride_ms, args.out,
                  short_policy=args.short_policy, min_session_ms=args.min_session_ms)

if __name__ == "__main__":
    main()
