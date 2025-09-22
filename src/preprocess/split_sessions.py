#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

def device_bucketize(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return "na"
    s = s.strip().lower()
    if s in ("mouse", "touch"):
        return s
    return "other"

def stratified_session_split(sumdf: pd.DataFrame,
                             train_ratio: float,
                             val_ratio: float,
                             seed: int = 42):
    """
    세션 단위 층화 분할.
    strata = f"{label}_{device_bucket}"
    """
    rng = np.random.default_rng(seed)
    sumdf = sumdf.copy()
    sumdf["device_bucket"] = sumdf["device_type"].apply(device_bucketize)
    sumdf["strata"] = sumdf["label"].astype(str) + "_" + sumdf["device_bucket"]

    train, val, test = [], [], []

    # 각 strata 그룹에서 70/15/15 비율로 분할하되,
    # 그룹 크기가 작아도 val/test에 최소 1건은 가도록(가능하면) 보정.
    for strata, grp in sumdf.groupby("strata", dropna=False):
        idx = np.arange(len(grp))
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(n * train_ratio))
        n_va = int(round(n * val_ratio))
        # 나머지는 test
        n_te = n - n_tr - n_va

        # 보정: test가 0이면 뒤에서 한 장 땡겨오기
        if n_te <= 0 and n >= 3:
            n_te = 1
            if n_va > 1:
                n_va -= 1
            elif n_tr > 1:
                n_tr -= 1

        # 아주 작은 그룹(n==1~2)에서는 최소 한 split만 채워도 OK
        # (누수만 없게 같은 세션이 복수 split로 가지만 않으면 됨)
        n_tr = max(0, min(n_tr, n))
        n_va = max(0, min(n_va, n - n_tr))
        n_te = n - n_tr - n_va

        tr_idx = idx[:n_tr]
        va_idx = idx[n_tr:n_tr+n_va]
        te_idx = idx[n_tr+n_va:]

        train += grp.iloc[tr_idx]["file"].tolist()
        val   += grp.iloc[va_idx]["file"].tolist()
        test  += grp.iloc[te_idx]["file"].tolist()

    return {"train": train, "val": val, "test": test}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="processed 폴더 루트 (sessions_summary.csv, window_map.csv가 있는 폴더)")
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    p_sum = root / "sessions_summary.csv"
    p_wm  = root / "window_map.csv"

    if not p_sum.exists() or not p_wm.exists():
        print(f"[ERR] {p_sum} 또는 {p_wm} 가 없습니다.", file=sys.stderr)
        sys.exit(1)

    sumdf = pd.read_csv(p_sum)
    wm = pd.read_csv(p_wm)

    # sanity: 필요한 컬럼 체크
    for col in ["file","label","device_type"]:
        if col not in sumdf.columns:
            print(f"[ERR] sessions_summary.csv에 '{col}' 컬럼이 없습니다.", file=sys.stderr)
            sys.exit(2)

    # 세션 단위 층화 분할
    ratios_ok = 0.999 <= (args.train_ratio + args.val_ratio) <= 1.001
    if not ratios_ok:
        print("[WARN] train_ratio + val_ratio != 1 - test는 나머지로 자동 설정됩니다.")
    split = stratified_session_split(sumdf, args.train_ratio, args.val_ratio, seed=args.seed)

    # 저장: 세션 리스트
    out_json = root / "split_sessions.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved {out_json}")

    # 각 윈도우별 split 매핑 CSV 생성 (운영/학습 편의를 위해)
    # window_map.csv: file,window_index,local_index,start_ms,end_ms,label
    wm2 = wm.copy()
    wm2["split"] = "train"
    wm2.loc[wm2["file"].isin(split["val"]), "split"] = "val"
    wm2.loc[wm2["file"].isin(split["test"]), "split"] = "test"
    out_wcsv = root / "split_windows.csv"
    wm2.to_csv(out_wcsv, index=False, encoding="utf-8")
    print(f"[OK] saved {out_wcsv}")

    # 리포트 출력
    sumdf["device_bucket"] = sumdf["device_type"].apply(device_bucketize)
    sumdf["split"] = np.where(sumdf["file"].isin(split["val"]), "val",
                              np.where(sumdf["file"].isin(split["test"]), "test", "train"))
    print("\n=== 세션 분할 결과 (count) ===")
    print(sumdf["split"].value_counts())

    print("\n=== split × label ===")
    print(pd.crosstab(sumdf["split"], sumdf["label"]))

    print("\n=== split × device_bucket ===")
    print(pd.crosstab(sumdf["split"], sumdf["device_bucket"]))

    print("\n=== split × label × device_bucket (count) ===")
    triple = sumdf.groupby(["split","label","device_bucket"])["file"].count()
    print(triple)

if __name__ == "__main__":
    main()

# python src\preprocess\split_sessions.py --root .\data\processed\all --train-ratio 0.70 --val-ratio 0.15 --seed 42
