#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_cnn.py (수정 버전)

- 학습 DataLoader(shuffle=True)와 최종 추론용 DataLoader(shuffle=False) 분리  ← 중요!
- GPU(CUDA) 자동 사용, 없으면 MPS→CPU
- CUDA일 때만 pin_memory=True
- torch.amp.autocast(device_type='cuda', enabled=use_cuda)
- 출력: artifacts/cnn/best.pt, probs_*.npy, logits_*.npy, labels_*.npy, thresholds.json
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss, confusion_matrix
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.amp import autocast
from torch.amp import GradScaler
import random

# ---------------- Utils ----------------

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_dataset(root: Path):
    npz = np.load(root / "dataset_windows.npz", allow_pickle=False)
    X = npz["X"].astype(np.float32)    # (N,T,7)
    y = npz["y"].astype(np.int64)      # (N,)
    N = X.shape[0]

    split_csv = root / "split_windows.csv"
    if split_csv.exists():
        df = pd.read_csv(split_csv)
        if {"window_index","split"}.issubset(df.columns):
            splits = np.array(df.sort_values("window_index")["split"].values)
            assert len(splits) == N, "split_windows.csv 길이가 X와 다릅니다."
        else:
            raise ValueError("split_windows.csv 에 window_index, split 컬럼 필요")
    else:
        idx = np.arange(N); np.random.shuffle(idx)
        n_tr = int(N*0.8); n_va = int(N*0.1)
        splits = np.array(["train"]*N, dtype=object)
        splits[idx[n_tr:n_tr+n_va]] = "val"
        splits[idx[n_tr+n_va:]] = "test"
    return X, y, splits

class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        xi = np.transpose(self.X[i], (1,0))  # (7,T)
        return torch.from_numpy(xi).float(), torch.tensor(self.y[i]).float()

# ---------------- Model ----------------

class CNN1D(nn.Module):
    def __init__(self, in_ch=7, c1=96, c2=192, dropout=0.2, input_bn=True):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_ch) if input_bn else nn.Identity()
        self.feat = nn.Sequential(
            nn.Conv1d(in_ch, c1, kernel_size=7, padding=3),
            nn.BatchNorm1d(c1), nn.ReLU(inplace=True),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(c2, 1)
    def forward(self, x):               # x: (B,7,T)
        x = self.input_bn(x)
        h = self.feat(x).mean(dim=-1)   # GAP
        return self.head(h).squeeze(1)  # (B,)

# ---------------- Inference helpers ----------------

@torch.no_grad()
def infer_logits_probs_labels(model, loader, device):
    model.eval()
    logits_all, probs_all, labels_all = [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        logit = model(xb); prob = torch.sigmoid(logit)
        logits_all.append(logit.detach().cpu().numpy())
        probs_all.append(prob.detach().cpu().numpy())
        labels_all.append(yb.detach().cpu().numpy())
    return (np.concatenate(logits_all), np.concatenate(probs_all), np.concatenate(labels_all))

def pick_threshold_at_fpr(probs, labels, target_fpr=0.01):
    fpr, tpr, thr = roc_curve(labels, probs)
    mask = fpr <= target_fpr
    if not np.any(mask):
        j = int(np.argmin(fpr)); return float(thr[j])
    i = int(np.argmax(tpr[mask])); return float(thr[mask][i])

def calculate_metrics(probs, labels, target_fpr=0.01):
    """Calculate ROC-AUC, Precision, Recall, FPR for given predictions"""
    # ROC-AUC
    roc_auc = roc_auc_score(labels, probs)
    
    # Get threshold at target FPR
    fpr_curve, tpr_curve, thresholds = roc_curve(labels, probs)
    mask = fpr_curve <= target_fpr
    if np.any(mask):
        idx = np.argmax(tpr_curve[mask])
        threshold = thresholds[mask][idx]
        fpr_at_threshold = fpr_curve[mask][idx]
    else:
        idx = np.argmin(fpr_curve)
        threshold = thresholds[idx]
        fpr_at_threshold = fpr_curve[idx]
    
    # Calculate metrics at threshold
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)  # Same as TPR
    actual_fpr = fp / max(1, fp + tn)
    
    return {
        'roc_auc': float(roc_auc),
        'precision': float(precision),
        'recall': float(recall),
        'fpr': float(actual_fpr),
        'threshold': float(threshold)
    }

def summarize_split(name, probs, labels, thr):
    roc = roc_auc_score(labels, probs); pr = average_precision_score(labels, probs)
    pred = (probs >= thr).astype(np.int64)
    TP = int(np.sum((pred==1)&(labels==1))); FP = int(np.sum((pred==1)&(labels==0)))
    TN = int(np.sum((pred==0)&(labels==0))); FN = int(np.sum((pred==0)&(labels==1)))
    TPR = TP / max(1, TP+FN); FPR = FP / max(1, FP+TN)
    prec = TP / max(1, TP+FP); rec = TPR
    brier = brier_score_loss(labels, probs)
    print(json.dumps({
        "split": name, "roc_auc": roc, "pr_auc": pr, "threshold": float(thr),
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "TPR": TPR, "FPR": FPR, "Precision": prec, "Recall": rec, "Brier": brier
    }, ensure_ascii=False, indent=2))

def plot_training_metrics(train_metrics, val_metrics, save_dir):
    """Plot training metrics over epochs with best model highlight"""
    epochs = list(range(1, len(val_metrics) + 1))
    
    # Find best epoch (highest ROC-AUC)
    roc_aucs = [m['roc_auc'] for m in val_metrics]
    best_idx = np.argmax(roc_aucs)
    best_epoch = epochs[best_idx]
    best_metrics = val_metrics[best_idx]
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Validation Metrics Over Epochs\n(Best Model: Epoch {best_epoch})', fontsize=16, fontweight='bold')
    
    metrics_info = [
        ('roc_auc', 'ROC-AUC', 'lower right'),
        ('precision', 'Precision', 'lower right'),
        ('recall', 'Recall (TPR)', 'lower right'),
        ('fpr', 'False Positive Rate', 'upper right')
    ]
    
    for idx, (metric, title, legend_loc) in enumerate(metrics_info):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        
        # Extract validation metric values only
        val_values = [m[metric] for m in val_metrics]
        
        # Plot validation line
        ax.plot(epochs, val_values, 'o-', color='#2E86AB', linewidth=3, markersize=6, 
                markerfacecolor='white', markeredgewidth=2, markeredgecolor='#2E86AB')
        
        # Highlight best epoch point
        best_value = best_metrics[metric]
        ax.plot(best_epoch, best_value, 'o', color='#FF6B35', markersize=12, 
                markerfacecolor='#FF6B35', markeredgewidth=3, markeredgecolor='white',
                label=f'Best (Epoch {best_epoch})')
        
        # Add text annotation for best point
        if metric == 'roc_auc':
            ax.annotate(f'Best: {best_value:.3f}\n(Epoch {best_epoch})', 
                       xy=(best_epoch, best_value), xytext=(best_epoch-5, best_value-0.05),
                       fontsize=10, ha='center', va='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))
        elif metric == 'precision':
            ax.annotate(f'Best: {best_value:.3f}\n(Epoch {best_epoch})', 
                       xy=(best_epoch, best_value), xytext=(best_epoch-5, best_value-0.05),
                       fontsize=10, ha='center', va='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))
        elif metric == 'recall':
            ax.annotate(f'Best: {best_value:.3f}\n(Epoch {best_epoch})', 
                       xy=(best_epoch, best_value), xytext=(best_epoch+5, best_value+0.05),
                       fontsize=10, ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))
        elif metric == 'fpr':
            ax.annotate(f'Best: {best_value:.4f}\n(Epoch {best_epoch})', 
                       xy=(best_epoch, best_value), xytext=(best_epoch+5, best_value+0.002),
                       fontsize=10, ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))
        
        # Styling
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Special handling for FPR (add target line)
        if metric == 'fpr':
            ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                      label='Target FPR (1%)')
            ax.legend(loc=legend_loc, fontsize=10)
            ax.set_ylim(0, max(0.02, max(val_values) * 1.2))
        else:
            ax.legend(loc=legend_loc, fontsize=10)
        
        # Special formatting for percentages
        if metric in ['roc_auc', 'precision', 'recall']:
            ax.set_ylim(0, 1.05)
            # Add percentage formatting
            from matplotlib.ticker import FuncFormatter
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1%}'))
        elif metric == 'fpr':
            # Add percentage formatting for FPR
            from matplotlib.ticker import FuncFormatter
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2%}'))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Save plot with best model highlight
    save_path = save_dir / "validation_metrics_with_best.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Validation metrics plot with best model highlight saved to: {save_path}")
    print(f"Best model (Epoch {best_epoch}) metrics:")
    print(f"  - ROC-AUC: {best_metrics['roc_auc']:.4f} ({best_metrics['roc_auc']:.1%})")
    print(f"  - Precision: {best_metrics['precision']:.4f} ({best_metrics['precision']:.1%})")
    print(f"  - Recall: {best_metrics['recall']:.4f} ({best_metrics['recall']:.1%})")
    print(f"  - FPR: {best_metrics['fpr']:.4f} ({best_metrics['fpr']:.2%})")
    
    # Save metrics data as JSON
    metrics_data = {
        'epochs': epochs,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'best_epoch': int(best_epoch),
        'best_metrics': best_metrics
    }
    with open(save_dir / "training_metrics.json", 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"Training metrics data saved to: {save_dir / 'training_metrics.json'}")

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input-bn", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    root = Path(args.root); ensure_dir(root); set_seed(args.seed)
    X, y, splits = load_dataset(root)
    idx_tr = np.where(splits=="train")[0]
    idx_va = np.where(splits=="val")[0]
    idx_te = np.where(splits=="test")[0]

    if torch.cuda.is_available(): device="cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): device="mps"
    else: device="cpu"
    use_cuda = (device=="cuda")
    if use_cuda: torch.set_float32_matmul_precision("high")
    pin_mem = bool(use_cuda)

    # 학습용 로더 (shuffle=True)
    dl_tr_train = DataLoader(WindowDataset(X[idx_tr], y[idx_tr]), batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers, pin_memory=pin_mem)
    # 검증/테스트 로더 (shuffle=False)
    dl_va = DataLoader(WindowDataset(X[idx_va], y[idx_va]), batch_size=args.batch_size,
                       shuffle=False, num_workers=args.num_workers, pin_memory=pin_mem)
    dl_te = DataLoader(WindowDataset(X[idx_te], y[idx_te]), batch_size=args.batch_size,
                       shuffle=False, num_workers=args.num_workers, pin_memory=pin_mem)
    print(f"[INFO] device={device}, pin_memory={pin_mem}, workers={args.num_workers}")

    model = CNN1D(in_ch=7, c1=96, c2=192, dropout=0.2, input_bn=bool(args.input_bn)).to(device)
    opt = Adam(model.parameters(), lr=args.lr)
    crit = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda', enabled=use_cuda)

    # Create plots directory
    plots_dir = Path("artifacts/cnn/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Training metrics tracking
    train_metrics_history = []
    val_metrics_history = []
    
    # Also create a dataloader for train evaluation (no shuffle)
    dl_tr_eval = DataLoader(WindowDataset(X[idx_tr], y[idx_tr]), batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers, pin_memory=pin_mem)

    best_auc, best_state, patience, bad = -1.0, None, 8, 0
    for ep in range(1, args.epochs+1):
        model.train(); losses=[]
        for xb, yb in dl_tr_train:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_cuda):
                loss = crit(model(xb), yb)
            if use_cuda:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            losses.append(loss.item())

        # Evaluation
        model.eval()
        with torch.no_grad():
            # Train metrics (for tracking purposes)
            _, probs_tr_ep, labels_tr_ep = infer_logits_probs_labels(model, dl_tr_eval, device)
            train_metrics = calculate_metrics(probs_tr_ep, labels_tr_ep, args.target_fpr)
            train_metrics_history.append(train_metrics)
            
            # Validation metrics
            _, probs_va, labels_va = infer_logits_probs_labels(model, dl_va, device)
            val_metrics = calculate_metrics(probs_va, labels_va, args.target_fpr)
            val_metrics_history.append(val_metrics)
        
        val_auc = val_metrics['roc_auc']
        print(f"[Epoch {ep:02d}] TrainLoss={np.mean(losses):.6f}  VAL ROC-AUC={val_auc:.6f}  "
              f"VAL Precision={val_metrics['precision']:.4f}  VAL Recall={val_metrics['recall']:.4f}  "
              f"VAL FPR={val_metrics['fpr']:.4f}")

        if val_auc > best_auc:
            best_auc, best_state, bad = val_auc, model.state_dict(), 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {ep} (best VAL ROC-AUC={best_auc:.6f})"); break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Generate training metrics plots
    if len(train_metrics_history) > 0:
        plot_training_metrics(train_metrics_history, val_metrics_history, plots_dir)
        print(f"Training metrics plots saved to: {plots_dir}")
    
    # dl_tr_eval already created above for tracking

    # 저장 경로
    art_dir = Path("artifacts/cnn"); art_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), art_dir / "best.pt")

    # 최종 추론 & 저장
    logits_tr, probs_tr, labels_tr = infer_logits_probs_labels(model, dl_tr_eval, device)
    logits_va, probs_va, labels_va = infer_logits_probs_labels(model, dl_va, device)
    logits_te, probs_te, labels_te = infer_logits_probs_labels(model, dl_te, device)

    np.save(root / "probs_train.npy",  probs_tr)
    np.save(root / "probs_val.npy",    probs_va)
    np.save(root / "probs_test.npy",   probs_te)
    np.save(root / "train_logits.npy", logits_tr); np.save(root / "val_logits.npy", logits_va); np.save(root / "test_logits.npy", logits_te)
    np.save(root / "train_labels.npy", labels_tr); np.save(root / "val_labels.npy", labels_va); np.save(root / "test_labels.npy", labels_te)

    thr = pick_threshold_at_fpr(probs_va, labels_va, target_fpr=args.target_fpr)
    with open(root / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump({"val_threshold": float(thr), "val_roc_auc": float(best_auc),
                   "target_fpr": float(args.target_fpr)}, f, ensure_ascii=False, indent=2)

    print("\n=== SUMMARY (window-level) ===")
    summarize_split("TRAIN", probs_tr, labels_tr, thr)
    summarize_split("VAL",   probs_va, labels_va, thr)
    summarize_split("TEST",  probs_te, labels_te, thr)

if __name__ == "__main__":
    main()
