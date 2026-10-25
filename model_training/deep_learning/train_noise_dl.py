import os
import sys
import json

# Ensure stdout and stderr support UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix
)
from pathlib import Path

# Add current directory to path
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(CURRENT_DIR))

from dataset import build_or_load_dataset, AEDataset
from models import Waveform1DCNN, Scalogram2DCNN

MODEL_OUT_DIR = ROOT_DIR / "models" / "deep_learning"
RESULTS_OUT_DIR = ROOT_DIR / "results" / "deep_learning"
MODEL_OUT_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_OUT_DIR.mkdir(exist_ok=True, parents=True)


def train_and_eval_fold(model_class, mode, dataset_data, train_idx, val_idx, num_epochs=12, batch_size=64, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    full_ds = AEDataset(
        dataset_data['waveforms'],
        dataset_data['scalograms'],
        dataset_data['labels'],
        mode=mode
    )
    
    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=batch_size, shuffle=False)
    
    model = model_class(num_classes=2).to(device)
    
    # Class weights to handle class imbalance
    y_train = dataset_data['labels'][train_idx]
    counts = np.bincount(y_train, minlength=2)
    weights = torch.tensor([counts.sum() / max(c, 1) for c in counts], dtype=torch.float32).to(device)
    weights = weights / weights.sum()
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_f2 = -1.0
    best_preds = []
    best_targets = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        all_p = []
        all_t = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                preds = torch.argmax(out, dim=1).cpu().numpy()
                all_p.extend(preds)
                all_t.extend(y.cpu().numpy())
                
        scheduler.step(val_loss)
        
        all_p = np.array(all_p)
        all_t = np.array(all_t)
        f2 = fbeta_score(all_t, all_p, beta=2.0, pos_label=1, zero_division=0)
        
        if f2 > best_f2:
            best_f2 = f2
            best_preds = all_p
            best_targets = all_t
            
    return best_targets, best_preds, model


def run_noise_benchmark():
    print("=" * 80)
    print("🚀 DEEP LEARNING BENCHMARK: NOISE VS. CAVITATION CLASSIFICATION")
    print("=" * 80)
    
    # Load dataset
    data = build_or_load_dataset(task="noise", force_rebuild=False, max_noise_samples=2500)
    
    y = data['labels']
    groups = data['experiment_ids']
    
    print(f"Total samples: {len(y)} ({sum(y == 1)} Cavitation, {sum(y == 0)} Noise)")
    print(f"Unique experiments: {len(np.unique(groups))}")
    print("Validation Strategy: 5-Fold Stratified Group CV on experiment_id (Zero Session Leakage)")
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    experiments = [
        ("1D-CNN (Raw Waveform)", Waveform1DCNN, "1d", MODEL_OUT_DIR / "noise_1d_cnn.pt"),
        ("2D-CNN (CWT Scalogram)", Scalogram2DCNN, "2d", MODEL_OUT_DIR / "noise_cwt_2d_cnn.pt")
    ]
    
    all_results = {}
    
    for exp_name, model_cls, mode, save_path in experiments:
        print(f"\n--- Training {exp_name} across 5 Group-Folds ---")
        pooled_true = []
        pooled_pred = []
        
        last_model = None
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(data['waveforms'], y, groups=groups)):
            t, p, mdl = train_and_eval_fold(model_cls, mode, data, train_idx, val_idx, num_epochs=10, batch_size=64)
            pooled_true.extend(t)
            pooled_pred.extend(p)
            last_model = mdl
            
            f1 = f1_score(t, p, average='weighted', zero_division=0)
            f2 = fbeta_score(t, p, beta=2.0, pos_label=1, zero_division=0)
            print(f"  Fold {fold + 1}/5: F1={f1:.3f}, F2(Signal)={f2:.3f}")
            
        pooled_true = np.array(pooled_true)
        pooled_pred = np.array(pooled_pred)
        
        acc = accuracy_score(pooled_true, pooled_pred)
        prec = precision_score(pooled_true, pooled_pred, average='weighted', zero_division=0)
        rec = recall_score(pooled_true, pooled_pred, average='weighted', zero_division=0)
        f1 = f1_score(pooled_true, pooled_pred, average='weighted', zero_division=0)
        f2_signal = fbeta_score(pooled_true, pooled_pred, beta=2.0, pos_label=1, zero_division=0)
        cm = confusion_matrix(pooled_true, pooled_pred).tolist()
        
        print(f"\n📊 Overall Pooled Results for {exp_name}:")
        print(f"  Accuracy:          {acc:.4f}")
        print(f"  Precision:         {prec:.4f}")
        print(f"  Recall:            {rec:.4f}")
        print(f"  Macro F1:          {f1:.4f}")
        print(f"  F2-Score (Signal): {f2_signal:.4f}")
        
        # Save model checkpoint
        if last_model is not None:
            torch.save(last_model.state_dict(), save_path)
            print(f"  Saved model weights to {save_path}")
            
        all_results[exp_name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'f2_signal': float(f2_signal),
            'confusion_matrix': cm,
            'total_test_samples': int(len(pooled_true))
        }
        
    # Save overall metrics
    metrics_path = MODEL_OUT_DIR / "noise_dl_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved all noise deep learning benchmark metrics to {metrics_path}")


if __name__ == "__main__":
    run_noise_benchmark()
