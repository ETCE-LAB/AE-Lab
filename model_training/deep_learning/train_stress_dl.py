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
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(CURRENT_DIR))

from dataset import build_or_load_dataset, AEDataset
from models import Waveform1DCNN, Scalogram2DCNN

MODEL_OUT_DIR = ROOT_DIR / "models" / "deep_learning"
RESULTS_OUT_DIR = ROOT_DIR / "results" / "deep_learning"
MODEL_OUT_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_OUT_DIR.mkdir(exist_ok=True, parents=True)


def train_and_eval_fold(model_class, mode, dataset_data, train_idx, val_idx, num_classes=2, num_epochs=12, batch_size=32, lr=0.001, balance_train=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_train = dataset_data['labels'][train_idx]
    
    # 1:1 Downsample majority class in training split to prevent single-class bias
    if balance_train:
        trn_cut = train_idx[y_train == 0]
        trn_drought = train_idx[y_train == 1]
        n_match = min(len(trn_cut), len(trn_drought))
        np.random.seed(42)
        bal_cut = np.random.choice(trn_cut, size=n_match, replace=False)
        bal_drought = np.random.choice(trn_drought, size=n_match, replace=False)
        actual_train_idx = np.concatenate([bal_cut, bal_drought])
    else:
        actual_train_idx = train_idx

    full_ds = AEDataset(
        dataset_data['waveforms'],
        dataset_data['scalograms'],
        dataset_data['labels'],
        mode=mode
    )

    train_loader = DataLoader(Subset(full_ds, actual_train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=batch_size, shuffle=False)

    model = model_class(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_macro_f1 = -1.0
    best_preds = []
    best_targets = []

    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        all_p = []
        all_t = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds = torch.argmax(out, dim=1).cpu().numpy()
                all_p.extend(preds)
                all_t.extend(y.cpu().numpy())

        all_p = np.array(all_p)
        all_t = np.array(all_t)
        macro_f1 = f1_score(all_t, all_p, average='macro', zero_division=0)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_preds = all_p
            best_targets = all_t

    return best_targets, best_preds, model


def run_stress_benchmark():
    print("=" * 80)
    print("🍂 STRESS CLASSIFICATION BENCHMARK (IDENTICAL BALANCED 5-FOLD BASIS)")
    print("=" * 80)

    data = build_or_load_dataset(task="stress", force_rebuild=False)

    y = data['labels']
    groups = data['plant_ids']
    class_names = data['class_names']
    num_classes = len(class_names)

    print(f"Total Cavitation events: {len(y)}")
    print(f"Classes: {class_names} (Cut: {sum(y == 0)}, Drought: {sum(y == 1)})")
    print(f"Unique plants: {len(np.unique(groups))}")
    print("Validation Strategy: 5-Fold Stratified Group CV on plant_id with 1:1 Balanced Training")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    experiments = [
        ("1D-CNN (Raw Waveform)", Waveform1DCNN, "1d", MODEL_OUT_DIR / "stress_1d_cnn.pt"),
        ("2D-CNN (CWT Scalogram)", Scalogram2DCNN, "2d", MODEL_OUT_DIR / "stress_cwt_2d_cnn.pt")
    ]

    all_results = {}

    for exp_name, model_cls, mode, save_path in experiments:
        print(f"\n--- Training {exp_name} across 5 Group-Folds on plant_id (Balanced Train) ---")
        pooled_true = []
        pooled_pred = []
        fold_baccs = []
        fold_f1s = []

        last_model = None
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(data['waveforms'], y, groups=groups)):
            t, p, mdl = train_and_eval_fold(
                model_cls, mode, data, train_idx, val_idx,
                num_classes=num_classes, num_epochs=12, batch_size=32, lr=0.001, balance_train=True
            )
            pooled_true.extend(t)
            pooled_pred.extend(p)
            last_model = mdl

            f1 = f1_score(t, p, average='macro', zero_division=0)
            bacc = balanced_accuracy_score(t, p)
            fold_baccs.append(bacc)
            fold_f1s.append(f1)
            print(f"  Fold {fold + 1}/5: Balanced Acc={bacc:.3f}, Macro F1={f1:.3f} (Val Size: {len(t)})")

        pooled_true = np.array(pooled_true)
        pooled_pred = np.array(pooled_pred)

        acc = accuracy_score(pooled_true, pooled_pred)
        pooled_bacc = balanced_accuracy_score(pooled_true, pooled_pred)
        mean_fold_bacc = float(np.mean(fold_baccs))
        prec = precision_score(pooled_true, pooled_pred, average='macro', zero_division=0)
        rec = recall_score(pooled_true, pooled_pred, average='macro', zero_division=0)
        macro_f1 = f1_score(pooled_true, pooled_pred, average='macro', zero_division=0)
        cm = confusion_matrix(pooled_true, pooled_pred).tolist()

        print(f"\n📊 Results for {exp_name}:")
        print(f"  Mean Fold Balanced Acc: {mean_fold_bacc:.4f} ({mean_fold_bacc*100:.2f}%)")
        print(f"  Pooled Balanced Acc:    {pooled_bacc:.4f} ({pooled_bacc*100:.2f}%)")
        print(f"  Overall Accuracy:       {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Macro F1-Score:         {macro_f1:.4f}")
        print(f"  Confusion Matrix:       {cm}")

        if last_model is not None:
            torch.save(last_model.state_dict(), save_path)

        all_results[exp_name] = {
            'mean_fold_balanced_accuracy': mean_fold_bacc,
            'balanced_accuracy': float(pooled_bacc),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'macro_f1': float(macro_f1),
            'confusion_matrix': cm,
            'class_names': class_names,
            'total_test_samples': int(len(pooled_true))
        }

    metrics_path = MODEL_OUT_DIR / "stress_dl_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved stress deep learning metrics (balanced basis) to {metrics_path}")


if __name__ == "__main__":
    run_stress_benchmark()
