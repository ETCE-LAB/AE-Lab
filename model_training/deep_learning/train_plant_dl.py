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
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
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


def train_and_eval_lopo(model_class, mode, dataset_data, num_classes=3, num_epochs=10, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    plant_ids = dataset_data['plant_ids']
    unique_plants = np.unique(plant_ids)
    y_all = dataset_data['labels']
    class_names = dataset_data['class_names']
    
    full_ds = AEDataset(
        dataset_data['waveforms'],
        dataset_data['scalograms'],
        dataset_data['labels'],
        mode=mode
    )
    
    lopo_true = []
    lopo_pred = []
    per_plant_results = []
    
    print(f"Running Leave-One-Plant-Out across {len(unique_plants)} unique plants...")
    
    last_trained_model = None
    
    for held_out_plant in unique_plants:
        test_mask = (plant_ids == held_out_plant)
        train_mask = ~test_mask
        
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        
        # Check that training part has all classes
        if len(np.unique(y_all[train_idx])) < num_classes:
            print(f"  Plant {held_out_plant}: skipped (species would be missing from training set)")
            continue
            
        train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(full_ds, test_idx), batch_size=batch_size, shuffle=False)
        
        model = model_class(num_classes=num_classes).to(device)
        
        # Compute class weights for training subset
        counts = np.bincount(y_all[train_idx], minlength=num_classes)
        weights = torch.tensor([counts.sum() / max(c, 1) for c in counts], dtype=torch.float32).to(device)
        weights = weights / weights.sum()
        
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Train
        for epoch in range(num_epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                
        # Test on held-out plant
        model.eval()
        p_plant = []
        t_plant = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds = torch.argmax(out, dim=1).cpu().numpy()
                p_plant.extend(preds)
                t_plant.extend(y.cpu().numpy())
                
        p_plant = np.array(p_plant)
        t_plant = np.array(t_plant)
        plant_acc = float((p_plant == t_plant).mean())
        
        species = class_names[np.bincount(t_plant, minlength=num_classes).argmax()]
        
        lopo_true.extend(t_plant)
        lopo_pred.extend(p_plant)
        
        per_plant_results.append({
            'plant_id': str(held_out_plant),
            'species': str(species),
            'n_events': int(len(test_idx)),
            'accuracy': plant_acc,
            'predicted_as': {str(class_names[k]): int((p_plant == k).sum()) for k in range(num_classes)}
        })
        print(f"  Plant {held_out_plant} ({species}, n={len(test_idx)}): LOPO Accuracy: {plant_acc:.1%}")
        last_trained_model = model

    return np.array(lopo_true), np.array(lopo_pred), per_plant_results, last_trained_model


def run_plant_benchmark():
    print("=" * 80)
    print("🌿 DEEP LEARNING BENCHMARK: PLANT SPECIES CLASSIFICATION (LOPO)")
    print("=" * 80)
    
    # Load dataset
    data = build_or_load_dataset(task="plant", force_rebuild=False)
    
    y = data['labels']
    class_names = data['class_names']
    num_classes = len(class_names)
    
    print(f"Total Cavitation events: {len(y)}")
    print(f"Classes: {class_names}")
    for idx, c in enumerate(class_names):
        print(f"  - {c}: {sum(y == idx)} events")
        
    experiments = [
        ("1D-CNN (Raw Waveform)", Waveform1DCNN, "1d", MODEL_OUT_DIR / "plant_1d_cnn.pt"),
        ("2D-CNN (CWT Scalogram)", Scalogram2DCNN, "2d", MODEL_OUT_DIR / "plant_cwt_2d_cnn.pt")
    ]
    
    all_results = {}
    
    for exp_name, model_cls, mode, save_path in experiments:
        print(f"\n--- Evaluating {exp_name} under Leave-One-Plant-Out ---")
        y_true, y_pred, per_plant, last_model = train_and_eval_lopo(
            model_cls, mode, data, num_classes=num_classes, num_epochs=8, batch_size=32
        )
        
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        mean_plant_acc = float(np.mean([p['accuracy'] for p in per_plant]))
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()
        
        majority_baseline = float(np.bincount(y_true, minlength=num_classes).max() / len(y_true))
        
        print(f"\n📊 Overall LOPO Results for {exp_name}:")
        print(f"  Overall Accuracy:          {acc:.4f} (Majority baseline: {majority_baseline:.4f})")
        print(f"  Balanced Accuracy:         {bal_acc:.4f}")
        print(f"  Mean Per-Plant Accuracy:   {mean_plant_acc:.4f}")
        print(f"  Macro Precision:           {prec:.4f}")
        print(f"  Macro Recall:              {rec:.4f}")
        print(f"  Macro F1-Score:            {f1:.4f}")
        
        if last_model is not None:
            torch.save(last_model.state_dict(), save_path)
            print(f"  Saved model weights to {save_path}")
            
        all_results[exp_name] = {
            'accuracy': float(acc),
            'balanced_accuracy': float(bal_acc),
            'mean_per_plant_accuracy': float(mean_plant_acc),
            'macro_precision': float(prec),
            'macro_recall': float(rec),
            'macro_f1': float(f1),
            'majority_baseline': majority_baseline,
            'per_plant': per_plant,
            'confusion_matrix': cm,
            'classes': class_names,
            'total_test_events': int(len(y_true))
        }
        
    metrics_path = MODEL_OUT_DIR / "plant_dl_metadata.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved all plant deep learning benchmark metrics to {metrics_path}")


if __name__ == "__main__":
    run_plant_benchmark()
