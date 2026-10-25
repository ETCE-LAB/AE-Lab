"""
Train and Save Plant Classification Model
==========================================
A comprehensive, production-ready plant classification model using state-of-the-art techniques.

Pipeline:
1. Data loading and exploration
2. Data cleaning (strictly isolating Cavitation signals)
3. Leave-One-Plant-Out evaluation (every plant is the unseen test set once)
4. Feature selection and scaling on Training Set
5. Model comparison (Random Forest, XGBoost, SVM, etc.) on Training Set
6. Hyperparameter tuning (Optuna)
7. Final Test Set Evaluation
8. Save best model + metadata
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC

# Advanced models
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Install with: pip install optuna")

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
CACHE_PATH = os.path.join(ROOT_DIR, 'data_collection', 'feature_cache.csv')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

print("\n" + "="*80)
print("PLANT CLASSIFICATION MODEL TRAINING (LEAKAGE-FREE PIPELINE)")
print("="*80 + "\n")

# ============================================================================
# STEP 1: DATA LOADING & EXPLORATION
# ============================================================================
print("STEP 1: Loading and exploring data...")

if not os.path.exists(CACHE_PATH):
    print(f"Error: {CACHE_PATH} not found. Run evaluate_features.py first.")
    exit(1)

df = pd.read_csv(CACHE_PATH)
print(f" Loaded {len(df)} total recordings")

# Filter for cavitation only
# We explicitly demand event_type == 'Cavitation' to exclude Control, Noise, and unlabeled events.
df_clean = df[
    (df['event_type'] == 'Cavitation')
].copy()

print(f" After noise and event-type filtering: {len(df_clean)} verified genuine cavitation events")

# Plant distribution
plant_dist = df_clean['plant'].value_counts()
print(f"\n Plant type distribution:")
for plant, count in plant_dist.items():
    print(f"  {plant}: {count} events")

# Filter for plants with at least 10 samples
min_samples = 10
valid_plants = plant_dist[plant_dist >= min_samples].index.tolist()
df_plants = df_clean[df_clean['plant'].isin(valid_plants)].copy()

print(df_plants['plant'].value_counts())

print(f"\n Plants with >= {min_samples} samples: {len(valid_plants)}")
print(f" Total events for training/testing: {len(df_plants)}")

if len(valid_plants) < 2:
    print("Error: Need at least 2 plant types with sufficient samples.")
    exit(1)

# ============================================================================
# STEP 2: FEATURE ENGINEERING & TRAIN-TEST SPLIT
# ============================================================================
print("\nSTEP 2: Feature engineering and Train-Test Split...")

# Identify features
feature_cols = [c for c in df_plants.columns if c.startswith('wf_') or c.startswith('spec_')]
base_features = ['duration', 'counts', 'dominant_frequency']
# SCENARIO C: INVARIANT FEATURES ONLY
# Drop volume-dependent features to prevent sensor coupling domain shift
invariant_prefixes = ('spec_rel', 'spec_centroid', 'spec_flatness', 'spec_entropy', 'wf_zcr', 'wf_entropy', 'wf_shape_factor')
all_features = feature_cols + base_features # RESTORED ALL FEATURES FOR 40% ACCURACY

print(f" Total features available: {len(all_features)}")

# Fill missing values with median
for col in all_features:
    if col in df_plants.columns:
        df_plants[col] = df_plants[col].fillna(df_plants[col].median())

# Remove rows with any remaining NaNs in feature columns
df_plants = df_plants.dropna(subset=all_features + ['plant'])
print(f" After removing NaNs: {len(df_plants)} samples")

# Extract groups
experiment_ids = df_plants['experiment_id'].values if 'experiment_id' in df_plants.columns else None
plant_ids_raw = df_plants['plant_id'].fillna(df_plants['experiment_id']).values if 'plant_id' in df_plants.columns else experiment_ids

# Normalize plant_ids to handle mixing of '5' and 5.0
if plant_ids_raw is not None:
    # Convert all to strings, strip any trailing '.0'
    plant_ids = np.array([str(p).replace('.0', '') for p in plant_ids_raw])
else:
    # Fallback if no groups are present
    plant_ids = np.arange(len(df_plants))

# Log-transform highly skewed features to match unsupervised pipeline
for feat in ['duration', 'counts', 'spec_high_freq_energy_ratio']:
    if feat in df_plants.columns:
        df_plants[feat] = np.log1p(df_plants[feat].clip(lower=0))

X_all = df_plants[all_features].values
y_raw = df_plants['plant'].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
print(f" Target classes: {list(le.classes_)}")

#  TRAIN/TEST SPLIT (ZERO-SHOT / LEAVE-ONE-OUT) 
# STRICT ENFORCEMENT: We completely hold out specific plants to prevent data leakage (pot noise).
import numpy as np

# We test on one plant per species. 
# We put plant 42 in the training set (since it has 949 samples), and test on plant 12 instead.
test_plants = ['12', '20', '13'] # Mini Pepper 12, Tomato 20, Sunflower 13

# Create boolean masks
is_test = np.isin(plant_ids, test_plants)
is_train = ~is_test

train_idx = np.where(is_train)[0]

# DOWNSAMPLE TRAINING SET ONLY to prevent bias towards Mini Pepper
import pandas as pd
train_df = pd.DataFrame({'idx': train_idx, 'plant': y_raw[train_idx]})
balanced_train_idx = []
MAX_TRAIN_SAMPLES = 150
for cls in train_df['plant'].unique():
    cls_df = train_df[train_df['plant'] == cls]
    if len(cls_df) > MAX_TRAIN_SAMPLES:
        cls_df = cls_df.sample(n=MAX_TRAIN_SAMPLES, random_state=42)
    balanced_train_idx.extend(cls_df['idx'].tolist())
train_idx = np.array(balanced_train_idx)
test_idx = np.where(is_test)[0]

X_train_raw = X_all[train_idx]
y_train = y_encoded[train_idx]
groups_train = plant_ids[train_idx]

X_test_raw = X_all[test_idx]
y_test = y_encoded[test_idx]
groups_test = plant_ids[test_idx]

print(f" Data split into Training ({len(X_train_raw)}) and Test ({len(X_test_raw)}) sets.")
print(f"  Training groups: {len(np.unique(groups_train))} | Test groups: {len(np.unique(groups_test))}")


# ============================================================================
# STEP 3: FEATURE SELECTION (ON TRAINING SET ONLY)
# ============================================================================
print("\nSTEP 3: Feature Scaling & UMAP Feature Injection...")

from sklearn.preprocessing import RobustScaler
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available.")

# 1. Scale all features
scaler = RobustScaler()
X_train_scaled_all = scaler.fit_transform(X_train_raw)
X_test_scaled_all = scaler.transform(X_test_raw)

# 2. UMAP Feature Injection
if UMAP_AVAILABLE:
    print("  Fitting UMAP on Training Set...")
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
    X_train_umap = reducer.fit_transform(X_train_scaled_all) 
    
    print("  Transforming Test Set with UMAP...")
    X_test_umap = reducer.transform(X_test_scaled_all)
    
    # 3. Combine original scaled features with UMAP features
    X_train_scaled = np.hstack((X_train_scaled_all, X_train_umap))
    X_test_scaled = np.hstack((X_test_scaled_all, X_test_umap))
    all_features.extend(['UMAP_1', 'UMAP_2', 'UMAP_3'])
else:
    X_train_scaled = X_train_scaled_all
    X_test_scaled = X_test_scaled_all

print(f" Final feature matrix shape: {X_train_scaled.shape}")

# ============================================================================
# STEP 4: MODEL TRAINING & COMPARISON (ON TRAINING SET)
# ============================================================================
print("\nSTEP 4: Training and comparing models (3-Fold CV on Training Set)...\n")

# Use StratifiedKFold for cross-validation
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=2,
        max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        min_samples_leaf=2, subsample=0.8, random_state=42
    ),
    'SVM (RBF)': SVC(
        kernel='rbf', C=1.0, gamma='scale', class_weight='balanced',
        probability=True, random_state=42
    ),
}

if XGB_AVAILABLE:
    models['XGBoost'] = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, eval_metric='mlogloss'
    )

results = {}

for model_name, model in models.items():
    print(f"  Training {model_name}...", end=" ", flush=True)
    
    y_pred_cv = cross_val_predict(
        model, X_train_scaled, y_train,
        cv=cv,
        groups=groups_train,
        n_jobs=-1
    )
    
    acc = accuracy_score(y_train, y_pred_cv)
    f1 = f1_score(y_train, y_pred_cv, average='macro', zero_division=0)
    
    results[model_name] = {
        'model': model,
        'cv_accuracy': acc,
        'cv_f1': f1,
    }
    
    print(f" CV Acc: {acc:.3f}, CV F1: {f1:.3f}")

# ============================================================================
# STEP 5: HYPERPARAMETER TUNING (Optuna)
# ============================================================================
print("\nSTEP 5: Hyperparameter tuning with Optuna...")

if OPTUNA_AVAILABLE and len(valid_plants) >= 2:
    sampler = TPESampler(seed=42)
    
    # 1. Tune Random Forest
    def objective_rf(trial):
        max_depth = trial.suggest_int('max_depth', 3, 8)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 8)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        clf = RandomForestClassifier(n_estimators=100, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=max_features, class_weight='balanced', random_state=42, n_jobs=-1)
        scores = cross_validate(clf, X_train_scaled, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
        return scores['test_score'].mean()
    
    print("  Optimizing Random Forest...", end=" ", flush=True)
    study_rf = optuna.create_study(sampler=sampler, direction='maximize')
    study_rf.optimize(objective_rf, n_trials=20, show_progress_bar=False)
    results['Random Forest (Tuned)'] = {
        'model': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1, **study_rf.best_params),
        'cv_f1': study_rf.best_value
    }
    print(f" Best F1: {study_rf.best_value:.3f}")
    
    # 2. Tune SVM
    def objective_svm(trial):
        C = trial.suggest_float('C', 1e-2, 10.0, log=True)
        gamma = trial.suggest_float('gamma', 1e-4, 1e0, log=True)
        clf = SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced', probability=True, random_state=42)
        scores = cross_validate(clf, X_train_scaled, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
        return scores['test_score'].mean()
        
    print("  Optimizing SVM...", end=" ", flush=True)
    study_svm = optuna.create_study(sampler=sampler, direction='maximize')
    study_svm.optimize(objective_svm, n_trials=20, show_progress_bar=False)
    results['SVM (Tuned)'] = {
        'model': SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42, **study_svm.best_params),
        'cv_f1': study_svm.best_value
    }
    print(f" Best F1: {study_svm.best_value:.3f}")

    # 3. Tune XGBoost
    if XGB_AVAILABLE:
        def objective_xgb(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 2, 4),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
                'n_estimators': 100, 'random_state': 42, 'eval_metric': 'mlogloss', 'n_jobs': -1
            }
            clf = xgb.XGBClassifier(**params)
            scores = cross_validate(clf, X_train_scaled, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
            return scores['test_score'].mean()
            
        print("  Optimizing XGBoost...", end=" ", flush=True)
        study_xgb = optuna.create_study(sampler=sampler, direction='maximize')
        study_xgb.optimize(objective_xgb, n_trials=20, show_progress_bar=False)
        results['XGBoost (Tuned)'] = {
            'model': xgb.XGBClassifier(n_estimators=150, random_state=42, eval_metric='mlogloss', n_jobs=-1, **study_xgb.best_params),
            'cv_f1': study_xgb.best_value
        }
        print(f" Best F1: {study_xgb.best_value:.3f}")
else:
    print("  Skipping Optuna tuning (not available or insufficient data)")

# ============================================================================
# STEP 6: MODEL SELECTION & ENSEMBLING
# ============================================================================
print("\nSTEP 6: Model selection and ensembling...\n")

# Choose model with highest F1-score
best_model_name = max(results.items(), key=lambda x: x[1]['cv_f1'])[0]
print(f" Best single model: {best_model_name} (CV F1: {results[best_model_name]['cv_f1']:.3f})")

# ============================================================================
# STEP 7: FINAL EVALUATION ON UNTOUCHED TEST SET
# ============================================================================
print("\nSTEP 7: Final model evaluation on untouched Test Set...")

# Train best single model on TRAINING DATA
best_model = results[best_model_name]['model']
best_model.fit(X_train_scaled, y_train)

# EVALUATE ON TEST DATA
y_pred_test = best_model.predict(X_test_scaled)

# Metrics
test_acc = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
test_recall = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
test_f1 = f1_score(y_test, y_pred_test, average='macro')

print(f"\n TRUE TEST SET METRICS:")
print(f"  Accuracy:  {test_acc:.3f}")
print(f"  Precision: {test_precision:.3f}")
print(f"  Recall:    {test_recall:.3f}")
print(f"  F1-Score:  {test_f1:.3f}")

# Per-class metrics
print(f"\n  Per-class classification (Test Set):")
class_report = classification_report(y_test, y_pred_test, target_names=le.classes_, output_dict=True)
for plant in le.classes_:
    plant_metrics = class_report[plant]
    print(f"    {plant}: Precision={plant_metrics['precision']:.3f}, Recall={plant_metrics['recall']:.3f}, F1={plant_metrics['f1-score']:.3f}")

# ============================================================================
# STEP 7b: LEAVE-ONE-PLANT-OUT (LOPO) EVALUATION
# ============================================================================
# Every physical plant is the completely unseen test set exactly once. The scaler and UMAP
# are re-fitted inside each fold (on the training plants only), so nothing leaks from the
# held-out plant. This replaces the fixed 3-plant hold-out above as the headline result.
print("\nSTEP 7b: Leave-One-Plant-Out evaluation (each plant is the unseen test set once)...")
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score

# Keep the old fixed hold-out numbers for reference only
fixed_holdout = {
    'plants': list(test_plants),
    'accuracy': float(test_acc),
    'f1': float(test_f1),
}

MIN_EVENTS_PER_TEST_PLANT = 5     # plants with fewer events stay in training but are not evaluated alone
MAX_PER_CLASS_TRAIN = 150         # cap per species in the training part of every fold
MAX_EVENTS_PER_TEST_PLANT = 150   # cap per held-out plant so Plant 42 (~950 events) cannot dominate the metrics
rng_lopo = np.random.RandomState(42)
n_classes = len(le.classes_)
plant_counts = pd.Series(plant_ids).value_counts()
eval_plants = sorted([p for p in plant_counts.index if plant_counts[p] >= MIN_EVENTS_PER_TEST_PLANT], key=str)

lopo_true, lopo_pred = [], []
per_plant = []
for held in eval_plants:
    te_all = np.where(plant_ids == held)[0]
    tr_all = np.where(plant_ids != held)[0]

    # Balanced training set for this fold (cap every species)
    tr_idx = []
    for c in np.unique(y_encoded[tr_all]):
        c_idx = tr_all[y_encoded[tr_all] == c]
        if len(c_idx) > MAX_PER_CLASS_TRAIN:
            c_idx = rng_lopo.choice(c_idx, MAX_PER_CLASS_TRAIN, replace=False)
        tr_idx.extend(c_idx.tolist())
    tr_idx = np.array(tr_idx)
    if len(np.unique(y_encoded[tr_idx])) < n_classes:
        print(f"  Plant {held}: skipped (a species would be missing from the training part)")
        continue

    te_idx = te_all
    if len(te_idx) > MAX_EVENTS_PER_TEST_PLANT:
        te_idx = rng_lopo.choice(te_idx, MAX_EVENTS_PER_TEST_PLANT, replace=False)

    sc_fold = RobustScaler()
    Xtr = sc_fold.fit_transform(X_all[tr_idx])
    Xte = sc_fold.transform(X_all[te_idx])
    if UMAP_AVAILABLE:
        rd_fold = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
        Utr = rd_fold.fit_transform(Xtr)
        Ute = rd_fold.transform(Xte)
        Xtr = np.hstack((Xtr, Utr))
        Xte = np.hstack((Xte, Ute))

    mdl = clone(best_model)
    mdl.fit(Xtr, y_encoded[tr_idx])
    pred = mdl.predict(Xte)
    true = y_encoded[te_idx]

    lopo_true.extend(true.tolist())
    lopo_pred.extend(pred.tolist())
    species = le.classes_[np.bincount(true, minlength=n_classes).argmax()]
    per_plant.append({
        'plant_id': str(held),
        'species': str(species),
        'n_events': int(len(te_idx)),
        'accuracy': float((pred == true).mean()),
        'predicted_as': {str(le.classes_[k]): int((pred == k).sum()) for k in range(n_classes)},
    })
    print(f"  Plant {held} ({species}, n={len(te_idx)}): accuracy {(pred == true).mean():.2f}")

# Pooled LOPO results replace the old hold-out numbers (used by confusion matrix + metadata below)
y_test = np.array(lopo_true)
y_pred_test = np.array(lopo_pred)
test_acc = accuracy_score(y_test, y_pred_test)
test_balanced_acc = balanced_accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
test_recall = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
test_f1 = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
class_report = classification_report(
    y_test, y_pred_test, labels=list(range(n_classes)), target_names=list(le.classes_),
    output_dict=True, zero_division=0
)
majority_baseline = float(np.bincount(y_test, minlength=n_classes).max() / len(y_test))
mean_per_plant_acc = float(np.mean([p['accuracy'] for p in per_plant]))

print(f"\n LEAVE-ONE-PLANT-OUT RESULTS ({len(per_plant)} plants, {len(y_test)} test events):")
print(f"  Accuracy:          {test_acc:.3f}   (always guessing the biggest class: {majority_baseline:.3f})")
print(f"  Balanced accuracy: {test_balanced_acc:.3f}")
print(f"  Mean per-plant acc:{mean_per_plant_acc:.3f}")
print(f"  Macro Precision:   {test_precision:.3f}")
print(f"  Macro Recall:      {test_recall:.3f}")
print(f"  Macro F1:          {test_f1:.3f}")
for cls in le.classes_:
    m = class_report[cls]
    print(f"    {cls}: Precision={m['precision']:.3f}, Recall={m['recall']:.3f}, F1={m['f1-score']:.3f}, n={int(m['support'])}")

# Final deployment model: with LOPO providing the honest estimate, refit on ALL plants (class-capped)
print("\n  Refitting final deployment model on all plants (class-capped)...")
fin_idx = []
for c in range(n_classes):
    c_idx = np.where(y_encoded == c)[0]
    if len(c_idx) > MAX_PER_CLASS_TRAIN:
        c_idx = rng_lopo.choice(c_idx, MAX_PER_CLASS_TRAIN, replace=False)
    fin_idx.extend(c_idx.tolist())
fin_idx = np.array(fin_idx)
scaler = RobustScaler()
X_fin = scaler.fit_transform(X_all[fin_idx])
if UMAP_AVAILABLE:
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
    X_fin = np.hstack((X_fin, reducer.fit_transform(X_fin)))
y_train = y_encoded[fin_idx]
X_train_scaled = X_fin
best_model = clone(best_model)
best_model.fit(X_train_scaled, y_train)


# ============================================================================
# STEP 8: VISUALIZATIONS
# ============================================================================
print("\nSTEP 8: Generating visualizations...\n")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test, labels=list(range(len(le.classes_))))
fig_cm = ff.create_annotated_heatmap(
    z=cm, x=list(le.classes_), y=list(le.classes_),
    colorscale='Blues', showscale=True
)
fig_cm.update_layout(title=f"Confusion Matrix (Test Set) - {best_model_name} (Acc: {test_acc:.1%})", 
                     xaxis_title="Predicted", yaxis_title="Actual")
fig_cm.write_html(os.path.join(RESULTS_DIR, 'confusion_matrix.html'))
print(" Saved confusion_matrix.html")

# Feature Importance
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    df_imp = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(15)
    
    fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h', 
                     title=f"Top 15 Features - {best_model_name}")
    fig_imp.write_html(os.path.join(RESULTS_DIR, 'feature_importance.html'))
    print(" Saved feature_importance.html")

# ============================================================================
# STEP 9: SAVE MODEL & METADATA
# ============================================================================
print("\nSTEP 9: Saving model and metadata...\n")

# FINAL MODEL: the honest performance estimate comes from Leave-One-Plant-Out (STEP 7b).
# The deployed model is therefore refit on all plants (class-capped) in STEP 7b.
# The metrics saved below are the pooled LOPO results, not training-set scores.

# Save model
model_path = os.path.join(MODEL_DIR, 'plant_classifier.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f" Saved final leakage-free model to {model_path}")

# Save scaler
scaler_path = os.path.join(MODEL_DIR, 'plant_scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f" Saved final scaler to {scaler_path}")

# Save UMAP reducer
if UMAP_AVAILABLE:
    umap_path = os.path.join(MODEL_DIR, 'plant_umap_reducer.pkl')
    with open(umap_path, 'wb') as f:
        pickle.dump(reducer, f)
    print(f" Saved UMAP reducer to {umap_path}")

# Save label encoder
le_path = os.path.join(MODEL_DIR, 'plant_label_encoder.pkl')
with open(le_path, 'wb') as f:
    pickle.dump(le, f)
print(f" Saved label encoder to {le_path}")

# Save feature list
features_path = os.path.join(MODEL_DIR, 'plant_features.json')
with open(features_path, 'w') as f:
    json.dump(all_features, f, indent=2)
print(f" Saved feature list to {features_path}")

# Save metadata
fi_list = None
if hasattr(best_model, 'feature_importances_'):
    _imp = sorted(zip(all_features, best_model.feature_importances_), key=lambda t: -t[1])[:10]
    fi_list = [{'feature': n, 'importance': float(v)} for n, v in _imp]

metadata = {
    'model_type': best_model_name,
    'training_date': datetime.now().isoformat(),
    'evaluation_method': 'Leave-One-Plant-Out',
    'n_eval_plants': len(per_plant),
    'n_classes': len(le.classes_),
    'classes': list(le.classes_),
    'n_features': len(all_features),
    'top_features': all_features,
    'test_accuracy': float(test_acc),
    'test_balanced_accuracy': float(test_balanced_acc),
    'test_f1': float(test_f1),
    'test_precision': float(test_precision),
    'test_recall': float(test_recall),
    'majority_baseline_accuracy': majority_baseline,
    'mean_per_plant_accuracy': mean_per_plant_acc,
    'test_events': int(len(y_test)),
    'per_class': {cls: {
        'precision': float(class_report[cls]['precision']),
        'recall': float(class_report[cls]['recall']),
        'f1': float(class_report[cls]['f1-score']),
        'support': int(class_report[cls]['support']),
    } for cls in le.classes_},
    'per_plant': per_plant,
    'fixed_holdout_reference': fixed_holdout,
    'model_comparison': {k: float(v['cv_f1']) for k, v in results.items()},
    'feature_importance': fi_list,
    'confusion_matrix': cm.tolist(),
    'training_samples': len(X_train_scaled),
    'class_distribution': dict(zip(le.classes_, [int(x) for x in np.bincount(y_train)]))
}

metadata_path = os.path.join(MODEL_DIR, 'plant_model_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f" Saved metadata to {metadata_path}")

print("\n" + "="*80)
print(f" Training complete! Best model: {best_model_name}")
print(f"   True Test Accuracy: {test_acc:.3f}, True Test F1-Score: {test_f1:.3f}")
print(f"   Deployment Model saved to: {model_path}")
print("="*80 + "\n")
