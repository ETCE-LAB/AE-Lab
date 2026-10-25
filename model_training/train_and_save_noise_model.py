"""
Train and Save Supervised Noise Classification Model (Noise vs. Signal)
========================================================================
A comprehensive machine learning pipeline following best practices.
1. Data loading and preprocessing
2. Feature engineering and selection
3. Model comparison using 5-Fold Stratified Cross-Validation
4. Best model retraining and asset packaging
5. Model diagnostics and feature importances
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import sqlite3
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score, make_scorer,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

# Advanced models
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Visualization
import plotly.graph_objects as go
import plotly.express as px
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
print("🚨 SUPERVISED NOISE CLASSIFICATION MODEL TRAINING")
print("="*80 + "\n")

# ============================================================================
# STEP 1: DATA LOADING & PREPROCESSING
# ============================================================================
print("STEP 1: Loading and exploring data...")

if not os.path.exists(CACHE_PATH):
    print(f"❌ Error: {CACHE_PATH} not found. Run evaluate_features.py first.")
    exit(1)

df = pd.read_csv(CACHE_PATH)
print(f"✓ Loaded {len(df)} total recordings from cache")

# Pull fresh labels from DB
print("Fetching freshest manual labels from database...")
db_path = os.path.join(ROOT_DIR, 'metadata.db')
conn = sqlite3.connect(db_path)
query = "SELECT recording_id, event_type FROM recordings"
db_labels = pd.read_sql_query(query, conn)
conn.close()

if 'event_type' in df.columns:
    df = df.drop(columns=['event_type'])
df = pd.merge(df, db_labels, on='recording_id', how='left')

# Target label: noise (event_type == 'noise') vs signal (event_type != 'noise')
df['is_noise'] = (df['event_type'].isin(['Noise', 'Kontrolle', 'Control'])).astype(int)

# Drop rows with missing labels before splitting
df = df.dropna(subset=['event_type'])

# ============================================================================
# STEP 2: TRAIN/TEST SPLIT (Before balancing/scaling to prevent leakage)
# ============================================================================
print("\nSTEP 2: Splitting Data...")

df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['is_noise'], random_state=42)
print(f"✓ Split data into Train ({len(df_train)}) and Test ({len(df_test)})")

# Separate classes IN TRAINING DATA ONLY
df_noise_train = df_train[df_train['is_noise'] == 1].copy()
df_signal_train = df_train[df_train['is_noise'] == 0].copy()

n_noise_train = len(df_noise_train)
n_signal_train = len(df_signal_train)

# Imbalance handling: downsample noise to 5x the size of signal IN TRAINING DATA ONLY
ratio = 5
n_noise_sampled = min(n_noise_train, n_signal_train * ratio)
df_noise_sampled = df_noise_train.sample(n=n_noise_sampled, random_state=42)

df_train_balanced = pd.concat([df_signal_train, df_noise_sampled]).copy()
print(f"✓ Created balanced training subset (Ratio {ratio}:1):\n  - Labeled Noise: {len(df_noise_sampled)} events\n  - Labeled Signal: {len(df_signal_train)} events")

# ============================================================================
# STEP 3: FEATURE ENGINEERING & SELECTION
# ============================================================================
print("\nSTEP 3: Preprocessing features...")

# Identify features
feature_cols = [c for c in df_train_balanced.columns if c.startswith('wf_') or c.startswith('spec_')]
base_features = ['duration', 'counts', 'dominant_frequency']
all_features = feature_cols + base_features

print(f"✓ Total features available: {len(all_features)}")

# Fill missing values using the median of the TRAINING set to avoid leakage
for col in all_features:
    if col in df_train_balanced.columns:
        median_val = df_train_balanced[col].median()
        df_train_balanced[col] = df_train_balanced[col].fillna(median_val)
        df_test[col] = df_test[col].fillna(median_val)

# Drop remaining NaNs
df_train_balanced = df_train_balanced.dropna(subset=all_features)
df_test = df_test.dropna(subset=all_features)
print(f"✓ After removing NaNs: {len(df_train_balanced)} train samples, {len(df_test)} test samples")

# Extract X and y
X_train_raw = df_train_balanced[all_features].values
y_train_str = np.where(df_train_balanced['event_type'].isin(['Noise', 'Kontrolle', 'Control']), 'noise', 'signal')

X_test_raw = df_test[all_features].values
y_test_str = np.where(df_test['event_type'].isin(['Noise', 'Kontrolle', 'Control']), 'noise', 'signal')

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_str)
y_test_encoded = le.transform(y_test_str)
print(f"✓ Target classes: {list(le.classes_)}")

# Force model to use the curated feature list from UMAP analysis
top_features = [
    'duration',
    'dominant_frequency',
    'spec_rel_energy_0.1_0.2', 
    'wf_crest_factor',
    'wf_zcr',
    'wf_impulse_factor',
    'wf_margin_factor',
    'wf_kurtosis',
    'wf_high_amp_counts',
    'wf_build_up_energy_ratio'
]

print(f"✓ Using {len(top_features)} curated, mathematically robust features for noise classification")

print(f"  Top selected features: {top_features}")

# Prepare training data with selected features
X_train_selected = df_train_balanced[top_features].values
X_test_selected = df_test[top_features].values

# Fit scaler only on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

print("✓ Scaling complete (fit on train, applied to test)")

# ============================================================================
# STEP 4: MODEL TRAINING & COMPARISON
# ============================================================================
print("\nSTEP 4: Setting up cross-validation on Training Set...\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=3,
        subsample=0.8, random_state=42
    ),
    'SVM (RBF)': SVC(
        kernel='rbf', C=1.0, class_weight='balanced',
        probability=True, random_state=42
    )
}

if XGB_AVAILABLE:
    # Scale pos_weight to handle class imbalance
    # signal=1, noise=0
    scale_pos = (y_train_encoded == 0).sum() / (y_train_encoded == 1).sum()
    models['XGBoost'] = xgb.XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos,
        random_state=42, n_jobs=-1, eval_metric='logloss'
    )

f2_scorer = make_scorer(fbeta_score, beta=2.0, pos_label=1)

results = {}

for name, model in models.items():
    print(f"  Training {name}...", end=" ", flush=True)
    cv_results = cross_validate(
        model, X_train_scaled, y_train_encoded,
        cv=cv,
        scoring={
            'accuracy': 'accuracy', 
            'precision': 'precision_weighted', 
            'recall': 'recall_weighted', 
            'f2_signal': f2_scorer
        },
        n_jobs=-1
    )
    acc_mean = cv_results['test_accuracy'].mean()
    f2_mean = cv_results['test_f2_signal'].mean()
    
    results[name] = {
        'model': model,
        'accuracy': cv_results['test_accuracy'],
        'f2_signal': cv_results['test_f2_signal'],
        'precision': cv_results['test_precision'],
        'recall': cv_results['test_recall']
    }
    print(f"✓ Acc: {acc_mean:.3f} ± {cv_results['test_accuracy'].std():.3f}, F2 (Signal): {f2_mean:.3f}")

# ============================================================================
# STEP 5: MODEL EVALUATION ON HELD-OUT TEST SET
# ============================================================================
print("\nSTEP 5: Evaluating best model on held-out TEST SET...")

# Choose model with highest mean F2-score for Signal
best_model_name = max(results.items(), key=lambda x: x[1]['f2_signal'].mean())[0]
best_model = results[best_model_name]['model']

print(f"✓ Best model selected: {best_model_name}")

# Fit best model on all training data
best_model.fit(X_train_scaled, y_train_encoded)
print(f"✓ Fitted final {best_model_name} model on full training dataset")

# Evaluate strictly on Test Set
y_pred = best_model.predict(X_test_scaled)

# Performance metrics
final_acc = accuracy_score(y_test_encoded, y_pred)
final_prec = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
final_rec = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
final_f1 = f1_score(y_test_encoded, y_pred, average='weighted')
final_f2_signal = fbeta_score(y_test_encoded, y_pred, beta=2.0, pos_label=1)

print(f"\n📈 Final Test Set Metrics (True Distribution, Zero Leakage):")
print(f"  - Accuracy:  {final_acc:.3f}")
print(f"  - Precision: {final_prec:.3f}")
print(f"  - Recall:    {final_rec:.3f}")
print(f"  - F1-Score:  {final_f1:.3f}")
print(f"  - F2-Score (Signal): {final_f2_signal:.3f}")

cm = confusion_matrix(y_test_encoded, y_pred)

# Per-class metrics
print("\n  Per-class metrics (Test Set):")
report = classification_report(y_test_encoded, y_pred, target_names=le.classes_, output_dict=True)
for c in le.classes_:
    c_metrics = report[c]
    print(f"    - {c}: Precision={c_metrics['precision']:.3f}, Recall={c_metrics['recall']:.3f}, F1-Score={c_metrics['f1-score']:.3f}")

# ============================================================================
# STEP 5: MODEL DIAGNOSTICS & FEATURE IMPORTANCES
# ============================================================================
print("\nSTEP 5: Computing feature importances and diagnostics...")

# Compute Permutation Feature Importance (robust for all model types)
r = permutation_importance(best_model, X_train_scaled, y_train_encoded, n_repeats=10, random_state=42, n_jobs=-1)
importances = np.maximum(r.importances_mean, 0.0)

# If permutation importances are all 0, fallback to equal importances
if importances.sum() == 0.0:
    importances = np.ones(len(top_features)) / len(top_features)

feat_imp = pd.DataFrame({
    'Feature': top_features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(f"✓ Feature rankings computed successfully")
print(f"  Top features: {feat_imp['Feature'].head(5).tolist()}")

# Generate confusion matrix for Test Set
cm = confusion_matrix(y_test_encoded, y_pred)
fig_cm = ff.create_annotated_heatmap(
    z=cm, x=list(le.classes_), y=list(le.classes_),
    colorscale='Purples', showscale=True
)
fig_cm.update_layout(
    title=f"Confusion Matrix - Noise Model ({best_model_name})",
    xaxis_title="Predicted Class",
    yaxis_title="Actual Class",
    template="plotly_dark"
)
cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix_noise.html')
fig_cm.write_html(cm_path)
print(f"✓ Saved confusion matrix to {cm_path}")

# Save Feature Importance plot
fig_imp = px.bar(
    feat_imp.head(15), x='Importance', y='Feature', orientation='h',
    title=f"Feature Importances - Noise Model ({best_model_name})",
    labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'},
    template="plotly_dark"
)
fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
imp_path = os.path.join(RESULTS_DIR, 'feature_importance_noise.html')
fig_imp.write_html(imp_path)
print(f"✓ Saved feature importances plot to {imp_path}")

# ============================================================================
# STEP 6: SAVE ASSETS
# ============================================================================
print("\nSTEP 6: Saving model assets...")

# Model file
model_path = os.path.join(MODEL_DIR, 'noise_classifier.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"✓ Saved classifier model: {model_path}")

# Scaler
scaler_path = os.path.join(MODEL_DIR, 'noise_scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Saved scaler: {scaler_path}")

# Label encoder
le_path = os.path.join(MODEL_DIR, 'noise_label_encoder.pkl')
with open(le_path, 'wb') as f:
    pickle.dump(le, f)
print(f"✓ Saved label encoder: {le_path}")

# Feature list
features_json_path = os.path.join(MODEL_DIR, 'noise_features.json')
with open(features_json_path, 'w') as f:
    json.dump(top_features, f, indent=2)
print(f"✓ Saved feature list: {features_json_path}")

# Metadata
metadata = {
    'model_type': best_model_name,
    'training_date': datetime.now().isoformat(),
    'n_classes': len(le.classes_),
    'classes': list(le.classes_),
    'n_features': len(top_features),
    'top_features': top_features,
    'test_accuracy': float(final_acc),
    'test_f1': float(final_f1),
    'test_f2_signal': float(final_f2_signal),
    'test_precision': float(final_prec),
    'test_recall': float(final_rec),
    'confusion_matrix': cm.tolist(),
    'training_samples': len(df_train_balanced),
    'class_distribution': {'noise': int(n_noise_sampled), 'signal': int(n_signal_train)}
}
metadata_path = os.path.join(MODEL_DIR, 'noise_model_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Saved metadata: {metadata_path}")

print("\n" + "="*80)
print(f"✅ Noise training complete! Best model: {best_model_name}")
print(f"   Accuracy: {final_acc:.3f}, F1-Score: {final_f1:.3f}")
print("="*80 + "\n")
