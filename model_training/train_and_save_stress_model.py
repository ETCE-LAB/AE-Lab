"""
Train and Save Stress Classification Model
==========================================
A comprehensive, production-ready stress classification model (Drought vs. Cut)
using state-of-the-art techniques and strictly preventing data leakage.
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

import plotly.express as px
import plotly.figure_factory as ff

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
CACHE_PATH = os.path.join(ROOT_DIR, 'data_collection', 'feature_cache.csv')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

print("\n" + "="*80)
print("🍂 STRESS CLASSIFICATION MODEL TRAINING (LEAKAGE-FREE PIPELINE)")
print("="*80 + "\n")

print("STEP 1: Loading and exploring data...")
if not os.path.exists(CACHE_PATH):
    print(f"Error: {CACHE_PATH} not found.")
    exit(1)

df = pd.read_csv(CACHE_PATH)
df_clean = df[
    (df['noise_probability'] < 0.5) & 
    (df['event_type'] == 'Cavitation')
].copy()

valid_stresses = ['drought', 'cut']
df_stress = df_clean[df_clean['stress_type'].isin(valid_stresses)].copy()

print(f"✓ Events with target stress types ({valid_stresses}): {len(df_stress)}")

print("\nSTEP 2: Feature engineering and Train-Test Split...")
feature_cols = [c for c in df_stress.columns if c.startswith('wf_') or c.startswith('spec_')]
base_features = ['duration', 'counts', 'dominant_frequency']
all_features = feature_cols + base_features

for col in all_features:
    if col in df_stress.columns:
        df_stress[col] = df_stress[col].fillna(df_stress[col].median())

df_stress = df_stress.dropna(subset=all_features + ['stress_type'])

experiment_ids = df_stress['experiment_id'].values if 'experiment_id' in df_stress.columns else None
plant_ids_raw = df_stress['plant_id'].fillna(df_stress['experiment_id']).values if 'plant_id' in df_stress.columns else experiment_ids

if plant_ids_raw is not None:
    plant_ids = np.array([str(p).replace('.0', '') for p in plant_ids_raw])
else:
    plant_ids = np.arange(len(df_stress))

# Log-transform highly skewed features to match unsupervised pipeline
for feat in ['duration', 'counts', 'spec_high_freq_energy_ratio']:
    if feat in df_stress.columns:
        df_stress[feat] = np.log1p(df_stress[feat].clip(lower=0))

X_all = df_stress[all_features].values
y_raw = df_stress['stress_type'].values

le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)

sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(X_all, y_encoded, groups=plant_ids))

X_train_raw = X_all[train_idx]
y_train = y_encoded[train_idx]
groups_train = plant_ids[train_idx]

X_test_raw = X_all[test_idx]
y_test = y_encoded[test_idx]

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
else:
    X_train_scaled = X_train_scaled_all
    X_test_scaled = X_test_scaled_all

print(f"✓ Final feature matrix shape: {X_train_scaled.shape}")


