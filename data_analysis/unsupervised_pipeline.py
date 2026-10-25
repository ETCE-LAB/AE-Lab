import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
try:
    import hdbscan
    import umap
    ADVANCED_LIBS = True
except ImportError:
    ADVANCED_LIBS = False
    print("UMAP/HDBSCAN not available. Using PCA and GaussianMixture.")

import plotly.express as px
import plotly.graph_objects as go

def run_pipeline(cache_path: Path, results_dir: Path, db_path: Path = None, mode: str = 'all'):
    if not cache_path.exists():
        raise FileNotFoundError(f"Features cache not found at {cache_path}. Run evaluate_features.py first.")

    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data (Mode: {mode})...")
    df = pd.read_csv(cache_path, low_memory=False)
    
    if db_path is not None and db_path.exists():
        import sqlite3
        conn = sqlite3.connect(db_path)
        labels_df = pd.read_sql_query("SELECT recording_id, event_type, plant, stress_type FROM recordings", conn)
        conn.close()
        
        labels_df['recording_id'] = labels_df['recording_id'].astype(str)
        df['recording_id'] = df['recording_id'].astype(str)
        
        # Drop columns if they exist in df to avoid _x _y suffixes
        cols_to_drop = [c for c in ['event_type', 'plant', 'stress_type'] if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            
        df = pd.merge(df, labels_df, on='recording_id', how='left')
        
        if mode in ['plants', 'stress']:
            df = df[df['event_type'] == 'Cavitation'].copy()
            if df.empty:
                raise ValueError(f"No Cavitation events found in cache for mode '{mode}'.")
    
    # USING HULL/SHAPE FEATURES FOR NATURAL SEPARATION
    # 1. dominant_frequency (>100kHz for cavitation)
    # 2. wf_build_up_energy_ratio (Energy before peak vs after peak. Near 0 for cavitation).
    # 3. wf_kurtosis (Spikiness of the signal. Very high for cavitation).
    # 4. wf_crest_factor (Peak vs RMS. High for cavitation).
    # 5. wf_skewness (Symmetry of the signal. Cavitation is a symmetric acoustic wave near 0. Noise is often asymmetric).
    all_features = [
        'dominant_frequency',
        'wf_build_up_energy_ratio',
        'wf_kurtosis',
        'wf_crest_factor',
        'wf_skewness'
    ]
    
    available_features = [f for f in all_features if f in df.columns]
    if not available_features:
        raise ValueError("No required features found in the cache.")
        
    df_clean = df.dropna(subset=available_features).copy()
    if df_clean.empty:
        raise ValueError("No rows left after dropping NAs for required features.")
        
    # Log-transform highly skewed features
    for feat in ['duration', 'counts', 'spec_high_freq_energy_ratio']:
        if feat in available_features:
            df_clean[feat] = np.log1p(df_clean[feat].clip(lower=0))
        
    X = df_clean[available_features].values
    
    print("Scaling features...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    with open(results_dir / 'unsupervised_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Reducing dimensions...")
    # Prepare Semi-Supervised Labels
    y_labels = None
    if mode == 'all' and 'event_type' in df_clean.columns:
        conditions = [
            df_clean['event_type'] == 'Cavitation',
            df_clean['event_type'].isin(['Noise', 'Kontrolle', 'Control'])
        ]
        choices = [1, 0]
        y_labels = np.select(conditions, choices, default=-1)
    elif mode == 'plants' and 'plant' in df_clean.columns:
        plants = df_clean['plant'].dropna().unique()
        mapping = {p: i for i, p in enumerate(plants)}
        y_labels = df_clean['plant'].map(mapping).fillna(-1).astype(int).values
    elif mode == 'stress' and 'stress_type' in df_clean.columns:
        stress_types = df_clean['stress_type'].dropna().unique()
        mapping = {s: i for i, s in enumerate(stress_types)}
        y_labels = df_clean['stress_type'].map(mapping).fillna(-1).astype(int).values

    if ADVANCED_LIBS:
        reducer = umap.UMAP(n_components=3, n_neighbors=50, min_dist=0.1, random_state=42)
        X_reduced = reducer.fit_transform(X_scaled, y=y_labels)
        dim_name = 'UMAP'
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=3, perplexity=min(100, len(X_scaled)-1), random_state=42, n_jobs=-1)
        X_reduced = reducer.fit_transform(X_scaled)
        dim_name = 'TSNE'
        
    df_clean[f'{dim_name}_1'] = X_reduced[:, 0]
    df_clean[f'{dim_name}_2'] = X_reduced[:, 1]
    df_clean[f'{dim_name}_3'] = X_reduced[:, 2]

    print("Running Isolation Forest (Anomaly Detection)...")
    iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    iso_preds = iso_forest.fit_predict(X_scaled)
    df_clean['Isolation_Forest_Label'] = np.where(iso_preds == -1, 'Anomaly (Cavitation?)', 'Normal (Noise?)')

    print("Running Clustering...")
    if ADVANCED_LIBS:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, prediction_data=True)
        cluster_labels = clusterer.fit_predict(X_scaled)
        df_clean['Cluster_Label'] = [f"Cluster {l}" if l != -1 else "Noise/Outlier" for l in cluster_labels]
    else:
        n_comp = min(2, len(X_scaled))
        gmm = GaussianMixture(n_components=n_comp, random_state=42)
        cluster_labels = gmm.fit_predict(X_scaled)
        df_clean['Cluster_Label'] = [f"Cluster {l}" for l in cluster_labels]

    output_csv = results_dir / f'unsupervised_results_{mode}.csv'
    df_clean.to_csv(output_csv, index=False)
    print(f"Saved unsupervised results to {output_csv}")
    
    return True, f"Successfully trained models and saved to {output_csv}"
