import pandas as pd
import numpy as np
import sqlite3
import os
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("Warning: PyWavelets (pywt) not found. CWT features will be skipped. Install with: pip install PyWavelets")

from pathlib import Path

# ==========================================
# Configuration & Paths
# ==========================================
# Determine project root assuming this script is in project_root/data_analysis/
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DB_PATH = PROJECT_ROOT / 'metadata.db'
DATA_DIR = PROJECT_ROOT / 'data'
CACHE_PATH = PROJECT_ROOT / 'data_collection' / 'feature_cache.csv'
REPORT_PATH = PROJECT_ROOT / 'feature_analysis.html'


def extract_features(recording_id: str) -> dict:
    """
    Extracts time-domain and frequency-domain features from scope and spectrum CSV files 
    for a given recording ID.
    
    Args:
        recording_id (str): The unique identifier for the recording.
        
    Returns:
        dict: A dictionary containing the extracted features.
    """
    features = {}
    scope_path = DATA_DIR / f"{recording_id}-measurement-scope.csv"
    spec_path = DATA_DIR / f"{recording_id}-measurement-spectrum.csv"

    
    if scope_path.exists():
        try:
            df_scope = pd.read_csv(scope_path)
            volt_col = [c for c in df_scope.columns if 'Voltage' in c or 'V' in c]
            if volt_col:
                v_raw = df_scope[volt_col[0]].values
                abs_v_raw = np.abs(v_raw)
                global_peak_idx = np.argmax(abs_v_raw)
                
                # --- NEW BUILD-UP FEATURES (Calculated on RAW UNWINDOWED data) ---
                raw_energy_before = np.sum(v_raw[:global_peak_idx]**2)
                raw_energy_after = np.sum(v_raw[global_peak_idx:]**2)
                features['wf_build_up_energy_ratio'] = raw_energy_before / (raw_energy_after + 1e-9)
                
                high_thresh = 0.5 * abs_v_raw[global_peak_idx]
                is_high = abs_v_raw > high_thresh
                features['wf_high_amp_counts'] = ((is_high[:-1] == False) & (is_high[1:] == True)).sum()
                
                # --- TRANSIENT WINDOWING ---
                # Isolate a 5 ms window around the peak (1 ms before, 4 ms after @ 1 MHz)
                win_start = max(0, global_peak_idx - 1000)
                win_end = min(len(v_raw), global_peak_idx + 4000)
                
                v = v_raw[win_start:win_end]
                abs_v = abs_v_raw[win_start:win_end]
                features['wf_variance'] = np.var(v)
                features['wf_skewness'] = skew(v)
                features['wf_kurtosis'] = kurtosis(v)
                features['wf_rms'] = np.sqrt(np.mean(v**2))
                features['wf_crest_factor'] = np.max(abs_v) / (features['wf_rms'] + 1e-9)
                features['wf_zcr'] = ((v[:-1] * v[1:]) < 0).sum()
                
                peak_idx = np.argmax(abs_v)
                max_v = abs_v[peak_idx]
                threshold = 0.1 * max_v
                
                # Walk backwards from peak to find true rise start
                rev_before_peak = abs_v[:peak_idx][::-1]
                try: 
                    dist_to_start = np.where(rev_before_peak < threshold)[0][0]
                    rise_start = peak_idx - dist_to_start
                except IndexError: 
                    rise_start = 0
                    
                # Walk forwards from peak to find true decay end
                after_peak = abs_v[peak_idx:]
                try:
                    dist_to_end = np.where(after_peak < threshold)[0][0]
                    decay_end = peak_idx + dist_to_end
                except IndexError:
                    decay_end = len(v) - 1
                    
                features['wf_rise_time'] = peak_idx - rise_start
                features['wf_decay_time'] = decay_end - peak_idx
                
                # New Waveform Features
                mean_abs = np.mean(abs_v) + 1e-9
                mean_sqrt = np.mean(np.sqrt(abs_v))**2 + 1e-9
                
                features['wf_shape_factor'] = features['wf_rms'] / mean_abs
                features['wf_impulse_factor'] = max_v / mean_abs
                features['wf_margin_factor'] = max_v / mean_sqrt
                
                # Shannon Entropy in time domain (normalized histogram)
                hist, _ = np.histogram(abs_v, bins=50, density=True)
                p = hist / (np.sum(hist) + 1e-9)
                features['wf_entropy'] = -np.sum(p * np.log2(p + 1e-9))
                
                # Hjorth Parameters
                dy = np.diff(v)
                ddy = np.diff(dy)
                var_y = features['wf_variance'] + 1e-9
                var_dy = np.var(dy) + 1e-9
                var_ddy = np.var(ddy) + 1e-9
                
                mobility = np.sqrt(var_dy / var_y)
                features['wf_hjorth_mobility'] = mobility
                features['wf_hjorth_complexity'] = np.sqrt(var_ddy / var_dy) / (mobility + 1e-9)
                
                # --- CWT FEATURES ---
                if PYWT_AVAILABLE and len(v) > 0:
                    # Use Morlet wavelet which is standard for acoustic signals
                    # 10 scales (frequencies)
                    scales = np.arange(1, 11)
                    coefs, freqs = pywt.cwt(v, scales, 'morl')
                    
                    # Energy at each scale
                    cwt_energies = np.sum(np.abs(coefs)**2, axis=1)
                    total_cwt_energy = np.sum(cwt_energies) + 1e-9
                    
                    # Normalize
                    cwt_energy_ratios = cwt_energies / total_cwt_energy
                    
                    for s_idx, energy_ratio in enumerate(cwt_energy_ratios):
                        features[f'wf_cwt_energy_scale_{s_idx+1}'] = energy_ratio
                        
                    # Wavelet entropy
                    p_cwt = cwt_energy_ratios[cwt_energy_ratios > 0]
                    features['wf_cwt_entropy'] = -np.sum(p_cwt * np.log2(p_cwt))
                    
        except Exception: pass
            
    if spec_path.exists():
        try:
            df_spec = pd.read_csv(spec_path)
            freq_col = [c for c in df_spec.columns if 'Frequency' in c or 'MHz' in c]
            mag_col = [c for c in df_spec.columns if 'Magnitude' in c or 'dBu' in c]
            if freq_col and mag_col:
                f = df_spec[freq_col[0]].values
                m_db = df_spec[mag_col[0]].values
                m_lin = 10**(m_db / 20.0)
                
                total_energy = np.sum(m_lin**2) + 1e-9
                features['spec_centroid'] = np.sum(f * m_lin) / (np.sum(m_lin) + 1e-9)
                geo_mean = np.exp(np.mean(np.log(m_lin + 1e-9)))
                ari_mean = np.mean(m_lin)
                features['spec_flatness'] = geo_mean / (ari_mean + 1e-9)
                
                # New Spectral Features
                # Spectral Entropy: p_i = |M(f_i)|^2 / sum_j|M(f_j)|^2  (matches thesis formula)
                p_spec = (m_lin**2) / total_energy  # total_energy = sum(m_lin**2), so p_spec sums to 1
                features['spec_entropy'] = -np.sum(p_spec * np.log2(p_spec + 1e-9))
                
                # Spectral Skewness & Kurtosis
                features['spec_skewness'] = skew(m_lin)
                features['spec_kurtosis'] = kurtosis(m_lin)
                
                # Spectral Rolloff (85%)
                cum_energy = np.cumsum(m_lin**2)
                rolloff_idx = np.where(cum_energy >= 0.85 * total_energy)[0]
                if len(rolloff_idx) > 0:
                    features['spec_rolloff'] = f[rolloff_idx[0]]
                else:
                    features['spec_rolloff'] = f[-1]
                
                bands = np.linspace(0, 1.0, 11)
                for i in range(10):
                    low, high = bands[i], bands[i+1]
                    mask = (f >= low) & (f < high)
                    features[f'spec_rel_energy_{low:.1f}_{high:.1f}'] = np.sum(m_lin[mask]**2) / total_energy
                
                # Additional Frequency Band Features
                # Energy concentration in first 50% of spectrum
                low_freq_mask = f <= 0.5
                features['spec_low_freq_energy_ratio'] = np.sum(m_lin[low_freq_mask]**2) / total_energy
                
                # Energy concentration in high frequency (> 0.5 MHz)
                high_freq_mask = f > 0.5
                features['spec_high_freq_energy_ratio'] = np.sum(m_lin[high_freq_mask]**2) / total_energy
                
                # Peak magnitude in different bands
                bands_detailed = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
                for j in range(len(bands_detailed) - 1):
                    low, high = bands_detailed[j], bands_detailed[j+1]
                    mask = (f >= low) & (f <= high)
                    if np.any(mask):
                        band_mag = m_lin[mask]
                        features[f'spec_peak_mag_{low:.1f}_{high:.1f}'] = np.max(band_mag)
                        features[f'spec_mean_mag_{low:.1f}_{high:.1f}'] = np.mean(band_mag)
                
                # Spectral spread (variance around centroid)
                centroid = features['spec_centroid']
                spread = np.sqrt(np.sum(((f - centroid)**2) * m_lin) / (np.sum(m_lin) + 1e-9))
                features['spec_spread'] = spread
                
                # Frequency ratio: energy in upper bands vs lower bands
                if features['spec_low_freq_energy_ratio'] > 1e-9:
                    features['spec_high_low_ratio'] = features['spec_high_freq_energy_ratio'] / (features['spec_low_freq_energy_ratio'] + 1e-9)
                else:
                    features['spec_high_low_ratio'] = 0.0
        except Exception: pass
    return features

def main():
    """Main execution block to generate feature cache and train exploratory models."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM recordings", conn)
    conn.close()
    df['recording_id'] = df['recording_id'].astype(str)

    if CACHE_PATH.exists():
        print(f"Loading features from {CACHE_PATH}...")
        df_combined = pd.read_csv(CACHE_PATH, low_memory=False)
        df_combined['recording_id'] = df_combined['recording_id'].astype(str)
        
        existing_rids = set(df_combined['recording_id'])
        new_rids = df[~df['recording_id'].isin(existing_rids)]['recording_id'].tolist()
        
        # Merge latest metadata
        feature_cols = [c for c in df_combined.columns if c.startswith('wf_') or c.startswith('spec_')]
        df_features_only = df_combined[['recording_id'] + feature_cols]
        df_combined = pd.merge(df, df_features_only, on='recording_id', how='left')
        
        if new_rids:
            print(f"Found {len(new_rids)} new recordings. Extracting features...")
            extracted = []
            for i, rid in enumerate(new_rids):
                if i % 100 == 0: print(f"   {i}/{len(new_rids)}...")
                f_dict = extract_features(rid)
                f_dict['recording_id'] = rid
                extracted.append(f_dict)
                
            df_new_features = pd.DataFrame(extracted)
            
            df_combined = df_combined.set_index('recording_id')
            df_new_features = df_new_features.set_index('recording_id')
            df_combined.update(df_new_features)
            df_combined = df_combined.reset_index()
            
        base_features = ['duration', 'counts', 'dominant_frequency', 'stem_diameter'] 
        all_features = feature_cols + base_features
        for c in all_features:
            if c in df_combined.columns:
                df_combined[c] = pd.to_numeric(df_combined[c], errors='coerce')
                df_combined[c] = df_combined[c].fillna(df_combined[c].median())
                
        df_combined.to_csv(CACHE_PATH, index=False)
        print(f"Cache updated and verified at {CACHE_PATH}")
    else:
        print("Extracting features to build cache (this will take a few minutes)...")
        extracted = []
        for i, rid in enumerate(df['recording_id']):
            if i % 500 == 0: print(f"   {i}/{len(df)}...")
            f_dict = extract_features(rid)
            f_dict['recording_id'] = rid
            extracted.append(f_dict)
            
        df_features = pd.DataFrame(extracted)
        df_combined = pd.merge(df, df_features, on='recording_id')
        
        feature_cols = [c for c in df_combined.columns if c.startswith('wf_') or c.startswith('spec_')]
        base_features = ['duration', 'counts', 'dominant_frequency', 'stem_diameter'] 
        all_features = feature_cols + base_features
        
        for c in all_features:
            if c in df_combined.columns:
                df_combined[c] = pd.to_numeric(df_combined[c], errors='coerce')
                df_combined[c] = df_combined[c].fillna(df_combined[c].median())
        
        df_combined.to_csv(CACHE_PATH, index=False)
        print(f"Features cached to {CACHE_PATH}")
        
    df_combined['Class'] = np.where(
        (df_combined['event_type'].isin(['Kontrolle', 'Noise'])) | (df_combined['plant'].isin(['Totes Holz', 'Dead wood'])),
        'Pure Noise',
        'Tentative Cavitation'
    )
    
    # Isolate numeric features
    feature_cols = [c for c in df_combined.columns if c.startswith('wf_') or c.startswith('spec_')]
    all_features = feature_cols + ['duration', 'counts', 'dominant_frequency', 'stem_diameter']
    all_features = [f for f in all_features if f in df_combined.columns]
    
    # 1. Train a quick Random Forest to find the MOST important features
    print("Training Random Forest to determine feature importance...")
    
    df_noise = df_combined[df_combined['Class'] == 'Pure Noise']
    # To balance, take an equal random sample of confident Cavitation events
    # We use events that have low noise_probability if available, else just random
    if 'noise_probability' in df_combined.columns:
        df_cav_pool = df_combined[df_combined['noise_probability'] < 0.3]
    else:
        df_cav_pool = df_combined[df_combined['Class'] == 'Tentative Cavitation']
    
    # Sample with replacement if needed, or take what's available
    sample_size = min(len(df_cav_pool), len(df_noise) * 3)
    df_cav = df_cav_pool.sample(n=sample_size, random_state=42, replace=False)
    
    df_train = pd.concat([df_noise, df_cav])
    X = df_train[all_features].values
    y = (df_train['Class'] == 'Pure Noise').astype(int)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    feat_imp = pd.DataFrame({'Feature': all_features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    # Save the importance bar chart
    fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', title="Feature Importance for identifying Noise")
    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
    
    top_8_features = feat_imp['Feature'].head(8).tolist()
    
    # 2. Create Violin Plots for the Top 8 features
    print(f"Top features: {top_8_features}")
    
    # We plot the violin plots on the whole dataset (or a large sample)
    df_plot = df_combined.copy()
    
    fig_violins = make_subplots(rows=4, cols=2, subplot_titles=top_8_features)
    
    for i, feature in enumerate(top_8_features):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        # Add Noise violin
        fig_violins.add_trace(go.Violin(
            x=df_plot['Class'][df_plot['Class'] == 'Pure Noise'],
            y=df_plot[feature][df_plot['Class'] == 'Pure Noise'],
            name='Pure Noise',
            line_color='red',
            showlegend=(i==0)
        ), row=row, col=col)
        
        # Add Cavitation violin
        fig_violins.add_trace(go.Violin(
            x=df_plot['Class'][df_plot['Class'] == 'Tentative Cavitation'],
            y=df_plot[feature][df_plot['Class'] == 'Tentative Cavitation'],
            name='Tentative Cavitation',
            line_color='blue',
            showlegend=(i==0)
        ), row=row, col=col)

    fig_violins.update_layout(height=1200, title_text="Distribution of Top 8 Features (Noise vs Cavitation)", violinmode='group')
    
    # --- PLANT CLASSIFICATION ANALYSIS ---
    print("\n" + "="*80)
    print("PLANT CLASSIFICATION ANALYSIS")
    print("="*80)
    
    # Filter for non-noise events only (real cavitation signals)
    df_cav_only = df_combined[df_combined['Class'] == 'Tentative Cavitation'].copy()
    
    if 'plant' in df_cav_only.columns and df_cav_only['plant'].nunique() > 1:
        # Remove null/missing plant values
        df_cav_plant = df_cav_only.dropna(subset=['plant'])
        
        # Only use plant types with at least 5 samples (to have enough data)
        plant_counts = df_cav_plant['plant'].value_counts()
        valid_plants = plant_counts[plant_counts >= 5].index.tolist()
        
        if len(valid_plants) >= 2:
            print(f"Found {len(valid_plants)} plant types with >= 5 samples: {valid_plants}")
            df_plant_train = df_cav_plant[df_cav_plant['plant'].isin(valid_plants)].copy()
            
            # Prepare data for plant classification
            X_plant = df_plant_train[all_features].values
            y_plant = pd.Categorical(df_plant_train['plant']).codes
            
            # Train Random Forest for plant classification
            rf_plant = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_plant.fit(X_plant, y_plant)
            
            importances_plant = rf_plant.feature_importances_
            feat_imp_plant = pd.DataFrame({
                'Feature': all_features, 
                'Importance': importances_plant
            }).sort_values(by='Importance', ascending=False)
            
            # Save plant classification importance chart
            fig_imp_plant = px.bar(
                feat_imp_plant, 
                x='Importance', 
                y='Feature', 
                orientation='h', 
                title="Feature Importance for Plant Classification"
            )
            fig_imp_plant.update_layout(yaxis={'categoryorder':'total ascending'})
            
            top_8_plant_features = feat_imp_plant['Feature'].head(8).tolist()
            print(f"Top plant classification features: {top_8_plant_features}")
            
            # Create Violin Plots for Top 8 plant features
            fig_violins_plant = make_subplots(rows=4, cols=2, subplot_titles=top_8_plant_features)
            
            colors_plant = px.colors.qualitative.Set2 if len(valid_plants) <= 8 else px.colors.qualitative.Plotly
            
            for i, feature in enumerate(top_8_plant_features):
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                for idx, plant in enumerate(valid_plants):
                    plant_data = df_plant_train[df_plant_train['plant'] == plant]
                    fig_violins_plant.add_trace(go.Violin(
                        x=plant_data['plant'],
                        y=plant_data[feature],
                        name=plant,
                        line_color=colors_plant[idx % len(colors_plant)],
                        showlegend=(i==0)
                    ), row=row, col=col)
            
            fig_violins_plant.update_layout(
                height=1200, 
                title_text=f"Distribution of Top 8 Features for Plant Classification ({', '.join(valid_plants)})", 
                violinmode='group'
            )
        else:
            print("Not enough plant types with sufficient samples for classification.")
            fig_imp_plant = None
            fig_violins_plant = None
    else:
        print("Not enough plant types or missing plant data for classification.")
        fig_imp_plant = None
        fig_violins_plant = None
    
    # --- STRESS TYPE CLASSIFICATION ANALYSIS ---
    print("\n" + "="*80)
    print("STRESS TYPE CLASSIFICATION ANALYSIS")
    print("="*80)
    
    # Filter for non-noise events only
    df_stress = df_cav_only.copy()
    
    if 'stress_type' in df_stress.columns and df_stress['stress_type'].nunique() > 1:
        # Remove null/missing stress values
        df_stress_clean = df_stress.dropna(subset=['stress_type'])
        
        # Only use stress types with at least 5 samples
        stress_counts = df_stress_clean['stress_type'].value_counts()
        valid_stresses = stress_counts[stress_counts >= 5].index.tolist()
        
        if len(valid_stresses) >= 2:
            print(f"Found {len(valid_stresses)} stress types with >= 5 samples: {valid_stresses}")
            df_stress_train = df_stress_clean[df_stress_clean['stress_type'].isin(valid_stresses)].copy()
            
            # Prepare data for stress classification
            X_stress = df_stress_train[all_features].values
            y_stress = pd.Categorical(df_stress_train['stress_type']).codes
            
            # Train Random Forest for stress classification
            rf_stress = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_stress.fit(X_stress, y_stress)
            
            importances_stress = rf_stress.feature_importances_
            feat_imp_stress = pd.DataFrame({
                'Feature': all_features, 
                'Importance': importances_stress
            }).sort_values(by='Importance', ascending=False)
            
            # Save stress classification importance chart
            fig_imp_stress = px.bar(
                feat_imp_stress, 
                x='Importance', 
                y='Feature', 
                orientation='h', 
                title="Feature Importance for Stress Type Classification"
            )
            fig_imp_stress.update_layout(yaxis={'categoryorder':'total ascending'})
            
            top_8_stress_features = feat_imp_stress['Feature'].head(8).tolist()
            print(f"Top stress classification features: {top_8_stress_features}")
            
            # Create Violin Plots for Top 8 stress features
            fig_violins_stress = make_subplots(rows=4, cols=2, subplot_titles=top_8_stress_features)
            
            colors_stress = px.colors.qualitative.Set2 if len(valid_stresses) <= 8 else px.colors.qualitative.Plotly
            
            for i, feature in enumerate(top_8_stress_features):
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                for idx, stress in enumerate(valid_stresses):
                    stress_data = df_stress_train[df_stress_train['stress_type'] == stress]
                    fig_violins_stress.add_trace(go.Violin(
                        x=stress_data['stress_type'],
                        y=stress_data[feature],
                        name=stress,
                        line_color=colors_stress[idx % len(colors_stress)],
                        showlegend=(i==0)
                    ), row=row, col=col)
            
            fig_violins_stress.update_layout(
                height=1200, 
                title_text=f"Distribution of Top 8 Features for Stress Type Classification ({', '.join(valid_stresses)})", 
                violinmode='group'
            )
        else:
            print("Not enough stress types with sufficient samples for classification.")
            fig_imp_stress = None
            fig_violins_stress = None
    else:
        print("Not enough stress types or missing stress data for classification.")
        fig_imp_stress = None
        fig_violins_stress = None
    
    # Combine everything into an HTML string
    html_content = f"""
    <html>
    <head><title>Feature Impact Analysis</title></head>
    <body style="font-family: sans-serif; background-color: #f5f5f5; margin: 20px;">
        <h1 style="text-align:center; color: #333;">🔬 Feature Impact Analysis Report</h1>
        
        <!-- NOISE CLASSIFICATION -->
        <section style="background-color: white; padding: 20px; margin: 20px auto; width: 90%; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h2 style="color: #d32f2f; border-bottom: 2px solid #d32f2f; padding-bottom: 10px;">🚨 Noise vs Cavitation Classification</h2>
            <p>This analysis identifies features most critical in separating Pure Noise from Cavitation events.</p>
            <div style="width: 100%; margin: 20px auto;">
                {fig_imp.to_html(full_html=False, include_plotlyjs='cdn')}
            </div>
            <div style="width: 100%; margin: 20px auto;">
                {fig_violins.to_html(full_html=False, include_plotlyjs='cdn')}
            </div>
        </section>
        
        <!-- PLANT CLASSIFICATION -->
        <section style="background-color: white; padding: 20px; margin: 20px auto; width: 90%; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h2 style="color: #1976d2; border-bottom: 2px solid #1976d2; padding-bottom: 10px;">🌱 Plant Classification</h2>
            <p>This analysis identifies features that best distinguish between different plant species.</p>
            {'<p style="color: #666;">⚠️ Not enough plant types with sufficient samples for meaningful classification.</p>' if fig_imp_plant is None else f'''
            <div style="width: 100%; margin: 20px auto;">
                {fig_imp_plant.to_html(full_html=False, include_plotlyjs='cdn')}
            </div>
            <div style="width: 100%; margin: 20px auto;">
                {fig_violins_plant.to_html(full_html=False, include_plotlyjs='cdn')}
            </div>
            '''}
        </section>
        
        <!-- STRESS TYPE CLASSIFICATION -->
        <section style="background-color: white; padding: 20px; margin: 20px auto; width: 90%; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h2 style="color: #388e3c; border-bottom: 2px solid #388e3c; padding-bottom: 10px;">💧 Stress Type Classification</h2>
            <p>This analysis identifies features that best distinguish between different stress types (drought, freeze, cut, etc).</p>
            {'<p style="color: #666;">⚠️ Not enough stress types with sufficient samples for meaningful classification.</p>' if fig_imp_stress is None else f'''
            <div style="width: 100%; margin: 20px auto;">
                {fig_imp_stress.to_html(full_html=False, include_plotlyjs='cdn')}
            </div>
            <div style="width: 100%; margin: 20px auto;">
                {fig_violins_stress.to_html(full_html=False, include_plotlyjs='cdn')}
            </div>
            '''}
        </section>
        
        <footer style="text-align: center; margin-top: 40px; padding: 20px; color: #999; border-top: 1px solid #ddd;">
            <p>Generated by AE-Lab Feature Evaluation Pipeline</p>
        </footer>
    </body>
    </html>
    """
    
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"Successfully generated '{REPORT_PATH.name}' at {REPORT_PATH}")

if __name__ == '__main__':
    main()