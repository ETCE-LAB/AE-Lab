import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import json
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from PIL import Image
from unsupervised_pipeline import run_pipeline

try:
    st_fragment = st.fragment
except AttributeError:
    try:
        st_fragment = st.experimental_fragment
    except AttributeError:
        st_fragment = lambda x: x

@st_fragment
def draw_static_line_chart(df, x_col):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import io
    y_col = [c for c in df.columns if c != x_col][0]
    fig = Figure(figsize=(5, 2.5), dpi=80)
    fig.patch.set_facecolor('#0e1117')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0e1117')
    ax.plot(df[x_col], df[y_col], color='#1f77b4', linewidth=1)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    return buf.getvalue()

def _update_label(rid, label, db_path):
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE recordings SET event_type = ? WHERE recording_id = ?", (label, rid))
    conn.commit()
    conn.close()

@st.fragment
def render_signal_inspector(rid, current_label, mode, project_root, db_path):
    import sqlite3
    import pandas as pd
    import glob
    
    conn = sqlite3.connect(db_path)
    res = conn.execute("SELECT event_type FROM recordings WHERE recording_id = ?", (rid,)).fetchone()
    conn.close()
    if res and res[0]:
        current_label = res[0]
        
    with st.expander(f"Signal: {rid} (Current: {current_label})", expanded=True):
        col_b1, col_b2, _ = st.columns([1, 1, 4])
        with col_b1:
            st.button("Cavitation", key=f"cav_{mode}_{rid}", on_click=_update_label, args=(rid, 'Cavitation', db_path))
        with col_b2:
            st.button("Noise", key=f"noise_{mode}_{rid}", on_click=_update_label, args=(rid, 'Noise', db_path))
        
        w1, w2 = st.columns(2)
        w_files = glob.glob(str(project_root / "data" / f"{rid}-measurement-scope.csv"))
        if w_files:
            w_df = pd.read_csv(w_files[0])
            if len(w_df) > 5000:
                w_df = w_df.iloc[::max(1, len(w_df)//5000), :]
            time_col = [c for c in w_df.columns if 'Time' in c or 'ms' in c][0]
            with w1:
                st.caption("Scope (Time Domain)")
                st.image(draw_static_line_chart(w_df, time_col), use_container_width=True)
        
        s_files = glob.glob(str(project_root / "data" / f"{rid}-measurement-spectrum.csv"))
        if s_files:
            s_df = pd.read_csv(s_files[0])
            freq_col = [c for c in s_df.columns if 'Frequency' in c or 'Hz' in c][0]
            s_df = s_df[s_df[freq_col] > 0]
            if len(s_df) > 5000:
                s_df = s_df.iloc[::max(1, len(s_df)//5000), :]
            with w2:
                st.caption("Spectrum (Frequency Domain)")
                st.image(draw_static_line_chart(s_df, freq_col), use_container_width=True)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
LOGO_PATH = PROJECT_ROOT / ".documentation" / "ae_lab_logo.jpg"
logo = Image.open(LOGO_PATH) if LOGO_PATH.exists() else "🔬"

st.set_page_config(page_title="AE-Lab Data Analysis", layout="wide", page_icon=logo)

# ==========================================
# Configuration & Paths
# ==========================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DB_PATH = PROJECT_ROOT / "metadata.db"
DATA_DIR = PROJECT_ROOT / "data"
CACHE_PATH = PROJECT_ROOT / "data_collection" / "feature_cache.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
UNSUPERVISED_FILE = RESULTS_DIR / "unsupervised_results.csv"
MODELS_DIR = PROJECT_ROOT / "models"

# ==========================================
# Helper Functions
# ==========================================
def render_model_insights(meta, colorscale="Blues", key_prefix="model"):
    """Shared Model Eval view: headline metrics, baseline comparison, annotated confusion matrix,
    per-class table, and (if available) per-plant / feature-importance / model-comparison insights.
    Everything derivable from the confusion matrix is computed here, so older metadata files still work."""
    classes = list(meta.get('classes', []))
    method = meta.get('evaluation_method', '20% hold-out test set')
    cm = np.array(meta['confusion_matrix']) if 'confusion_matrix' in meta else None

    if cm is not None and cm.size and cm.sum() > 0:
        total = cm.sum()
        diag = np.diag(cm).astype(float)
        row_sum = cm.sum(axis=1).astype(float)   # actual events per class
        col_sum = cm.sum(axis=0).astype(float)   # predicted events per class
        recalls = np.divide(diag, row_sum, out=np.zeros_like(diag), where=row_sum > 0)
        precisions = np.divide(diag, col_sum, out=np.zeros_like(diag), where=col_sum > 0)
        f1s = np.divide(2 * precisions * recalls, precisions + recalls,
                        out=np.zeros_like(diag), where=(precisions + recalls) > 0)
        has_support = row_sum > 0
        acc = diag.sum() / total
        bal_acc = float(recalls[has_support].mean())
        baseline = float(row_sum.max() / total)
    else:
        cm = None
        acc = meta.get('test_accuracy', 0)
        bal_acc = meta.get('test_balanced_accuracy')
        baseline = meta.get('majority_baseline_accuracy')

    # ---- headline metrics ----
    cols = st.columns(6 if 'test_f2_signal' in meta else 5)
    delta = f"{(acc - baseline) * 100:+.1f} pp vs. guessing the biggest class" if baseline is not None else None
    cols[0].metric("Accuracy", f"{acc:.2%}", delta, help="Share of all test events predicted correctly. Misleading on imbalanced data - compare with the baseline in the delta.")
    cols[1].metric("Balanced accuracy", f"{bal_acc:.2%}" if bal_acc is not None else "n/a",
                   help="Average recall over all classes. Every class counts equally, so a tiny class matters as much as a huge one.")
    cols[2].metric("Precision (macro)", f"{meta.get('test_precision', 0):.2f}",
                   help="When the model predicts a class, how often is it right? Averaged over classes.")
    cols[3].metric("Recall (macro)", f"{meta.get('test_recall', 0):.2f}",
                   help="Of all real events of a class, how many did the model find? Averaged over classes.")
    cols[4].metric("F1 (macro)", f"{meta.get('test_f1', 0):.2f}",
                   help="Harmonic mean of precision and recall, averaged over classes. The most honest single number for imbalanced data.")
    if 'test_f2_signal' in meta:
        cols[5].metric("F2 (signal)", f"{meta.get('test_f2_signal', 0):.2f}",
                       help="Like F1 but recall counts 4x as much as precision.")

    n_events = meta.get('test_events', int(cm.sum()) if cm is not None else 0)
    st.caption(
        f"Model: {meta.get('model_type', 'n/a')} | Evaluation: **{method}** on {n_events} unseen events | "
        f"Trained on {meta.get('training_samples', 0)} samples | Last updated: {meta.get('training_date', 'Unknown')[:10]}"
    )

    if baseline is not None and acc <= baseline + 0.05:
        st.warning(
            f"Accuracy ({acc:.0%}) is barely above what you would get by always guessing the biggest class ({baseline:.0%}). "
            "Judge the model by balanced accuracy / macro F1 and the per-class recall below."
        )
    if cm is not None:
        for i, cls in enumerate(classes):
            if has_support[i] and recalls[i] == 0:
                st.error(f"The model never correctly identified a single '{cls}' event (recall = 0%).")

    # ---- confusion matrix ----
    if cm is not None:
        st.markdown("#### Confusion matrix")
        view = st.radio("Show", ["Event counts", "% of each actual class (recall)"], horizontal=True,
                        key=f"{key_prefix}_cm_view")
        if view.startswith("Event"):
            z, text_fmt, zmax = cm, "d", None
        else:
            z = np.divide(cm, row_sum[:, None], out=np.zeros_like(cm, dtype=float), where=row_sum[:, None] > 0) * 100
            text_fmt, zmax = ".0f", 100
        fig_cm = px.imshow(
            z, text_auto=text_fmt, color_continuous_scale=colorscale, x=classes, y=classes, zmin=0, zmax=zmax,
            labels=dict(x="Predicted by the model", y="Actual (ground truth)", color="Events" if zmax is None else "%"),
        )
        fig_cm.update_xaxes(side="top")
        fig_cm.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=340,
                             title=f"Confusion matrix - {method}")
        st.plotly_chart(fig_cm, use_container_width=True)

        with st.expander("How to read this confusion matrix", expanded=False):
            st.markdown(
                "- **Rows = what the event really is** (ground truth / your labels).\n"
                "- **Columns = what the model predicted.**\n"
                "- **Diagonal = correct predictions.** Everything off the diagonal is a mistake: "
                "row *A*, column *B* means *'a real A was predicted as B'*.\n"
                "- **Recall of a class** = diagonal cell / sum of its **row** ('of all real A, how many did we find?').\n"
                "- **Precision of a class** = diagonal cell / sum of its **column** ('when the model says A, how often is it right?').\n"
                "- Use the *% of each actual class* view to compare classes of very different size."
            )
            st.markdown("**Reading of your current matrix:**")
            for i, cls in enumerate(classes):
                if not has_support[i]:
                    continue
                off = cm[i].copy()
                off[i] = 0
                mistake = ""
                if off.sum() > 0:
                    j = int(off.argmax())
                    mistake = f" Most common mistake: predicted as **{classes[j]}** ({int(off[j])} events)."
                prec_txt = f"; when it says {cls} it is right {precisions[i]:.0%} of the time" if col_sum[i] > 0 else "; it never predicted this class"
                st.markdown(f"- Of **{int(row_sum[i])}** real **{cls}** events, **{int(diag[i])}** were found ({recalls[i]:.0%} recall).{mistake}{prec_txt}.")

        st.markdown("#### Per-class performance")
        df_cls = pd.DataFrame({
            "Class": classes, "Real events (support)": row_sum.astype(int), "Predicted events": col_sum.astype(int),
            "Precision": precisions.round(3), "Recall": recalls.round(3), "F1": f1s.round(3),
        })
        st.dataframe(df_cls, use_container_width=True, hide_index=True)

    # ---- per-plant breakdown (Leave-One-Plant-Out only) ----
    per_plant = meta.get('per_plant')
    if per_plant:
        st.markdown("#### Per-plant results (Leave-One-Plant-Out)")
        st.caption(
            "Each bar is one physical plant: the model was trained on all OTHER plants and then tested on this one. "
            f"Mean per-plant accuracy: {meta.get('mean_per_plant_accuracy', 0):.1%}. "
            "Plants with few events are noisy - check the event count."
        )
        df_pp = pd.DataFrame(per_plant)
        df_pp["predicted_as"] = df_pp["predicted_as"].apply(lambda d: ", ".join(f"{k}: {v}" for k, v in d.items()))
        fig_pp = px.bar(df_pp, x="plant_id", y="accuracy", color="species", hover_data=["n_events", "predicted_as"],
                        text=df_pp["n_events"].apply(lambda n: f"n={n}"), range_y=[0, 1])
        fig_pp.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=300, xaxis_title="Held-out plant ID",
                             yaxis_title="Accuracy on that plant", xaxis_type="category")
        st.plotly_chart(fig_pp, use_container_width=True)
        st.dataframe(df_pp.rename(columns={"plant_id": "Plant", "species": "Species", "n_events": "Events",
                                           "accuracy": "Accuracy", "predicted_as": "Predicted as"}),
                     use_container_width=True, hide_index=True)

    # ---- model comparison + feature importance ----
    ci1, ci2 = st.columns(2)
    comparison = meta.get('model_comparison')
    if comparison:
        with ci1:
            st.markdown("#### Candidate models (selection score)")
            df_cmp = pd.DataFrame({"Model": list(comparison.keys()), "CV macro F1": list(comparison.values())})
            fig_cmp = px.bar(df_cmp, x="CV macro F1", y="Model", orientation="h", range_x=[0, 1])
            fig_cmp.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=260)
            st.plotly_chart(fig_cmp, use_container_width=True)
            st.caption("Used only to pick the model type. It comes from random folds inside the training data, "
                       "so it is optimistic. The real performance is the Leave-One-Plant-Out result above.")
    fi = meta.get('feature_importance')
    if fi:
        with ci2:
            st.markdown("#### Top features")
            df_fi = pd.DataFrame(fi).sort_values("importance")
            fig_fi = px.bar(df_fi, x="importance", y="feature", orientation="h")
            fig_fi.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=260)
            st.plotly_chart(fig_fi, use_container_width=True)



@st.cache_data
def load_metadata():
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    query = "SELECT * FROM recordings"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if CACHE_PATH.exists():
        try:
            df_feats = pd.read_csv(CACHE_PATH, low_memory=False)
            feat_cols = [c for c in df_feats.columns if c.startswith('wf_') or c.startswith('spec_')] + ['recording_id']
            df['recording_id'] = df['recording_id'].astype(str)
            df_feats['recording_id'] = df_feats['recording_id'].astype(str)
            df = pd.merge(df, df_feats[feat_cols], on='recording_id', how='left')
        except Exception as e:
            st.warning(f"Could not load feature cache: {e}")
    
    if not df.empty and 'event_timestamp' in df.columns:
        df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], errors='coerce')
        df = df.sort_values(by='event_timestamp')
    return df

@st.cache_data
def load_waveform(recording_id):
    path = DATA_DIR / f"{recording_id}-measurement-scope.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_spectrum(recording_id):
    path = DATA_DIR / f"{recording_id}-measurement-spectrum.csv"
    if path.exists():
        return pd.read_csv(path)
    return None



def render_clustering_ui(mode, df_filtered, cache_path, results_dir, db_path, project_root):
    if st.button(f"⚙️ Run Clustering Pipeline ({mode.capitalize()})", key=f"run_cluster_{mode}", type="primary"):
        with st.spinner(f"Running clustering pipeline for {mode}..."):
            try:
                import subprocess
                st.info("Running feature extraction...")
                subprocess.run(["python", str(project_root / "helper_scripts" / "evaluate_features.py")], check=True)
                st.info("Running clustering pipeline...")
                from unsupervised_pipeline import run_pipeline
                success, msg = run_pipeline(cache_path, results_dir, db_path, mode=mode)
                st.success(msg)
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                
    unsup_file = results_dir / f"unsupervised_results_{mode}.csv"
    if unsup_file.exists():
        df_semi_sup = pd.read_csv(unsup_file, low_memory=False)
        try:
            import sqlite3
            conn = sqlite3.connect(db_path, timeout=10.0)
            fresh_labels = pd.read_sql_query("SELECT recording_id, event_type FROM recordings", conn)
            conn.close()
            fresh_labels['recording_id'] = fresh_labels['recording_id'].astype(str)
            df_semi_sup['recording_id'] = df_semi_sup['recording_id'].astype(str)
            if 'event_type' in df_semi_sup.columns:
                df_semi_sup = df_semi_sup.drop(columns=['event_type'])
            df_semi_sup = pd.merge(df_semi_sup, fresh_labels, on='recording_id', how='left')
        except Exception as e:
            st.warning(f"Could not refresh labels: {e}")
            
        if 'recording_id' in df_filtered.columns:
            valid_rids = df_filtered['recording_id'].astype(str).unique()
            df_semi_sup = df_semi_sup[df_semi_sup['recording_id'].astype(str).isin(valid_rids)]
            st.caption(f"Showing {len(df_semi_sup)} points based on your sidebar filters.")
            
        dim_cols = [c for c in df_semi_sup.columns if c.endswith('_1') or c.endswith('_2') or c.endswith('_3')]
        if dim_cols:
            dim_prefix = dim_cols[0].split('_')[0]
            

            
            c1, c2 = st.columns([1, 2])
            with c1:
                color_options = [c for c in ['event_type', 'Cluster_Label', 'Isolation_Forest_Label', 'plant', 'stress_type'] if c in df_semi_sup.columns]
                
                default_color = 'event_type'
                if mode == 'plants' and 'plant' in color_options:
                    default_color = 'plant'
                elif mode == 'stress' and 'stress_type' in color_options:
                    default_color = 'stress_type'
                
                default_idx = color_options.index(default_color) if default_color in color_options else 0
                
                color_col = st.selectbox("Color by", color_options, index=default_idx, key=f"color_{mode}")
                x_default_idx = next((i for i, c in enumerate(dim_cols) if 'UMAP_1' in c), 0)
                y_default_idx = next((i for i, c in enumerate(dim_cols) if 'UMAP_2' in c), 1 if len(dim_cols) > 1 else 0)
                
                x_col = st.selectbox("X-Axis", dim_cols, index=x_default_idx, key=f"x_{mode}")
                y_col = st.selectbox("Y-Axis", dim_cols, index=y_default_idx, key=f"y_{mode}")
            
            # --- START DUAL VIEW INJECTION ---
            if 'Cluster_Label' in df_semi_sup.columns:
                st.markdown("### 🤖 Algorithm Performance Comparison")
                st.caption("Visually verify if the unsupervised clustering algorithm found natural separations between Cavitation and Noise.")
                comp1, comp2 = st.columns(2)
                with comp1:
                    fig_a = px.scatter(df_semi_sup, x=x_col, y=y_col, color='event_type', template="plotly_dark", title="Ground Truth (Human Labels)", hover_data=['recording_id'])
                    fig_a.update_layout(dragmode='lasso', height=350, margin=dict(l=0,r=0,t=30,b=0))
                    try:
                        event_a = st.plotly_chart(fig_a, use_container_width=True, on_select="rerun", key=f"chart_a_{mode}")
                    except TypeError:
                        st.plotly_chart(fig_a, use_container_width=True)
                        event_a = None
                with comp2:
                    fig_b = px.scatter(df_semi_sup, x=x_col, y=y_col, color='Cluster_Label', template="plotly_dark", title="Algorithm Output (Unsupervised Clusters)", hover_data=['recording_id'])
                    fig_b.update_layout(dragmode='lasso', height=350, margin=dict(l=0,r=0,t=30,b=0))
                    try:
                        event_b = st.plotly_chart(fig_b, use_container_width=True, on_select="rerun", key=f"chart_b_{mode}")
                    except TypeError:
                        st.plotly_chart(fig_b, use_container_width=True)
                        event_b = None
                st.markdown("---")
            # --- END DUAL VIEW INJECTION ---
            
            with c2:
                fig_key = f"fig_semi_sup_{mode}"
                state_hash = hash((len(df_semi_sup), color_col, x_col, y_col))
                
                if fig_key not in st.session_state or st.session_state.get(f"{fig_key}_hash") != state_hash:
                    fig_semi_sup = px.scatter(
                        df_semi_sup, x=x_col, y=y_col, color=color_col,
                        hover_data=['recording_id', 'plant'] if 'plant' in df_semi_sup.columns else ['recording_id'],
                        template="plotly_dark", title=f"{dim_prefix} 2D Projection ({mode.capitalize()})"
                    )
                    fig_semi_sup.update_layout(dragmode='zoom', height=500, margin=dict(l=0,r=0,t=30,b=0))
                    st.session_state[fig_key] = fig_semi_sup
                    st.session_state[f"{fig_key}_hash"] = state_hash
                else:
                    fig_semi_sup = st.session_state[fig_key]
                    
                try:
                    event = st.plotly_chart(fig_semi_sup, use_container_width=True, on_select="rerun", key=f"chart_{mode}")
                except TypeError:
                    st.plotly_chart(fig_semi_sup, use_container_width=True, key=f"chart_{mode}")
                    st.warning("Your Streamlit version does not support interactive 'on_select'. Upgrade Streamlit to 1.35+ to view waveforms on click.")
                    event = None
                    
            selected_points = []
            if 'event_a' in locals() and event_a and "selection" in event_a and "points" in event_a.selection and len(event_a.selection["points"]) > 0:
                selected_points = event_a.selection["points"]
            elif 'event_b' in locals() and event_b and "selection" in event_b and "points" in event_b.selection and len(event_b.selection["points"]) > 0:
                selected_points = event_b.selection["points"]
            elif event and "selection" in event and "points" in event.selection and len(event.selection["points"]) > 0:
                selected_points = event.selection["points"]
            
            if len(selected_points) > 0:
                st.subheader(f"🔍 Inspecting {len(selected_points)} Selected Events")
                
                valid_rids = [p["customdata"][0] for p in selected_points if "customdata" in p]
                if valid_rids:
                    st.markdown("**Batch Action:**")
                    c_all1, c_all2, _ = st.columns([1, 1, 4])
                    with c_all1:
                        if st.button(f"🔴 Relabel All ({len(valid_rids)}) to Cavitation", key=f"relabel_all_cav_{mode}"):
                            import sqlite3
                            conn = sqlite3.connect(db_path)
                            conn.executemany("UPDATE recordings SET event_type = 'Cavitation' WHERE recording_id = ?", [(r,) for r in valid_rids])
                            conn.commit()
                            conn.close()
                            # st.rerun()
                    with c_all2:
                        if st.button(f"🔵 Relabel All ({len(valid_rids)}) to Noise", key=f"relabel_all_noise_{mode}"):
                            import sqlite3
                            conn = sqlite3.connect(db_path)
                            conn.executemany("UPDATE recordings SET event_type = 'Noise' WHERE recording_id = ?", [(r,) for r in valid_rids])
                            conn.commit()
                            conn.close()
                            # st.rerun()
                
                if len(selected_points) > 30:
                    st.warning(f"You selected {len(selected_points)} points. Displaying the first 30 waveforms to prevent browser crash.")
                
                with st.spinner("Loading charts..."):
                    for p in selected_points[:30]:
                        rid = None
                        if "customdata" in p:
                            rid = p["customdata"][0]
                        
                        if rid:
                            current_label = df_semi_sup[df_semi_sup['recording_id'] == rid]['event_type'].values
                            current_label = current_label[0] if len(current_label) > 0 and pd.notna(current_label[0]) else "Unknown"
                            
                            render_signal_inspector(rid, current_label, mode, project_root, db_path)
                
def main():
    col1, col2 = st.columns([1, 8])
    with col1:
        if LOGO_PATH.exists():
            st.image(logo, width=160)
    with col2:
        st.title("AE-Lab Dashboard: Comprehensive Data Analysis")
        st.markdown("Get a comprehensive visual overview of all recorded AE signals, features, and model predictions.")
    
    df = load_metadata()
    if df.empty:
        st.warning("No metadata found in the database or database does not exist.")
        return

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filter")
    
    event_types = df["event_type"].dropna().unique().tolist() if "event_type" in df.columns else []
    stress_types = df["stress_type"].dropna().unique().tolist() if "stress_type" in df.columns else []
    plant_types = df["plant"].dropna().unique().tolist() if "plant" in df.columns else []
    experiment_ids = df["experiment_id"].dropna().unique().tolist() if "experiment_id" in df.columns else []
    
    selected_experiments = st.sidebar.multiselect("Experiment ID", experiment_ids, default=experiment_ids)
    selected_events = st.sidebar.multiselect("Event Type", event_types, default=event_types)
    selected_stress = st.sidebar.multiselect("Stress Type", stress_types, default=stress_types)
    selected_plants = st.sidebar.multiselect("Plant", plant_types, default=plant_types)
    
    df_filtered = df.copy()
    if selected_experiments and "experiment_id" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["experiment_id"].isin(selected_experiments)]
    if selected_events and "event_type" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["event_type"].isin(selected_events)]
    if selected_stress and "stress_type" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["stress_type"].isin(selected_stress)]
    if selected_plants and "plant" in df_filtered.columns:
        df_filtered = df_filtered[
            df_filtered["plant"].isin(selected_plants) | df_filtered["plant"].isna()
        ]
        
    if "noise_probability" in df_filtered.columns:
        max_noise = st.sidebar.slider(
            "Max Noise Probability (Filter out likely noise)", 
            min_value=0.0, max_value=1.0, value=1.0, step=0.05,
            help="1.0 = Show all events. 0.5 = Show only events that are very likely real cavitation."
        )
        df_filtered = df_filtered[df_filtered["noise_probability"] <= max_noise]
    else:
        max_noise = 1.0

    if "event_timestamp" in df_filtered.columns and not df_filtered["event_timestamp"].isna().all():
        min_date = df_filtered["event_timestamp"].min().date()
        max_date = df_filtered["event_timestamp"].max().date()
        
        import datetime
        date_range = st.sidebar.date_input(
            "Time Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date + datetime.timedelta(days=1)
        )
        
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df_filtered = df_filtered[
                (df_filtered["event_timestamp"] >= start_dt) & 
                (df_filtered["event_timestamp"] < end_dt + pd.Timedelta(days=1))
            ]

    # Numeric columns for plotting
    base_numeric = ["peak_amplitude", "energy", "duration", "dominant_frequency", "counts", "temperature", "humidity", "soil_moisture"]
    extra_numeric = [c for c in df_filtered.columns if c.startswith('wf_') or c.startswith('spec_')]
    numeric_cols = [c for c in base_numeric + extra_numeric if c in df_filtered.columns]

    # --- TABS ---
    tab_overview, tab_explorer, tab_timeline, tab_time, tab_signal, tab_semi_supervised, tab_test_eval = st.tabs([
        "📋 Overview", "📊 Explorer", "📈 Timeline", "⏱️ Time", "🔍 Signal", "🗺️ Clustering & Relabeling", "✅ Model Eval"
    ])
    
    # --- TAB OVERVIEW ---
    with tab_overview:
        st.header("📋 Dataset Overview")
        st.write("A comprehensive summary of the available data, filters, and distributions.")
        
        st.subheader("Key Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("📊 Total Recordings", len(df))
        m2.metric("🔍 After Filter", len(df_filtered))
        m3.metric("🗓️ Experiments", df["experiment_id"].nunique() if "experiment_id" in df.columns else "N/A")
        m4.metric("🌱 Plants", df["plant"].nunique() if "plant" in df.columns else "N/A")
        m5.metric("⏰ Time Span", f"{(df['event_timestamp'].max() - df['event_timestamp'].min()).days if 'event_timestamp' in df.columns else 'N/A'} days")
        
        st.markdown("---")
        st.subheader("Dataset Composition")
        c1, c2, c3 = st.columns(3)
        with c1:
            if "event_type" in df_filtered.columns:
                event_dist = df_filtered["event_type"].value_counts()
                fig_event = px.pie(values=event_dist.values, names=event_dist.index, template="plotly_dark", title="Event Type Distribution")
                st.plotly_chart(fig_event, use_container_width=True)
        with c2:
            if "stress_type" in df_filtered.columns:
                stress_dist = df_filtered["stress_type"].value_counts()
                fig_stress = px.pie(values=stress_dist.values, names=stress_dist.index, template="plotly_dark", title="Stress Type Distribution")
                st.plotly_chart(fig_stress, use_container_width=True)
        with c3:
            if "plant" in df_filtered.columns:
                plant_dist = df_filtered["plant"].value_counts()
                fig_plant = px.pie(values=plant_dist.values, names=plant_dist.index, template="plotly_dark", title="Plant Distribution")
                st.plotly_chart(fig_plant, use_container_width=True)
                
        st.markdown("---")
        st.subheader("Signal Properties (filtered)")
        sc1, sc2, sc3, sc4 = st.columns(4)
        if "peak_amplitude" in df_filtered.columns:
            sc1.metric("Peak Amplitude (V)", f"{df_filtered['peak_amplitude'].mean():.4f}", f"σ={df_filtered['peak_amplitude'].std():.4f}")
        if "dominant_frequency" in df_filtered.columns:
            sc2.metric("Avg Dom. Frequency (MHz)", f"{df_filtered['dominant_frequency'].mean():.3f}", f"σ={df_filtered['dominant_frequency'].std():.3f}")
        if "duration" in df_filtered.columns:
            sc3.metric("Avg Duration (s)", f"{df_filtered['duration'].mean():.2e}", f"σ={df_filtered['duration'].std():.2e}")
        if "energy" in df_filtered.columns:
            sc4.metric("Avg Energy (V²s)", f"{df_filtered['energy'].mean():.2e}", f"σ={df_filtered['energy'].std():.2e}")

    # --- TAB EXPLORER ---
    with tab_explorer:
        st.header("🔍 Feature Explorer")
        st.write("Explore correlations between different extracted features.")
        
        if numeric_cols:
            c1, c2, c3 = st.columns(3)
            x_axis = c1.selectbox("X-Axis", numeric_cols, index=0)
            y_axis = c2.selectbox("Y-Axis", numeric_cols, index=1 if len(numeric_cols)>1 else 0)
            
            group_cols = [c for c in ["experiment_id", "event_type", "stress_type", "plant", "sensor_position"] if c in df_filtered.columns]
            color_by = c3.selectbox("Color (Grouping)", group_cols) if group_cols else None

            if x_axis and y_axis:
                fig_scatter = px.scatter(
                    df_filtered, x=x_axis, y=y_axis, color=color_by,
                    hover_data=['recording_id'],
                    template="plotly_dark",
                    title=f"{y_axis} vs {x_axis}"
                )
                fig_scatter.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
                fig_scatter.update_layout(dragmode='lasso')
                
                try:
                    event = st.plotly_chart(fig_scatter, use_container_width=True, on_select="rerun", selection_mode=('points', 'lasso', 'box'), key="explorer_scatter")
                except TypeError:
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    st.warning("Update Streamlit to use interactive selection.")
                    event = None
                    
                if event and "selection" in event and "points" in event.selection and len(event.selection["points"]) > 0:
                    selected_points = event.selection["points"]
                    st.subheader(f"🔍 Inspecting {len(selected_points)} Selected Events")
                    
                    valid_rids = [p["customdata"][0] for p in selected_points if "customdata" in p]
                    if valid_rids:
                        st.markdown("**Batch Action:**")
                        c_all1, c_all2, _ = st.columns([1, 1, 4])
                        with c_all1:
                            if st.button(f"✔️ Relabel All ({len(valid_rids)}) to Cavitation", key="relabel_all_cav_explorer"):
                                import sqlite3
                                conn = sqlite3.connect(DB_PATH)
                                conn.executemany("UPDATE recordings SET event_type = 'Cavitation' WHERE recording_id = ?", [(r,) for r in valid_rids])
                                conn.commit()
                                conn.close()
                                st.rerun()
                        with c_all2:
                            if st.button(f"✖️ Relabel All ({len(valid_rids)}) to Noise", key="relabel_all_noise_explorer"):
                                import sqlite3
                                conn = sqlite3.connect(DB_PATH)
                                conn.executemany("UPDATE recordings SET event_type = 'Noise' WHERE recording_id = ?", [(r,) for r in valid_rids])
                                conn.commit()
                                conn.close()
                                st.rerun()
                    
                    if len(selected_points) > 30:
                        st.warning(f"You selected {len(selected_points)} points. Displaying the first 30 waveforms to prevent browser crash.")
                    
                    with st.spinner("Loading charts..."):
                        for p in selected_points[:30]:
                            rid = None
                            if "customdata" in p:
                                rid = p["customdata"][0]
                            
                            if rid:
                                rid_str = str(rid)
                                df_filtered_str = df_filtered.copy()
                                df_filtered_str['recording_id'] = df_filtered_str['recording_id'].astype(str)
                                current_label = df_filtered_str[df_filtered_str['recording_id'] == rid_str]['event_type'].values
                                current_label = current_label[0] if len(current_label) > 0 and pd.notna(current_label[0]) else "Unknown"
                                
                                render_signal_inspector(rid, current_label, "explorer", PROJECT_ROOT, DB_PATH)

            with st.expander("Feature Distribution (Boxplots)", expanded=False):
                box_feature = st.selectbox("Feature for Boxplot", numeric_cols, index=0, key='box')
                box_group = st.selectbox("Group by", group_cols, key='box_group') if group_cols else None
                
                fig_box = px.box(df_filtered, x=box_group, y=box_feature, color=box_group, template="plotly_dark", points="all")
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("No numeric features available.")

    # --- TAB TIMELINE ---
    with tab_timeline:
        st.header("📈 Measurement Timeline")
        if "event_timestamp" in df_filtered.columns and not df_filtered["event_timestamp"].dropna().empty and numeric_cols:
            time_feature = st.selectbox("Metric over Time", numeric_cols, index=1 if len(numeric_cols)>1 else 0)
            color_by_time = "event_type" if "event_type" in df_filtered.columns else None
            
            fig_time = px.line(
                df_filtered.dropna(subset=['event_timestamp']), 
                x="event_timestamp", y=time_feature, color=color_by_time, 
                markers=True, template="plotly_dark", title=f"Trend of {time_feature} over all Measurements"
            )
            fig_time.update_layout(
                title=dict(font=dict(size=22, family="Arial, sans-serif")),
                xaxis=dict(title=dict(font=dict(size=20, family="Arial, sans-serif")), tickfont=dict(size=16)),
                yaxis=dict(title=dict(font=dict(size=20, family="Arial, sans-serif")), tickfont=dict(size=16)),
                legend=dict(
                    yanchor="top",
                    y=0.97,
                    xanchor="right",
                    x=0.98,
                    bgcolor="rgba(20, 24, 35, 0.85)",
                    bordercolor="rgba(255, 255, 255, 0.25)",
                    borderwidth=1.5,
                    font=dict(size=18, family="Arial, sans-serif")
                )
            )
            st.plotly_chart(fig_time, use_container_width=True)
            
            st.subheader("Time-of-Day Distribution of Events")
            col_tod1, col_tod2 = st.columns([1.1, 1.1])
            with col_tod1:
                show_active_tod = st.checkbox(
                    "Overlay Active Experiments Curve (Background)", 
                    value=True, 
                    key="show_active_tod_curve",
                    help="Adds a secondary axis curve showing how many experiments were actively recording at each hour of the day."
                )
            with col_tod2:
                tod_theme = st.radio(
                    "Chart Theme (Thesis Export)", 
                    ["Dark Theme", "Thesis Light (White)"], 
                    horizontal=True, 
                    key="tod_theme_radio"
                )
            
            df_tod = df_filtered.dropna(subset=['event_timestamp']).copy()
            if not df_tod.empty:
                df_tod['Hour'] = df_tod['event_timestamp'].dt.hour
                
                is_light = (tod_theme == "Thesis Light (White)")
                theme_template = "plotly_white" if is_light else "plotly_dark"
                font_color = "#111111" if is_light else "#f0f2f6"
                grid_color = "rgba(0, 0, 0, 0.08)" if is_light else "rgba(255, 255, 255, 0.1)"
                zero_color = "rgba(0, 0, 0, 0.25)" if is_light else "rgba(255, 255, 255, 0.3)"
                leg_bg = "rgba(255, 255, 255, 0.88)" if is_light else "rgba(20, 24, 35, 0.85)"
                leg_border = "rgba(0, 0, 0, 0.25)" if is_light else "rgba(255, 255, 255, 0.25)"
                active_color = "#D96E14" if is_light else "#FFA15A"

                event_colors = {
                    'Cavitation': '#00CC96',  # Emerald green
                    'Noise': '#EF553B',       # Coral red
                    'Control': '#636EFA'      # Blue
                }
                
                fig_tod = px.histogram(
                    df_tod, x='Hour', color=color_by_time, range_x=[-0.5, 23.5], nbins=24,
                    template=theme_template, 
                    labels={"Hour": "Time of Day (Hour)", "count": "Events"},
                    color_discrete_map=event_colors if color_by_time == 'event_type' else None
                )
                
                if show_active_tod and 'experiment_id' in df_filtered.columns:
                    exp_spans = []
                    for exp_id in df_filtered['experiment_id'].dropna().unique():
                        sub_exp = df[df['experiment_id'] == exp_id]
                        start_t = pd.to_datetime(sub_exp['started_at'], errors='coerce').dropna().min()
                        end_t = pd.to_datetime(sub_exp['event_timestamp'], errors='coerce').dropna().max()
                        if pd.isna(start_t):
                            start_t = pd.to_datetime(sub_exp['event_timestamp'], errors='coerce').dropna().min()
                        if pd.isna(end_t):
                            end_t = start_t
                        if pd.notna(start_t) and pd.notna(end_t):
                            exp_spans.append((start_t, end_t))
                            
                    def get_active_hours(start_t, end_t):
                        if pd.isna(start_t) or pd.isna(end_t) or end_t < start_t:
                            return set()
                        if (end_t - start_t) >= pd.Timedelta(hours=24):
                            return set(range(24))
                        hours = set()
                        cur = start_t.floor('h')
                        while cur <= end_t:
                            hour_start = cur
                            hour_end = cur + pd.Timedelta(hours=1)
                            if start_t == end_t:
                                if hour_start <= start_t <= hour_end:
                                    hours.add(cur.hour)
                            else:
                                if max(start_t, hour_start) < min(end_t, hour_end):
                                    hours.add(cur.hour)
                            cur += pd.Timedelta(hours=1)
                        return hours
                        
                    hourly_active = [0] * 24
                    for s, e in exp_spans:
                        for h in get_active_hours(s, e):
                            hourly_active[h] += 1
                            
                    fig_tod.add_trace(
                        go.Scatter(
                            x=list(range(24)),
                            y=hourly_active,
                            name="Active Experiments",
                            mode="lines",
                            line=dict(color=active_color, width=3.0, dash="dot"),
                            yaxis="y2",
                            hovertemplate=f"Time: %{{x:02d}}:00<br>Active Experiments: %{{y}} / {len(exp_spans)}<extra></extra>"
                        )
                    )
                    fig_tod.update_layout(
                        yaxis2=dict(
                            title=dict(text="Active Experiments", font=dict(size=24, family="Arial, sans-serif", color=active_color)),
                            tickfont=dict(size=20, color=active_color),
                            overlaying="y",
                            side="right",
                            showgrid=False
                        )
                    )

                fig_tod.update_layout(
                    barmode='stack',
                    height=520,
                    margin=dict(l=75, r=75, t=65, b=65),
                    title=dict(
                        text="Time-of-Day Distribution of Events & Active Experiments" if show_active_tod else "Time-of-Day Distribution of Events",
                        font=dict(size=24, family="Arial, sans-serif", color=font_color)
                    ),
                    xaxis=dict(
                        title=dict(text="Time of Day (Hour)", font=dict(size=24, family="Arial, sans-serif", color=font_color)),
                        tickfont=dict(size=18, color=font_color),
                        tickmode='array',
                        tickvals=list(range(24)),
                        ticktext=[f"{h:02d}:00" for h in range(24)],
                        showgrid=True,
                        gridcolor=grid_color
                    ),
                    yaxis=dict(
                        title=dict(text="Event Count", font=dict(size=24, family="Arial, sans-serif", color=font_color)),
                        tickfont=dict(size=20, color=font_color),
                        showgrid=True,
                        gridcolor=grid_color,
                        zeroline=True,
                        zerolinecolor=zero_color
                    ),
                    legend=dict(
                        title=dict(text=""),
                        yanchor="top",
                        y=0.97,
                        xanchor="right",
                        x=0.96,
                        bgcolor=leg_bg,
                        bordercolor=leg_border,
                        borderwidth=1.5,
                        font=dict(size=22, family="Arial, sans-serif", color=font_color)
                    )
                )

                plot_config = {
                    'toImageButtonOptions': {
                        'format': 'png',
                        'width': None,
                        'height': None,
                        'scale': 2
                    }
                }
                st.plotly_chart(fig_tod, use_container_width=True, config=plot_config)
        else:
            st.info("No valid timestamps found.")

    # --- TAB TIME (Zeit-Vorhersage) ---
    with tab_time:
        st.header("⏱️ Event & Cavitation Time Analysis")
        st.write("Analyzes the elapsed time since the experiment started, categorizing events by type (Cavitation, Noise, Control) and plant species.")
        
        df_time = df_filtered.dropna(subset=['started_at', 'event_timestamp']).copy()
        if not df_time.empty:
            df_time['started_at'] = pd.to_datetime(df_time['started_at'], errors='coerce')
            df_time['event_timestamp'] = pd.to_datetime(df_time['event_timestamp'], errors='coerce')
            df_time = df_time.dropna(subset=['started_at', 'event_timestamp'])
            df_time['time_elapsed_min'] = (df_time['event_timestamp'] - df_time['started_at']).dt.total_seconds() / 60.0
            df_time = df_time[(df_time['time_elapsed_min'] >= 0) & (df_time['time_elapsed_min'] < 500000)]
            
            df_time_all = df_time.copy()
            selected_exp = "All"
            if 'experiment_id' in df_time.columns:
                unique_exps = sorted(df_time['experiment_id'].dropna().unique().tolist())
                if unique_exps:
                    selected_exp = st.selectbox("🔍 Filter Time Charts by Experiment ID", ["All"] + unique_exps, index=0, key="time_exp_filter")
                    if selected_exp != "All":
                        df_time = df_time[df_time['experiment_id'] == selected_exp].copy()

            def _format_td(td):
                if pd.isna(td) or td.total_seconds() < 0:
                    return "0m"
                sec = int(td.total_seconds())
                d, rem = divmod(sec, 86400)
                h, rem = divmod(rem, 3600)
                m, _ = divmod(rem, 60)
                parts = []
                if d > 0:
                    parts.append(f"{d}d")
                if h > 0 or d > 0:
                    parts.append(f"{h}h")
                parts.append(f"{m}m")
                return " ".join(parts)

            # Build Experiment Duration Summary Table
            exp_table_records = []
            if 'experiment_id' in df_time_all.columns:
                for exp_id, sub in df_time_all.groupby('experiment_id'):
                    start_t = sub['started_at'].min()
                    end_t = sub['event_timestamp'].max()
                    dur = end_t - start_t if pd.notna(start_t) and pd.notna(end_t) else pd.Timedelta(0)
                    dur = max(dur, pd.Timedelta(0))
                    total_ev = len(sub)
                    cav_ev = (sub['event_type'] == 'Cavitation').sum() if 'event_type' in sub.columns else 0
                    noise_ev = (sub['event_type'] == 'Noise').sum() if 'event_type' in sub.columns else 0
                    plant_name = sub['plant'].dropna().iloc[0] if ('plant' in sub.columns and not sub['plant'].dropna().empty) else 'Unknown'
                    exp_table_records.append({
                        'Experiment ID': str(exp_id),
                        'Plant': str(plant_name),
                        'Start Time': start_t.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(start_t) else 'N/A',
                        'Latest Event': end_t.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(end_t) else 'N/A',
                        'Duration': _format_td(dur),
                        'Duration (Hours)': round(dur.total_seconds() / 3600.0, 1),
                        'Total Events': int(total_ev),
                        'Cavitation': int(cav_ev),
                        'Cavitation %': f"{(cav_ev / total_ev) * 100:.1f}%" if total_ev else "0%",
                        'Noise': int(noise_ev),
                        'Noise %': f"{(noise_ev / total_ev) * 100:.1f}%" if total_ev else "0%",
                        '_dur_td': dur
                    })
            df_exp_table = pd.DataFrame(exp_table_records)

            # --- TOP-LEVEL RECORDING TIME METRICS ---
            if not df_exp_table.empty:
                if selected_exp == "All":
                    total_rec_td = sum((r['_dur_td'] for r in exp_table_records), pd.Timedelta(0))
                    total_rec_hours = total_rec_td.total_seconds() / 3600.0
                    rt_c1, rt_c2, rt_c3, rt_c4 = st.columns(4)
                    rt_c1.metric("Total Recording Time", _format_td(total_rec_td), help="Total recording time summed across all experiments (Start Time to Latest Event).")
                    rt_c2.metric("Total Hours", f"{total_rec_hours:,.1f} h")
                    rt_c3.metric("Experiments", f"{len(df_exp_table)}")
                    rt_c4.metric("Total Events", f"{len(df_time_all):,}")
                else:
                    match_row = df_exp_table[df_exp_table['Experiment ID'] == selected_exp]
                    if not match_row.empty:
                        row = match_row.iloc[0]
                        rt_c1, rt_c2, rt_c3, rt_c4 = st.columns(4)
                        rt_c1.metric("Recording Duration", row['Duration'], help="Elapsed duration from Experiment Start Time to the Latest Event.")
                        rt_c2.metric("Duration (Hours)", f"{row['Duration (Hours)']:.1f} h")
                        rt_c3.metric("Start Time", str(row['Start Time']))
                        rt_c4.metric("Latest Event", str(row['Latest Event']))
            
            if not df_time.empty:
                event_colors = {
                    'Cavitation': '#00CC96',  # Emerald green
                    'Noise': '#EF553B',       # Coral red
                    'Control': '#636EFA'      # Blue
                }

                # --- 1. EVENT TYPE CATEGORIZATION OVER TIME ---
                st.subheader("📊 Event Type Categorization (Cavitation vs. Noise)")
                
                if 'event_type' in df_time.columns:
                    et_counts = df_time['event_type'].value_counts()
                    total_events = len(df_time)
                    
                    m_cols = st.columns(min(len(et_counts) + 1, 5))
                    m_cols[0].metric("Total Events", f"{total_events:,}")
                    for idx, (etype, cnt) in enumerate(et_counts.items()):
                        if idx + 1 < len(m_cols):
                            pct = (cnt / total_events) * 100
                            m_cols[idx + 1].metric(f"{etype}", f"{cnt:,}", f"{pct:.1f}%")

                    col_ctrl1, col_ctrl2 = st.columns([1, 1])
                    with col_ctrl1:
                        barmode = st.radio(
                            "Bar Mode", 
                            ["stack", "group", "relative"], 
                            format_func=lambda x: {"stack": "Stacked", "group": "Grouped (Side-by-side)", "relative": "Relative (%)"}[x],
                            horizontal=True,
                            key="time_barmode"
                        )
                    with col_ctrl2:
                        time_unit = st.radio("Time Axis Unit", ["Minutes", "Hours"], horizontal=True, key="time_unit")
                    
                    time_col = 'time_elapsed_display'
                    if time_unit == "Hours":
                        df_time[time_col] = df_time['time_elapsed_min'] / 60.0
                        df_time_all[time_col] = df_time_all['time_elapsed_min'] / 60.0
                        x_label = "Elapsed Time (Hours)"
                    else:
                        df_time[time_col] = df_time['time_elapsed_min']
                        df_time_all[time_col] = df_time_all['time_elapsed_min']
                        x_label = "Elapsed Time (Minutes)"

                    fig_event_time = px.histogram(
                        df_time, 
                        x=time_col, 
                        color='event_type',
                        color_discrete_map=event_colors,
                        nbins=60,
                        barmode=barmode,
                        barnorm="" if barmode != "relative" else "percent",
                        title="Event Distribution Over Elapsed Time by Category (Cavitation / Noise / Control)",
                        template="plotly_dark",
                        opacity=0.8,
                        labels={time_col: x_label, 'count': 'Events', 'event_type': 'Event Type'}
                    )
                    fig_event_time.update_layout(
                        margin=dict(l=0, r=0, t=40, b=0),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_event_time, use_container_width=True)

                    with st.expander("📈 Cumulative Events Over Time (Cavitation vs. Noise Rate)", expanded=False):
                        df_cum = df_time.sort_values(time_col).copy()
                        df_cum['cumulative_count'] = df_cum.groupby('event_type').cumcount() + 1
                        
                        fig_cum = px.line(
                            df_cum,
                            x=time_col,
                            y='cumulative_count',
                            color='event_type',
                            color_discrete_map=event_colors,
                            title="Cumulative Events Over Time by Category",
                            template="plotly_dark",
                            labels={time_col: x_label, 'cumulative_count': 'Cumulative Events', 'event_type': 'Event Type'}
                        )
                        fig_cum.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                        st.plotly_chart(fig_cum, use_container_width=True)

                # --- 2. EVENTS OVER TIME BY EXPERIMENT ID ---
                st.subheader("🧪 Events Over Time by Experiment ID")
                st.caption("Distribution of all events across elapsed time, with the Experiment ID as the legend category.")
                if 'experiment_id' in df_time_all.columns:
                    exp_time_source = df_time_all if selected_exp == "All" else df_time
                    unique_chart_exps = sorted(exp_time_source['experiment_id'].dropna().unique().tolist())
                    
                    c_et1, c_et2 = st.columns([1, 1])
                    with c_et1:
                        exp_filter_mode = st.radio(
                            "Experiments Shown",
                            ["All Experiments", "Top 10 by Event Count", "Custom Selection"],
                            horizontal=True,
                            key="exp_time_filter_mode"
                        )
                    with c_et2:
                        exp_barmode = st.radio(
                            "Bar Mode",
                            ["stack", "group"],
                            format_func=lambda x: {"stack": "Stacked", "group": "Grouped (Side-by-side)"}[x],
                            horizontal=True,
                            key="exp_time_barmode"
                        )
                    
                    if exp_filter_mode == "Top 10 by Event Count":
                        top_10 = exp_time_source['experiment_id'].value_counts().head(10).index.tolist()
                        df_exp_plot = exp_time_source[exp_time_source['experiment_id'].isin(top_10)].copy()
                    elif exp_filter_mode == "Custom Selection":
                        chosen_exps = st.multiselect(
                            "Select Experiments to Plot",
                            unique_chart_exps,
                            default=unique_chart_exps[:5] if len(unique_chart_exps) >= 5 else unique_chart_exps,
                            key="exp_time_custom_select"
                        )
                        df_exp_plot = exp_time_source[exp_time_source['experiment_id'].isin(chosen_exps)].copy()
                    else:
                        df_exp_plot = exp_time_source.copy()

                    if time_col not in df_exp_plot.columns and 'time_elapsed_min' in df_exp_plot.columns:
                        df_exp_plot[time_col] = df_exp_plot['time_elapsed_min'] / (60.0 if time_unit == "Hours" else 1.0)

                    if not df_exp_plot.empty:
                        fig_time_by_exp = px.histogram(
                            df_exp_plot,
                            x=time_col,
                            color='experiment_id',
                            nbins=60,
                            barmode=exp_barmode,
                            title=f"All Events Over Elapsed Time by Experiment ID ({time_unit})",
                            template="plotly_dark",
                            opacity=0.8,
                            labels={time_col: x_label, 'count': 'Events', 'experiment_id': 'Experiment ID'}
                        )
                        fig_time_by_exp.update_layout(
                            margin=dict(l=0, r=0, t=40, b=0),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_time_by_exp, use_container_width=True)
                    else:
                        st.info("No events match the selected experiment filter.")

                    with st.expander("📊 Total Event Counts by Experiment ID (Summary Bar Chart)", expanded=False):
                        c_exp1, c_exp2 = st.columns([1, 1])
                        with c_exp1:
                            exp_sort_by = st.selectbox(
                                "Sort Summary", 
                                ["Count (Desc)", "Chronological / ID"], 
                                key="exp_sort_by"
                            )
                        with c_exp2:
                            exp_log_y = st.checkbox("Log scale (Y-axis)", value=False, key="exp_log_y")

                        exp_grouped = exp_time_source.groupby(['experiment_id', 'event_type']).size().reset_index(name='count')
                        if exp_sort_by == "Count (Desc)":
                            exp_order = exp_time_source['experiment_id'].value_counts().index.tolist()
                        else:
                            exp_order = sorted(exp_time_source['experiment_id'].unique().tolist())
                            
                        fig_exp_summary = px.bar(
                            exp_grouped,
                            x='experiment_id',
                            y='count',
                            color='event_type',
                            color_discrete_map=event_colors,
                            category_orders={'experiment_id': exp_order},
                            title="Total Events per Experiment ID (Colored by Event Type)",
                            template="plotly_dark",
                            log_y=exp_log_y,
                            labels={'experiment_id': 'Experiment ID', 'count': 'Total Events', 'event_type': 'Event Type'}
                        )
                        fig_exp_summary.update_layout(
                            margin=dict(l=0, r=0, t=40, b=80),
                            xaxis_tickangle=-45,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_exp_summary, use_container_width=True)

                # --- EXPERIMENT RECORDING DURATIONS TABLE ---
                st.subheader("📋 Experiment Recording Durations Table")
                st.caption("Recording durations calculated from the experiment start time to the latest event timestamp.")
                if not df_exp_table.empty:
                    display_table = df_exp_table.drop(columns=['_dur_td']).sort_values('Duration (Hours)', ascending=False)
                    if selected_exp != "All":
                        tbl_filter = st.checkbox("Show only selected experiment in table", value=False, key="tbl_filter_selected")
                        if tbl_filter:
                            display_table = display_table[display_table['Experiment ID'] == selected_exp]
                    st.dataframe(
                        display_table,
                        use_container_width=True,
                        hide_index=True
                    )

                # --- 3. PLANT-SPECIFIC CAVITATION TIME DISTRIBUTION ---
                st.subheader("🌱 Cavitation Frequency by Plant Over Time")
                st.caption("Distribution of genuine cavitation events over elapsed time. Shows background experiment durations to explain why event rates taper as shorter experiments conclude.")
                
                df_cav = df_time[df_time['event_type'] == 'Cavitation'] if 'event_type' in df_time.columns else df_time
                if not df_cav.empty:
                    df_cav = df_cav.copy()
                    if time_col not in df_cav.columns and 'time_elapsed_min' in df_cav.columns:
                        df_cav[time_col] = df_cav['time_elapsed_min'] / (60.0 if time_unit == "Hours" else 1.0)

                    col_opt1, col_opt2 = st.columns([1, 1])
                    with col_opt1:
                        show_active_curve = st.checkbox(
                            "Overlay Active Experiments Curve (Background)", 
                            value=True, 
                            key="show_active_curve",
                            help="Adds a secondary axis curve showing how many experiments were still recording at each point in time."
                        )
                    with col_opt2:
                        show_exp_ends = st.checkbox(
                            "Show Experiment End Markers", 
                            value=(selected_exp != "All"), 
                            key="show_exp_ends",
                            help="Shows dashed marker(s) where experiments completed."
                        )

                    # Build Figure
                    unit_div = 60.0 if time_unit == "Hours" else 1.0
                    if show_active_curve and selected_exp == "All" and not df_exp_table.empty:
                        fig_time_dist = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        plants = df_cav['plant'].dropna().unique().tolist() if 'plant' in df_cav.columns else ['All']
                        for plant_name in sorted(plants):
                            sub_p = df_cav[df_cav['plant'] == plant_name] if 'plant' in df_cav.columns else df_cav
                            fig_time_dist.add_trace(
                                go.Histogram(
                                    x=sub_p[time_col],
                                    name=str(plant_name),
                                    nbinsx=50,
                                    opacity=0.75
                                ),
                                secondary_y=False
                            )
                        
                        exp_durations_unit = [r['_dur_td'].total_seconds() / (60.0 * unit_div) for r in exp_table_records if r['_dur_td'].total_seconds() > 0]
                        if exp_durations_unit:
                            max_dur = max(exp_durations_unit)
                            grid_t = np.linspace(0, max_dur, 100)
                            active_counts = [(sum(1 for d in exp_durations_unit if d >= t)) for t in grid_t]
                            
                            fig_time_dist.add_trace(
                                go.Scatter(
                                    x=grid_t,
                                    y=active_counts,
                                    name="Active Experiments",
                                    mode="lines",
                                    line=dict(color="#FFA15A", width=2.5, dash="dot"),
                                    hovertemplate=f"Elapsed: %{{x:.1f}} {time_unit.lower()}<br>Active Experiments: %{{y}} / {len(exp_durations_unit)}<extra></extra>"
                                ),
                                secondary_y=True
                            )
                        
                        fig_time_dist.update_layout(
                            barmode='stack',
                            title=f"Genuine Cavitation Events Over Time & Active Experiments ({time_unit})",
                            template="plotly_dark",
                            margin=dict(l=0, r=0, t=40, b=0),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        fig_time_dist.update_xaxes(title_text=x_label)
                        fig_time_dist.update_yaxes(title_text="Cavitation Events", secondary_y=False)
                        fig_time_dist.update_yaxes(title_text="Active Experiments", secondary_y=True, showgrid=False)
                    else:
                        fig_time_dist = px.histogram(
                            df_cav, 
                            x=time_col, 
                            nbins=50, 
                            color='plant' if 'plant' in df_cav.columns else None,
                            title=f"Genuine Cavitation Events Over Time by Plant ({time_unit})",
                            template="plotly_dark", 
                            opacity=0.75,
                            labels={time_col: x_label, 'plant': 'Plant', 'count': 'Cavitation Events'}
                        )
                        fig_time_dist.update_layout(
                            margin=dict(l=0, r=0, t=40, b=0),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )

                    # Add experiment end markers if requested or if single experiment is selected
                    if show_exp_ends:
                        if selected_exp != "All" and not df_exp_table.empty:
                            m_row = df_exp_table[df_exp_table['Experiment ID'] == selected_exp]
                            if not m_row.empty:
                                dur_val = m_row.iloc[0]['_dur_td'].total_seconds() / (60.0 * unit_div)
                                fig_time_dist.add_vline(
                                    x=dur_val, 
                                    line_dash="dash", 
                                    line_color="#FFA15A", 
                                    line_width=2,
                                    annotation_text=f"Exp End ({dur_val:.1f} {time_unit.lower()})",
                                    annotation_position="top right"
                                )
                        elif selected_exp == "All" and not df_exp_table.empty:
                            top_ends = df_exp_table.nlargest(5, 'Duration (Hours)')
                            for _, r in top_ends.iterrows():
                                dur_val = r['_dur_td'].total_seconds() / (60.0 * unit_div)
                                fig_time_dist.add_vline(
                                    x=dur_val,
                                    line_dash="dot",
                                    line_color="rgba(255, 161, 90, 0.4)",
                                    line_width=1,
                                    annotation_text=f"{r['Experiment ID'][-6:]} end",
                                    annotation_position="top"
                                )

                    st.plotly_chart(fig_time_dist, use_container_width=True)

                    # --- GANTT-STYLE EXPERIMENT LIFESPAN OVERVIEW ---
                    with st.expander("📊 Experiment Lifespans Overview (Gantt Chart)", expanded=False):
                        if not df_exp_table.empty:
                            df_gantt = df_exp_table.copy()
                            df_gantt['Length'] = df_gantt['_dur_td'].apply(lambda td: round(td.total_seconds() / (60.0 * unit_div), 1))
                            df_gantt = df_gantt.sort_values('Length', ascending=True)
                            
                            fig_gantt = px.bar(
                                df_gantt,
                                x='Length',
                                y='Experiment ID',
                                color='Plant',
                                orientation='h',
                                title=f"Experiment Total Recording Lengths ({time_unit})",
                                template='plotly_dark',
                                labels={'Length': f'Total Recording Length ({time_unit})', 'Experiment ID': 'Experiment ID'},
                                hover_data=['Duration', 'Start Time', 'Latest Event', 'Cavitation']
                            )
                            fig_gantt.update_layout(
                                margin=dict(l=0, r=0, t=40, b=0),
                                height=max(350, len(df_gantt) * 18),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            st.plotly_chart(fig_gantt, use_container_width=True)

                    # --- 4. CAVITATION PARAMETER DYNAMICS OVER TIME ---
                    st.subheader("💧 Cavitation Events & Parameter Dynamics")
                    st.caption("Analyze how environmental and signal parameters (like Soil Moisture, Temperature, Humidity) evolve over elapsed time alongside cavitation events.")

                    candidate_params = {
                        'soil_moisture': 'Soil Moisture (%)',
                        'temperature': 'Temperature (°C)',
                        'humidity': 'Relative Humidity (%)',
                        'stem_diameter': 'Stem Diameter (mm)',
                        'peak_amplitude': 'Peak Amplitude (V)',
                        'energy': 'Energy',
                        'duration': 'Duration (µs)',
                        'counts': 'Counts',
                        'dominant_frequency': 'Dominant Frequency (kHz)'
                    }
                    valid_param_cols = [c for c in candidate_params.keys() if c in df_cav.columns and df_cav[c].dropna().nunique() > 1]
                    for c in df_cav.select_dtypes(include=[np.number]).columns:
                        if c not in valid_param_cols and not c.startswith('_') and c not in ['time_elapsed_min', 'time_elapsed_display', 'index']:
                            if df_cav[c].dropna().nunique() > 1:
                                valid_param_cols.append(c)

                    if valid_param_cols:
                        default_idx = valid_param_cols.index('soil_moisture') if 'soil_moisture' in valid_param_cols else 0
                        cp1, cp2 = st.columns([1.5, 1.5])
                        with cp1:
                            param_chosen = st.selectbox(
                                "Parameter on Primary Y-Axis",
                                valid_param_cols,
                                index=default_idx,
                                format_func=lambda x: candidate_params.get(x, x.replace('_', ' ').title()),
                                key="param_chart_y"
                            )
                        with cp2:
                            chart_view_mode = st.radio(
                                "Chart View",
                                ["Parameter vs Time", "Parameter Distribution Histogram"],
                                horizontal=True,
                                key="param_chart_view"
                            )

                        param_label = candidate_params.get(param_chosen, param_chosen.replace('_', ' ').title())

                        if chart_view_mode == "Parameter vs Time":
                            co1, co2 = st.columns([1, 1])
                            with co1:
                                show_param_active = st.checkbox(
                                    "Overlay Active Experiments Curve (Background)", 
                                    value=True, 
                                    key="param_show_active_curve",
                                    help="Adds an orange dashed curve on the secondary Y-axis showing how many experiments were still actively recording at each point in time."
                                )
                            with co2:
                                show_param_rate = st.checkbox(
                                    "Overlay Cavitation Event Rate Histogram", 
                                    value=False, 
                                    key="param_show_rate_curve",
                                    help="Adds background bars showing cavitation event counts over time."
                                )

                            fig_param = make_subplots(specs=[[{"secondary_y": True}]])
                            
                            plants = df_cav['plant'].dropna().unique().tolist() if 'plant' in df_cav.columns else ['All']
                            for plant_name in sorted(plants):
                                sub_p = df_cav[df_cav['plant'] == plant_name].dropna(subset=[time_col, param_chosen]) if 'plant' in df_cav.columns else df_cav.dropna(subset=[time_col, param_chosen])
                                if not sub_p.empty:
                                    fig_param.add_trace(
                                        go.Scatter(
                                            x=sub_p[time_col],
                                            y=sub_p[param_chosen],
                                            mode='markers',
                                            name=str(plant_name),
                                            marker=dict(size=6, opacity=0.75),
                                            hovertemplate=f"Elapsed: %{{x:.1f}} {time_unit.lower()}<br>{param_label}: %{{y:.2f}}<extra>{plant_name}</extra>"
                                        ),
                                        secondary_y=False
                                    )

                            if show_param_rate:
                                fig_param.add_trace(
                                    go.Histogram(
                                        x=df_cav[time_col],
                                        nbinsx=50,
                                        name="Cavitation Events Count",
                                        marker_color="rgba(0, 204, 150, 0.25)",
                                        opacity=0.3
                                    ),
                                    secondary_y=True
                                )

                            unit_div = 60.0 if time_unit == "Hours" else 1.0
                            if show_param_active and selected_exp == "All" and not df_exp_table.empty:
                                exp_durations_unit = [r['_dur_td'].total_seconds() / (60.0 * unit_div) for r in exp_table_records if r['_dur_td'].total_seconds() > 0]
                                if exp_durations_unit:
                                    max_dur = max(exp_durations_unit)
                                    grid_t = np.linspace(0, max_dur, 100)
                                    active_counts = [(sum(1 for d in exp_durations_unit if d >= t)) for t in grid_t]
                                    fig_param.add_trace(
                                        go.Scatter(
                                            x=grid_t,
                                            y=active_counts,
                                            name="Active Experiments",
                                            mode="lines",
                                            line=dict(color="#FFA15A", width=2.5, dash="dot"),
                                            hovertemplate=f"Elapsed: %{{x:.1f}} {time_unit.lower()}<br>Active Experiments: %{{y}} / {len(exp_durations_unit)}<extra></extra>"
                                        ),
                                        secondary_y=True
                                    )

                            # If single experiment, add end line
                            if selected_exp != "All" and not df_exp_table.empty:
                                m_row = df_exp_table[df_exp_table['Experiment ID'] == selected_exp]
                                if not m_row.empty:
                                    dur_val = m_row.iloc[0]['_dur_td'].total_seconds() / (60.0 * unit_div)
                                    fig_param.add_vline(
                                        x=dur_val,
                                        line_dash="dash",
                                        line_color="#FFA15A",
                                        line_width=2,
                                        annotation_text=f"Exp End ({dur_val:.1f} {time_unit.lower()})",
                                        annotation_position="top right"
                                    )

                            fig_param.update_layout(
                                title=f"Cavitation Events: {param_label} vs. Elapsed Time ({time_unit})",
                                template="plotly_dark",
                                margin=dict(l=0, r=0, t=40, b=0),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            fig_param.update_xaxes(title_text=x_label)
                            fig_param.update_yaxes(title_text=param_label, secondary_y=False)
                            if show_param_active or show_param_rate:
                                sec_title = "Active Experiments" if show_param_active and not show_param_rate else ("Cavitation Event Count" if not show_param_active and show_param_rate else "Active Experiments / Event Count")
                                fig_param.update_yaxes(
                                    title_text=sec_title, 
                                    secondary_y=True, 
                                    showgrid=False
                                )
                            st.plotly_chart(fig_param, use_container_width=True)
                        else:
                            df_param_hist = df_cav.dropna(subset=[param_chosen])
                            fig_param = px.histogram(
                                df_param_hist,
                                x=param_chosen,
                                nbins=40,
                                color='plant' if 'plant' in df_param_hist.columns else None,
                                title=f"Cavitation Event Frequency by {param_label} (Colored by Plant)",
                                template="plotly_dark",
                                opacity=0.75,
                                labels={param_chosen: param_label, 'count': 'Cavitation Events', 'plant': 'Plant'}
                            )
                            fig_param.update_layout(
                                margin=dict(l=0, r=0, t=40, b=0),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            st.plotly_chart(fig_param, use_container_width=True)
                    else:
                        st.info("No parameters with variable values available for plotting.")
                else:
                    st.info("No cavitation events found in the filtered selection.")
                
                # Check for model
                time_model_path = MODELS_DIR / 'time_model.joblib'
                if time_model_path.exists():
                    st.success("Time prediction model is available (Not run dynamically in this basic view).")
                else:
                    st.info("No trained time prediction model found.")
            else:
                st.warning("No valid time data found after filtering.")
        else:
            st.info("No data with 'started_at' and 'event_timestamp' available.")

    # --- TAB SIGNAL ---
    with tab_signal:
        st.header("🔍 Signal Overlay & Comparison")
        st.write("Overlay waveforms and spectra to visually compare Acoustic Emissions.")
        
        if "recording_id" in df_filtered.columns:
            available_ids = df_filtered["recording_id"].tolist()
            selected_ids = st.multiselect("Select Signals to Overlay", available_ids, default=available_ids[:2] if len(available_ids)>=2 else available_ids)

            if selected_ids:
                if len(selected_ids) > 10:
                    st.warning("Maximum 10 signals recommended. Showing first 10.")
                    selected_ids = selected_ids[:10]

                # Style & Layout Controls for Thesis Export
                c_opt1, c_opt2, c_opt3, c_opt4 = st.columns([1.1, 1.1, 0.85, 1.05])
                with c_opt1:
                    sig_layout = st.radio(
                        "Chart Layout", 
                        ["Stacked (Full Width / Large)", "Side-by-Side"], 
                        horizontal=True, 
                        key="sig_layout_radio"
                    )
                with c_opt2:
                    sig_theme = st.radio(
                        "Chart Theme (Thesis Export)", 
                        ["Thesis Light (White)", "Dark Theme"], 
                        horizontal=True, 
                        key="sig_theme_radio"
                    )
                with c_opt3:
                    sig_freq_unit = st.radio(
                        "Frequency Unit", 
                        ["kHz", "MHz"], 
                        horizontal=True, 
                        key="sig_freq_unit_radio"
                    )
                with c_opt4:
                    st.write("Reference Shape")
                    show_ideal_hull = st.checkbox(
                        "Ideal Cavitation Hull", 
                        value=False, 
                        key="sig_ideal_hull_chk"
                    )

                DEFAULT_COLORS = [
                    '#1f77b4',  # blue
                    '#d62728',  # red
                    '#2ca02c',  # green
                    '#9467bd',  # purple
                    '#8c564b',  # brown
                    '#e377c2',  # pink
                    '#7f7f7f',  # gray
                    '#bcbd22',  # olive
                    '#17becf',  # cyan
                    '#ff7f0e'   # orange
                ]

                def hex_to_rgba(hex_str, alpha=0.12):
                    hex_str = hex_str.lstrip('#')
                    if len(hex_str) == 6:
                        r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
                        return f"rgba({r}, {g}, {b}, {alpha})"
                    return f"rgba(255, 127, 14, {alpha})"

                # Expandable Color & Style Customization
                with st.expander("🎨 Line Colors & Style Customization", expanded=False):
                    c_st1, c_st2 = st.columns([1.2, 3.8])
                    with c_st1:
                        line_width = st.slider("Line Width", min_value=1.0, max_value=4.0, value=1.6, step=0.2, key="sig_line_width_slider")
                        if show_ideal_hull:
                            hull_color = st.color_picker("Hull Outline Color", value="#ff7f0e", key="sig_hull_color_picker")
                        else:
                            hull_color = "#ff7f0e"
                    with c_st2:
                        st.write("**Signal Colors:**")
                        num_pick_cols = min(max(len(selected_ids), 1), 4)
                        pick_cols = st.columns(num_pick_cols)
                        sig_colors = {}
                        for i, rid in enumerate(selected_ids):
                            def_col = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                            with pick_cols[i % num_pick_cols]:
                                sig_colors[rid] = st.color_picker(
                                    f"ID {str(rid)[-6:]}", 
                                    value=def_col, 
                                    key=f"sig_col_{rid}",
                                    help=f"Color for signal {rid}"
                                )

                is_light = (sig_theme == "Thesis Light (White)")
                theme_template = "plotly_white" if is_light else "plotly_dark"
                font_color = "#111111" if is_light else "#f0f2f6"
                grid_color = "rgba(0, 0, 0, 0.08)" if is_light else "rgba(255, 255, 255, 0.1)"
                zero_color = "rgba(0, 0, 0, 0.25)" if is_light else "rgba(255, 255, 255, 0.3)"
                leg_bg = "rgba(255, 255, 255, 0.88)" if is_light else "rgba(20, 24, 35, 0.82)"
                leg_border = "rgba(0, 0, 0, 0.25)" if is_light else "rgba(255, 255, 255, 0.25)"
                
                chart_height = 520 if sig_layout == "Stacked (Full Width / Large)" else 480

                fig_scope = go.Figure()
                fig_spec = go.Figure()

                if show_ideal_hull:
                    # Ideal physical cavitation hull (Hüllkurve): pre-trigger baseline 0 V + shock step + exponential decay
                    t_pre_hull = np.linspace(-0.8, 0.0, 150, endpoint=False)
                    t_post_hull = np.linspace(0.0, 1.2, 600)
                    t_hull = np.concatenate([t_pre_hull, t_post_hull])
                    
                    a0_ideal = 0.88
                    tau_ideal = 0.135  # ms (135 us)
                    
                    y_upper = np.concatenate([np.zeros_like(t_pre_hull), a0_ideal * np.exp(-t_post_hull / tau_ideal)])
                    y_lower = -y_upper
                    
                    x_poly = np.concatenate([t_hull, t_hull[::-1]])
                    y_poly = np.concatenate([y_upper, y_lower[::-1]])
                    
                    hull_fill = hex_to_rgba(hull_color, 0.12 if is_light else 0.16)
                    fig_scope.add_trace(go.Scatter(
                        x=x_poly,
                        y=y_poly,
                        fill='toself',
                        fillcolor=hull_fill,
                        line=dict(
                            color=hull_color,
                            width=line_width,
                            dash='dash'
                        ),
                        name="Ideal Cavitation Hull",
                        hoverinfo='skip',
                        showlegend=True
                    ))
                
                for i, rid in enumerate(selected_ids):
                    # Non-breaking spaces anchored with a zero-width space (\u200b) so SVG never strips them as trailing whitespace
                    trace_name = f"ID: {str(rid)}\u00A0\u00A0\u00A0\u00A0\u200b"
                    cur_color = sig_colors.get(rid, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])

                    # Scope
                    df_scope = load_waveform(rid)
                    if df_scope is not None and not df_scope.empty:
                        if len(df_scope) > 5000:
                            df_scope = df_scope.iloc[::max(1, len(df_scope)//5000), :]
                        t_col = df_scope.columns[0]
                        v_col = df_scope.columns[1]
                        fig_scope.add_trace(go.Scatter(
                            x=df_scope[t_col], 
                            y=df_scope[v_col], 
                            mode='lines', 
                            name=trace_name, 
                            line=dict(color=cur_color, width=line_width),
                            opacity=0.88,
                            showlegend=True
                        ))
                    
                    # Spectrum
                    df_spec = load_spectrum(rid)
                    if df_spec is not None and not df_spec.empty:
                        f_col = df_spec.columns[0]
                        m_col = df_spec.columns[1]
                        df_spec = df_spec[df_spec[f_col] >= 0]
                        if len(df_spec) > 5000:
                            df_spec = df_spec.iloc[::max(1, len(df_spec)//5000), :]
                        
                        if sig_freq_unit == "kHz":
                            x_freq = df_spec[f_col] * 1000.0 if df_spec[f_col].max() <= 10.0 else df_spec[f_col]
                            mask_freq = (x_freq >= 0) & (x_freq <= 1000.0)
                        else:
                            x_freq = df_spec[f_col] if df_spec[f_col].max() <= 10.0 else df_spec[f_col] / 1000.0
                            mask_freq = (x_freq >= 0) & (x_freq <= 1.0)

                        fig_spec.add_trace(go.Scatter(
                            x=x_freq[mask_freq], 
                            y=df_spec[m_col][mask_freq], 
                            mode='lines', 
                            name=trace_name, 
                            line=dict(color=cur_color, width=line_width),
                            opacity=0.88,
                            showlegend=True
                        ))

                plot_config = {
                    'toImageButtonOptions': {
                        'format': 'png',
                        'width': None,
                        'height': None,
                        'scale': 2
                    }
                }

                # Layout configuration with inside top-right larger legend and larger English labels
                fig_scope.update_layout(
                    title=dict(text="Time Domain Waveform", font=dict(size=24, family="Arial, sans-serif", color=font_color)),
                    template=theme_template,
                    height=chart_height,
                    showlegend=True,
                    margin=dict(l=75, r=45, t=55, b=60),
                    xaxis=dict(
                        title=dict(text="Time (ms)", font=dict(size=24, family="Arial, sans-serif", color=font_color)),
                        tickfont=dict(size=24, color=font_color),
                        showgrid=True,
                        gridcolor=grid_color,
                        zeroline=True,
                        zerolinecolor=zero_color
                    ),
                    yaxis=dict(
                        title=dict(text="Amplitude (V)", font=dict(size=24, family="Arial, sans-serif", color=font_color)),
                        tickfont=dict(size=24, color=font_color),
                        showgrid=True,
                        gridcolor=grid_color,
                        zeroline=True,
                        zerolinecolor=zero_color
                    ),
                    legend=dict(
                        yanchor="top",
                        y=0.97,
                        xanchor="right",
                        x=0.96,
                        bgcolor=leg_bg,
                        bordercolor=leg_border,
                        borderwidth=1.5,
                        font=dict(size=24, family="Arial, sans-serif", color=font_color)
                    )
                )

                freq_label = f"Frequency ({sig_freq_unit})"
                fig_spec.update_layout(
                    title=dict(text="Frequency Domain Spectrum (FFT)", font=dict(size=24, family="Arial, sans-serif", color=font_color)),
                    template=theme_template,
                    height=chart_height,
                    showlegend=True,
                    margin=dict(l=75, r=45, t=55, b=60),
                    xaxis=dict(
                        title=dict(text=freq_label, font=dict(size=24, family="Arial, sans-serif", color=font_color)),
                        tickfont=dict(size=24, color=font_color),
                        range=[0, 1000] if sig_freq_unit == "kHz" else [0, 1.0],
                        showgrid=True,
                        gridcolor=grid_color,
                        zeroline=True,
                        zerolinecolor=zero_color
                    ),
                    yaxis=dict(
                        title=dict(text="Magnitude (dBu)", font=dict(size=24, family="Arial, sans-serif", color=font_color)),
                        tickfont=dict(size=24, color=font_color),
                        showgrid=True,
                        gridcolor=grid_color,
                        zeroline=True,
                        zerolinecolor=zero_color
                    ),
                    legend=dict(
                        yanchor="top",
                        y=0.97,
                        xanchor="right",
                        x=0.96,
                        bgcolor=leg_bg,
                        bordercolor=leg_border,
                        borderwidth=1.5,
                        font=dict(size=24, family="Arial, sans-serif", color=font_color)
                    )
                )
                
                if sig_layout == "Stacked (Full Width / Large)":
                    st.plotly_chart(fig_scope, use_container_width=True, config=plot_config)
                    st.plotly_chart(fig_spec, use_container_width=True, config=plot_config)
                else:
                    c_scope, c_spec = st.columns(2)
                    with c_scope:
                        st.plotly_chart(fig_scope, use_container_width=True, config=plot_config)
                    with c_spec:
                        st.plotly_chart(fig_spec, use_container_width=True, config=plot_config)
            else:
                st.info("Select signals above to compare.")
        else:
            st.error("No recording IDs available.")
    # --- TAB TEST SET EVAL ---
    with tab_test_eval:
        st.header("✅ Model Evaluation")
        st.write("Train models and evaluate them on data they have never seen. Plant model: Leave-One-Plant-Out (every plant is the unseen test set once). Noise/Stress models: 20% hold-out.")
        
        tab_noise, tab_plant, tab_stress = st.tabs(["🚨 Noise Model", "🌱 Plant Model", "🍂 Stress Model"])
        
        with tab_plant:
            st.subheader("🌱 Plant Model")
            st.caption("Classifies specific plant species based on cavitation events.")
            if st.button("🚀 Train Plant Model", type="primary"):
                with st.spinner("Training plant model..."):
                    try:
                        import subprocess
                        subprocess.run(["python", str(PROJECT_ROOT / "model_training" / "train_and_save_plant_model.py")], check=True)
                        st.success("Plant model trained successfully!")
                    except Exception as e:
                        st.error(f"Training failed: {e}")
            
            plant_meta_path = MODELS_DIR / "plant_model_metadata.json"
            if plant_meta_path.exists():
                with open(plant_meta_path, "r") as f:
                    p_meta = json.load(f)
                
                render_model_insights(p_meta, "Blues", key_prefix="plant")
            else:
                st.info("No plant model metadata found. Please train the model.")
        
        with tab_stress:
            st.subheader("🍂 Stress Model")
            st.caption("Classifies physiological stress states (e.g. Drought) from cavitation events.")
            if st.button("🚀 Train Stress Model", type="primary"):
                with st.spinner("Training stress model..."):
                    try:
                        import subprocess
                        subprocess.run(["python", str(PROJECT_ROOT / "model_training" / "train_and_save_stress_model.py")], check=True)
                        st.success("Stress model trained successfully!")
                    except Exception as e:
                        st.error(f"Training failed: {e}")
            
            stress_meta_path = MODELS_DIR / "stress_model_metadata.json"
            if stress_meta_path.exists():
                with open(stress_meta_path, "r") as f:
                    s_meta = json.load(f)
                
                render_model_insights(s_meta, "Greens", key_prefix="stress")
            else:
                st.info("No stress model metadata found. Please train the model.")
        
        with tab_noise:
            st.subheader("🚨 Noise Model")
            st.caption("Aggressively filters noise from genuine signals, prioritizing F2 score.")
            if st.button("🚀 Train Noise Model", type="primary"):
                with st.spinner("Training noise model (evaluation)..."):
                    try:
                        import subprocess
                        subprocess.run(["python", str(PROJECT_ROOT / "model_training" / "train_and_save_noise_model.py")], check=True)
                        st.success("Noise model trained successfully!")
                    except Exception as e:
                        st.error(f"Training failed: {e}")
            
            noise_meta_path = MODELS_DIR / "noise_model_metadata.json"
            if noise_meta_path.exists():
                with open(noise_meta_path, "r") as f:
                    n_meta = json.load(f)
                
                render_model_insights(n_meta, "Reds", key_prefix="noise")
            else:
                st.info("No noise model metadata found. Please train the model.")

    # --- TAB SEMI-SUPERVISED ---
    with tab_semi_supervised:
        st.header("🗺️ Interactive Clustering Discovery")
        st.write("Discover hidden patterns using dimensionality reduction (UMAP).")
        
        c_tab_all, c_tab_plant, c_tab_stress = st.tabs(["🌐 Global (All Data)", "🌱 Plants (Cavitation Only)", "🍂 Stress (Cavitation Only)"])
        
        with c_tab_all:
            st.subheader("Global Discovery")
            st.caption("Good for finding new noise clusters or generic patterns.")
            render_clustering_ui('all', df_filtered, CACHE_PATH, RESULTS_DIR, DB_PATH, PROJECT_ROOT)
            
        with c_tab_plant:
            st.subheader("Plant Species Discovery")
            st.caption("UMAP projection calculated ONLY on Cavitation events. Highly sensitive to species differences.")
            render_clustering_ui('plants', df_filtered, CACHE_PATH, RESULTS_DIR, DB_PATH, PROJECT_ROOT)
            
        with c_tab_stress:
            st.subheader("Stress State Discovery")
            st.caption("UMAP projection calculated ONLY on Cavitation events. Highly sensitive to physiological stress states.")
            render_clustering_ui('stress', df_filtered, CACHE_PATH, RESULTS_DIR, DB_PATH, PROJECT_ROOT)

if __name__ == "__main__":
    main()
