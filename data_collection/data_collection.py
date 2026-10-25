import streamlit as st
import pandas as pd
import sqlite3
import os
import glob
import plotly.express as px
from datetime import datetime
import json
import subprocess
import sys
from pathlib import Path
from PIL import Image
from typing import Dict, Any, List, Optional

# Determine project root assuming this script is in project_root/data_collection/
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
LOGO_PATH = PROJECT_ROOT / ".documentation" / "ae_lab_logo.jpg"
logo = Image.open(LOGO_PATH) if LOGO_PATH.exists() else "🌱"

# ==========================================
# Configuration & Paths
# ==========================================
st.set_page_config(page_title="AE-Lab Data Viz", layout="wide", page_icon=logo)

# Determine project root assuming this script is in project_root/data_collection/
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DB_PATH = PROJECT_ROOT / "metadata.db"
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_PATH = Path(__file__).parent / "config.json"
STOP_FLAG_PATH = Path(__file__).parent / "stop_measurement.flag"
MEASURE_SCRIPT = Path(__file__).parent / "picoscope_measure.py"


# ==========================================
# Database Functions
# ==========================================
def init_db() -> None:
    """Initialize the SQLite database with required tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    c = conn.cursor()
    
    # Session metadata table
    c.execute('''
        CREATE TABLE IF NOT EXISTS session_metadata (
            id INTEGER PRIMARY KEY,
            plant TEXT, species TEXT, stem_diameter REAL, size REAL,
            leaf_count INTEGER, leaf_size REAL, sensor_position TEXT,
            contact_pressure TEXT, sensor_type TEXT, gain TEXT, filter TEXT,
            temperature REAL, humidity REAL, soil_moisture REAL, light TEXT,
            stress_type TEXT, stress_type_details TEXT, event_type TEXT,
            started_at TEXT, auto_sensor INTEGER, arduino_ip TEXT,
            experiment_id TEXT, experiment_description TEXT, plant_id TEXT
        )
    ''')
    
    # Ensure at least one row exists for global session state
    c.execute('SELECT COUNT(*) FROM session_metadata')
    if c.fetchone()[0] == 0:
         c.execute('INSERT INTO session_metadata (id) VALUES (1)')
         
    # Recordings table
    c.execute('''
        CREATE TABLE IF NOT EXISTS recordings (
            recording_id TEXT PRIMARY KEY,
            plant TEXT, species TEXT, stem_diameter REAL, size REAL,
            leaf_count INTEGER, leaf_size REAL, sensor_position TEXT,
            contact_pressure TEXT, sensor_type TEXT, gain TEXT, filter TEXT,
            temperature REAL, humidity REAL, soil_moisture REAL, light TEXT,
            stress_type TEXT, stress_type_details TEXT, event_type TEXT, started_at TEXT,
            event_timestamp TEXT, trigger_amplitude REAL, peak_amplitude REAL,
            duration REAL, energy REAL, counts INTEGER, dominant_frequency REAL,
            experiment_id TEXT, experiment_description TEXT, plant_id TEXT
            -- Note: Other columns like noise_probability may be added dynamically by ML scripts
        )
    ''')
    conn.commit()
    conn.close()

def get_session_meta() -> Dict[str, Any]:
    """Retrieve the current session metadata."""
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM session_metadata WHERE id=1')
    row = c.fetchone()
    conn.close()
    return dict(row) if row else {}

def save_session_meta(meta: Dict[str, Any]) -> None:
    """Update the session metadata in the database."""
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    c = conn.cursor()
    cols = [k for k in meta.keys() if k != 'id']
    updates = ', '.join([f"{k}=?" for k in cols])
    vals = [meta[k] for k in cols] + [1]
    
    # Check if we need to add columns dynamically (basic migration support)
    c.execute(f"PRAGMA table_info(session_metadata)")
    existing_cols = {col[1] for col in c.fetchall()}
    for col in cols:
        if col not in existing_cols:
            c.execute(f"ALTER TABLE session_metadata ADD COLUMN {col} TEXT")
            
    c.execute(f'UPDATE session_metadata SET {updates} WHERE id=?', vals)
    conn.commit()
    conn.close()

def get_recording_meta(rec_id: str) -> Optional[Dict[str, Any]]:
    """Fetch metadata for a specific recording ID."""
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM recordings WHERE recording_id=?', (rec_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None

def get_existing_plant_ids() -> List[str]:
    """Get a sorted list of unique plant IDs from past recordings."""
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    c = conn.cursor()
    # Handle possible missing column for backwards compatibility
    try:
        c.execute('SELECT DISTINCT plant_id FROM recordings WHERE plant_id IS NOT NULL AND plant_id != ""')
        rows = c.fetchall()
        plant_ids = sorted([r[0] for r in rows])
    except sqlite3.OperationalError:
        plant_ids = []
    conn.close()
    return plant_ids

def get_latest_plant_details(plant_id: str) -> Dict[str, Any]:
    """Retrieve the most recent details for a specific plant ID."""
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    try:
        c.execute('''
            SELECT plant, species, stem_diameter, size, leaf_count, leaf_size 
            FROM recordings 
            WHERE plant_id=? 
            ORDER BY started_at DESC LIMIT 1
        ''', (plant_id,))
        row = c.fetchone()
    except sqlite3.OperationalError:
        row = None
    conn.close()
    return dict(row) if row else {}


# ==========================================
# UI Components
# ==========================================
def render_sidebar():
    """Renders the sidebar for PicoScope control and returns the configuration."""
    st.sidebar.header("PicoScope Control")
    
    # Load default hardware config
    default_config = {
        "N_SAMPLES": 12500,
        "TIMEBASE": 2,
        "TRIGGER_MV": 20.0,
        "CH_RANGE": 7,
        "PRE_TRIGGER_PCT": 40.0,
        "MIN_FREQUENCY_KHZ": 30.0
    }
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                default_config.update(json.load(f))
        except json.JSONDecodeError:
            st.sidebar.warning("Failed to load config.json. Using defaults.")

    with st.sidebar.expander("Hardware Config", expanded=True):
        n_samples = st.number_input("Samples", value=int(default_config["N_SAMPLES"]))
        timebase = st.number_input("Timebase (index)", value=int(default_config["TIMEBASE"]))
        trigger_mv = st.number_input("Trigger Threshold (mV)", value=float(default_config["TRIGGER_MV"]))
        
        range_options = [3, 4, 5, 6, 7, 8, 9, 10]
        range_labels = {3:'100mV', 4:'200mV', 5:'500mV', 6:'1V', 7:'2V', 8:'5V', 9:'10V', 10:'20V'}
        ch_idx = range_options.index(default_config["CH_RANGE"]) if default_config["CH_RANGE"] in range_options else 4
        ch_range = st.selectbox("Channel Range", range_options, index=ch_idx, format_func=lambda x: range_labels[x])
        pre_trigger = st.slider("Pre-Trigger %", 0, 100, int(default_config["PRE_TRIGGER_PCT"]))
        min_frequency_khz = st.number_input("Min Frequency (kHz)", value=float(default_config.get("MIN_FREQUENCY_KHZ", 30.0)))
    
    config = {
        "N_SAMPLES": n_samples,
        "TIMEBASE": timebase,
        "TRIGGER_MV": trigger_mv,
        "CH_RANGE": ch_range,
        "PRE_TRIGGER_PCT": float(pre_trigger),
        "MIN_FREQUENCY_KHZ": float(min_frequency_khz)
    }
    
    c1, c2 = st.sidebar.columns(2)
    start_btn = c1.button("▶️ Start", use_container_width=True, type="primary")
    stop_btn = c2.button("⏹️ Stop", use_container_width=True)

    # Handle process start/stop
    meta = get_session_meta()
    
    if start_btn:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=4)
        if STOP_FLAG_PATH.exists():
            STOP_FLAG_PATH.unlink()
            
        # Update started_at and experiment_id in session meta
        started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        experiment_id = f"EXP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        updated_meta = meta.copy()
        updated_meta["started_at"] = started_at
        updated_meta["experiment_id"] = experiment_id
        save_session_meta(updated_meta)
            
        # Launch measurement script
        kwargs = {}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE
            
        subprocess.Popen([sys.executable, str(MEASURE_SCRIPT)], **kwargs)
        st.sidebar.success(f"Measurement process started at {started_at}!")

    if stop_btn:
        with open(STOP_FLAG_PATH, "w") as f:
            f.write("stop")
        st.sidebar.info("Stop signal sent. Measurement will abort.")
        
    st.sidebar.markdown(f"**Status:** {'🟥 Stopped' if STOP_FLAG_PATH.exists() else '🟩 Running (or idle)'}")
    
    return config

def render_session_tab():
    """Renders the Session Settings & Context tab."""
    st.header("Global Meta Information")
    st.write("Variables defined here represent your **current experimental setup**. When a measurement triggers, these settings will be *automatically* stored along with the recording data and calculated signal features into the database.")
    
    meta = get_session_meta()
    st.info(f"Experiment ID: **{meta.get('experiment_id', 'Not started')}** \n\nSession started at: **{meta.get('started_at', 'Not started yet')}**")
    
    with st.expander("Plant Auto-Fill"):
        existing_pids = get_existing_plant_ids()
        selected_pid = st.selectbox("Select an existing Plant ID to auto-fill details (or pick 'New Plant')", ["New Plant"] + existing_pids)
        
    plant_defaults = get_latest_plant_details(selected_pid) if selected_pid != "New Plant" else {}
        
    with st.form("session_meta_form"):
        st.subheader("Experiment Information")
        exp_desc = st.text_input("Experiment Description (Why & What)", value=meta.get('experiment_description', ''))
        
        st.subheader("1. Plant Details")
        c1, c2, c3 = st.columns(3)
        plant_id_val = selected_pid if selected_pid != "New Plant" else meta.get('plant_id', '')
        plant_id = c1.text_input("Plant ID (e.g. TOM_01)", value=plant_id_val)
        plant = c2.text_input("Plant", value=plant_defaults.get('plant') or meta.get('plant', ''))
        species = c3.text_input("Species", value=plant_defaults.get('species') or meta.get('species', ''))
        
        c4, c5, c6, c7 = st.columns(4)
        stem_diameter = c4.number_input("Stem Diameter (mm)", value=float(plant_defaults.get('stem_diameter') or meta.get('stem_diameter') or 0.0))
        size = c5.number_input("Size (cm)", value=float(plant_defaults.get('size') or meta.get('size') or 0.0))
        leaf_count = c6.number_input("Leaf Count", value=int(plant_defaults.get('leaf_count') or meta.get('leaf_count') or 0))
        leaf_size = c7.number_input("Avg Leaf Size (cm²)", value=float(plant_defaults.get('leaf_size') or meta.get('leaf_size') or 0.0))
        
        st.subheader("2. Stress & Environment")
        c8, c9 = st.columns(2)
        stress_opts = ["drought", "cut", "freeze", "none", "other"]
        stress_idx = stress_opts.index(meta.get('stress_type', 'none')) if meta.get('stress_type') in stress_opts else 3
        stress_type = c8.selectbox("Stress Type (Water Stress Level)", stress_opts, index=stress_idx)
        stress_type_details = c9.text_input("Stress Type Details", value=meta.get('stress_type_details', ''))
        
        st.subheader("3. Environment")
        e1, e2 = st.columns(2)
        auto_sensor = e1.checkbox("Automatic Sensor Values via Arduino (DHT22 & Soil Moisture)", value=bool(meta.get('auto_sensor', 1)))
        arduino_ip = e2.text_input("Arduino IP Address", value=meta.get('arduino_ip', '192.168.1.100'))
        
        e3, e4, e5, e6 = st.columns(4)
        temperature = e3.number_input("Temperature (°C) [Manual]", value=float(meta.get('temperature') or 20.0), disabled=auto_sensor)
        humidity = e4.number_input("Humidity (%) [Manual]", value=float(meta.get('humidity') or 50.0), disabled=auto_sensor)
        soil_moisture = e5.number_input("Soil Moisture (%) [Manual]", value=float(meta.get('soil_moisture') or 50.0), disabled=auto_sensor)
        light = e6.text_input("Light (e.g. 1000 µmol)", value=meta.get('light', ''))
        
        st.subheader("4. Sensor Details & Events")
        s1, s2, s3, s4 = st.columns(4)
        sensor_type = s1.text_input("Sensor Type", value=meta.get('sensor_type', ''))
        sensor_position = s2.text_input("Sensor Position", value=meta.get('sensor_position', ''))
        contact_pressure = s3.text_input("Contact Pressure", value=meta.get('contact_pressure', ''))
        
        event_opts = ["Noise", "Cavitation", "other"]
        event_idx = event_opts.index(meta.get('event_type', 'Kontrolle')) if meta.get('event_type') in event_opts else 0
        event_type = s4.selectbox("Event Type", event_opts, index=event_idx)

        s5, s6 = st.columns(2)
        gain = s5.text_input("Gain / Amplification", value=meta.get('gain', ''))
        filter_typ = s6.text_input("Filter", value=meta.get('filter', ''))

        if st.form_submit_button("Save Session Context", type="primary"):
            updated = {
                'plant_id': plant_id, 'plant': plant, 'species': species, 'stem_diameter': stem_diameter,
                'size': size, 'leaf_count': leaf_count, 'leaf_size': leaf_size,
                'sensor_position': sensor_position, 'contact_pressure': contact_pressure,
                'sensor_type': sensor_type, 'gain': gain, 'filter': filter_typ,
                'temperature': temperature, 'humidity': humidity,
                'soil_moisture': soil_moisture, 'light': light,
                'stress_type': stress_type, 'stress_type_details': stress_type_details, 'event_type': event_type,
                'auto_sensor': int(auto_sensor), 'arduino_ip': arduino_ip,
                'experiment_description': exp_desc,
                'experiment_id': meta.get('experiment_id', ''),
                'started_at': meta.get('started_at', '') # Keep existing
            }
            save_session_meta(updated)
            st.success("Context saved successfully! These settings will now lock in for all subsequent recordings.")

def render_analysis_tab(config: Dict[str, Any]):
    """Renders the Recordings & Analysis tab."""
    st.button("🔄 Refresh Data List")
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
    scope_files = list(DATA_DIR.glob("*scope.csv"))
    
    if not scope_files:
        st.info(f"No recordings found yet in '{DATA_DIR}'. Run a measurement first!")
        return

    recording_ids = sorted([f.name.replace("-scope.csv", "") for f in scope_files], reverse=True)
    selected_recording = st.selectbox("Select Recording to Analyze", recording_ids)
    
    scope_path = DATA_DIR / f"{selected_recording}-scope.csv"
    spectrum_path = DATA_DIR / f"{selected_recording}-spectrum.csv"
    
    # Features Display
    rec_meta = get_recording_meta(selected_recording)
    if rec_meta:
        st.subheader(f"📊 Signal Features (Event Time: {rec_meta.get('event_timestamp', '')})")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Peak Ampl. (V)", f"{rec_meta.get('peak_amplitude', 0.0):.4f}")
        m2.metric("Dominant Freq. (MHz)", f"{rec_meta.get('dominant_frequency', 0.0):.4f}")
        m3.metric("Duration (s)", f"{rec_meta.get('duration', 0.0):.4e}")
        m4.metric("Energy", f"{rec_meta.get('energy', 0.0):.4e}")
        m5.metric("Counts", f"{rec_meta.get('counts', 0.0)}")
        m6.metric("Trigger (V)", f"{rec_meta.get('trigger_amplitude', 0.0):.4f}")

        # Environmental values
        e1, e2, e3 = st.columns(3)
        if rec_meta.get('temperature'):
            e1.metric("Temperature", f"{rec_meta['temperature']:.1f} °C")
        if rec_meta.get('humidity'):
            e2.metric("Humidity", f"{rec_meta['humidity']:.1f} %")
        if rec_meta.get('soil_moisture'):
            e3.metric("Soil Moisture", f"{rec_meta['soil_moisture']:.1f} %")

        # ML Predictions (if present in DB)
        st.subheader("🤖 ML Predictions (if available)")
        c1, c2, c3, c4 = st.columns(4)
        
        noise_prob = rec_meta.get('noise_probability')
        if noise_prob is not None:
            noise_color = "red" if noise_prob > 0.5 else "green"
            c1.markdown(f"**Noise Likelihood:** <span style='color:{noise_color};'>{noise_prob:.1%}</span>", unsafe_allow_html=True)
        
        plant = rec_meta.get('pred_plant')
        plant_prob = rec_meta.get('pred_plant_prob')
        if plant and plant_prob:
            c2.markdown(f"**Predicted Plant:** {plant} ({plant_prob:.1%})")
        
        stress = rec_meta.get('pred_stress')
        stress_prob = rec_meta.get('pred_stress_prob')
        if stress and stress_prob:
            c3.markdown(f"**Predicted Stress:** {stress} ({stress_prob:.1%})")
            
        leaf_area = rec_meta.get('pred_leaf_area')
        if leaf_area:
            c4.markdown(f"**Transpiration Capacity (Leaf Area):** {leaf_area:.1f} cm²")

    # Visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Time Domain (Scope)")
        if scope_path.exists():
            try:
                df_scope = pd.read_csv(scope_path)
                # Downsample for visualization performance if very large
                if len(df_scope) > 5000:
                    df_scope_display = df_scope.iloc[::max(1, len(df_scope)//5000), :]
                else:
                    df_scope_display = df_scope
                
                fig_scope = px.line(df_scope_display, x="Time (ms)", y="Voltage (V)", template="plotly_dark", color_discrete_sequence=['#00CC96'])
                fig_scope.update_layout(margin=dict(l=0, r=0, t=10, b=0))
                
                trigger_mv = config.get("TRIGGER_MV", 20.0)
                fig_scope.add_hline(y=(trigger_mv/1000.0), line_dash="dash", line_color="rgba(255,255,255,0.3)")
                fig_scope.add_vline(x=0.0, line_dash="dash", line_color="rgba(255,0,0,0.5)")
                
                st.plotly_chart(fig_scope, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading scope data: {e}")
            
    with col2:
        st.write("### Frequency Domain (Spectrum)")
        if spectrum_path.exists():
            try:
                df_spectrum = pd.read_csv(spectrum_path)
                df_spectrum_display = df_spectrum[df_spectrum['Frequency (MHz)'] > 0]
                if len(df_spectrum_display) > 5000:
                    df_spectrum_display = df_spectrum_display.iloc[::max(1, len(df_spectrum_display)//5000), :]
                
                fig_spec = px.line(df_spectrum_display, x="Frequency (MHz)", y="Magnitude (dBu)", template="plotly_dark", color_discrete_sequence=['#EF553B'])
                fig_spec.update_layout(margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_spec, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading spectrum data: {e}")

    if rec_meta:
        with st.expander("Show Static Metadata captured for this recording"):
            meta_dict = {k:v for k,v in rec_meta.items() if k not in [
                'id', 'recording_id', 'event_timestamp', 'trigger_amplitude', 
                'peak_amplitude', 'duration', 'energy', 'counts', 'dominant_frequency'
            ]}
            st.json(meta_dict)

def render_feed_tab():
    """Renders the Live Events Feed tab."""
    st.header("📡 Live Events Feed")
    st.write("View the most recent acoustic events and their ML predictions.")
    
    col_auto, col_refresh = st.columns([1, 4])
    auto_refresh = col_auto.checkbox("🔄 Auto-Refresh (Live)", value=False)
    if col_refresh.button("Manual Refresh"):
        st.rerun()
        
    meta = get_session_meta()
    current_experiment_id = meta.get('experiment_id')
    
    if current_experiment_id:
        # Gracefully handle potentially missing columns in older schemas
        feed_query = """
            SELECT 
                recording_id as Timestamp, 
                event_type as 'Event Type', 
                plant_id as 'Plant ID', 
                ROUND(peak_amplitude, 4) as 'Peak Ampl. (V)', 
                ROUND(dominant_frequency, 2) as 'Dominant Freq. (Hz)'
            FROM recordings 
            WHERE experiment_id = ?
            ORDER BY recording_id DESC 
            LIMIT 50
        """
        try:
            conn = sqlite3.connect(DB_PATH, timeout=20.0)
            df_feed = pd.read_sql_query(feed_query, conn, params=(current_experiment_id,))
            
            # Optionally add ML columns if they exist
            c = conn.cursor()
            c.execute(f"PRAGMA table_info(recordings)")
            cols = {col[1] for col in c.fetchall()}
            
            if 'noise_probability' in cols:
                df_feed['Noise Prob (%)'] = pd.read_sql_query("SELECT ROUND(noise_probability * 100, 1) FROM recordings WHERE experiment_id = ? ORDER BY recording_id DESC LIMIT 50", conn, params=(current_experiment_id,))
            if 'pred_plant' in cols:
                df_feed['Pred. Plant'] = pd.read_sql_query("SELECT pred_plant FROM recordings WHERE experiment_id = ? ORDER BY recording_id DESC LIMIT 50", conn, params=(current_experiment_id,))
            if 'pred_stress' in cols:
                df_feed['Pred. Stress'] = pd.read_sql_query("SELECT pred_stress FROM recordings WHERE experiment_id = ? ORDER BY recording_id DESC LIMIT 50", conn, params=(current_experiment_id,))
            
            conn.close()
        except Exception as e:
            df_feed = pd.DataFrame()
            st.warning(f"Could not fetch events. Have any been recorded yet? ({e})")
    else:
        df_feed = pd.DataFrame()
    
    if df_feed.empty:
        st.info("No events recorded yet.")
    else:
        def highlight_noise(val):
            if pd.isna(val) or val == '': return ''
            try:
                return 'color: red; font-weight: bold' if float(val) > 50.0 else 'color: #00FF00'
            except ValueError:
                return ''
                
        # Handle style formatting if ML columns exist
        if 'Noise Prob (%)' in df_feed.columns:
            styled_df = df_feed.style.applymap(highlight_noise, subset=['Noise Prob (%)'])
            st.dataframe(styled_df, height=600)
        else:
            st.dataframe(df_feed, height=600)
        
    if auto_refresh:
        import time
        time.sleep(3)
        st.rerun()

# ==========================================
# Main Application
# ==========================================
def main():
    col1, col2 = st.columns([1, 8])
    with col1:
        if LOGO_PATH.exists():
            st.image(logo, width=80)
    with col2:
        st.title("AE-Lab Data Collection & Visualization")
    init_db()
    
    config = render_sidebar()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["⚙️ Session Settings & Context", "📈 Recordings & Analysis", "📡 Live Events Feed"])
    
    with tab1:
        render_session_tab()
    with tab2:
        render_analysis_tab(config)
    with tab3:
        render_feed_tab()

if __name__ == "__main__":
    main()
