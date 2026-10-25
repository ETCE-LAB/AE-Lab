import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = ROOT_DIR / "models"
DL_MODEL_DIR = MODEL_DIR / "deep_learning"
RESULTS_DIR = ROOT_DIR / "results" / "deep_learning"

# 1. Noise Data (Table 6.1: 6,648 samples)
noise_data = [
    {"Model": "Classical (RF, Table 6.1)", "Accuracy": 99.6, "F1-Score": 99.6, "F2 (Signal)": 98.6},
    {"Model": "1D-CNN (Raw)", "Accuracy": 99.4, "F1-Score": 99.4, "F2 (Signal)": 99.7},
    {"Model": "2D-CNN (CWT)", "Accuracy": 98.0, "F1-Score": 98.0, "F2 (Signal)": 98.9}
]
df_noise = pd.DataFrame(noise_data)

# 2. Plant Data (Table 6.1: Exact 575 LOPO Test Events across 13 Plants)
plant_data = [
    {"Model": "Classical (XGB, Table 6.1)", "Overall Acc": 15.3, "Balanced Acc": 16.2, "F1-Score": 14.5},
    {"Model": "1D-CNN (Raw)", "Overall Acc": 17.1, "Balanced Acc": 15.7, "F1-Score": 15.1},
    {"Model": "2D-CNN (CWT)", "Overall Acc": 14.7, "Balanced Acc": 14.2, "F1-Score": 12.0}
]
df_plant = pd.DataFrame(plant_data)

# 3. Stress Data (Thesis Section 6.4: 196 Balanced Samples - 98 Drought vs 98 Cut)
stress_data = [
    {"Model": "Classical (SVM, Table 6.1)", "Overall Acc": 54.0, "F1-Score": 66.9},
    {"Model": "1D-CNN (Raw)", "Overall Acc": 55.6, "F1-Score": 55.5},
    {"Model": "2D-CNN (CWT)", "Overall Acc": 49.5, "F1-Score": 49.5}
]
df_stress = pd.DataFrame(stress_data)

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=(
        "<b>Task 1: Noise vs. Cavitation</b><br><sup>6,648 samples (Table 6.1 baseline)</sup>",
        "<b>Task 2: Plant Species (3 Classes)</b><br><sup>Exact 575 LOPO Test Events (13 Plants)</sup>",
        "<b>Task 3: Stress Type (Cut vs Drought)</b><br><sup>196 samples (98 drought vs 98 cut)</sup>"
    )
)

# Colors
c1, c2, c3 = "#2b5c8f", "#10b981", "#f59e0b"

# Subplot 1: Noise
for metric, col in zip(["Accuracy", "F1-Score", "F2 (Signal)"], [c1, c2, c3]):
    fig.add_trace(
        go.Bar(
            name=metric,
            x=df_noise["Model"],
            y=df_noise[metric],
            marker_color=col,
            text=[f"{v:.1f}%" for v in df_noise[metric]],
            textposition="auto"
        ),
        row=1, col=1
    )

# Subplot 2: Plant
for metric, col in zip(["Overall Acc", "Balanced Acc", "F1-Score"], [c1, c2, c3]):
    fig.add_trace(
        go.Bar(
            name=metric,
            x=df_plant["Model"],
            y=df_plant[metric],
            marker_color=col,
            text=[f"{v:.1f}%" for v in df_plant[metric]],
            textposition="auto"
        ),
        row=1, col=2
    )

# Subplot 3: Stress
for metric, col in zip(["Overall Acc", "F1-Score"], [c1, c2]):
    fig.add_trace(
        go.Bar(
            name=metric,
            x=df_stress["Model"],
            y=df_stress[metric],
            marker_color=col,
            text=[f"{v:.1f}%" for v in df_stress[metric]],
            textposition="auto"
        ),
        row=1, col=3
    )

fig.update_layout(
    title=dict(
        text="<b>Acoustic Emission Machine Learning Benchmark: Thesis Baselines vs. Deep Learning</b><br><sup>Audited comparison on exact dataset partitions and evaluation splits defined in the thesis</sup>",
        font=dict(size=18, family="Arial, sans-serif")
    ),
    barmode="group",
    template="plotly_white",
    height=550,
    margin=dict(l=50, r=40, t=100, b=60),
    showlegend=False
)

fig.update_yaxes(title_text="Score (%)", range=[0, 105], row=1, col=1)
fig.update_yaxes(title_text="Score (%)", range=[0, 70], row=1, col=2)
fig.update_yaxes(title_text="Score (%)", range=[0, 85], row=1, col=3)

out_html = RESULTS_DIR / "benchmark_comparison_chart.html"
fig.write_html(out_html)
print(f"Saved interactive comparison chart to {out_html}")
