import os
import sqlite3
import numpy as np
import pandas as pd
import pywt
import torch
from torch.utils.data import Dataset
from pathlib import Path

# Base Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
DB_PATH = ROOT_DIR / "metadata.db"
CACHE_DIR = ROOT_DIR / "data_collection"
CACHE_DIR.mkdir(exist_ok=True, parents=True)


def load_dataset_metadata(filter_cavitation_only=False):
    """
    Load recording metadata from SQLite database.
    """
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT recording_id, plant, species, plant_id, experiment_id, event_type, stress_type, peak_amplitude
        FROM recordings
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Verify scope CSV file existence
    df['file_exists'] = df['recording_id'].apply(lambda rid: (DATA_DIR / f"{rid}-measurement-scope.csv").exists())
    df = df[df['file_exists']].copy()

    if filter_cavitation_only:
        df = df[df['event_type'] == 'Cavitation'].copy()

    return df


def extract_waveform(recording_id, decimate_factor=4):
    """
    Load raw scope voltage waveform, decimate by 4 (12500 -> 3125 points at 12.5 MHz),
    and apply peak normalization.
    """
    csv_path = DATA_DIR / f"{recording_id}-measurement-scope.csv"
    if not csv_path.exists():
        return None

    try:
        data = pd.read_csv(csv_path, usecols=[1]).values.flatten()
        if len(data) == 0:
            return None
        v = data[::decimate_factor].astype(np.float32)
        peak = np.max(np.abs(v))
        if peak > 1e-9:
            v_norm = v / peak
        else:
            v_norm = v
        return v_norm
    except Exception:
        return None


def compute_cwt_scalogram(v, n_scales=32, target_time_steps=128):
    """
    Compute 2D continuous wavelet transform (Morlet) scalogram.
    Returns 2D matrix of shape (n_scales, target_time_steps).
    """
    scales = np.geomspace(2, 64, num=n_scales)
    coefs, _ = pywt.cwt(v, scales, 'morl')
    power = np.abs(coefs) ** 2

    # Downsample time dimension
    step = max(1, power.shape[1] // target_time_steps)
    power_down = power[:, ::step][:, :target_time_steps]

    p_max = np.max(power_down)
    if p_max > 1e-9:
        power_down = power_down / p_max

    return power_down.astype(np.float32)


def build_or_load_dataset(task="noise", force_rebuild=False, max_noise_samples=2500):
    """
    Build or load dataset tensors for training.
    task: "noise" (Cavitation vs Noise), "plant" (Cavitation across species), or "stress" (Drought vs Cut).
    """
    cache_tag = f"{task}_max{max_noise_samples}" if task == "noise" else task
    cache_path = CACHE_DIR / f"dl_cache_{cache_tag}.npz"

    if cache_path.exists() and not force_rebuild:
        print(f"Loading cached dataset from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        return {
            'waveforms': data['waveforms'],
            'scalograms': data['scalograms'],
            'labels': data['labels'],
            'recording_ids': data['recording_ids'],
            'experiment_ids': data['experiment_ids'],
            'plant_ids': data['plant_ids'],
            'plants': data['plants'],
            'class_names': data['class_names'].tolist()
        }

    print(f"Building dataset for task '{task}'...")
    df_meta = load_dataset_metadata(filter_cavitation_only=(task in ["plant", "stress"]))

    if task == "noise":
        df_cav = df_meta[df_meta['event_type'] == 'Cavitation'].copy()
        df_noise = df_meta[df_meta['event_type'].isin(['Noise', 'Control', 'Kontrolle'])].copy()

        if len(df_noise) > max_noise_samples:
            df_noise = df_noise.sample(n=max_noise_samples, random_state=42)

        df_target = pd.concat([df_cav, df_noise]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        df_target['label_str'] = np.where(df_target['event_type'] == 'Cavitation', 'Cavitation', 'Noise')
        class_names = ['Noise', 'Cavitation']
        df_target['label_idx'] = (df_target['label_str'] == 'Cavitation').astype(int)

    elif task == "plant":
        valid_plants = ['Mini Pepper', 'Sunflower', 'Tomato']
        df_target = df_meta[df_meta['plant'].isin(valid_plants)].copy()
        df_target = df_target.dropna(subset=['plant_id']).reset_index(drop=True)
        class_names = sorted(valid_plants)
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        df_target['label_idx'] = df_target['plant'].map(class_to_idx)

    elif task == "stress":
        valid_stresses = ['cut', 'drought']
        df_target = df_meta[df_meta['stress_type'].isin(valid_stresses)].copy()
        df_target = df_target.dropna(subset=['plant_id']).reset_index(drop=True)
        class_names = sorted(valid_stresses)
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        df_target['label_idx'] = df_target['stress_type'].map(class_to_idx)
    else:
        raise ValueError(f"Unknown task: {task}")

    print(f"Total candidate samples: {len(df_target)}")

    waveforms = []
    scalograms = []
    valid_indices = []

    for idx, row in df_target.iterrows():
        rid = row['recording_id']
        wf = extract_waveform(rid)
        if wf is not None and len(wf) == 3125:
            sc = compute_cwt_scalogram(wf)
            if sc.shape == (32, 128):
                waveforms.append(wf)
                scalograms.append(sc)
                valid_indices.append(idx)

        if (idx + 1) % 500 == 0 or (idx + 1) == len(df_target):
            print(f"  Processed {idx + 1}/{len(df_target)} files...")

    df_valid = df_target.iloc[valid_indices].reset_index(drop=True)
    waveforms = np.array(waveforms, dtype=np.float32)
    scalograms = np.array(scalograms, dtype=np.float32)

    # Add channel dimension: (N, 1, 3125) and (N, 1, 32, 128)
    waveforms = np.expand_dims(waveforms, axis=1)
    scalograms = np.expand_dims(scalograms, axis=1)

    labels = df_valid['label_idx'].values.astype(np.int64)
    recording_ids = df_valid['recording_id'].values
    experiment_ids = df_valid['experiment_id'].fillna('unknown').values
    plant_ids = df_valid['plant_id'].fillna('unknown').values
    plants = df_valid['plant'].fillna('unknown').values

    print(f"Saving preprocessed tensors ({len(df_valid)} items) to {cache_path}...")
    np.savez_compressed(
        cache_path,
        waveforms=waveforms,
        scalograms=scalograms,
        labels=labels,
        recording_ids=recording_ids,
        experiment_ids=experiment_ids,
        plant_ids=plant_ids,
        plants=plants,
        class_names=np.array(class_names)
    )

    return {
        'waveforms': waveforms,
        'scalograms': scalograms,
        'labels': labels,
        'recording_ids': recording_ids,
        'experiment_ids': experiment_ids,
        'plant_ids': plant_ids,
        'plants': plants,
        'class_names': class_names
    }


class AEDataset(Dataset):
    """
    PyTorch Dataset yielding either 1D waveforms, 2D CWT scalograms, or both.
    """
    def __init__(self, waveforms, scalograms, labels, mode="1d"):
        self.waveforms = torch.from_numpy(waveforms)
        self.scalograms = torch.from_numpy(scalograms)
        self.labels = torch.from_numpy(labels)
        self.mode = mode

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        if self.mode == "1d":
            return self.waveforms[idx], label
        elif self.mode == "2d":
            return self.scalograms[idx], label
        elif self.mode == "both":
            return self.waveforms[idx], self.scalograms[idx], label
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
