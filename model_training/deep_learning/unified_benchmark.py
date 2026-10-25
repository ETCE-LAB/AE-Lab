"""
Unified, leakage-audited benchmark: classical ML vs. 1D-CNN vs. 2D-CWT-CNN.

For every task the three model families are evaluated on EXACTLY the same
train/test indices (same recordings, same folds). Nothing is selected on the test
data: the CNNs are trained for a FIXED number of epochs and the last-epoch model is
evaluated (the older train_*_dl.py scripts picked the best epoch on the evaluation
fold, which is optimistic; their numbers are superseded by this script).

Tasks
  noise  : replicates the classical hold-out of train_and_save_noise_model.py
           (80/20 stratified split, rs=42, 5:1 noise down-sampling of the TRAIN part only)
  plant  : replicates the exact Leave-One-Plant-Out folds of train_and_save_plant_model.py
           (same RNG sequence, train cap 150/species, test cap 150/plant, >=5 events)
  stress : grouped 5-fold CV on plant_id (StratifiedGroupKFold) on the balanced
           196-sample (98/98) and 502-sample (251/251) sets, 3 repeated split seeds.
           Classical models and CNNs use identical folds.

Usage: python unified_benchmark.py <noise|plant|stress|all> [--seeds 3]
Results: results/deep_learning/unified_<task>.json
"""
import sys
import json
import time
import sqlite3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, fbeta_score,
                             precision_score, recall_score, confusion_matrix)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(CURRENT_DIR))
from dataset import extract_waveform, compute_cwt_scalogram, CACHE_DIR, DB_PATH  # noqa: E402
from models import Waveform1DCNN, Scalogram2DCNN  # noqa: E402

FEATURE_CACHE = ROOT_DIR / "data_collection" / "feature_cache.csv"
RESULTS_DIR = ROOT_DIR / "results" / "deep_learning"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DL_MODELS = [("1D-CNN (raw waveform)", Waveform1DCNN, "wf"),
             ("2D-CNN (CWT scalogram)", Scalogram2DCNN, "sc")]

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def get_tensors(rids, tag):
    """Waveform (N,1,3125) + scalogram (N,1,32,128) for exactly these recording ids (cached)."""
    rids = np.asarray([str(r) for r in rids])
    path = CACHE_DIR / f"dl_unified_{tag}.npz"
    if path.exists():
        d = np.load(path, allow_pickle=True)
        if len(d['rids']) == len(rids) and (d['rids'] == rids).all():
            return d['wf'], d['sc']
    print(f"  building waveform/CWT tensors for {len(rids)} recordings ({tag}) ...", flush=True)
    wf = np.zeros((len(rids), 1, 3125), np.float32)
    sc = np.zeros((len(rids), 1, 32, 128), np.float32)
    for i, r in enumerate(rids):
        w = extract_waveform(r)
        assert w is not None and len(w) == 3125, f"waveform missing/invalid for {r}"
        wf[i, 0] = w
        sc[i, 0] = compute_cwt_scalogram(w)
        if (i + 1) % 2000 == 0:
            print(f"    {i + 1}/{len(rids)}", flush=True)
    np.savez(path, rids=rids, wf=wf, sc=sc)
    return wf, sc


def fit_predict(model_cls, X, y, tr, te, n_classes, epochs, bs, seed, lr=1e-3):
    """Train for a FIXED number of epochs, return predictions of the last-epoch model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_cls(num_classes=n_classes)
    counts = np.bincount(y[tr], minlength=n_classes)
    w = torch.tensor(len(tr) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y[tr])),
                        batch_size=bs, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(te), 256):
            out.append(torch.argmax(model(torch.from_numpy(X[te[i:i + 256]])), 1).numpy())
    return np.concatenate(out)


def summarize(runs, keys):
    return {k: {"mean": float(np.mean([r[k] for r in runs])),
                "std": float(np.std([r[k] for r in runs])),
                "values": [float(r[k]) for r in runs]} for k in keys}


def feature_names(df):
    cols = [c for c in df.columns if c.startswith('wf_') or c.startswith('spec_')]
    return cols + ['duration', 'counts', 'dominant_frequency']


# ----------------------------------------------------------------------------
# TASK 1: noise vs. cavitation  (replicates the classical hold-out)
# ----------------------------------------------------------------------------

def noise_metrics(y, p):
    return dict(accuracy=accuracy_score(y, p), balanced_accuracy=balanced_accuracy_score(y, p),
                macro_f1=f1_score(y, p, average='macro'), weighted_f1=f1_score(y, p, average='weighted'),
                signal_precision=precision_score(y, p, pos_label=1, zero_division=0),
                signal_recall=recall_score(y, p, pos_label=1, zero_division=0),
                signal_f2=fbeta_score(y, p, beta=2.0, pos_label=1, zero_division=0),
                confusion_matrix=confusion_matrix(y, p, labels=[0, 1]).tolist())


def task_noise(n_seeds, epochs=12):
    print("\n=== TASK 1: NOISE vs CAVITATION (identical classical hold-out) ===")
    df = pd.read_csv(FEATURE_CACHE, low_memory=False)
    conn = sqlite3.connect(DB_PATH)
    lab = pd.read_sql_query("SELECT recording_id, event_type FROM recordings", conn)
    conn.close()
    if 'event_type' in df.columns:
        df = df.drop(columns=['event_type'])
    df = df.merge(lab, on='recording_id', how='left')
    df['is_noise'] = df['event_type'].isin(['Noise', 'Kontrolle', 'Control']).astype(int)
    df = df.dropna(subset=['event_type'])

    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['is_noise'], random_state=42)
    sig = df_train[df_train['is_noise'] == 0]
    noi = df_train[df_train['is_noise'] == 1]
    noi = noi.sample(n=min(len(noi), len(sig) * 5), random_state=42)
    df_tr = pd.concat([sig, noi]).copy()
    df_te = df_test.copy()

    top = ['duration', 'dominant_frequency', 'spec_rel_energy_0.1_0.2', 'wf_crest_factor', 'wf_zcr',
           'wf_impulse_factor', 'wf_margin_factor', 'wf_kurtosis', 'wf_high_amp_counts',
           'wf_build_up_energy_ratio']
    all_f = feature_names(df)
    for c in all_f:
        med = df_tr[c].median()
        df_tr[c] = df_tr[c].fillna(med)
        df_te[c] = df_te[c].fillna(med)
    df_tr = df_tr.dropna(subset=all_f)
    df_te = df_te.dropna(subset=all_f)
    y_tr = (df_tr['is_noise'] == 0).astype(int).values   # 1 = signal (cavitation)
    y_te = (df_te['is_noise'] == 0).astype(int).values
    print(f"  pool={len(df)}  train(balanced 5:1)={len(df_tr)} "
          f"(signal {y_tr.sum()}, noise {(1 - y_tr).sum()})  test={len(df_te)} "
          f"(signal {y_te.sum()}, noise {(1 - y_te).sum()})")
    overlap = len(set(df_tr['experiment_id'].dropna()) & set(df_te['experiment_id'].dropna()))
    print(f"  experiment_ids shared between train and test: {overlap} of {df_te['experiment_id'].nunique()} test experiments")

    res = {"task": "noise",
           "basis": dict(split="stratified 80/20 hold-out (not grouped), random_state=42",
                         pool=int(len(df)), train_total_before_balancing=int(len(df_train)),
                         train_samples=int(len(df_tr)), train_signal=int(y_tr.sum()),
                         train_noise=int((1 - y_tr).sum()), test_samples=int(len(df_te)),
                         test_signal=int(y_te.sum()), test_noise=int((1 - y_te).sum()),
                         experiments_shared_train_test=int(overlap),
                         test_experiments=int(df_te['experiment_id'].nunique()),
                         epochs=epochs, seeds=n_seeds),
           "models": {}}

    sc = StandardScaler().fit(df_tr[top].values)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=2,
                                class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(sc.transform(df_tr[top].values), y_tr)
    res["models"]["Classical RF (10 features)"] = {"single_run": noise_metrics(y_te, rf.predict(sc.transform(df_te[top].values)))}
    print("  RF:", {k: round(v, 4) for k, v in res["models"]["Classical RF (10 features)"]["single_run"].items() if k != 'confusion_matrix'})

    rids = np.concatenate([df_tr['recording_id'].values, df_te['recording_id'].values])
    wf, scg = get_tensors(rids, "noise_holdout")
    tr = np.arange(len(df_tr))
    te = np.arange(len(df_tr), len(rids))
    y_all = np.concatenate([y_tr, y_te]).astype(np.int64)
    for name, cls, kind in DL_MODELS:
        X = wf if kind == "wf" else scg
        runs = []
        for s in range(n_seeds):
            t0 = time.time()
            p = fit_predict(cls, X, y_all, tr, te, 2, epochs, 64, s)
            m = noise_metrics(y_te, p)
            runs.append(m)
            print(f"  {name} seed {s}: acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
                  f"sigRecall={m['signal_recall']:.4f} F2={m['signal_f2']:.4f} cm={m['confusion_matrix']} ({time.time() - t0:.0f}s)", flush=True)
        res["models"][name] = {"runs": runs, "summary": summarize(runs, ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'signal_recall', 'signal_precision', 'signal_f2'])}
    (RESULTS_DIR / "unified_noise.json").write_text(json.dumps(res, indent=2))


# ----------------------------------------------------------------------------
# TASK 2: plant species (replicates the exact LOPO folds of the classical script)
# ----------------------------------------------------------------------------

def task_plant(n_seeds, epochs=10):
    print("\n=== TASK 2: PLANT SPECIES (identical LOPO folds) ===")
    df = pd.read_csv(FEATURE_CACHE, low_memory=False)
    df_clean = df[df['event_type'] == 'Cavitation'].copy()
    dist = df_clean['plant'].value_counts()
    valid = dist[dist >= 10].index.tolist()
    dfp = df_clean[df_clean['plant'].isin(valid)].copy()
    all_f = feature_names(dfp)
    for c in all_f:
        if c in dfp.columns:
            dfp[c] = dfp[c].fillna(dfp[c].median())
    dfp = dfp.dropna(subset=all_f + ['plant'])
    plant_ids = np.array([str(p).replace('.0', '') for p in dfp['plant_id'].fillna(dfp['experiment_id']).values])
    classes = sorted(dfp['plant'].unique())
    y = np.array([classes.index(p) for p in dfp['plant'].values])
    n_classes = len(classes)

    # ---- exact replication of the classical LOPO loop (same RNG consumption order)
    rng = np.random.RandomState(42)
    counts = pd.Series(plant_ids).value_counts()
    eval_plants = sorted([p for p in counts.index if counts[p] >= 5], key=str)
    folds = []
    for held in eval_plants:
        te_all = np.where(plant_ids == held)[0]
        tr_all = np.where(plant_ids != held)[0]
        tr_idx = []
        for c in np.unique(y[tr_all]):
            c_idx = tr_all[y[tr_all] == c]
            if len(c_idx) > 150:
                c_idx = rng.choice(c_idx, 150, replace=False)
            tr_idx.extend(c_idx.tolist())
        tr_idx = np.array(tr_idx)
        if len(np.unique(y[tr_idx])) < n_classes:
            continue
        te_idx = te_all
        if len(te_idx) > 150:
            te_idx = rng.choice(te_idx, 150, replace=False)
        folds.append((held, tr_idx, te_idx))

    meta = json.load(open(ROOT_DIR / "models" / "plant_model_metadata.json"))
    mine = {f[0]: len(f[2]) for f in folds}
    theirs = {p['plant_id']: p['n_events'] for p in meta['per_plant']}
    pooled_y = np.concatenate([y[f[2]] for f in folds])
    sup = np.bincount(pooled_y, minlength=n_classes).tolist()
    print(f"  folds={len(folds)}  test events={len(pooled_y)}  per-class test support={sup}  train/fold={[len(f[1]) for f in folds][:3]}...")
    print(f"  per-plant test counts identical to classical metadata: {mine == theirs}")
    print(f"  per-class support identical to classical metadata: {sup == [meta['per_class'][c]['support'] for c in classes]}")
    assert mine == theirs, "fold replication failed"

    wf, scg = get_tensors(dfp['recording_id'].values, "plant_lopo")
    y64 = y.astype(np.int64)
    maj = float(np.bincount(pooled_y).max() / len(pooled_y))
    res = {"task": "plant",
           "basis": dict(protocol="Leave-One-Plant-Out, exact classical folds", eval_plants=len(folds),
                         total_events_in_pool=int(len(dfp)),
                         train_samples_per_fold=sorted(set(len(f[1]) for f in folds)),
                         test_events=int(len(pooled_y)), test_support=dict(zip(classes, sup)),
                         majority_baseline=maj, epochs=epochs, seeds=n_seeds,
                         matches_classical_metadata=True),
           "models": {"Classical XGBoost (65 features + UMAP)": {"single_run": dict(
               accuracy=meta['test_accuracy'], balanced_accuracy=meta['test_balanced_accuracy'],
               macro_f1=meta['test_f1'], mean_per_plant_accuracy=meta['mean_per_plant_accuracy'],
               source="models/plant_model_metadata.json (original LOPO run)")}}}
    for name, cls, kind in DL_MODELS:
        X = wf if kind == "wf" else scg
        runs = []
        for s in range(n_seeds):
            yt, yp, pp = [], [], []
            for held, tr_idx, te_idx in folds:
                p = fit_predict(cls, X, y64, tr_idx, te_idx, n_classes, epochs, 32, s)
                yt.extend(y[te_idx]); yp.extend(p)
                pp.append(float((p == y[te_idx]).mean()))
            yt, yp = np.array(yt), np.array(yp)
            m = dict(accuracy=accuracy_score(yt, yp), balanced_accuracy=balanced_accuracy_score(yt, yp),
                     macro_f1=f1_score(yt, yp, average='macro', zero_division=0),
                     mean_per_plant_accuracy=float(np.mean(pp)), n_test=int(len(yt)),
                     confusion_matrix=confusion_matrix(yt, yp, labels=list(range(n_classes))).tolist())
            runs.append(m)
            print(f"  {name} seed {s}: n={m['n_test']} acc={m['accuracy']:.4f} balAcc={m['balanced_accuracy']:.4f} "
                  f"macroF1={m['macro_f1']:.4f} meanPlantAcc={m['mean_per_plant_accuracy']:.4f}", flush=True)
        res["models"][name] = {"runs": runs, "summary": summarize(runs, ['accuracy', 'balanced_accuracy', 'macro_f1', 'mean_per_plant_accuracy'])}
    (RESULTS_DIR / "unified_plant.json").write_text(json.dumps(res, indent=2))


# ----------------------------------------------------------------------------
# TASK 3: stress type (grouped CV, identical folds for all models)
# ----------------------------------------------------------------------------

def build_stress_sets():
    df_feat = pd.read_csv(FEATURE_CACHE, low_memory=False)
    conn = sqlite3.connect(DB_PATH)
    meta = pd.read_sql_query("""SELECT recording_id, plant_id, experiment_id, stress_type FROM recordings
                                WHERE event_type='Cavitation' AND stress_type IN ('cut','drought')""", conn)
    conn.close()
    m = pd.merge(meta, df_feat.drop(columns=[c for c in ['plant_id', 'experiment_id', 'stress_type'] if c in df_feat.columns]),
                 on='recording_id')
    m = m.dropna(subset=['plant_id']).reset_index(drop=True)
    drought = m[m.stress_type == 'drought']
    cut = m[m.stress_type == 'cut']
    # --- set A: 98 / 98
    sa_d = drought.sample(n=98, random_state=42)
    parts = [g.sample(n=min(len(g), 25), random_state=42) for _, g in cut.groupby('plant_id')]
    cand = pd.concat(parts)
    sa_c = cand.sample(n=98, random_state=42) if len(cand) > 98 else pd.concat(
        [cand, cut.loc[~cut.index.isin(cand.index)].sample(n=98 - len(cand), random_state=42)])
    A = pd.concat([sa_c, sa_d]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    # --- set B: 251 / 251
    parts = []
    for pid, g in cut.groupby('plant_id'):
        parts.append(g.sample(n=min(len(g), 100), random_state=42) if str(pid).replace('.0', '') in ('40', '42') else g)
    cand = pd.concat(parts)
    sb_c = cand.sample(n=251, random_state=42) if len(cand) > 251 else pd.concat(
        [cand, cut.loc[~cut.index.isin(cand.index)].sample(n=251 - len(cand), random_state=42)])
    B = pd.concat([sb_c, drought]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return {"196 (98 drought / 98 cut)": A, "502 (251 drought / 251 cut)": B}, feature_names(m)


def stress_metrics(y, p):
    return dict(accuracy=accuracy_score(y, p), balanced_accuracy=balanced_accuracy_score(y, p),
                macro_f1=f1_score(y, p, average='macro', zero_division=0),
                predicted_drought_share=float(np.mean(p == 1)),
                confusion_matrix=confusion_matrix(y, p, labels=[0, 1]).tolist())


def task_stress(n_repeats, epochs=12):
    print("\n=== TASK 3: STRESS TYPE (identical grouped 5-fold CV) ===")
    sets, feats = build_stress_sets()
    res = {"task": "stress", "sets": {}}
    for sname, d in sets.items():
        y = (d['stress_type'] == 'drought').astype(np.int64).values
        groups = d['plant_id'].astype(str).str.replace('.0', '', regex=False).values
        n_pl = len(np.unique(groups))
        Xc = d[feats].copy()
        for c in ['duration', 'counts', 'spec_high_freq_energy_ratio']:
            Xc[c] = np.log1p(Xc[c].clip(lower=0))
        Xc = Xc.values
        print(f"\n  -- set {sname}: n={len(d)}, plants={n_pl}, classical features={len(feats)}")
        wf, scg = get_tensors(d['recording_id'].values, "stress_" + sname.split()[0])
        classical = {
            "Classical SVM-RBF": lambda: make_pipeline(SimpleImputer(strategy='median'), RobustScaler(),
                                                       SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')),
            "Classical RandomForest": lambda: make_pipeline(SimpleImputer(strategy='median'),
                                                            RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=2,
                                                                                   class_weight='balanced', random_state=42, n_jobs=-1)),
        }
        all_models = list(classical) + [m[0] for m in DL_MODELS]
        runs = {k: [] for k in all_models}
        single_class_folds = []
        for r in range(n_repeats):
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42 + r)
            folds = list(sgkf.split(Xc, y, groups=groups))
            single_class_folds.append(int(sum(len(np.unique(y[v])) < 2 for _, v in folds)))
            for k, mk in classical.items():
                P = np.zeros(len(y), int)
                for tr, va in folds:
                    mdl = mk(); mdl.fit(Xc[tr], y[tr]); P[va] = mdl.predict(Xc[va])
                runs[k].append(stress_metrics(y, P))
            for name, cls, kind in DL_MODELS:
                X = wf if kind == "wf" else scg
                P = np.zeros(len(y), int)
                for tr, va in folds:
                    P[va] = fit_predict(cls, X, y, tr, va, 2, epochs, 16 if len(d) < 300 else 32, r)
                runs[name].append(stress_metrics(y, P))
            print(f"    repeat {r} (split seed {42 + r}): " + " | ".join(
                f"{k.split(' (')[0]}: acc={runs[k][-1]['accuracy']:.3f} F1={runs[k][-1]['macro_f1']:.3f}" for k in all_models), flush=True)
        res["sets"][sname] = {
            "n": int(len(d)), "plants": int(n_pl), "classical_features": len(feats), "epochs": epochs,
            "folds_with_single_class_per_repeat": single_class_folds,
            "models": {k: {"runs": v, "summary": summarize(v, ['accuracy', 'balanced_accuracy', 'macro_f1', 'predicted_drought_share'])}
                       for k, v in runs.items()}}
    (RESULTS_DIR / "unified_stress.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["noise", "plant", "stress", "all"])
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    if a.task in ("plant", "all"):
        task_plant(a.seeds)
    if a.task in ("stress", "all"):
        task_stress(a.seeds)
    if a.task in ("noise", "all"):
        task_noise(a.seeds)
