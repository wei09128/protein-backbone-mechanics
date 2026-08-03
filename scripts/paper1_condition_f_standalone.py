"""
paper1_condition_f_standalone.py
===================================
Full-scale (not 20-structure-sample) rerun of Condition F -- does NOT
require tau_phi_lj/tau_psi_lj, so it can run on features_v3_fixed.csv
right now, without waiting for add_lj_torques.py's background job.

Two things tested, both structure-grouped CV (GroupKFold by pdb_id):
  1. Regression: do the corrected virtual-rotation-only features predict
     continuous psi (circular R^2, RF)?
  2. Classification: do the same features predict basin identity, beating
     majority-class baseline? (Direct comparison point: the ORIGINAL
     broken-probe diagnostic scored EXACTLY at baseline, 0.427/0.427,
     zero information.)

Usage:
  python paper1_condition_f_standalone.py --csv features_v3_fixed.csv
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

try:
    import cupy as cp
    from cuml.ensemble import RandomForestRegressor as cuRF, RandomForestClassifier as cuRFC
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    from sklearn.ensemble import RandomForestRegressor as skRF, RandomForestClassifier as skRFC

VIRTUAL_ROTATION_FEATURES = [
    "steric_clash_phi_plus", "steric_clash_phi_minus",
    "steric_clash_psi_plus", "steric_clash_psi_minus",
]
MINDIST_FEATURES = ["steric_mindist_phi", "steric_mindist_psi"]


def circular_r2(y_true_deg, sin_pred, cos_pred):
    pred_deg = np.degrees(np.arctan2(sin_pred, cos_pred))
    resid = (y_true_deg - pred_deg + 180) % 360 - 180
    ss_res = np.sum(resid ** 2)
    mean_true = np.degrees(np.arctan2(
        np.mean(np.sin(np.radians(y_true_deg))),
        np.mean(np.cos(np.radians(y_true_deg)))
    ))
    resid_null = (y_true_deg - mean_true + 180) % 360 - 180
    ss_tot = np.sum(resid_null ** 2)
    return 1 - ss_res / ss_tot


def assign_basin(phi, psi):
    if -180 <= phi < 0 and -100 <= psi < 45:
        if -100 <= phi < -30 and -70 <= psi < -15:
            return 'alphaR'
        return 'other_neg_phi'
    if -180 <= phi < -30 and (psi >= 45 or psi < -170):
        return 'beta'
    if -100 <= phi < -30 and 45 <= psi < 180:
        return 'PPII'
    if 0 <= phi < 180 and -20 <= psi < 100:
        return 'alphaL'
    return 'other'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--n_splits', type=int, default=5)
    args = ap.parse_args()

    print(f"GPU backend: {'ENABLED' if GPU_AVAILABLE else 'NOT FOUND -- CPU sklearn'}")
    print("Loading CSV...")
    df = pd.read_csv(args.csv, low_memory=False)

    feature_cols = [c for c in VIRTUAL_ROTATION_FEATURES if c in df.columns]
    mindist_cols = [c for c in MINDIST_FEATURES if c in df.columns]
    all_feats = feature_cols + mindist_cols
    print(f"  Using features: {all_feats}")
    if not feature_cols:
        raise ValueError("No steric_clash_* columns found -- wrong CSV?")

    needed = all_feats + ['phi_deg', 'psi_deg', 'pdb_id']
    df = df.dropna(subset=needed).reset_index(drop=True)
    print(f"  {len(df):,} residues, {df['pdb_id'].nunique():,} structures")

    X = df[all_feats].values.astype(np.float32)
    y_sin = np.sin(np.radians(df['psi_deg'].values)).astype(np.float32)
    y_cos = np.cos(np.radians(df['psi_deg'].values)).astype(np.float32)
    psi_true = df['psi_deg'].values
    groups = df['pdb_id'].values

    splitter = GroupKFold(n_splits=args.n_splits)
    splits = list(splitter.split(X, y_sin, groups=groups))

    print(f"\n{'='*60}")
    print("  REGRESSION: virtual-rotation features -> continuous psi")
    print(f"{'='*60}")
    fold_r2 = []
    for train_idx, test_idx in splits:
        if GPU_AVAILABLE:
            Xtr, Xte = cp.asarray(X[train_idx]), cp.asarray(X[test_idx])
            ysin_tr, ycos_tr = cp.asarray(y_sin[train_idx]), cp.asarray(y_cos[train_idx])
            rf_sin = cuRF(n_estimators=200, max_depth=12, min_samples_leaf=5, n_bins=128, random_state=0)
            rf_cos = cuRF(n_estimators=200, max_depth=12, min_samples_leaf=5, n_bins=128, random_state=0)
        else:
            Xtr, Xte = X[train_idx], X[test_idx]
            ysin_tr, ycos_tr = y_sin[train_idx], y_cos[train_idx]
            rf_sin = skRF(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=0, n_jobs=-1)
            rf_cos = skRF(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=0, n_jobs=-1)
        rf_sin.fit(Xtr, ysin_tr)
        rf_cos.fit(Xtr, ycos_tr)
        sin_pred = rf_sin.predict(Xte)
        cos_pred = rf_cos.predict(Xte)
        if GPU_AVAILABLE:
            sin_pred = cp.asnumpy(sin_pred); cos_pred = cp.asnumpy(cos_pred)
        fold_r2.append(circular_r2(psi_true[test_idx], sin_pred, cos_pred))
    print(f"  circular CV R^2 (psi): mean={np.mean(fold_r2):.4f}  per-fold={[round(r,4) for r in fold_r2]}")

    print(f"\n{'='*60}")
    print("  CLASSIFICATION: virtual-rotation features -> basin identity")
    print("  (compare directly: ORIGINAL broken probe scored EXACTLY at")
    print("   baseline, 0.427/0.427 -- zero information)")
    print(f"{'='*60}")
    df['basin'] = [assign_basin(p, s) for p, s in zip(df['phi_deg'], df['psi_deg'])]
    print(df['basin'].value_counts())
    baseline_acc = df['basin'].value_counts(normalize=True).max()
    
    # --- ADD THIS: Encode labels to integers ---
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(df['basin'].values)
    # -------------------------------------------
    fold_acc = []
    for train_idx, test_idx in splits:
        if GPU_AVAILABLE:
            Xtr, Xte = cp.asarray(X[train_idx]), cp.asarray(X[test_idx])
            # Use encoded y
            ytr = y_encoded[train_idx]
            clf = cuRFC(n_estimators=100, max_depth=8, random_state=0)
        else:
            Xtr, Xte = X[train_idx], X[test_idx]
            # Use original strings (sklearn handles this fine)
            ytr = df['basin'].values[train_idx]
            clf = skRFC(n_estimators=100, max_depth=8, n_jobs=-1, random_state=0)
            
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        
        if GPU_AVAILABLE:
            # Map back to strings if you want to compare with strings, 
            # or compare using integers. 
            # Easiest: convert predictions to string labels to match y_basin
            pred_decoded = le.inverse_transform(cp.asnumpy(pred))
            fold_acc.append(np.mean(pred_decoded == df['basin'].values[test_idx]))
        else:
            fold_acc.append(np.mean(pred == df['basin'].values[test_idx]))

    print(f"\n  Baseline (majority class): {baseline_acc:.3f}")
    print(f"  5-fold structure-grouped CV accuracy: {np.mean(fold_acc):.3f} +/- {np.std(fold_acc):.3f}")
    print(f"  Delta vs baseline: {np.mean(fold_acc)-baseline_acc:+.3f}")

    print(f"""
{'='*60}
SUMMARY FOR MAJOR COMMENT 1 RESPONSE:
  n = {len(df):,} residues, {df['pdb_id'].nunique():,} structures (full dataset,
  not a 20-structure sample), structure-grouped CV throughout.
  Regression R^2 (psi): {np.mean(fold_r2):.4f}
  Classification accuracy: {np.mean(fold_acc):.3f} (baseline {baseline_acc:.3f},
  delta {np.mean(fold_acc)-baseline_acc:+.3f})
{'='*60}
""")


if __name__ == '__main__':
    main()
