"""
paper1_table1_preprocess.py — Run ONCE, caches preprocessed data for the
parallel model runners below.
================================================================================
Loading + scaling the 1.7M-row feature matrix is itself real work; doing it
independently in 4 separate terminal processes wastes time and risks the 4
runs seeing slightly different data if anything about the pipeline changes
between them. Run this once, then point all 4 parallel model jobs at the
cached .npz file.

Usage:
  python paper1_table1_preprocess.py --csv features_lj_v3_FINAL_CLEAN.csv
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

REF_PHI, REF_PSI = -63.0, -43.0


def wrap(angle):
    return ((angle + 180.0) % 360.0) - 180.0


# CORRECTED feature list: EXCLUDES tau_phi_lj/tau_psi_lj.
# Rationale: Paper 1's entire Section 2.3 narrative (ablation numbers,
# +0.163/+0.142 figures, "tau_phi_steric remains #1 ranked feature") is
# anchored specifically to tau_phi_steric. Including tau_phi_lj alongside
# it creates redundant/collinear features representing the same physical
# quantity (steric torque) computed two different ways, diluting feature
# importance between them for no interpretive benefit. The LJ-vs-proxy
# comparison has its own dedicated home (Condition E in the circularity
# ablation) -- it doesn't need to also live permanently in the main model.
FULL_FEATURES = [
    'steric_N_3A', 'steric_N_4A', 'steric_N_5A',
    'steric_CA_3A', 'steric_CA_4A', 'steric_CA_5A',
    'steric_C_3A', 'steric_C_4A', 'steric_C_5A',
    'steric_O_3A', 'steric_O_4A', 'steric_O_5A',
    'steric_asym_x', 'steric_asym_y', 'steric_asym_z', 'improper_ca',
    'steric_clash_phi_plus', 'steric_clash_phi_minus',
    'steric_clash_psi_plus', 'steric_clash_psi_minus',
    'steric_mindist_phi', 'steric_mindist_psi',
    'sc_contact_nm1_to_bb', 'sc_contact_np1_to_bb',
    'tau_phi_correct', 'tau_psi_correct', 'tau_phi_bb_donor', 'tau_psi_bb_donor',
    'tau_phi_bb_acc', 'tau_psi_bb_acc', 'tau_phi_sc_hb', 'tau_psi_sc_hb',
    'tau_phi_steric', 'tau_psi_steric', 'tau_phi_elec_corr', 'tau_psi_elec_corr',
    'chi1_rad', 'has_chi1',
    'chi2_rad', 'has_chi2', 'sc_mass', 'sc_n_heavy',
    'sc_n_rotatable', 'sc_rigidity', 'sc_is_branched', 'sc_is_aromatic',
    'sc_lever_arm', 'hb_n_strong', 'hb_best_e', 'bfactor_ca', 'is_pro_np1',
    'angle_NCaC', 'angle_CaCN', 'angle_CNCa', 'dist_ca_m2', 'dist_ca_p2',
    'sc_mass_nm1', 'sc_mass_np1'
]
TORQUE_FEATURES = ["tau_phi_correct", "tau_psi_correct"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='table1_cache.npz')
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    print(f"Loaded {len(df):,} residues")
    df = df.dropna(subset=["phi_deg", "psi_deg"])

    phi = df["phi_deg"].values
    psi = df["psi_deg"].values
    y_phi = np.array([wrap(x - REF_PHI) for x in phi])
    y_psi = np.array([wrap(x - REF_PSI) for x in psi])
    y_sin_phi = np.sin(np.radians(phi)); y_cos_phi = np.cos(np.radians(phi))
    y_sin_psi = np.sin(np.radians(psi)); y_cos_psi = np.cos(np.radians(psi))

    if "chain" in df.columns:
        groups_raw = (df["pdb_id"].astype(str) + "_" + df["chain"].astype(str)).values
    else:
        groups_raw = df["pdb_id"].astype(str).values
    # Convert to integer codes -- GroupKFold only needs equal values to mean
    # "same group," and integer arrays avoid the object-dtype pickle issue
    # that string arrays trigger in np.savez/np.load.
    _, groups = np.unique(groups_raw, return_inverse=True)
    print(f"Unique groups: {len(np.unique(groups)):,}")

    missing = [f for f in FULL_FEATURES if f not in df.columns]
    if missing:
        print(f"  WARNING: missing features: {missing}")
    feats = [f for f in FULL_FEATURES if f in df.columns]
    print(f"Using {len(feats)} full-model features (tau_phi_lj/tau_psi_lj excluded)")

    X_full = df[feats].fillna(0).values
    X_tau = df[TORQUE_FEATURES].fillna(0).values

    X_full_sc = StandardScaler().fit_transform(X_full)
    X_tau_sc = StandardScaler().fit_transform(X_tau)

    cv = GroupKFold(n_splits=5)
    splits = list(cv.split(X_full_sc, y_phi, groups=groups))
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        intersection = set(df["pdb_id"].values[train_idx]) & set(df["pdb_id"].values[test_idx])
        assert not intersection, f"FOLD {fold_idx}: leaking across train/test!"
    print("Group-leakage assertion passed for all folds.")

    np.savez(args.out,
             X_full_sc=X_full_sc, X_tau_sc=X_tau_sc,
             y_phi=y_phi, y_psi=y_psi,
             y_sin_phi=y_sin_phi, y_cos_phi=y_cos_phi,
             y_sin_psi=y_sin_psi, y_cos_psi=y_cos_psi,
             phi=phi, psi=psi, groups=groups,
             feat_names=np.array(feats, dtype='<U50'))
    print(f"\nSaved cache to {args.out} -- point all model-runner jobs at this file.")


if __name__ == '__main__':
    main()
