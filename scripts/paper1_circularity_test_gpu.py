"""
paper1_circularity_test_gpu.py
================================
Extends paper1_phi_psi_ablation_gpu.py with two new conditions addressing
ACS Omega Reviewer Major Comment #1 (circularity), not just #2/#8.

Condition E (LJ swap): replace tau_phi_steric/tau_psi_steric (v5 1/r^2
Cbeta-Calpha proxy) with tau_phi_lj/tau_psi_lj (real pairwise 12-6
Lennard-Jones torque, computed this session via add_lj_torques.py).
Both are computed from the TRUE observed coordinates, so this tests
whether a more physically faithful redescription of the same real
geometry predicts psi at least as well -- consistent with either a real
mechanical-transmission story OR pure circularity; it does not
distinguish the two, but it is a legitimate check of whether the paper's
steric-torque formula is defensible as physics per se.

Condition F (virtual-rotation only): uses ONLY the features that are
already computed on a *virtually rotated, non-observed* geometry per
Methods 4.3 (steric_clash_phi_plus/minus, steric_clash_psi_plus/minus --
the "rotational clash" feature the original reviewer explicitly flagged
as the one legitimate off-equilibrium probe in the whole feature set).
If R^2 collapses toward the residue-level-CV floor from these features
ALONE, that is direct evidence the rest of the paper's predictive power
comes from features that are strict functions of the observed
conformation (supporting the circularity concern, #1). If R^2 stays
substantial, that is genuine evidence of predictive signal from a
non-circular probe.

Usage:
    python paper1_circularity_test_gpu.py --csv features_lj_FINAL_CLEAN.csv
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

try:
    import cupy as cp
    from cuml.ensemble import RandomForestRegressor as cuRF
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    from sklearn.ensemble import RandomForestRegressor as skRF

# --------------------------------------------------------------------------
# Feature groups
# --------------------------------------------------------------------------

STERIC_FEATURES = [  # Group A
    "steric_N_3A", "steric_N_4A", "steric_N_5A",
    "steric_CA_3A", "steric_CA_4A", "steric_CA_5A",
    "steric_C_3A", "steric_C_4A", "steric_C_5A",
    "steric_O_3A", "steric_O_4A", "steric_O_5A",
    "steric_asym_x", "steric_asym_y", "steric_asym_z",
    "steric_clash_phi_plus", "steric_clash_phi_minus",
    "steric_clash_psi_plus", "steric_clash_psi_minus",
    "sc_contact_nm1_to_bb", "sc_contact_np1_to_bb",
]

# The ONLY features computed on a virtually rotated (non-observed) geometry.
VIRTUAL_ROTATION_FEATURES = [
    "steric_clash_phi_plus", "steric_clash_phi_minus",
    "steric_clash_psi_plus", "steric_clash_psi_minus",
]

TORQUE_FEATURES_PROXY = [  # Group B, original v5 proxy
    "tau_phi_correct", "tau_psi_correct",
    "tau_phi_bb_donor", "tau_psi_bb_donor",
    "tau_phi_bb_acc", "tau_psi_bb_acc",
    "tau_phi_sc_hb", "tau_psi_sc_hb",
    "tau_phi_steric", "tau_psi_steric",
    "tau_phi_elec_corr", "tau_psi_elec_corr",
    "chi1_rad", "has_chi1",
]

TORQUE_FEATURES_LJ = [  # Group B, with real LJ swapped in for the proxy
    "tau_phi_correct", "tau_psi_correct",
    "tau_phi_bb_donor", "tau_psi_bb_donor",
    "tau_phi_bb_acc", "tau_psi_bb_acc",
    "tau_phi_sc_hb", "tau_psi_sc_hb",
    "tau_phi_lj", "tau_psi_lj",
    "tau_phi_elec_corr", "tau_psi_elec_corr",
    "chi1_rad", "has_chi1",
]

CONTEXT_FEATURES = [  # Group C
    "chi2_rad", "has_chi2",
    "sc_mass", "sc_n_heavy", "sc_n_rotatable", "sc_rigidity",
    "sc_is_branched", "sc_is_aromatic", "sc_lever_arm",
    "bfactor_ca", "is_pro_np1",
    "angle_NCaC", "angle_CaCN", "angle_CNCa",
    "dist_ca_m2", "dist_ca_p2",
    "sc_mass_nm1", "sc_mass_np1",
    "hb_n_strong", "hb_best_e",
]

FULL_PROXY = STERIC_FEATURES + TORQUE_FEATURES_PROXY + CONTEXT_FEATURES
FULL_LJ = STERIC_FEATURES + TORQUE_FEATURES_LJ + CONTEXT_FEATURES


def load_data(csv_path, needed_cols):
    df = pd.read_csv(csv_path, low_memory=False)
    missing = [c for c in needed_cols + ["phi_deg", "psi_deg", "pdb_id"] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")
    df = df.dropna(subset=needed_cols + ["phi_deg", "psi_deg"]).reset_index(drop=True)
    return df


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


def _make_rf(seed):
    if GPU_AVAILABLE:
        return cuRF(n_estimators=200, max_depth=12, min_samples_leaf=5,
                    n_bins=128, random_state=seed)
    else:
        return skRF(n_estimators=200, max_depth=12, min_samples_leaf=5,
                    random_state=seed, n_jobs=-1)


def _to_device(arr):
    if GPU_AVAILABLE:
        return cp.asarray(arr, dtype=cp.float32)
    return arr


def _to_host(arr):
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def run_variant(df, feature_cols, group_col, label, n_splits=5, track_feature=None):
    X_np = df[feature_cols].values.astype(np.float32)
    y_sin_np = np.sin(np.radians(df["psi_deg"].values)).astype(np.float32)
    y_cos_np = np.cos(np.radians(df["psi_deg"].values)).astype(np.float32)
    psi_true = df["psi_deg"].values

    if group_col is not None:
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(X_np, y_sin_np, groups=df[group_col].values))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        splits = list(splitter.split(X_np))

    fold_r2 = []
    importances = []
    for train_idx, test_idx in splits:
        X_train = _to_device(X_np[train_idx])
        X_test = _to_device(X_np[test_idx])
        y_sin_train = _to_device(y_sin_np[train_idx])
        y_cos_train = _to_device(y_cos_np[train_idx])

        rf_sin = _make_rf(seed=0)
        rf_cos = _make_rf(seed=0)
        rf_sin.fit(X_train, y_sin_train)
        rf_cos.fit(X_train, y_cos_train)

        sin_pred = _to_host(rf_sin.predict(X_test))
        cos_pred = _to_host(rf_cos.predict(X_test))
        fold_r2.append(circular_r2(psi_true[test_idx], sin_pred, cos_pred))

        imp_sin = _to_host(rf_sin.feature_importances_)
        imp_cos = _to_host(rf_cos.feature_importances_)
        importances.append((imp_sin + imp_cos) / 2)

    mean_importance = np.mean(importances, axis=0)
    imp_series = pd.Series(mean_importance, index=feature_cols).sort_values(ascending=False)

    backend = "GPU (cuML)" if GPU_AVAILABLE else "CPU (sklearn fallback)"
    print(f"\n=== {label} ===")
    print(f"backend={backend}  n_features={len(feature_cols)}  CV folds={n_splits}  "
          f"grouped_by={group_col if group_col else 'none (residue-level, INFLATED)'}")
    print(f"circular CV R^2 (psi): mean={np.mean(fold_r2):.4f}  "
          f"per-fold={[round(r,4) for r in fold_r2]}")
    print("Top 10 feature importances:")
    print(imp_series.head(10).to_string())

    result = {"r2": np.mean(fold_r2)}
    if track_feature and track_feature in imp_series.index:
        rank = list(imp_series.index).index(track_feature) + 1
        val = imp_series[track_feature]
        print(f"{track_feature}: importance={val:.4f}  rank={rank}/{len(feature_cols)}")
        result[f"{track_feature}_importance"] = val
        result[f"{track_feature}_rank"] = rank
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n_splits", type=int, default=5)
    args = ap.parse_args()

    print(f"GPU backend: {'ENABLED (cuML/cupy found)' if GPU_AVAILABLE else 'NOT FOUND -- falling back to CPU sklearn'}")

    all_needed = list(set(FULL_PROXY + FULL_LJ + VIRTUAL_ROTATION_FEATURES))
    df = load_data(args.csv, all_needed)
    print(f"Loaded {len(df):,} residues from {df['pdb_id'].nunique():,} structures.")

    results = {}

    # ── Condition E: LJ swap (structure-level CV only -- the honest one) ──
    print("\n\n########## CONDITION E: real Lennard-Jones swapped for v5 proxy ##########")
    results['proxy_structcv'] = run_variant(
        df, FULL_PROXY, 'pdb_id',
        "Baseline (tau_phi_steric proxy) -- structure-level CV",
        args.n_splits, track_feature='tau_phi_steric')
    results['lj_structcv'] = run_variant(
        df, FULL_LJ, 'pdb_id',
        "LJ swap (tau_phi_lj) -- structure-level CV",
        args.n_splits, track_feature='tau_phi_lj')

    # ── Condition F: virtual-rotation features ONLY ──
    print("\n\n########## CONDITION F: virtual-rotation features ONLY (Methods 4.3 clash probe) ##########")
    results['virtual_only_structcv'] = run_variant(
        df, VIRTUAL_ROTATION_FEATURES, 'pdb_id',
        "ONLY steric_clash_phi/psi_plus/minus (genuinely off-equilibrium) -- structure-level CV",
        args.n_splits)

    # For reference: full feature set minus tau_phi_steric/tau_psi_steric,
    # i.e. "if we drop the suspect feature entirely, what's left?"
    full_minus_suspect = [c for c in FULL_PROXY if c not in ('tau_phi_steric', 'tau_psi_steric')]
    print("\n\n########## CONDITION G: full feature set MINUS tau_phi_steric/tau_psi_steric ##########")
    results['minus_suspect_structcv'] = run_variant(
        df, full_minus_suspect, 'pdb_id',
        "All features except tau_phi_steric/tau_psi_steric -- structure-level CV",
        args.n_splits)

    print("\n\n================ SUMMARY ================")
    print(f"{'condition':55s} {'R2_psi':>8s}")
    print(f"{'Baseline (v5 proxy, full features)':55s} {results['proxy_structcv']['r2']:8.4f}")
    print(f"{'LJ swap (full features)':55s} {results['lj_structcv']['r2']:8.4f}")
    print(f"{'Virtual-rotation clash features ONLY':55s} {results['virtual_only_structcv']['r2']:8.4f}")
    print(f"{'Full minus tau_phi/psi_steric':55s} {results['minus_suspect_structcv']['r2']:8.4f}")

    print(f"""
Interpretation guide:

CONDITION E (LJ vs proxy): both tau_phi_steric and tau_phi_lj are computed
from the TRUE observed coordinates, so this does NOT distinguish real
mechanical transmission from circularity -- it only tests whether a more
physically faithful force law (real 12-6 LJ) is at least as predictive as
the ad hoc v5 proxy. If R^2({results['lj_structcv']['r2']:.4f}) >= R^2({results['proxy_structcv']['r2']:.4f}),
that supports using real LJ as the more defensible physical quantity in the
paper going forward, but does NOT by itself answer reviewer concern #1.

CONDITION F (the actual test of circularity, #1): steric_clash_phi/psi_plus/minus
are the ONLY features in the entire set computed on a virtually rotated,
non-observed geometry (Methods 4.3). Compare this R^2 against the residue-level
"floor" you'd expect from a genuinely weak/no signal, and against the full-feature
R^2 above:
  - If this R^2 is close to the full-feature R^2, that is strong evidence the
    paper's predictive power reflects real off-equilibrium mechanical signal,
    not circularity -- report this explicitly and prominently in the response
    letter as the direct answer to Major Comment #1.
  - If this R^2 collapses toward a small fraction of the full-feature R^2,
    that confirms most of the paper's predictive power comes from features
    that are strict functions of the observed conformation (supporting the
    circularity concern), and Section 2.3's causal "mechanical transmission"
    language needs to be tempered accordingly (also addresses Major Comment #7).

CONDITION G (diagnostic): shows how much R^2 survives when the single most
suspect feature (tau_phi_steric) is removed entirely, using everything else
(including the virtual-rotation clash features, other torques, and context).
This is a middle ground between the full model and Condition F.
""")


if __name__ == "__main__":
    main()
