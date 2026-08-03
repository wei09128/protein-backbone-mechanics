"""
paper1_phi_psi_ablation_gpu.py
===============================
GPU version of paper1_phi_psi_ablation.py, addressing ACS Omega Reviewer
Major Comments #2 and #8 on "Protein Backbone Mechanics I".

Same logic as the CPU script (with/without true-phi feature x
residue-level/structure-level CV = 4 conditions), but the RandomForestRegressor
fits run on GPU via RAPIDS cuML, which is where essentially all the wall-clock
time in the original script goes (200 trees x depth 12 x 2 heads x 5 folds x
4 conditions, on ~1.66M rows x ~57 features).

Requires a RAPIDS environment, e.g.:
    conda create -n rapids -c rapidsai -c conda-forge -c nvidia \
        cuml=25.02 cupy python=3.11 cuda-version=12.5
    conda activate rapids

Falls back to sklearn/CPU automatically if cuML/cupy are not importable,
so the script still runs (slowly) on a machine without a GPU -- useful for
sanity-checking output format before submitting to a GPU node.

Usage:
    python paper1_phi_psi_ablation_gpu.py --csv features_lj_FINAL_CLEAN.csv
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

# --------------------------------------------------------------------------
# GPU backend detection
# --------------------------------------------------------------------------
try:
    import cupy as cp
    from cuml.ensemble import RandomForestRegressor as cuRF
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    from sklearn.ensemble import RandomForestRegressor as skRF

# --------------------------------------------------------------------------
# Feature groups -- identical to the CPU script, matched to Methods 4.3-4.5.
# Deliberately EXCLUDES phi_deg/psi_deg (targets) and pdb_id/chain/res_idx
# (identifiers) from all feature sets below.
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

TORQUE_FEATURES = [  # Group B  (includes tau_phi_steric -- the feature under test)
    "tau_phi_correct", "tau_psi_correct",
    "tau_phi_bb_donor", "tau_psi_bb_donor",
    "tau_phi_bb_acc", "tau_psi_bb_acc",
    "tau_phi_sc_hb", "tau_psi_sc_hb",
    "tau_phi_steric", "tau_psi_steric",
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

ALL_FEATURES = STERIC_FEATURES + TORQUE_FEATURES + CONTEXT_FEATURES


def load_data(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    needed = ALL_FEATURES + ["phi_deg", "psi_deg", "pdb_id"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")
    df = df.dropna(subset=ALL_FEATURES + ["phi_deg", "psi_deg"]).reset_index(drop=True)
    return df


def circular_r2(y_true_deg, sin_pred, cos_pred):
    """Reconstruct angle from sin/cos predictions and score against true angle
    using wrapped angular residuals (matches Methods 4.6 circular R^2)."""
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
    """Construct a regressor on GPU (cuML) or CPU (sklearn), same hyperparams
    as the original CPU script (n_estimators=200, max_depth=12,
    min_samples_leaf=5)."""
    if GPU_AVAILABLE:
        # cuML uses a histogram/quantile split algorithm rather than sklearn's
        # exact-count algorithm; n_bins controls the quantile resolution.
        # 128 bins is a safe default that tracks sklearn's exact splits closely
        # for continuous features like these.
        return cuRF(n_estimators=200, max_depth=12, min_samples_leaf=5,
                    n_bins=128, random_state=seed)
    else:
        return skRF(n_estimators=200, max_depth=12, min_samples_leaf=5,
                    random_state=seed, n_jobs=-1)


def _to_device(arr):
    """Move a numpy array to GPU (float32, cuML's preferred dtype) if
    available, else return as-is."""
    if GPU_AVAILABLE:
        return cp.asarray(arr, dtype=cp.float32)
    return arr


def _to_host(arr):
    """Bring predictions/importances back to numpy for scoring/printing."""
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def run_variant(df, feature_cols, group_col, label, n_splits=5):
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

    # ==========================================================
    # Added Sanity Check for Group Leakage
    # ==========================================================
    if group_col is not None:
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            intersection = set(df[group_col].values[train_idx]) & set(df[group_col].values[test_idx])
            assert not intersection, \
                f"FOLD {fold_idx}: {group_col} is leaking across train/test in a supposedly grouped fold!"
    # ==========================================================

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
    tps_rank = list(imp_series.index).index("tau_phi_steric") + 1
    tps_val = imp_series["tau_phi_steric"]
    print(f"tau_phi_steric: importance={tps_val:.4f}  rank={tps_rank}/{len(feature_cols)}")
    return {"r2": np.mean(fold_r2), "tau_phi_steric_importance": tps_val,
            "tau_phi_steric_rank": tps_rank}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n_splits", type=int, default=5)
    args = ap.parse_args()

    print(f"GPU backend: {'ENABLED (cuML/cupy found)' if GPU_AVAILABLE else 'NOT FOUND -- falling back to CPU sklearn'}")

    df = load_data(args.csv)
    print(f"Loaded {len(df):,} residues from {df['pdb_id'].nunique():,} structures.")

    df["sin_phi_true"] = np.sin(np.radians(df["phi_deg"].values))
    df["cos_phi_true"] = np.cos(np.radians(df["phi_deg"].values))
    with_phi_cols = ALL_FEATURES + ["sin_phi_true", "cos_phi_true"]
    without_phi_cols = ALL_FEATURES

    results = {}
    for group_col, cv_label in [(None, "residue-level CV (inflated)"),
                                 ("pdb_id", "structure-level CV (honest)")]:
        results[("with_phi", group_col)] = run_variant(
            df, with_phi_cols, group_col,
            f"WITH true phi as feature -- {cv_label}", args.n_splits)
        results[("without_phi", group_col)] = run_variant(
            df, without_phi_cols, group_col,
            f"WITHOUT phi information -- {cv_label}", args.n_splits)

    print("\n\n================ SUMMARY ================")
    print(f"{'variant':30s} {'cv_split':22s} {'R2_psi':>8s} {'tau_phi_steric_imp':>20s} {'rank':>6s}")
    for (variant, group_col), res in results.items():
        cv_label = "residue-level" if group_col is None else "structure-level"
        print(f"{variant:30s} {cv_label:22s} {res['r2']:8.4f} "
              f"{res['tau_phi_steric_importance']:20.4f} {res['tau_phi_steric_rank']:6d}")

    print("""
Interpretation guide:
- Compare 'with_phi' vs 'without_phi' rows at the SAME cv_split level.
  If tau_phi_steric importance collapses (drops toward the bottom of the
  ranking) once true phi is removed, the reviewer's leakage concern (#2) is
  supported: the cross-axis coupling claim was largely phi geometrically
  constraining the feasible psi range, not tau_phi_steric mechanically
  transmitting strain through Calpha.
  If tau_phi_steric importance stays high (top few features) even WITHOUT
  phi, that is evidence FOR the mechanical-transmission interpretation
  and strengthens the paper -- report this explicitly in the response letter.
- Compare 'residue-level' vs 'structure-level' rows at the SAME with/without
  phi setting. A large R^2 drop under structure-level grouping indicates the
  original CV (#8) was inflated by shared structural context leaking between
  train/test folds; report the honest (grouped) numbers in the revision and
  update every R^2 and 'xN improvement' claim in Table 1 accordingly.
""")


if __name__ == "__main__":
    main()
