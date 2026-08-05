#!/usr/bin/env python3
"""
paper3_07_holdout_validation.py

Structure-level held-out predictive validation of the (phi,psi) coupling
surface. Answers: does the full tensor-product coupling model actually
predict backbone geometry better on UNSEEN structures than (a) the grand
mean or (b) the additive phi/psi model? This is the generalization test
that complements the in-sample ΔR²/eta^2 results in Table 1 -- it is the
"killer analysis" recommended to strengthen the novelty case beyond the
ACS Omega review.

Model correspondence to Methods 4.2:
    Model 0 (baseline):  y = mean(y)                     -- grand mean
    Model A (additive):  y = s(phi, 15) + s(psi, 15)      -- same as Table 1
    Model B (full):      y = te(phi, psi, 15x15)          -- same as Table 1

Split is by STRUCTURE (pdb_id), not by residue, to avoid leakage between
train and test (residues from the same structure share crystal-packing /
refinement idiosyncrasies).

Usage (mirrors paper3_06_gam_coupling.py conventions):
    python paper3_07_holdout_validation.py \
        --csv features_lj_v3_FINAL_CLEAN.csv --out ./figures/paper3/ \
        --n_splits 20 --test_frac 0.2 --n_sub 50000

Output:
    holdout_rmse_by_observable.csv   -- per-split, per-observable RMSE for all 3 models
    holdout_rmse_summary.csv         -- mean/CI of RMSE and % reduction, per observable
    holdout_rmse_reduction.png       -- bar chart: % RMSE reduction (full vs additive)
                                          matching the paper's figure-first direction
"""

import argparse
import numpy as np
import pandas as pd
from pygam import LinearGAM, te, s

# ------------------------------------------------------------------
# Observable -> column name mapping.
# EDIT THESE if your CSV uses different column names -- the script
# will print available columns and exit cleanly if a name is missing,
# rather than failing deep inside a fit.
# ------------------------------------------------------------------
OBSERVABLE_COLUMNS = {
    "tau":        "tau_deg",        # N-Ca-C angle
    "N_Ca_Cb":    "angle_N_CA_CB",  # sidechain-anchored angle
    "C_Ca_Cb":    "angle_C_CA_CB",
    "omega":      "omega_deg",      # peptide planarity (idealized/computed; use omega_measured_deg instead if that's what Table 1 used)
    "bond_N_Ca":  "bond_N_CA",
    "bond_Ca_C":  "bond_CA_C",
}

PHI_COL = "phi_deg"
PSI_COL = "psi_deg"
STRUCTURE_COL = "pdb_id"   # column identifying which crystal structure a residue came from


def resolve_columns(df, requested):
    """Fail fast and clearly if expected columns are missing."""
    missing = [c for c in requested if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Missing expected column(s): {missing}\n"
            f"Available columns in CSV:\n  " + "\n  ".join(sorted(df.columns))
        )


def fit_and_predict(train, test, phi, psi, y_col, n_splines=15):
    """Fit baseline / additive / full models on train, return test predictions.
    Caller is responsible for re-centering circular observables (e.g. omega)
    before calling this -- see circular_shift() and its use in main()."""
    Xtr = train[[phi, psi]].values
    ytr = train[y_col].values
    Xte = test[[phi, psi]].values

    # Model 0: grand mean
    pred_mean = np.full(len(test), ytr.mean())

    # Model A: additive s(phi) + s(psi)
    gam_add = LinearGAM(s(0, n_splines=n_splines) + s(1, n_splines=n_splines)).fit(Xtr, ytr)
    pred_add = gam_add.predict(Xte)

    # Model B: full tensor-product te(phi, psi)
    gam_full = LinearGAM(te(0, 1, n_splines=n_splines)).fit(Xtr, ytr)
    pred_full = gam_full.predict(Xte)

    return pred_mean, pred_add, pred_full


def circular_shift(y, center=180.0):
    """Re-center a circular quantity so values near `center` (e.g. trans omega
    at 180 deg) sit near 0 instead of near the +/-180 wrap boundary. Without
    this, both grand-mean AND GAM fitting are corrupted: a GAM minimizes
    ordinary squared error during training, so residues at +179 and -179
    (same physical geometry) get averaged toward 0 instead of toward 180."""
    return ((np.asarray(y, dtype=float) - center + 180) % 360) - 180


def rmse(y_true, y_pred, circular=False):
    if circular:
        # safety net for any residual wraparound after re-centering (e.g. cis
        # outliers landing near the new +/-180 boundary)
        diff = (np.asarray(y_true) - np.asarray(y_pred) + 180) % 360 - 180
    else:
        diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(diff ** 2)))


CIRCULAR_OBSERVABLES = {"omega"}  # near +/-180 deg, can wrap; tau/N_Ca_Cb/C_Ca_Cb sit near ~110 deg, no wrap risk
CIRCULAR_CENTER = 180.0  # trans omega; if your data reports omega centered elsewhere, adjust


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="Path to features_lj_v3_FINAL_CLEAN.csv")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--n_splits", type=int, default=20,
                     help="Number of random structure-level train/test splits (default 20, matching existing bootstrap convention)")
    ap.add_argument("--test_frac", type=float, default=0.2, help="Fraction of STRUCTURES held out per split")
    ap.add_argument("--n_sub", type=int, default=50000, help="Max residues subsampled from train for GAM fitting speed")
    ap.add_argument("--n_splines", type=int, default=15, help="Spline knots, matching Methods 4.2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.csv)
    resolve_columns(df, [PHI_COL, PSI_COL, STRUCTURE_COL] + list(OBSERVABLE_COLUMNS.values()))

    structures = np.asarray(df[STRUCTURE_COL].unique(), dtype=object)
    n_test_structures = max(1, int(len(structures) * args.test_frac))

    records = []
    for split_i in range(args.n_splits):
        rng.shuffle(structures)
        test_structs = set(structures[:n_test_structures])
        train_mask = ~df[STRUCTURE_COL].isin(test_structs)
        test_mask = df[STRUCTURE_COL].isin(test_structs)

        train_full = df.loc[train_mask]
        test = df.loc[test_mask]

        # subsample train for fitting speed, same spirit as --n_sub elsewhere in the pipeline
        if len(train_full) > args.n_sub:
            train = train_full.sample(n=args.n_sub, random_state=split_i)
        else:
            train = train_full

        for obs_name, col in OBSERVABLE_COLUMNS.items():
            sub_train = train.dropna(subset=[PHI_COL, PSI_COL, col])
            sub_test = test.dropna(subset=[PHI_COL, PSI_COL, col])
            if len(sub_train) < 200 or len(sub_test) < 50:
                continue  # not enough data this split/observable, skip rather than fail

            is_circular = obs_name in CIRCULAR_OBSERVABLES
            if is_circular:
                # re-center BEFORE fitting -- both the grand mean and the GAM
                # training objective need this, not just the RMSE evaluation
                sub_train = sub_train.copy()
                sub_test = sub_test.copy()
                sub_train[col] = circular_shift(sub_train[col].values, CIRCULAR_CENTER)
                sub_test[col] = circular_shift(sub_test[col].values, CIRCULAR_CENTER)

            pred_mean, pred_add, pred_full = fit_and_predict(
                sub_train, sub_test, PHI_COL, PSI_COL, col, n_splines=args.n_splines
            )
            y_true = sub_test[col].values

            records.append({
                "split": split_i,
                "observable": obs_name,
                "n_test_residues": len(sub_test),
                "n_test_structures": n_test_structures,
                "rmse_mean_baseline": rmse(y_true, pred_mean, circular=is_circular),
                "rmse_additive": rmse(y_true, pred_add, circular=is_circular),
                "rmse_full_coupling": rmse(y_true, pred_full, circular=is_circular),
            })

    results = pd.DataFrame.from_records(records)
    if results.empty:
        raise SystemExit("No results produced -- check column names / data sizes.")

    results["pct_reduction_full_vs_additive"] = (
        100 * (results["rmse_additive"] - results["rmse_full_coupling"]) / results["rmse_additive"]
    )
    results["pct_reduction_full_vs_mean"] = (
        100 * (results["rmse_mean_baseline"] - results["rmse_full_coupling"]) / results["rmse_mean_baseline"]
    )

    import os
    os.makedirs(args.out, exist_ok=True)
    results.to_csv(os.path.join(args.out, "holdout_rmse_by_observable.csv"), index=False)

    summary = results.groupby("observable").agg(
        rmse_mean_baseline_mean=("rmse_mean_baseline", "mean"),
        rmse_additive_mean=("rmse_additive", "mean"),
        rmse_full_coupling_mean=("rmse_full_coupling", "mean"),
        pct_reduction_full_vs_additive_mean=("pct_reduction_full_vs_additive", "mean"),
        pct_reduction_full_vs_additive_ci_lo=("pct_reduction_full_vs_additive", lambda x: np.percentile(x, 2.5)),
        pct_reduction_full_vs_additive_ci_hi=("pct_reduction_full_vs_additive", lambda x: np.percentile(x, 97.5)),
        pct_reduction_full_vs_mean_mean=("pct_reduction_full_vs_mean", "mean"),
    ).reset_index()
    summary.to_csv(os.path.join(args.out, "holdout_rmse_summary.csv"), index=False)

    print(summary.to_string(index=False))

    # Figure: % RMSE reduction (full coupling vs additive), matching figure-first direction
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        order = summary.sort_values("pct_reduction_full_vs_additive_mean", ascending=True)
        ax.barh(order["observable"], order["pct_reduction_full_vs_additive_mean"],
                xerr=[order["pct_reduction_full_vs_additive_mean"] - order["pct_reduction_full_vs_additive_ci_lo"],
                      order["pct_reduction_full_vs_additive_ci_hi"] - order["pct_reduction_full_vs_additive_mean"]],
                capsize=3)
        ax.set_xlabel("Held-out RMSE reduction, full coupling vs. additive (%)")
        ax.set_title("Out-of-sample generalization of the coupling surface\n(structure-level holdout, n=%d splits)" % args.n_splits)
        ax.axvline(0, color="black", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "holdout_rmse_reduction.png"), dpi=200)
        print(f"\nFigure written: {os.path.join(args.out, 'holdout_rmse_reduction.png')}")
    except ImportError:
        print("matplotlib not available -- skipped figure, CSVs still written.")

    print(f"\nWrote: {os.path.join(args.out, 'holdout_rmse_by_observable.csv')}")
    print(f"Wrote: {os.path.join(args.out, 'holdout_rmse_summary.csv')}")


if __name__ == "__main__":
    main()
