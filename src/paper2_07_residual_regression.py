"""
paper2_07_residual_regression.py — How much of backbone angle variance
remains after accounting for local (backbone + sidechain) factors?
========================================================================

Move C. Paper 2 has established:
   (B) Universal Ramachandran response: τ, ω, ∠N-Cα-Cβ, ∠C-Cα-Cβ all
       deform systematically with (φ, ψ).
   (A) Sidechain lever: β-branching amplifies, χ1 rotamer modulates.

Move C asks: is that the whole story, or does a third channel exist?

Procedure:
  1. For each of the four angles, fit a Gradient Boosting regressor
     predicting angle from LOCAL features only:
         phi_deg, psi_deg, res_type, chi1_rad, chi2_rad, sc_mass,
         sc_is_branched, sc_is_aromatic
  2. Compute residuals = observed − predicted. These are what the
     local model CANNOT explain.
  3. Test whether residuals correlate with NON-LOCAL features:
         steric_CA_5A, steric_N_5A, steric_asym_{x,y,z},
         hb_n_bonds, hb_n_strong, hb_best_e, hb_mean_d_HO,
         sc_contact_nm1_to_bb, sc_contact_np1_to_bb, bfactor_ca
  4. Report: R² of local model (how much is local-explained),
     Pearson r and effect size of residuals vs each non-local feature.

Interpretation:
  - High R² (>0.5) + weak residual correlations → local story is complete
  - Moderate R² + strong residual correlations → non-local channel exists,
    we've identified which feature(s) carry it
  - Low R² + chaotic residuals → either local model is mis-specified
    or the non-local channel dominates

Uses a random sample (default 100k residues) for fitting to keep runtime
reasonable on the full 2.5M dataset. Statistics are already overpowered
at 100k.

Usage
-----
    python paper2_07_residual_regression.py --csv features.csv
    python paper2_07_residual_regression.py --csv features.csv --n_sample 200000
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


_ANGLES = [
    ('tau_deg',            'τ (N-Cα-C)'),
    ('omega_measured_deg', 'ω (peptide)'),
    ('angle_N_CA_CB',      '∠N-Cα-Cβ'),
    ('angle_C_CA_CB',      '∠C-Cα-Cβ'),
]

# Local features — available to the model
_LOCAL_FEATURES = [
    'phi_deg', 'psi_deg', 'res_type',
    'chi1_rad', 'chi2_rad',
    'sc_mass', 'sc_n_heavy',
    'sc_is_branched', 'sc_is_aromatic',
    'has_chi1', 'has_chi2',
]

# Non-local features — tested against residuals
_NONLOCAL_FEATURES = [
    # Through-space steric
    'steric_CA_5A', 'steric_N_5A', 'steric_C_5A', 'steric_O_5A',
    'steric_asym_x', 'steric_asym_y', 'steric_asym_z',
    # H-bonds
    'hb_n_bonds', 'hb_n_strong', 'hb_best_e',
    'hb_mean_d_HO', 'hb_best_d_HO',
    # Neighbour-sidechain contacts
    'sc_contact_nm1_to_bb', 'sc_contact_np1_to_bb',
    # Control: B-factor (tracks disorder, not mechanics)
    'bfactor_ca',
]


def fit_local_model(df, target, features, test_frac=0.3, random_state=42):
    """Fit GBR on local features. Returns (model, test_rmse, test_r2, residuals_series)."""
    d = df.dropna(subset=features + [target]).copy()
    X = d[features].values
    y = d[target].values

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, d.index, test_size=test_frac, random_state=random_state)

    # Compact model — fast, avoids overfitting at this dataset size
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=random_state)
    print(f"    fitting GBR on {len(X_tr):,} training rows ...")
    model.fit(X_tr, y_tr)

    pred_te = model.predict(X_te)
    rmse = float(np.sqrt(np.mean((y_te - pred_te) ** 2)))
    ss_res = float(np.sum((y_te - pred_te) ** 2))
    ss_tot = float(np.sum((y_te - np.mean(y_te)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Compute residuals on FULL sample (not just test) for the non-local analysis
    pred_full = model.predict(X)
    residuals = pd.Series(y - pred_full, index=d.index, name='residual')
    return model, rmse, r2, residuals, d


def correlate_residuals_with_nonlocal(d, residuals, nonlocal_features):
    """
    For each non-local feature, compute Pearson r vs residuals.
    Also compute the per-feature effect size: std of binned mean residuals
    across quartiles of that feature (relative to residual std).

    Returns a DataFrame sorted by |r|.
    """
    rows = []
    res_std = float(residuals.std())

    for feat in nonlocal_features:
        if feat not in d.columns:
            continue
        sub = d[feat].dropna()
        common = residuals.index.intersection(sub.index)
        if len(common) < 1000:
            continue

        x = sub.loc[common].values
        y = residuals.loc[common].values

        # Pearson r
        r, p = pearsonr(x, y)

        # Effect size: range of mean residual across quartiles, in residual stds
        try:
            q = pd.qcut(x, 4, labels=False, duplicates='drop')
            grp_means = pd.Series(y).groupby(q).mean()
            effect = float(grp_means.max() - grp_means.min()) / res_std
        except Exception:
            effect = float('nan')

        rows.append(dict(
            feature=feat,
            n=len(common),
            r=r, p=p,
            effect_size_sd=effect,
        ))

    rdf = pd.DataFrame(rows).sort_values('r', key=lambda s: -s.abs())
    return rdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--n_sample', type=int, default=100_000,
                    help='Random sample size for speed (default 100k)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='paper2_07_residual_regression.png')
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found"); sys.exit(1)

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} total rows")

    # Sample to keep runtime reasonable
    if len(df) > args.n_sample:
        df = df.sample(n=args.n_sample, random_state=args.seed)
        print(f"  sampled to {len(df):,} rows (seed={args.seed})")

    # Summary of available features
    available = [f for f in _LOCAL_FEATURES if f in df.columns]
    missing = [f for f in _LOCAL_FEATURES if f not in df.columns]
    if missing:
        print(f"  WARNING: missing local features: {missing}")
    print(f"  local features used: {len(available)}")

    # Run per-angle
    all_results = {}
    for target, label in _ANGLES:
        if target not in df.columns:
            print(f"  [SKIP] {target}: column not in CSV")
            continue

        # Filter to trans peptides for omega only — cis/twisted wrap the
        # numerical range and generate nonsense RMSE.
        # Also transform omega to deviation-from-trans to avoid circular
        # wrap between +179° and -179° (arithmetic mean of these is 0, not
        # 180, which breaks GBR's variance decomposition).
        df_angle = df
        eff_target = target
        if target == 'omega_measured_deg':
            n_before = len(df_angle)
            df_angle = df_angle[np.abs(df_angle[target]) > 150].copy()
            # Map +179° and -179° both to ~1° (deviation from trans)
            df_angle['omega_dev'] = 180.0 - np.abs(df_angle[target])
            eff_target = 'omega_dev'
            print(f"\n  [ω filter] kept {len(df_angle):,}/{n_before:,} "
                  f"trans peptides (|ω| > 150°)")
            print(f"  [ω transform] predicting omega_dev = 180 − |ω| "
                  f"to avoid circular wrap")
            print(f"  [ω range after transform] min={df_angle['omega_dev'].min():.2f}, "
                  f"max={df_angle['omega_dev'].max():.2f}, "
                  f"mean={df_angle['omega_dev'].mean():.2f}°")

        print("\n" + "=" * 80)
        print(f"LOCAL MODEL: {label}")
        print("=" * 80)

        model, rmse, r2, residuals, d = fit_local_model(
            df_angle, eff_target, available)
        all_results[target] = dict(
            label=label, rmse=rmse, r2=r2,
            residuals=residuals, d=d, model=model)

        print(f"  Test set R²:   {r2:.4f}")
        print(f"  Test set RMSE: {rmse:.4f}°")
        print(f"  Residual std:  {float(residuals.std()):.4f}°  "
              f"(full sample)")

        # Feature importances (top 5)
        imp = pd.Series(model.feature_importances_, index=available)
        imp = imp.sort_values(ascending=False)
        print(f"  Top 5 local features by importance:")
        for f, v in imp.head(5).items():
            print(f"    {f:<20s}  {v:.4f}")

        # Non-local correlation analysis
        print(f"\n  Residuals correlated with non-local features:")
        rdf = correlate_residuals_with_nonlocal(d, residuals, _NONLOCAL_FEATURES)
        print(f"    {'feature':<22s}  {'r':>7s}  {'effect(SD)':>10s}  {'p':>10s}  {'n':>8s}")
        print("    " + "-" * 70)
        for _, row in rdf.iterrows():
            sig = '***' if row['p'] < 1e-20 else (
                  '**' if row['p'] < 1e-5 else (
                  '*' if row['p'] < 1e-2 else ''))
            print(f"    {row['feature']:<22s}  {row['r']:>+7.3f}  "
                  f"{row['effect_size_sd']:>10.3f}  {row['p']:>10.1e}  "
                  f"{int(row['n']):>8,}  {sig}")

        all_results[target]['nonlocal_corr'] = rdf

    # ── Summary figure: R² bar chart + top non-local features per angle ──────
    print(f"\nGenerating {args.out} ...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for k, (target, _) in enumerate(_ANGLES):
        ax = axes[k]
        if target not in all_results:
            ax.axis('off'); continue
        res = all_results[target]
        rdf = res['nonlocal_corr'].copy()
        # Sort by |effect_size_sd|, top 8
        rdf['abs_eff'] = rdf['effect_size_sd'].abs()
        top = rdf.sort_values('abs_eff', ascending=False).head(8)
        y = np.arange(len(top))
        colors = ['#c0392b' if abs(e) > 0.08 else '#888'
                  for e in top['effect_size_sd']]
        ax.barh(y, top['effect_size_sd'], color=colors, edgecolor='white')
        ax.set_yticks(y)
        ax.set_yticklabels(top['feature'], fontsize=9)
        ax.set_xlabel('Effect size of non-local feature on residual (SD units)')
        ax.axvline(0, color='k', lw=0.5)
        ax.axvline(+0.08, color='#aaa', lw=0.5, ls=':')
        ax.axvline(-0.08, color='#aaa', lw=0.5, ls=':')
        ax.set_title(f"{res['label']}   local R² = {res['r2']:.3f}",
                      fontsize=11)
        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.25)

    plt.suptitle(
        'Move C: residuals after local (backbone+sidechain) model\n'
        'Red bars = |effect| > 0.08 SD (real non-local channel); '
        'grey = negligible',
        fontsize=12, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches='tight')
    print(f"Figure saved: {args.out}")

    # ── Verdict ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("VERDICT")
    print("(thresholds: local R² > 0.3 = 'explained'; max effect size > 0.08")
    print(" SD = 'real non-local channel detected')")
    print("=" * 80)
    for target, label in _ANGLES:
        if target not in all_results:
            continue
        res = all_results[target]
        rdf = res['nonlocal_corr']
        if len(rdf) == 0:
            continue

        max_eff = rdf['effect_size_sd'].abs().max()
        # Top feature by effect size (not by r — r misses non-linear effects)
        top_idx = rdf['effect_size_sd'].abs().idxmax()
        top_feat = rdf.loc[top_idx, 'feature']
        top_eff  = rdf.loc[top_idx, 'effect_size_sd']
        top_r    = rdf.loc[top_idx, 'r']

        if res['r2'] > 0.3 and max_eff < 0.08:
            verdict = ("ONE-CHANNEL: local model dominates, no meaningful "
                       "non-local signal")
        elif res['r2'] > 0.3 and max_eff >= 0.08:
            verdict = (f"TWO-CHANNEL: local R² = {res['r2']:.2f}, "
                       f"plus non-local via {top_feat} "
                       f"(effect={top_eff:+.3f} SD, r={top_r:+.3f})")
        elif res['r2'] <= 0.3 and max_eff >= 0.15:
            verdict = (f"NON-LOCAL DOMINANT: local R² = {res['r2']:.2f} "
                       f"is weak; top non-local {top_feat} "
                       f"has effect {top_eff:+.3f} SD")
        else:
            verdict = (f"WEAK: local R² = {res['r2']:.2f}, max effect "
                       f"= {max_eff:.3f} SD. Intrinsic noise dominates.")

        print(f"  {label:<16s}  {verdict}")


if __name__ == '__main__':
    main()