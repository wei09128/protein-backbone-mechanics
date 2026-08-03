#!/usr/bin/env python3
"""
paper3_vs_paper1_correlation.py — Is coupling actually linked to k_eff?
================================================================================
Section 3.3 of Paper 3 currently ASSERTS that the bond-angle coupling
reported there is "the direct manifestation" of Paper 1's k_eff curvature
findings, without ever testing this. These are two genuinely different
computed quantities:
  - Paper 1's k_eff = -tau_env / delta_theta (a torque-derived secant slope,
    Methods 4.7 in Paper 1)
  - Paper 3's coupling = Delta f_phipsi from the tau_deg ANOVA decomposition
    (a bond-angle geometric variance decomposition, Methods 4.3 in Paper 3)

This script computes BOTH quantities on the identical (phi,psi) 10x10 grid
and directly tests their correlation (Pearson + Spearman, weighted by bin
population), producing the scatter plot that should either support or
replace the current unsupported assertion in Section 3.3.

Usage:
  python paper3_vs_paper1_correlation.py --csv features_lj_FINAL_CLEAN.csv --bin_size 10
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import warnings

warnings.filterwarnings('ignore')

REF_PHI, REF_PSI = -63.0, -43.0
MIN_DTHETA_DEG = 5.0


def wrap180(x):
    return (x + 180) % 360 - 180


def anova2_decomposition(df, phi_col, psi_col, value_col, bin_size=10):
    """VERBATIM from paper3_03_coupling_decomposition.py -- unchanged."""
    sub = df[[phi_col, psi_col, value_col]].dropna().copy()
    if len(sub) < 100:
        return None

    phi_bins = np.arange(-180, 180 + bin_size, bin_size)
    psi_bins = np.arange(-180, 180 + bin_size, bin_size)

    sub['phi_bin'] = pd.cut(sub[phi_col], phi_bins, labels=False, right=False)
    sub['psi_bin'] = pd.cut(sub[psi_col], psi_bins, labels=False, right=False)
    sub = sub.dropna(subset=['phi_bin', 'psi_bin'])
    sub['phi_bin'] = sub['phi_bin'].astype(int)
    sub['psi_bin'] = sub['psi_bin'].astype(int)
    if len(sub) < 100:
        return None

    f0 = sub[value_col].mean()
    cell_stats = sub.groupby(['phi_bin', 'psi_bin'])[value_col].agg(['mean', 'count', 'var'])
    cell_stats.columns = ['cell_mean', 'cell_count', 'cell_var']
    MIN_COUNT = 5
    cell_stats = cell_stats[cell_stats['cell_count'] >= MIN_COUNT]
    if len(cell_stats) < 10:
        return None

    phi_marginal = (cell_stats.reset_index().groupby('phi_bin')
                     .apply(lambda g: np.average(g['cell_mean'], weights=g['cell_count']),
                            include_groups=False))
    psi_marginal = (cell_stats.reset_index().groupby('psi_bin')
                     .apply(lambda g: np.average(g['cell_mean'], weights=g['cell_count']),
                            include_groups=False))
    phi_effect = phi_marginal - f0
    psi_effect = psi_marginal - f0

    coupling = cell_stats[['cell_mean', 'cell_count']].copy()
    coupling['phi_effect'] = coupling.index.get_level_values('phi_bin').map(phi_effect)
    coupling['psi_effect'] = coupling.index.get_level_values('psi_bin').map(psi_effect)
    coupling['additive_prediction'] = f0 + coupling['phi_effect'] + coupling['psi_effect']
    coupling['coupling_residual'] = coupling['cell_mean'] - coupling['additive_prediction']
    coupling = coupling.dropna()
    if len(coupling) < 10:
        return None

    phi_bins_c = phi_bins[:-1] + bin_size / 2
    psi_bins_c = psi_bins[:-1] + bin_size / 2
    coupling = coupling.reset_index()
    coupling['phi_center'] = coupling['phi_bin'].map(lambda i: phi_bins_c[i])
    coupling['psi_center'] = coupling['psi_bin'].map(lambda i: psi_bins_c[i])
    return coupling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--bin_size', type=int, default=10)
    ap.add_argument('--min_n_per_bin', type=int, default=100,
                    help='Minimum residues per bin for both quantities to be trusted together')
    ap.add_argument('--out_dir', default='.')
    args = ap.parse_args()

    print("Loading data...")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"  {len(df):,} residues, {df['pdb_id'].nunique()} structures")

    print("Computing Paper 1's k_eff_phi (secant slope)...")
    df2 = df.dropna(subset=['phi_deg', 'psi_deg', 'tau_phi_correct']).copy()
    dphi_deg = wrap180(df2['phi_deg'].values - REF_PHI)
    valid = np.abs(dphi_deg) > MIN_DTHETA_DEG
    df2 = df2[valid].copy()
    dphi_rad = np.radians(dphi_deg[valid])
    df2['k_eff_phi'] = -df2['tau_phi_correct'].values / dphi_rad

    phi_bins = np.arange(-180, 180 + args.bin_size, args.bin_size)
    psi_bins = np.arange(-180, 180 + args.bin_size, args.bin_size)
    df2['pb'] = pd.cut(df2['phi_deg'], phi_bins, labels=False, right=False)
    df2['qb'] = pd.cut(df2['psi_deg'], psi_bins, labels=False, right=False)
    df2 = df2.dropna(subset=['pb', 'qb'])
    df2['pb'] = df2['pb'].astype(int); df2['qb'] = df2['qb'].astype(int)
    phi_c = phi_bins[:-1] + args.bin_size / 2
    psi_c = psi_bins[:-1] + args.bin_size / 2

    keff_bins = df2.groupby(['pb', 'qb'])['k_eff_phi'].agg(['mean', 'count']).reset_index()
    keff_bins['phi_center'] = keff_bins['pb'].map(lambda i: phi_c[i])
    keff_bins['psi_center'] = keff_bins['qb'].map(lambda i: psi_c[i])
    keff_bins = keff_bins.rename(columns={'mean': 'k_eff_phi_mean', 'count': 'n_keff'})

    print("Computing Paper 3's tau_deg coupling residual...")
    coupling = anova2_decomposition(df, 'phi_deg', 'psi_deg', 'tau_deg', args.bin_size)
    if coupling is None:
        print("ERROR: coupling decomposition failed -- check column names / data.")
        return

    merged = pd.merge(
        keff_bins[['phi_center', 'psi_center', 'k_eff_phi_mean', 'n_keff']],
        coupling[['phi_center', 'psi_center', 'coupling_residual', 'cell_count']],
        on=['phi_center', 'psi_center'], how='inner')
    merged = merged[(merged['n_keff'] >= args.min_n_per_bin) &
                     (merged['cell_count'] >= args.min_n_per_bin)]
    print(f"  {len(merged)} bins with >= {args.min_n_per_bin} residues in BOTH quantities")

    if len(merged) < 10:
        print("Too few well-populated overlapping bins for a reliable correlation test.")
        return

    x = merged['coupling_residual'].values
    y = merged['k_eff_phi_mean'].values
    x_abs = np.abs(x)

    r_pearson, p_pearson = sp_stats.pearsonr(x, y)
    r_pearson_abs, p_pearson_abs = sp_stats.pearsonr(x_abs, y)
    rho_spearman, p_spearman = sp_stats.spearmanr(x, y)
    rho_spearman_abs, p_spearman_abs = sp_stats.spearmanr(x_abs, y)

    print(f"\n{'='*70}")
    print(f"  CORRELATION: tau_deg coupling (Paper 3) vs. k_eff_phi (Paper 1)")
    print(f"  n = {len(merged)} well-populated (phi,psi) bins")
    print(f"{'='*70}")
    print(f"  Signed coupling vs. k_eff_phi:")
    print(f"    Pearson  r = {r_pearson:+.3f}  (p = {p_pearson:.2e})")
    print(f"    Spearman rho = {rho_spearman:+.3f}  (p = {p_spearman:.2e})")
    print(f"  |Coupling| vs. k_eff_phi:")
    print(f"    Pearson  r = {r_pearson_abs:+.3f}  (p = {p_pearson_abs:.2e})")
    print(f"    Spearman rho = {rho_spearman_abs:+.3f}  (p = {p_spearman_abs:.2e})")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    sc = ax.scatter(x, y, c=merged['n_keff'], cmap='viridis', s=18, alpha=0.7)
    ax.set_xlabel('tau coupling residual, Delta f_phipsi (deg)  [Paper 3]')
    ax.set_ylabel('k_eff_phi, secant slope  [Paper 1]')
    ax.set_title(f'Signed: Pearson r={r_pearson:+.3f} (p={p_pearson:.1e})', fontsize=10)
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    plt.colorbar(sc, ax=ax, label='n residues/bin')

    ax = axes[1]
    sc = ax.scatter(x_abs, y, c=merged['n_keff'], cmap='viridis', s=18, alpha=0.7)
    ax.set_xlabel('|tau coupling residual| (deg)  [Paper 3]')
    ax.set_ylabel('k_eff_phi, secant slope  [Paper 1]')
    ax.set_title(f'|Coupling|: Pearson r={r_pearson_abs:+.3f} (p={p_pearson_abs:.1e})', fontsize=10)
    ax.axhline(0, color='gray', lw=0.5)
    plt.colorbar(sc, ax=ax, label='n residues/bin')

    plt.tight_layout()
    out_path = f"{args.out_dir}/paper3_vs_paper1_correlation.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved {out_path}")

    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print("  A strong, significant correlation (|r| meaningfully > 0, tight")
    print("  scatter) would support Section 3.3's claim that these two")
    print("  independently-computed quantities track the same underlying")
    print("  phenomenon. A weak or near-zero correlation means that claim")
    print("  should be dropped or substantially softened -- these are simply")
    print("  two different quantities that both happen to be conformation-")
    print("  dependent, without a demonstrated direct relationship.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
