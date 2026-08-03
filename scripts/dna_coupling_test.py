#!/usr/bin/env python3
"""
dna_coupling_test.py — Does DNA sugar pucker (delta) genuinely couple with chi?
==================================================================================
Applies the same ANOVA-style separability test used in Protein Backbone
Mechanics III to the DNA backbone: for a geometric observable f(delta, chi),
decompose:

    f(delta, chi) = f0 + df_delta(delta) + df_chi(chi) + df_coupling(delta,chi) + eps

where df_delta and df_chi are marginal (row/column mean - grand mean) effects
and df_coupling is the interaction/coupling residual -- the part that CANNOT
be explained by delta and chi acting independently.

  eta^2 = Var(df_coupling) / Var(f - f0)

is the fraction of systematic variance attributable to genuine delta-chi
coupling. eta^2 near 0 means delta and chi act independently (separable);
eta^2 substantially > 0 means real interaction exists (like phi/psi in
protein backbone mechanics).

Usage:
  python dna_coupling_test.py --csv dna_features_v3_nonredundant.csv --out ./dna_coupling/
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def anova2_decomposition(df, delta_col, chi_col, value_col, bin_size=10):
    """
    2-way ANOVA decomposition of value_col over (delta_col, chi_col) bins.
    Returns eta_squared, and the binned mean grids for plotting.
    """
    sub = df.dropna(subset=[delta_col, chi_col, value_col]).copy()

    delta_bins = np.arange(0, 190, bin_size)   # delta/pucker phase is 0-180 for delta_deg range typically; but delta_deg here is the actual torsion (roughly 80-160 range) -- use observed range
    d_min, d_max = sub[delta_col].min(), sub[delta_col].max()
    c_min, c_max = sub[chi_col].min(), sub[chi_col].max()

    delta_bins = np.arange(np.floor(d_min / bin_size) * bin_size,
                            np.ceil(d_max / bin_size) * bin_size + bin_size, bin_size)
    chi_bins = np.arange(np.floor(c_min / bin_size) * bin_size,
                          np.ceil(c_max / bin_size) * bin_size + bin_size, bin_size)

    sub['db'] = pd.cut(sub[delta_col], delta_bins, labels=False, right=False)
    sub['cb'] = pd.cut(sub[chi_col], chi_bins, labels=False, right=False)
    sub = sub.dropna(subset=['db', 'cb'])
    sub['db'] = sub['db'].astype(int)
    sub['cb'] = sub['cb'].astype(int)

    grand_mean = sub[value_col].mean()

    # Marginal means
    row_mean = sub.groupby('db')[value_col].transform('mean')  # delta marginal
    col_mean = sub.groupby('cb')[value_col].transform('mean')  # chi marginal
    cell_mean = sub.groupby(['db', 'cb'])[value_col].transform('mean')

    df_delta = row_mean - grand_mean
    df_chi = col_mean - grand_mean
    df_coupling = cell_mean - row_mean - col_mean + grand_mean

    total_var = np.var(sub[value_col] - grand_mean)
    var_delta = np.var(df_delta)
    var_chi = np.var(df_chi)
    var_coupling = np.var(df_coupling)

    eta2_delta = var_delta / total_var
    eta2_chi = var_chi / total_var
    eta2_coupling = var_coupling / total_var

    # Build grid for plotting
    d_centers = delta_bins[:-1] + bin_size / 2
    c_centers = chi_bins[:-1] + bin_size / 2
    grid = np.full((len(d_centers), len(c_centers)), np.nan)
    counts = np.zeros((len(d_centers), len(c_centers)), dtype=int)
    for (db, cb), grp in sub.groupby(['db', 'cb']):
        if 0 <= db < len(d_centers) and 0 <= cb < len(c_centers):
            grid[db, cb] = grp[value_col].mean()
            counts[db, cb] = len(grp)

    return dict(eta2_delta=eta2_delta, eta2_chi=eta2_chi, eta2_coupling=eta2_coupling,
                grid=grid, counts=counts, d_centers=d_centers, c_centers=c_centers,
                n=len(sub))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='./dna_coupling')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Loading CSV...")
    df = pd.read_csv(args.csv, low_memory=False)
    df = df[df['res_name'].isin(['DA', 'DC', 'DG', 'DT'])].copy()
    print(f"  {len(df):,} true-DNA nucleotides")

    # Test several observables for delta-chi coupling
    observables = [
        'angle_C1_O4_C4',   # ring closure angle, plausible chi/pucker sensitivity
        'bond_C1_N',        # glycosidic bond length
        'angle_O4_C1_N',    # anomeric angle (O4'-C1'-N)
        'angle_C2_C1_N',    # C2'-C1'-N angle
        'angle_C5_C4_C3',   # ring angle, less obviously chi-related (control)
    ]

    print(f"\n{'='*70}")
    print(f"  ANOVA-style delta-chi coupling test (per observable)")
    print(f"{'='*70}")
    results = {}
    for obs in observables:
        r = anova2_decomposition(df, 'delta_deg', 'chi_deg', obs, bin_size=10)
        results[obs] = r
        print(f"  {obs:20s}  eta2_delta={r['eta2_delta']:.3f}  "
              f"eta2_chi={r['eta2_chi']:.3f}  eta2_coupling={r['eta2_coupling']:.3f}  "
              f"(n={r['n']:,})")

    # ── Plot the strongest coupling candidate ────────────────────────────
    best_obs = max(observables, key=lambda o: results[o]['eta2_coupling'])
    print(f"\nStrongest coupling signal: {best_obs} "
          f"(eta2_coupling = {results[best_obs]['eta2_coupling']:.3f})")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    r = results[best_obs]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(r['grid'].T, origin='lower', aspect='auto',
                   extent=[r['d_centers'][0], r['d_centers'][-1],
                           r['c_centers'][0], r['c_centers'][-1]],
                   cmap='RdBu_r')
    ax.set_xlabel('δ (sugar pucker torsion, °)')
    ax.set_ylabel('χ (glycosidic torsion, °)')
    ax.set_title(f'{best_obs} vs (δ, χ)\n'
                 f'η²(coupling) = {r["eta2_coupling"]:.3f}, '
                 f'η²(δ) = {r["eta2_delta"]:.3f}, η²(χ) = {r["eta2_chi"]:.3f}')
    plt.colorbar(im, ax=ax, label=best_obs)
    plt.tight_layout()
    out_path = os.path.join(args.out, 'dna_delta_chi_coupling.png')
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
