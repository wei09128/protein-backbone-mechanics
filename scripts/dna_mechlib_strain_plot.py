#!/usr/bin/env python3
"""
dna_mechlib_strain_plot.py — DNA backbone strain: AMBER vs MechLib-DNA
=========================================================================
DNA-backbone analog of paper4_04_strain_plot.py. Instead of a continuous
(phi,psi) grid (intractable for DNA's 6+1 backbone torsions), MechLib-DNA
groups by the low-dimensional classification that actually captures DNA
backbone conformational substates:

    (res_name, BI/BII, sugar_pucker_class)

  - BI/BII is the dominant discrete backbone conformational switch
    (analogous to protein's alpha/beta/PPII basins), set by the
    epsilon-zeta torsion pair.
  - sugar_pucker_class (10 Altona-Sundaralingam sectors) captures the
    other major degree of freedom, analogous to a chi1 rotamer class.
  - res_name (DA/DC/DG/DT) captures base-specific geometry differences,
    analogous to protein's per-residue-type grouping.

AMBER bond/angle equilibrium values are from parm10.dat (base AMBER
parameter set) verified directly from the real file; these are shared
identically across bsc0, bsc1, and OL15 since none of those refinements
change bond/angle equilibria -- only torsional (dihedral) potentials.
This makes the strain-energy comparison below force-field-agnostic
across all three DNA AMBER variants.

Usage:
  python dna_mechlib_strain_plot.py --csv dna_features_v2_nonredundant.csv --out ./dna_mechlib/
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Verified from parm10.dat (base AMBER parameter set; unchanged by bsc1/OL15)
AMBER_DNA = {
    'bond_P_O5':        {'eq': 1.610,  'k': 230.0, 'unit': 'Å'},
    'bond_O5_C5':       {'eq': 1.410,  'k': 320.0, 'unit': 'Å'},
    'bond_C5_C4':       {'eq': 1.526,  'k': 310.0, 'unit': 'Å'},
    'bond_C4_O4':       {'eq': 1.410,  'k': 320.0, 'unit': 'Å'},
    'bond_C4_C3':       {'eq': 1.526,  'k': 310.0, 'unit': 'Å'},
    'bond_C3_O3':       {'eq': 1.410,  'k': 320.0, 'unit': 'Å'},
    'bond_C3_C2':       {'eq': 1.526,  'k': 310.0, 'unit': 'Å'},
    'bond_C2_C1':       {'eq': 1.526,  'k': 310.0, 'unit': 'Å'},
    'bond_C1_O4':       {'eq': 1.410,  'k': 320.0, 'unit': 'Å'},
    'bond_C1_N':        {'eq': 1.475,  'k': 337.0, 'unit': 'Å'},
    'bond_O3_Pnext':     {'eq': 1.610,  'k': 230.0, 'unit': 'Å'},
    'angle_O5_P_O3prev': {'eq': 102.60, 'k': 45.0,  'unit': 'deg'},
    'angle_OP1_P_OP2':   {'eq': 119.90, 'k': 140.0, 'unit': 'deg'},
    'angle_C5_O5_P':     {'eq': 120.50, 'k': 100.0, 'unit': 'deg'},
    'angle_C5_C4_O4':    {'eq': 109.50, 'k': 50.0,  'unit': 'deg'},
    'angle_C5_C4_C3':    {'eq': 109.50, 'k': 40.0,  'unit': 'deg'},
    'angle_O4_C4_C3':    {'eq': 109.50, 'k': 50.0,  'unit': 'deg'},
    'angle_C4_C3_O3':    {'eq': 109.50, 'k': 50.0,  'unit': 'deg'},
    'angle_C4_C3_C2':    {'eq': 109.50, 'k': 40.0,  'unit': 'deg'},
    'angle_O3_C3_C2':    {'eq': 109.50, 'k': 50.0,  'unit': 'deg'},
    'angle_C3_C2_C1':    {'eq': 109.50, 'k': 40.0,  'unit': 'deg'},
    'angle_C2_C1_O4':    {'eq': 109.50, 'k': 50.0,  'unit': 'deg'},
    'angle_C1_O4_C4':    {'eq': 109.50, 'k': 60.0,  'unit': 'deg'},
    'angle_O4_C1_N':     {'eq': 109.50, 'k': 50.0,  'unit': 'deg'},
    'angle_C2_C1_N':     {'eq': 109.50, 'k': 50.0,  'unit': 'deg'},
    'angle_C3_O3_Pnext': {'eq': 120.50, 'k': 100.0, 'unit': 'deg'},
}


def build_library(df, min_count=10):
    """
    MechLib-DNA library keyed by (res_name, bi_bii, pucker_class), with
    fallback to (bi_bii, pucker_class) pooled across bases, then to the
    global mean, when a specific bin is too sparse (< min_count).
    """
    geo_cols = [c for c in AMBER_DNA if c in df.columns]
    sub = df.dropna(subset=['bi_bii', 'pucker_class']).copy()
    sub = sub[sub['bi_bii'] != 'undefined']

    lib = {}
    global_mean = sub[geo_cols].mean()

    for (res, bibii, pucker), grp in sub.groupby(['res_name', 'bi_bii', 'pucker_class']):
        if len(grp) >= min_count:
            lib[(res, bibii, pucker)] = grp[geo_cols].mean()

    pooled = {}
    for (bibii, pucker), grp in sub.groupby(['bi_bii', 'pucker_class']):
        pooled[(bibii, pucker)] = grp[geo_cols].mean()

    return lib, pooled, global_mean, geo_cols


def lookup_eq(lib, pooled, global_mean, res, bibii, pucker, col):
    key = (res, bibii, pucker)
    if key in lib and not np.isnan(lib[key][col]):
        return lib[key][col]
    pkey = (bibii, pucker)
    if pkey in pooled and not np.isnan(pooled[pkey][col]):
        return pooled[pkey][col]
    return global_mean[col]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='./dna_mechlib')
    ap.add_argument('--min_count', type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Loading CSV...")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"  {len(df):,} nucleotides")

    # Restrict to true DNA residues -- some "pure DNA" structures contain
    # embedded ribonucleotides (RNA primers, RNA-DNA junctions), which use
    # AMBER RNA-specific atom types/equilibria (2'-OH sugar, different
    # pucker preference) and would show artificially inflated "strain" if
    # compared against DNA-specific equilibrium values.
    n_before = len(df)
    df = df[df['res_name'].isin(['DA', 'DC', 'DG', 'DT'])].copy()
    n_rna_excluded = n_before - len(df)
    if n_rna_excluded > 0:
        print(f"  Excluded {n_rna_excluded} embedded RNA-named residues "
              f"({100*n_rna_excluded/n_before:.2f}%) -- not true DNA")

    print("Building MechLib-DNA library (res_name x BI/BII x sugar pucker)...")
    lib, pooled, global_mean, geo_cols = build_library(df, args.min_count)
    print(f"  {len(lib)} (res,bibii,pucker) bins with >= {args.min_count} nucleotides")
    print(f"  {len(pooled)} pooled (bibii,pucker) fallback bins")

    sub = df.dropna(subset=['bi_bii', 'pucker_class']).copy()
    sub = sub[sub['bi_bii'] != 'undefined'].reset_index(drop=True)

    print("Computing strain...")
    strain_amber = np.zeros(len(sub))
    strain_lib = np.zeros(len(sub))

    per_col_amber = {}
    per_col_lib = {}

    for col in geo_cols:
        obs = sub[col].values
        valid = ~np.isnan(obs)
        k = AMBER_DNA[col]['k']
        conv = np.pi / 180.0 if AMBER_DNA[col]['unit'] == 'deg' else 1.0
        eq = AMBER_DNA[col]['eq']

        ea = 0.5 * k * ((obs - eq) * conv) ** 2
        ea[~valid] = 0.0
        strain_amber += ea

        lib_eq = np.array([
            lookup_eq(lib, pooled, global_mean, sub['res_name'].iloc[i],
                      sub['bi_bii'].iloc[i], sub['pucker_class'].iloc[i], col)
            if valid[i] else eq
            for i in range(len(sub))
        ])
        el = 0.5 * k * ((obs - lib_eq) * conv) ** 2
        el[~valid] = 0.0
        strain_lib += el

        per_col_amber[col] = ea[valid].mean() if valid.sum() else np.nan
        per_col_lib[col] = el[valid].mean() if valid.sum() else np.nan

    overall_amber_mean = strain_amber.mean()
    overall_lib_mean = strain_lib.mean()
    overall_amber_median = np.median(strain_amber)
    overall_lib_median = np.median(strain_lib)
    overall_reduction_mean = 100 * (overall_amber_mean - overall_lib_mean) / overall_amber_mean
    overall_reduction_median = 100 * (overall_amber_median - overall_lib_median) / overall_amber_median

    print(f"\n{'='*60}")
    print(f"  DNA backbone strain: AMBER vs MechLib-DNA")
    print(f"  --- MEAN (sensitive to rare severe defects) ---")
    print(f"  Mean AMBER strain  : {overall_amber_mean:.5f} kcal/mol/nt")
    print(f"  Mean Library strain: {overall_lib_mean:.5f} kcal/mol/nt")
    print(f"  Reduction (mean)   : {overall_reduction_mean:.1f}%")
    print(f"  --- MEDIAN (robust to rare severe defects) ---")
    print(f"  Median AMBER strain  : {overall_amber_median:.5f} kcal/mol/nt")
    print(f"  Median Library strain: {overall_lib_median:.5f} kcal/mol/nt")
    print(f"  Reduction (median)   : {overall_reduction_median:.1f}%")
    print(f"{'='*60}")

    # ── Per-BI/BII breakdown ─────────────────────────────────────────────
    print("\nPer-BI/BII breakdown:")
    for bibii in ['BI', 'BII']:
        mask = sub['bi_bii'].values == bibii
        if mask.sum() < 10:
            continue
        a_mean, l_mean = strain_amber[mask].mean(), strain_lib[mask].mean()
        a_med, l_med = np.median(strain_amber[mask]), np.median(strain_lib[mask])
        print(f"  {bibii:4s} (n={mask.sum():6d}): "
              f"mean AMBER={a_mean:.4f} Lib={l_mean:.4f} red={100*(a_mean-l_mean)/a_mean:.1f}%  |  "
              f"median AMBER={a_med:.4f} Lib={l_med:.4f} red={100*(a_med-l_med)/a_med:.1f}%")

    # ── Per-base breakdown ────────────────────────────────────────────────
    print("\nPer-base breakdown:")
    for res in sorted(sub['res_name'].unique()):
        mask = sub['res_name'].values == res
        if mask.sum() < 10:
            continue
        a_mean, l_mean = strain_amber[mask].mean(), strain_lib[mask].mean()
        a_med, l_med = np.median(strain_amber[mask]), np.median(strain_lib[mask])
        print(f"  {res:4s} (n={mask.sum():6d}): "
              f"mean AMBER={a_mean:.4f} Lib={l_mean:.4f} red={100*(a_mean-l_mean)/a_mean:.1f}%  |  "
              f"median AMBER={a_med:.4f} Lib={l_med:.4f} red={100*(a_med-l_med)/a_med:.1f}%")

    # ── Plot ──────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: BI/BII strain comparison
    ax = axes[0]
    labels, amber_vals, lib_vals = [], [], []
    for bibii in ['BI', 'BII']:
        mask = sub['bi_bii'].values == bibii
        if mask.sum() < 10:
            continue
        labels.append(bibii)
        amber_vals.append(strain_amber[mask].mean())
        lib_vals.append(strain_lib[mask].mean())
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, amber_vals, w, label='AMBER (parm10)', color='#B2182B',
           edgecolor='#333333', linewidth=0.5)
    ax.bar(x + w/2, lib_vals, w, label='MechLib-DNA', color='#2166AC',
           edgecolor='#333333', linewidth=0.5)
    for i, (a, l) in enumerate(zip(amber_vals, lib_vals)):
        pct = (a - l) / a * 100
        ax.text(i, max(a, l) * 1.03, f'\u2212{pct:.0f}%', ha='center',
                fontsize=10, fontweight='bold', color='#2166AC')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Mean strain energy (kcal/mol/nt)')
    ax.set_title('A. Phantom strain by BI/BII state', fontweight='bold')
    ax.legend()

    # Panel B: per-observable strain
    ax = axes[1]
    obs_cols = list(geo_cols)
    amber_obs = [per_col_amber[c] for c in obs_cols]
    lib_obs = [per_col_lib[c] for c in obs_cols]
    x2 = np.arange(len(obs_cols))
    ax.bar(x2 - w/2, amber_obs, w, label='AMBER', color='#B2182B',
           edgecolor='#333333', linewidth=0.5)
    ax.bar(x2 + w/2, lib_obs, w, label='Library', color='#2166AC',
           edgecolor='#333333', linewidth=0.5)
    ax.set_xticks(x2)
    ax.set_xticklabels([c.replace('bond_', '').replace('angle_', '')
                         for c in obs_cols], fontsize=7, rotation=60, ha='right')
    ax.set_ylabel('Mean strain (kcal/mol/nt)')
    ax.set_title('B. Strain by observable', fontweight='bold')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'dna_strain_comparison.png'), dpi=200)
    plt.close()
    print(f"\nSaved dna_strain_comparison.png to {args.out}")


if __name__ == '__main__':
    main()
