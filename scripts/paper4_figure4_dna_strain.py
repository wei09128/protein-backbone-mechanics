#!/usr/bin/env python3
"""
figure4_dna_strain.py — Polished Figure 4: DNA backbone strain, AMBER vs MechLib-DNA
=======================================================================================
Uses MEDIAN-based statistics (not mean) throughout, since we've directly
confirmed the mean is inflated by a small number of genuine crystallographic
defects (same category of rare structural error found in the protein LJ
dataset). The BII breakdown is shown but flagged as an open methodological
question -- three sample bond/angle terms showed no bimodality or sparse-bin
issue, so the cause of BII's anomalous behavior is not yet understood and
should not be presented as resolved.

Usage:
  python figure4_dna_strain.py --csv dna_features_v2_nonredundant.csv --out ./figures/
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mechlib_style import setup_style, COLORS, clean_observable_name

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
    geo_cols = [c for c in AMBER_DNA if c in df.columns]
    sub = df.dropna(subset=['bi_bii', 'pucker_class']).copy()
    sub = sub[sub['bi_bii'] != 'undefined']

    lib, pooled = {}, {}
    global_mean = sub[geo_cols].mean()
    for (res, bibii, pucker), grp in sub.groupby(['res_name', 'bi_bii', 'pucker_class']):
        if len(grp) >= min_count:
            lib[(res, bibii, pucker)] = grp[geo_cols].mean()
    for (bibii, pucker), grp in sub.groupby(['bi_bii', 'pucker_class']):
        pooled[(bibii, pucker)] = grp[geo_cols].mean()
    return sub, lib, pooled, global_mean, geo_cols


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
    ap.add_argument('--out', default='./figures')
    ap.add_argument('--min_count', type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    setup_style()
    import matplotlib.pyplot as plt

    print("Loading CSV...")
    df = pd.read_csv(args.csv, low_memory=False)
    n_before = len(df)
    df = df[df['res_name'].isin(['DA', 'DC', 'DG', 'DT'])].copy()
    print(f"Excluded {n_before - len(df)} embedded RNA-named residues")

    sub, lib, pooled, global_mean, geo_cols = build_library(df, args.min_count)

    strain_amber = np.zeros(len(sub))
    strain_lib = np.zeros(len(sub))
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
            if valid[i] else eq for i in range(len(sub))
        ])
        el = 0.5 * k * ((obs - lib_eq) * conv) ** 2
        el[~valid] = 0.0
        strain_lib += el

    median_amber = np.median(strain_amber)
    median_lib = np.median(strain_lib)
    overall_reduction = 100 * (median_amber - median_lib) / median_amber
    print(f"Overall MEDIAN reduction: {overall_reduction:.1f}%")

    # Per-base medians
    base_labels, base_amber, base_lib = [], [], []
    for res in ['DA', 'DC', 'DG', 'DT']:
        mask = sub['res_name'].values == res
        if mask.sum() < 10:
            continue
        base_labels.append(res)
        base_amber.append(np.median(strain_amber[mask]))
        base_lib.append(np.median(strain_lib[mask]))

    # Per-BI/BII medians
    bibii_labels, bibii_amber, bibii_lib = [], [], []
    for bibii in ['BI', 'BII']:
        mask = sub['bi_bii'].values == bibii
        if mask.sum() < 10:
            continue
        bibii_labels.append(f'{bibii}\n(n={mask.sum():,})')
        bibii_amber.append(np.median(strain_amber[mask]))
        bibii_lib.append(np.median(strain_lib[mask]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Panel A: by base (the clean, trustworthy result)
    ax = axes[0]
    x = np.arange(len(base_labels)); w = 0.35
    ax.bar(x - w/2, base_amber, w, label='AMBER (parm10)', color=COLORS['amber'],
           edgecolor='#333333', linewidth=0.5)
    ax.bar(x + w/2, base_lib, w, label='MechLib-DNA', color=COLORS['mechlib'],
           edgecolor='#333333', linewidth=0.5)
    for i, (a, l) in enumerate(zip(base_amber, base_lib)):
        pct = (a - l) / a * 100
        ax.text(i, max(a, l) * 0.90, f'\u2212{pct:.0f}%', ha='center',
                fontsize=9, fontweight='bold', color=COLORS['mechlib'])
    ax.set_xticks(x); ax.set_xticklabels(base_labels)
    ax.set_ylabel('Median strain (kcal/mol/nt)')
    ax.set_title(f'A. By base (overall: \u2212{overall_reduction:.0f}%)')
    ax.legend(frameon=False)

    # Panel B: by BI/BII, explicitly flagged
    ax = axes[1]
    x2 = np.arange(len(bibii_labels))
    ax.bar(x2 - w/2, bibii_amber, w, color=COLORS['amber'], edgecolor='#333333', linewidth=0.5)
    ax.bar(x2 + w/2, bibii_lib, w, color=COLORS['mechlib'], edgecolor='#333333', linewidth=0.5)
    for i, (a, l) in enumerate(zip(bibii_amber, bibii_lib)):
        pct = (a - l) / a * 100
        color = COLORS['mechlib'] if pct > 0 else '#D6604D'
        ax.text(i, max(a, l) * 0.90, f'{"\u2212" if pct>0 else "+"}{abs(pct):.0f}%',
                ha='center', fontsize=9, fontweight='bold', color=color)
    ax.set_xticks(x2); ax.set_xticklabels(bibii_labels)
    ax.set_ylabel('Median strain (kcal/mol/nt)')
    ax.set_title('B. By BI/BII state')


    # Set generous y-limits so the tall bars and labels have headroom
    axes[0].set_ylim(0, 2.3)  # Raises Panel A ceiling from 2.0 to 2.3
    axes[1].set_ylim(0, 2.7)  # Raises Panel B ceiling to clear the BII bar (which is at ~2.3)

    plt.tight_layout()
    out_path = os.path.join(args.out, 'figure4_dna_strain.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
