#!/usr/bin/env python3
"""
figure8_dna_ridge_plot.py — Ridge plot: DNA strain-reduction distribution by base
=====================================================================================
Shows that MechLib-DNA's improvement over AMBER parm10 is a consistent
distributional narrowing across all four bases, not merely a mean shift.

Usage:
  python figure8_dna_ridge_plot.py --csv dna_features_v3_clean.csv --out ./figures/
"""

import argparse
import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mechlib_style import setup_style, COLORS

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
    from scipy.stats import gaussian_kde

    print("Loading CSV...")
    df = pd.read_csv(args.csv, low_memory=False)
    df = df[df['res_name'].isin(['DA', 'DC', 'DG', 'DT'])].copy()

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

    sub = sub.copy()
    sub['strain_amber'] = strain_amber
    sub['strain_lib'] = strain_lib
    valid_mask = sub['strain_amber'] > 0.01
    sub.loc[valid_mask, 'pct_reduction'] = 100 * (
        (sub.loc[valid_mask, 'strain_amber'] - sub.loc[valid_mask, 'strain_lib'])
        / sub.loc[valid_mask, 'strain_amber']
    )

    bases = ['DA', 'DC', 'DG', 'DT']
    fig, ax = plt.subplots(figsize=(9, 6))

    y_offset = 0
    colors_by_base = {'DA': COLORS['amber'], 'DC': COLORS['opls'],
                       'DG': COLORS['charmm'], 'DT': COLORS['mechlib']}

    for base in bases:
        vals = sub[(sub['res_name'] == base)]['pct_reduction'].dropna()
        vals = vals[(vals > -100) & (vals < 100)]
        if len(vals) < 50:
            continue
        kde = gaussian_kde(vals)
        x = np.linspace(-100, 100, 400)
        y = kde(x)
        y = y / y.max() * 3.0
        ax.fill_between(x, y_offset, y + y_offset, alpha=0.7,
                         color=colors_by_base[base], label=base)
        ax.plot(x, y + y_offset, color='black', linewidth=0.8)
        ax.text(102, y_offset + 0.3, base, fontsize=11, fontweight='bold')
        y_offset += 1.3

    ax.set_xlabel('Per-nucleotide strain reduction (%)')
    ax.set_yticks([])
    ax.set_xlim(-100, 115)
    ax.set_title('DNA strain-reduction distribution by base\n(ridge plot)')

    plt.tight_layout()
    out_path = os.path.join(args.out, 'figure8_dna_ridge_plot.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
