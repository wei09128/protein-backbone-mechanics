#!/usr/bin/env python3
"""
figure2_cross_forcefield.py — Polished Figure 2: AMBER vs OPLS-AA/M vs MechLib
=================================================================================
Usage:
  python figure2_cross_forcefield.py --csv features_lj_FINAL_CLEAN.csv --out ./figures/
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

AMBER = {
    'tau_deg':        {'eq': 110.10, 'k': 63.0,  'unit': 'deg'},
    'angle_N_CA_CB':  {'eq': 109.70, 'k': 80.0,  'unit': 'deg'},
    'angle_C_CA_CB':  {'eq': 111.10, 'k': 63.0,  'unit': 'deg'},
    'angle_CaCN':     {'eq': 116.6, 'k': 70.0,  'unit': 'deg'},
    'angle_CNCa':     {'eq': 121.9, 'k': 50.0,  'unit': 'deg'},
    'angle_CA_C_O':   {'eq': 120.4, 'k': 80.0,  'unit': 'deg'},
    'bond_N_CA':      {'eq': 1.449,  'k': 337.0, 'unit': 'Å'},
    'bond_CA_C':      {'eq': 1.522, 'k': 317.0, 'unit': 'Å'},
    'bond_C_O':       {'eq': 1.229, 'k': 570.0, 'unit': 'Å'},
    'bond_C_N_next':  {'eq': 1.335, 'k': 490.0, 'unit': 'Å'},
    'bond_CA_CB':     {'eq': 1.526, 'k': 317.0, 'unit': 'Å'},
}
OPLS_AAM = {
    'tau_deg':        {'eq': 110.10, 'k': 63.0,  'unit': 'deg'},
    'angle_N_CA_CB':  {'eq': 109.70, 'k': 80.0,  'unit': 'deg'},
    'angle_C_CA_CB':  {'eq': 111.10, 'k': 63.0,  'unit': 'deg'},
    'angle_CaCN':     {'eq': 116.60, 'k': 70.0,  'unit': 'deg'},
    'angle_CNCa':     {'eq': 121.90, 'k': 50.0,  'unit': 'deg'},
    'angle_CA_C_O':   {'eq': 120.40, 'k': 80.0,  'unit': 'deg'},
    'bond_N_CA':      {'eq': 1.449,  'k': 337.0, 'unit': 'Å'},
    'bond_CA_C':      {'eq': 1.522,  'k': 317.0, 'unit': 'Å'},
    'bond_C_O':       {'eq': 1.229,  'k': 570.0, 'unit': 'Å'},
    'bond_C_N_next':  {'eq': 1.335,  'k': 490.0, 'unit': 'Å'},
    'bond_CA_CB':     {'eq': 1.529,  'k': 268.0, 'unit': 'Å'},
}


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

    phi_bins = np.arange(-180, 190, 10)
    psi_bins = np.arange(-180, 190, 10)
    sub = df.copy()
    sub['pb'] = pd.cut(sub['phi_deg'], phi_bins, labels=False, right=False)
    sub['qb'] = pd.cut(sub['psi_deg'], psi_bins, labels=False, right=False)
    sub = sub.dropna(subset=['pb', 'qb'])
    sub['pb'] = sub['pb'].astype(int); sub['qb'] = sub['qb'].astype(int)

    geo_cols = [c for c in AMBER if c in sub.columns]
    amber_vals, opls_vals, lib_vals = {}, {}, {}

    for col in geo_cols:
        valid = sub[col].notna()
        sv = sub.loc[valid]
        lib_mean = sv.groupby(['res_name', 'pb', 'qb'])[col].transform('mean')
        counts = sv.groupby(['res_name', 'pb', 'qb'])[col].transform('count')
        pooled_mean = sv.groupby(['pb', 'qb'])[col].transform('mean')
        lib_eq = np.where(counts >= args.min_count, lib_mean, pooled_mean)

        conv = np.pi / 180.0 if AMBER[col]['unit'] == 'deg' else 1.0

        e_a = 0.5 * AMBER[col]['k'] * ((sv[col] - AMBER[col]['eq']) * conv) ** 2
        e_o = 0.5 * OPLS_AAM[col]['k'] * ((sv[col] - OPLS_AAM[col]['eq']) * conv) ** 2
        e_l = 0.5 * AMBER[col]['k'] * ((sv[col] - lib_eq) * conv) ** 2  # k held fixed to AMBER's for lib

        amber_vals[col] = e_a.mean()
        opls_vals[col] = e_o.mean()
        lib_vals[col] = e_l.mean()

    overall_a = np.mean(list(amber_vals.values()))
    overall_o = np.mean(list(opls_vals.values()))
    overall_l = np.mean(list(lib_vals.values()))
    print(f"Overall (unweighted mean of per-observable means): "
          f"AMBER={overall_a:.4f}  OPLS={overall_o:.4f}  MechLib={overall_l:.4f}")

    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [clean_observable_name(c) for c in geo_cols]
    x = np.arange(len(labels))
    w = 0.27

    ax.bar(x - w, [opls_vals[c] for c in geo_cols], w, label='OPLS-AA/M',
           color=COLORS['opls'], edgecolor='#333333', linewidth=0.5)
    ax.bar(x, [amber_vals[c] for c in geo_cols], w, label='AMBER ff14SB',
           color=COLORS['amber'], edgecolor='#333333', linewidth=0.5)
    ax.bar(x + w, [lib_vals[c] for c in geo_cols], w, label='MechLib',
           color=COLORS['mechlib'], edgecolor='#333333', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Mean strain (kcal/mol/residue)')
    ax.set_title('Protein backbone strain: cross-force-field comparison')
    ax.legend(frameon=False)

    plt.tight_layout()
    out_path = os.path.join(args.out, 'figure2_cross_forcefield.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
