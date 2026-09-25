#!/usr/bin/env python3
"""
figure1_protein_amber_mechlib_CORRECTED.py — Figure 1 with verified AMBER params
====================================================================================
CORRECTED 2026-09-11 relative to the originally-submitted
paper4_figure1_protein_amber_mechlib.py. Two errors found and fixed, both
verified directly against /home/wei/miniconda3/envs/amber/dat/leap/parm/parm10.dat
(AmberTools 24.8) -- the primary AMBER ff14SB parameter source, not a
derived/secondary file:

  1. angle_N_CA_CB's force constant was WRONG in the original script: 63.0,
     should be 80.0. parm10.dat line 415:
         CT-CX-N     80.0      109.70    AA ala, general
     k=80.0 kcal/mol/rad^2, read directly -- no unit conversion needed.
     This is a NEW finding, distinct from the equilibrium-value fix below,
     and affects the strain calculation for this term regardless of which
     equilibrium value is used.

  2. angle_C_CA_CB's equilibrium value in the ORIGINAL script (111.10) was
     already correct and is UNCHANGED here -- flagging this explicitly so
     it's clear this fix touches only the k value, not the equilibrium
     angle, for this particular term.

Everything else (tau, N_CA_CB eq, bond_N_CA, angle_CaCN, angle_CNCa,
angle_CA_C_O, all other bond constants) was already correct in the original
script and is unchanged here.

Run this alongside (not instead of) the original to see exactly how much
the corrected N-Ca-Cb force constant shifts the headline number -- the
printed "Overall reduction" from each run is directly comparable.

Usage:
  python figure1_protein_amber_mechlib_CORRECTED.py --csv features_lj_FINAL_CLEAN.csv --out ./figures/
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mechlib_style import setup_style, COLORS, panel_label, clean_observable_name

# CORRECTED: angle_N_CA_CB k changed from 63.0 -> 80.0 (parm10.dat line 415,
# verified directly). All other values unchanged from the original script.
AMBER = {
    'tau_deg':        {'eq': 110.10, 'k': 63.0,  'unit': 'deg'},
    'angle_N_CA_CB':  {'eq': 109.70, 'k': 80.0,  'unit': 'deg'},  # k: 63.0 -> 80.0
    'angle_C_CA_CB':  {'eq': 111.10, 'k': 63.0,  'unit': 'deg'},
    'angle_CaCN':     {'eq': 116.6,  'k': 70.0,  'unit': 'deg'},
    'angle_CNCa':     {'eq': 121.9,  'k': 50.0,  'unit': 'deg'},
    'angle_CA_C_O':   {'eq': 120.4,  'k': 80.0,  'unit': 'deg'},
    'bond_N_CA':      {'eq': 1.449,  'k': 337.0, 'unit': 'Å'},
    'bond_CA_C':      {'eq': 1.522,  'k': 317.0, 'unit': 'Å'},
    'bond_C_O':       {'eq': 1.229,  'k': 570.0, 'unit': 'Å'},
    'bond_C_N_next':  {'eq': 1.335,  'k': 490.0, 'unit': 'Å'},
    'bond_CA_CB':     {'eq': 1.526,  'k': 317.0, 'unit': 'Å'},
}

SS_LABELS = {0: 'αR', 1: 'β', 2: 'PPII', 3: '3₁₀', 4: 'coil', 5: 'αL'}
SS_ORDER = [0, 1, 2, 4, 5]


def build_library(df, bin_size=10, min_count=10):
    phi_bins = np.arange(-180, 180 + bin_size, bin_size)
    psi_bins = np.arange(-180, 180 + bin_size, bin_size)
    sub = df.copy()
    sub['pb'] = pd.cut(sub['phi_deg'], phi_bins, labels=False, right=False)
    sub['qb'] = pd.cut(sub['psi_deg'], psi_bins, labels=False, right=False)
    sub = sub.dropna(subset=['pb', 'qb'])
    sub['pb'] = sub['pb'].astype(int)
    sub['qb'] = sub['qb'].astype(int)
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='./figures')
    ap.add_argument('--min_count', type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    setup_style()

    print("Loading CSV...")
    df = pd.read_csv(args.csv, low_memory=False)
    sub = build_library(df)
    geo_cols = [c for c in AMBER if c in sub.columns]

    strain_amber = {}
    strain_lib = {}
    ss_amber = {ss: [] for ss in SS_ORDER}
    ss_lib = {ss: [] for ss in SS_ORDER}
    aa_amber = {}
    aa_lib = {}

    for col in geo_cols:
        k = AMBER[col]['k']
        conv = np.pi / 180.0 if AMBER[col]['unit'] == 'deg' else 1.0
        eq = AMBER[col]['eq']

        valid = sub[col].notna()
        sv = sub.loc[valid]
        lib_mean = sv.groupby(['res_name', 'pb', 'qb'])[col].transform('mean')
        counts = sv.groupby(['res_name', 'pb', 'qb'])[col].transform('count')
        pooled_mean = sv.groupby(['pb', 'qb'])[col].transform('mean')
        lib_eq = np.where(counts >= args.min_count, lib_mean, pooled_mean)

        e_amber = 0.5 * k * ((sv[col] - eq) * conv) ** 2
        e_lib = 0.5 * k * ((sv[col] - lib_eq) * conv) ** 2

        strain_amber[col] = e_amber.mean()
        strain_lib[col] = e_lib.mean()

        sub.loc[valid, f'_ea_{col}'] = e_amber.values
        sub.loc[valid, f'_el_{col}'] = e_lib.values

        if col == 'angle_N_CA_CB':
            print(f"  [corrected term] angle_N_CA_CB: mean AMBER strain={e_amber.mean():.5f}, "
                  f"mean MechLib strain={e_lib.mean():.5f}  (k=80.0, was 63.0 in original)")

    ea_cols = [f'_ea_{c}' for c in geo_cols]
    el_cols = [f'_el_{c}' for c in geo_cols]
    sub['total_ea'] = sub[ea_cols].sum(axis=1, skipna=True)
    sub['total_el'] = sub[el_cols].sum(axis=1, skipna=True)

    for ss in SS_ORDER:
        mask = sub['ss_bin'] == ss
        if mask.sum() < 100:
            continue
        ss_amber[ss] = sub.loc[mask, 'total_ea'].mean()
        ss_lib[ss] = sub.loc[mask, 'total_el'].mean()

    aa_order = ['GLY', 'PRO', 'ILE', 'VAL', 'ASN', 'ASP', 'HIS', 'THR',
                'CYS', 'PHE', 'LEU', 'TYR', 'LYS', 'SER', 'ARG', 'GLN',
                'GLU', 'TRP', 'MET', 'ALA']
    for aa in aa_order:
        mask = sub['res_name'] == aa
        if mask.sum() < 100:
            continue
        aa_amber[aa] = sub.loc[mask, 'total_ea'].mean()
        aa_lib[aa] = sub.loc[mask, 'total_el'].mean()

    overall_reduction = 100 * (sub['total_ea'].mean() - sub['total_el'].mean()) / sub['total_ea'].mean()
    print(f"\nOverall reduction (CORRECTED k for angle_N_CA_CB): {overall_reduction:.1f}%")
    print("Compare this to the original script's printed value to see the exact shift.")

    # ── Figure (identical layout to the original script) ────────────────────
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    ax = axes[0]
    labels = [SS_LABELS[ss] for ss in SS_ORDER if ss in ss_amber]
    a_vals = [ss_amber[ss] for ss in SS_ORDER if ss in ss_amber]
    l_vals = [ss_lib[ss] for ss in SS_ORDER if ss in ss_amber]
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x - w/2, a_vals, w, label='AMBER ff14SB (corrected)', color=COLORS['amber'],
           edgecolor='#333333', linewidth=0.5)
    ax.bar(x + w/2, l_vals, w, label='MechLib', color=COLORS['mechlib'],
           edgecolor='#333333', linewidth=0.5)
    for i, (a, l) in enumerate(zip(a_vals, l_vals)):
        pct = (a - l) / a * 100
        ax.text(i, max(a, l) * 1.04, f'\u2212{pct:.0f}%', ha='center',
                fontsize=9, fontweight='bold', color=COLORS['mechlib'])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Mean strain (kcal/mol/residue)')
    ax.set_title('A. By secondary structure (corrected)')
    ax.legend(frameon=False)

    ax = axes[1]
    labels_b = [clean_observable_name(c) for c in geo_cols]
    a_vals_b = [strain_amber[c] for c in geo_cols]
    l_vals_b = [strain_lib[c] for c in geo_cols]
    x2 = np.arange(len(labels_b))
    ax.bar(x2 - w/2, a_vals_b, w, color=COLORS['amber'], edgecolor='#333333', linewidth=0.5)
    ax.bar(x2 + w/2, l_vals_b, w, color=COLORS['mechlib'], edgecolor='#333333', linewidth=0.5)
    ax.set_xticks(x2); ax.set_xticklabels(labels_b, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean strain (kcal/mol/residue)')
    ax.set_title('B. By observable (corrected)')

    ax = axes[2]
    branched = {'VAL', 'ILE', 'THR'}
    colors_c = []
    aas = list(aa_amber.keys())
    pct_c = [(aa_amber[aa] - aa_lib[aa]) / aa_amber[aa] * 100 for aa in aas]
    order_idx = np.argsort(pct_c)[::-1]
    aas_sorted = [aas[i] for i in order_idx]
    pct_sorted = [pct_c[i] for i in order_idx]
    for aa in aas_sorted:
        if aa in branched: colors_c.append('#EF6548')
        elif aa == 'GLY': colors_c.append(COLORS['opls'])
        elif aa == 'PRO': colors_c.append(COLORS['charmm'])
        else: colors_c.append(COLORS['mechlib'])
    x3 = np.arange(len(aas_sorted))
    ax.bar(x3, pct_sorted, color=colors_c, edgecolor='#333333', linewidth=0.5)
    ax.set_xticks(x3); ax.set_xticklabels(aas_sorted, fontsize=8)
    ax.set_ylabel('Strain reduction (%)')
    ax.axhline(overall_reduction, color='gray', ls='--', lw=1,
               label=f'Overall ({overall_reduction:.1f}%)')
    ax.set_title('C. By residue type (corrected)')
    ax.legend(frameon=False, fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(args.out, 'figure1_protein_amber_mechlib_CORRECTED.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
