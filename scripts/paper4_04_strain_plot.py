#!/usr/bin/env python3
"""
Generate Figure 2 for Paper 4: Per-SS strain energy comparison.
Uses the strain data already computed by paper4_05_fair_benchmarks.py.

Usage:
  python paper4_06_strain_plot.py \
      --csv /mnt/f/Protein_Folding/v8_g/p3.csv \
      --out ./paper4_fair/
"""

import argparse
import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

AMBER = {
    'tau_deg':        {'eq': 111.1, 'k': 63.0,  'unit': 'deg'},
    'angle_N_CA_CB':  {'eq': 110.1, 'k': 63.0,  'unit': 'deg'},
    'angle_C_CA_CB':  {'eq': 110.1, 'k': 63.0,  'unit': 'deg'},
    'angle_CaCN':     {'eq': 116.6, 'k': 70.0,  'unit': 'deg'},
    'angle_CNCa':     {'eq': 121.9, 'k': 50.0,  'unit': 'deg'},
    'angle_CA_C_O':   {'eq': 120.4, 'k': 80.0,  'unit': 'deg'},
    'bond_N_CA':      {'eq': 1.458, 'k': 337.0, 'unit': 'Å'},
    'bond_CA_C':      {'eq': 1.522, 'k': 317.0, 'unit': 'Å'},
    'bond_C_O':       {'eq': 1.229, 'k': 570.0, 'unit': 'Å'},
    'bond_C_N_next':  {'eq': 1.335, 'k': 490.0, 'unit': 'Å'},
    'bond_CA_CB':     {'eq': 1.526, 'k': 317.0, 'unit': 'Å'},
}


def build_library(df, bin_size=10, min_count=10):
    phi_bins = np.arange(-180, 180 + bin_size, bin_size)
    psi_bins = np.arange(-180, 180 + bin_size, bin_size)
    phi_c = phi_bins[:-1] + bin_size / 2
    psi_c = psi_bins[:-1] + bin_size / 2

    sub = df.copy()
    sub['pb'] = pd.cut(sub['phi_deg'], phi_bins, labels=False, right=False)
    sub['qb'] = pd.cut(sub['psi_deg'], psi_bins, labels=False, right=False)
    sub = sub.dropna(subset=['pb', 'qb'])
    sub['pb'] = sub['pb'].astype(int)
    sub['qb'] = sub['qb'].astype(int)

    geo_cols = [c for c in AMBER if c in sub.columns]
    lib = {}
    for gn in list(sub['res_name'].unique()) + ['ALL']:
        grp = sub if gn == 'ALL' else sub[sub['res_name'] == gn]
        cm = grp.groupby(['pb', 'qb'])[geo_cols].mean()
        lib[gn] = {}
        for (pb, qb), row in cm.iterrows():
            pk = str(int(phi_c[int(pb)]))
            qk = str(int(psi_c[int(qb)]))
            if pk not in lib[gn]: lib[gn][pk] = {}
            lib[gn][pk][qk] = {c: float(row[c]) for c in geo_cols if not np.isnan(row[c])}
    return lib


def lookup(lib, phi, psi, res, param, bin_size=10):
    half = bin_size / 2.0
    centers = np.arange(-180 + half, 180 + half, bin_size)
    def bk(a):
        a = ((a + 180) % 360) - 180
        i = int(np.round((a - centers[0]) / bin_size))
        return str(int(centers[max(0, min(i, len(centers)-1))]))
    pk, qk = bk(phi), bk(psi)
    for cls in [res, 'ALL']:
        if cls in lib:
            cell = lib[cls].get(pk, {}).get(qk)
            if cell and param in cell:
                return cell[param]
    return AMBER[param]['eq']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='./paper4_fair')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Reading CSV...")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"  {len(df):,} rows")

    print("Building library...")
    lib = build_library(df)

    print("Computing strain...")
    phi = df['phi_deg'].values
    psi = df['psi_deg'].values
    res = df['res_name'].values
    geo_cols = [c for c in AMBER if c in df.columns]

    strain_amber = np.zeros(len(df))
    strain_lib = np.zeros(len(df))

    for col in geo_cols:
        obs = df[col].values
        valid = ~np.isnan(obs)
        k = AMBER[col]['k']
        conv = np.pi / 180.0 if AMBER[col]['unit'] == 'deg' else 1.0
        amber_eq = AMBER[col]['eq']

        ea = 0.5 * k * ((obs - amber_eq) * conv) ** 2
        ea[~valid] = 0.0
        strain_amber += ea

        lib_eq = np.array([
            lookup(lib, phi[i], psi[i], res[i], col)
            if valid[i] and not np.isnan(phi[i]) and not np.isnan(psi[i])
            else amber_eq for i in range(len(df))
        ])
        el = 0.5 * k * ((obs - lib_eq) * conv) ** 2
        el[~valid] = 0.0
        strain_lib += el

    print("Generating plots...")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ss_labels = {0: '\u03B1R', 1: '\u03B2', 2: 'PPII', 3: '3\u2081\u2080', 4: 'coil', 5: '\u03B1L'}

    # ── Figure 2A: Per-SS strain comparison ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Grouped bar chart
    ax = axes[0]
    ss_order = [0, 1, 2, 4, 5]  # αR, β, PPII, coil, αL
    labels, amber_vals, lib_vals = [], [], []

    for ss in ss_order:
        mask = df['ss_bin'].values == ss
        if mask.sum() < 100:
            continue
        labels.append(ss_labels[ss])
        amber_vals.append(strain_amber[mask].mean())
        lib_vals.append(strain_lib[mask].mean())

    x = np.arange(len(labels))
    w = 0.35
    bars_a = ax.bar(x - w/2, amber_vals, w, label='AMBER ff14SB',
                    color='#B2182B', edgecolor='#333333', linewidth=0.5)
    bars_l = ax.bar(x + w/2, lib_vals, w, label='Library',
                    color='#2166AC', edgecolor='#333333', linewidth=0.5)

    # Add improvement % labels
    for i, (a, l) in enumerate(zip(amber_vals, lib_vals)):
        pct = (a - l) / a * 100
        ax.text(i, max(a, l) + 0.008, f'\u2212{pct:.0f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='#2166AC')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Mean strain energy (kcal/mol/residue)', fontsize=11)
    ax.set_title('A. Phantom strain by secondary structure', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(0, max(amber_vals) * 1.25)

    # Panel B: Per-observable strain
    ax = axes[1]
    obs_labels = ['\u03C4', '\u2220NCaCb', '\u2220CCaCb', '\u2220CaCN',
                  '\u2220CNCa', '\u2220CaCO',
                  'N-Ca', 'Ca-C', 'C=O', 'C-N', 'Ca-Cb']
    obs_cols = ['tau_deg', 'angle_N_CA_CB', 'angle_C_CA_CB', 'angle_CaCN',
                'angle_CNCa', 'angle_CA_C_O',
                'bond_N_CA', 'bond_CA_C', 'bond_C_O', 'bond_C_N_next', 'bond_CA_CB']

    amber_obs, lib_obs, pct_obs = [], [], []
    valid_labels = []
    for col, lbl in zip(obs_cols, obs_labels):
        if col not in df.columns:
            continue
        obs = df[col].values
        valid = ~np.isnan(obs)
        k = AMBER[col]['k']
        conv = np.pi / 180.0 if AMBER[col]['unit'] == 'deg' else 1.0

        ea = np.mean(0.5 * k * ((obs[valid] - AMBER[col]['eq']) * conv) ** 2)
        lib_eq_arr = np.array([
            lookup(lib, phi[i], psi[i], res[i], col)
            if not np.isnan(phi[i]) and not np.isnan(psi[i])
            else AMBER[col]['eq'] for i in range(len(df)) if valid[i]
        ])
        el = np.mean(0.5 * k * ((obs[valid] - lib_eq_arr) * conv) ** 2)

        amber_obs.append(ea)
        lib_obs.append(el)
        pct_obs.append((ea - el) / ea * 100)
        valid_labels.append(lbl)

    x2 = np.arange(len(valid_labels))
    ax.bar(x2 - w/2, amber_obs, w, label='AMBER', color='#B2182B',
           edgecolor='#333333', linewidth=0.5)
    ax.bar(x2 + w/2, lib_obs, w, label='Library', color='#2166AC',
           edgecolor='#333333', linewidth=0.5)

    for i, pct in enumerate(pct_obs):
        y = max(amber_obs[i], lib_obs[i])
        ax.text(i, y + 0.001, f'\u2212{pct:.0f}%', ha='center', va='bottom',
                fontsize=7, fontweight='bold', color='#2166AC')

    ax.set_xticks(x2)
    ax.set_xticklabels(valid_labels, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('Mean strain (kcal/mol/residue)', fontsize=11)
    ax.set_title('B. Strain by observable', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'strain_comparison.png'), dpi=200)
    plt.close()
    print(f"  Saved strain_comparison.png")

    # ── Figure 3: Improvement by residue type ────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    aa_order = ['GLY', 'PRO', 'ILE', 'VAL', 'ASN', 'ASP', 'HIS', 'THR',
                'CYS', 'PHE', 'LEU', 'TYR', 'LYS', 'SER', 'ARG', 'GLN',
                'GLU', 'TRP', 'MET', 'ALA']
    pct_by_aa = []
    valid_aa = []

    for aa in aa_order:
        mask = res == aa
        if mask.sum() < 100:
            continue
        ea = strain_amber[mask].mean()
        el = strain_lib[mask].mean()
        pct_by_aa.append((ea - el) / ea * 100)
        valid_aa.append(aa)

    colors = []
    for aa in valid_aa:
        if aa in ('VAL', 'ILE', 'THR'):
            colors.append('#EF6548')
        elif aa == 'GLY':
            colors.append('#41AB5D')
        elif aa == 'PRO':
            colors.append('#807DBA')
        else:
            colors.append('#4292C6')

    x3 = np.arange(len(valid_aa))
    ax.bar(x3, pct_by_aa, color=colors, edgecolor='#333333', linewidth=0.5)
    ax.set_xticks(x3)
    ax.set_xticklabels(valid_aa, fontsize=9)
    ax.set_ylabel('Strain reduction (%)', fontsize=11)
    ax.set_title('Total strain reduction by amino acid type\n'
                 '(red = \u03B2-branched, green = Gly, purple = Pro)',
                 fontsize=11)
    ax.axhline(31.5, color='gray', ls='--', lw=1, alpha=0.5, label='Average (31.5%)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'strain_by_residue.png'), dpi=200)
    plt.close()
    print(f"  Saved strain_by_residue.png")

    print("Done!")


if __name__ == '__main__':
    main()