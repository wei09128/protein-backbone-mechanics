#!/usr/bin/env python3
"""
paper1_fig6_plot.py — Generate the actual Figure 6 image
=============================================================
paper1_minor3_bootstrap_ci.py computes the corrected, structure-level
bootstrap CIs for Fig 6 but only prints them to console -- it never
produces a plot. This script reuses the identical bootstrap logic and
renders the actual bar chart with proper CI error bars (replacing the
original figure's naive SEM bars), using the paper's own established
color palette (paper1_style.py).

Usage:
  python paper1_fig6_plot.py --csv features_lj_v3_FINAL_CLEAN.csv
"""

import argparse
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from paper1_style import COLORS, setup_style

PHI_REF, PSI_REF = -63.0, -43.0
MIN_DTHETA_DEG = 5.0

_SC_N_ROT = {
    'GLY': 0, 'ALA': 0, 'VAL': 1, 'LEU': 2, 'ILE': 2, 'PRO': 0,
    'PHE': 2, 'TYR': 2, 'TRP': 2, 'SER': 1, 'THR': 1, 'CYS': 1,
    'MET': 3, 'ASP': 2, 'ASN': 2, 'GLU': 3, 'GLN': 3, 'LYS': 4, 'ARG': 4, 'HIS': 2,
}


def wrap180(x):
    return (x + 180) % 360 - 180


def bootstrap_by_structure(values, pdb_ids, n_boot=2000, statistic=np.mean, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'v': values, 'pdb_id': pdb_ids})
    groups = df.groupby('pdb_id')['v'].apply(list).to_dict()
    structs = np.array(list(groups.keys()))
    n_struct = len(structs)
    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(structs, size=n_struct, replace=True)
        vals = np.concatenate([groups[s] for s in sampled])
        boot_stats[b] = statistic(vals)
    return statistic(values), np.percentile(boot_stats, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--out_dir', default='.')
    args = ap.parse_args()
    setup_style()
    import matplotlib.pyplot as plt

    print("Loading data...")
    df = pd.read_csv(args.csv, low_memory=False)
    df = df.dropna(subset=['phi_deg', 'psi_deg', 'tau_phi_correct', 'tau_psi_correct', 'res_name'])
    print(f"  {len(df):,} residues, {df['pdb_id'].nunique()} structures")

    dphi_deg = wrap180(df['phi_deg'].values - PHI_REF)
    dpsi_deg = wrap180(df['psi_deg'].values - PSI_REF)
    dphi_rad = np.radians(dphi_deg)
    dpsi_rad = np.radians(dpsi_deg)
    valid_phi = np.abs(dphi_deg) > MIN_DTHETA_DEG
    valid_psi = np.abs(dpsi_deg) > MIN_DTHETA_DEG

    df['k_phi'] = np.nan
    df['k_psi'] = np.nan
    df.loc[valid_phi, 'k_phi'] = np.clip(
        -df.loc[valid_phi, 'tau_phi_correct'].values / dphi_rad[valid_phi], -50, 50)
    df.loc[valid_psi, 'k_psi'] = np.clip(
        -df.loc[valid_psi, 'tau_psi_correct'].values / dpsi_rad[valid_psi], -50, 50)
    df['n_rot'] = df['res_name'].map(lambda a: _SC_N_ROT.get(a, np.nan))

    ex_labels = {0: 'GLY/ALA/PRO', 1: 'VAL/SER/THR', 2: 'LEU/PHE/ASP',
                 3: 'MET/GLU/GLN', 4: 'LYS/ARG'}

    print("Computing bootstrap CIs...")
    results = {'k_phi': [], 'k_psi': []}
    for nr in range(5):
        sub = df[df['n_rot'] == nr]
        for label, col in [('k_phi', 'k_phi'), ('k_psi', 'k_psi')]:
            vals = sub[col].dropna()
            ids = sub.loc[vals.index, 'pdb_id']
            if len(vals) < 10:
                results[label].append((nr, np.nan, np.nan, np.nan, 0))
                continue
            mean_abs, (ci_lo, ci_hi) = bootstrap_by_structure(
                vals.abs().values, ids.values, args.n_boot, np.mean)
            results[label].append((nr, mean_abs, ci_lo, ci_hi, len(vals)))
            print(f"  n_rot={nr} {label}: mean|k|={mean_abs:.3f} CI=[{ci_lo:.3f},{ci_hi:.3f}]")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, label, color, title in [
        (axes[0], 'k_phi', COLORS['green'], 'k_φ'),
        (axes[1], 'k_psi', COLORS['blue'], 'k_ψ'),
    ]:
        rows = results[label]
        x = [r[0] for r in rows]
        means = [r[1] for r in rows]
        err_lo = [r[1] - r[2] for r in rows]
        err_hi = [r[3] - r[1] for r in rows]
        ax.bar(x, means, color=color, edgecolor='#333333', linewidth=0.6,
               yerr=[err_lo, err_hi], capsize=4,
               error_kw={'elinewidth': 1.3, 'ecolor': '#222222'})
        ax.set_xticks(x)
        ax.set_xticklabels([f"{ex_labels[i]}\n({i}-chi)" for i in x], fontsize=8)
        ax.set_ylabel(f'Mean |{title}| (kcal/mol/rad2)')
        ax.set_title(f'{title} by rotatable-bond count\n(bootstrap 95% CI, structure-resampled)',
                     fontsize=10)
        ax.set_ylim(0, 3.0)

    plt.tight_layout()
    out_path = f"{args.out_dir}/figure6_corrected.png"
    plt.savefig(out_path)
    plt.close()
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
