#!/usr/bin/env python3
"""
figure3_cmap_comparison.py — Polished Figure 3: MechLib vs CHARMM36 CMAP
============================================================================
Usage:
  python figure3_cmap_comparison.py --csv features_lj_FINAL_CLEAN.csv \\
      --cmap cmap_grid.npz --out ./figures/
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mechlib_style import setup_style, COLORS

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


def load_cmap(npz_path, grid_idx=0):
    data = np.load(npz_path)
    grid = data[f'grid_{grid_idx}']
    n_phi, n_psi = grid.shape
    d_phi = 360.0 / n_phi
    d_psi = 360.0 / n_psi

    def lookup(phi_deg, psi_deg):
        p = ((phi_deg + 180.0) % 360.0) / d_phi
        q = ((psi_deg + 180.0) % 360.0) / d_psi
        i0, j0 = int(np.floor(p)) % n_phi, int(np.floor(q)) % n_psi
        i1, j1 = (i0 + 1) % n_phi, (j0 + 1) % n_psi
        fp, fq = p - np.floor(p), q - np.floor(q)
        v00, v10 = grid[i0, j0], grid[i1, j0]
        v01, v11 = grid[i0, j1], grid[i1, j1]
        v0 = v00 * (1 - fp) + v10 * fp
        v1 = v01 * (1 - fp) + v11 * fp
        return v0 * (1 - fq) + v1 * fq
    return grid, lookup


def build_mechlib_grid(df, bin_size=10, min_count=10):
    phi_bins = np.arange(-180, 180 + bin_size, bin_size)
    psi_bins = np.arange(-180, 180 + bin_size, bin_size)
    phi_c = phi_bins[:-1] + bin_size / 2
    psi_c = psi_bins[:-1] + bin_size / 2

    sub = df.copy()
    sub['pb'] = pd.cut(sub['phi_deg'], phi_bins, labels=False, right=False)
    sub['qb'] = pd.cut(sub['psi_deg'], psi_bins, labels=False, right=False)
    sub = sub.dropna(subset=['pb', 'qb'])
    sub['pb'] = sub['pb'].astype(int); sub['qb'] = sub['qb'].astype(int)

    geo_cols = [c for c in AMBER if c in sub.columns]
    grid_shape = (len(phi_c), len(psi_c))
    strain_grid = np.full(grid_shape, np.nan)
    count_grid = np.zeros(grid_shape, dtype=int)

    for (pb, qb), grp in sub.groupby(['pb', 'qb']):
        if len(grp) < min_count:
            continue
        total = 0.0
        for col in geo_cols:
            obs = grp[col].dropna().values
            if len(obs) == 0:
                continue
            k = AMBER[col]['k']
            conv = np.pi / 180.0 if AMBER[col]['unit'] == 'deg' else 1.0
            eq = AMBER[col]['eq']
            total += 0.5 * k * ((np.mean(obs) - eq) * conv) ** 2
        strain_grid[int(pb), int(qb)] = total
        count_grid[int(pb), int(qb)] = len(grp)
    return strain_grid, count_grid, phi_c, psi_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--cmap', required=True)
    ap.add_argument('--cmap_grid_idx', type=int, default=0)
    ap.add_argument('--out', default='./figures')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    setup_style()
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    print("Loading data...")
    df = pd.read_csv(args.csv, low_memory=False)
    cmap_grid, cmap_lookup = load_cmap(args.cmap, args.cmap_grid_idx)
    mechlib_grid, count_grid, phi_c, psi_c = build_mechlib_grid(df)

    cmap_on_grid = np.array([[cmap_lookup(p, q) for q in psi_c] for p in phi_c])
    valid = ~np.isnan(mechlib_grid) & (count_grid >= 10)
    x = cmap_on_grid[valid]
    y = mechlib_grid[valid]
    r, p = pearsonr(x, y)
    print(f"Pearson r = {r:.3f}, p = {p:.2e}, n = {valid.sum()}")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    im0 = axes[0].imshow(cmap_on_grid.T, origin='lower', extent=[-180, 180, -180, 180],
                          cmap='RdBu_r', aspect='auto')
    axes[0].set_title('A. CHARMM36 CMAP energy')
    axes[0].set_xlabel('φ (°)'); axes[0].set_ylabel('ψ (°)')
    plt.colorbar(im0, ax=axes[0], label='kcal/mol')

    im1 = axes[1].imshow(mechlib_grid.T, origin='lower', extent=[-180, 180, -180, 180],
                          cmap='RdBu_r', aspect='auto')
    axes[1].set_title('B. MechLib geometric strain')
    axes[1].set_xlabel('φ (°)'); axes[1].set_ylabel('ψ (°)')
    plt.colorbar(im1, ax=axes[1], label='kcal/mol')

    axes[2].scatter(x, y, s=10, alpha=0.45, color=COLORS['mechlib'],
                     edgecolor='none')
    axes[2].set_xlabel('CHARMM36 CMAP energy (kcal/mol)')
    axes[2].set_ylabel('MechLib strain (kcal/mol)')
    axes[2].set_title(f'C. Correlation (r = {r:.2f}, p < 0.001)')

    plt.tight_layout()
    out_path = os.path.join(args.out, 'figure3_cmap_comparison.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
