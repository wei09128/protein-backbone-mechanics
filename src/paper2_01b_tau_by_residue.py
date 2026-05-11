"""
paper2_01b_tau_by_residue.py — Per-residue τ(φ,ψ) decomposition
==================================================================

The main τ-map (paper2_01_tau_map.py) showed a huge signal:
  α-helix:  Δτ = +0.47° (expanded)
  β-sheet:  Δτ = −1.58° (compressed)
  peak-to-peak: 8.4°
  signal/noise: 3.56 over 2.17 M residues.

But the normalization subtracts each residue type's MEDIAN τ, which
doesn't fully control for composition if a residue's τ shifts differently
in different (φ,ψ) regions. E.g., VAL has narrow baseline τ (~109.8°)
and is enriched in β — if VAL's τ narrows FURTHER in β (vs its own
baseline), the global signal might partly be "VAL in β" not a universal
mechanical response.

This script tests the null: "if the effect is universal, every residue
type should show the same sign of Δτ in each canonical Ramachandran
region." We report:
  - For each of 20 residues, Δτ in α / β / PPII / αL (vs that residue's median)
  - Agreement table: how many residues show α expanded, β compressed, etc.
  - A 4x5 grid figure saved to disk

If 18+ of 20 agree → signal is universal → paper 2 is a real mechanical
finding. If only ~10 agree → composition confound, needs more work.

Usage
-----
    python paper2_01b_tau_by_residue.py --csv features.csv
    python paper2_01b_tau_by_residue.py --csv features.csv --out tau_per_residue.png
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


_RES20 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','HIS','ILE','LEU',
          'LYS','MET','PHE','SER','THR','TRP','TYR','VAL']  # 18 std (excl GLY/PRO)

# Canonical Ramachandran regions (same as paper2_01_tau_map.py)
_REGIONS = {
    'α':    {'phi': (-80, -40),  'psi': (-60, -20)},
    'β':    {'phi': (-150, -90), 'psi': (100, 160)},
    'PPII': {'phi': (-90, -60),  'psi': (120, 160)},
    'αL':   {'phi': (40, 80),    'psi': (20, 80)},
}

BIN_WIDTH = 10.0
MIN_BIN_COUNT = 20


def region_mean(df, region, value_col='delta_tau'):
    """Return (mean, sem, n) of value_col in a Ramachandran region."""
    p, q = _REGIONS[region]['phi'], _REGIONS[region]['psi']
    m = (df['phi_deg'].between(p[0], p[1])
         & df['psi_deg'].between(q[0], q[1]))
    n = int(m.sum())
    if n == 0:
        return float('nan'), float('nan'), 0
    v = df.loc[m, value_col]
    return float(v.mean()), float(v.std() / np.sqrt(n)), n


def compute_heatmap(df, value_col='delta_tau', bin_width=BIN_WIDTH,
                    min_count=MIN_BIN_COUNT):
    """Bin into (φ,ψ) cells and compute mean value_col per cell."""
    phi_edges = np.arange(-180, 181, bin_width)
    psi_edges = np.arange(-180, 181, bin_width)
    phi = df['phi_deg'].values
    psi = df['psi_deg'].values
    val = df[value_col].values
    pi = np.clip(np.digitize(phi, phi_edges) - 1, 0, len(phi_edges)-2)
    qi = np.clip(np.digitize(psi, psi_edges) - 1, 0, len(psi_edges)-2)
    n_p, n_q = len(phi_edges)-1, len(psi_edges)-1
    grid = np.full((n_q, n_p), np.nan)
    for i in range(n_p):
        for j in range(n_q):
            mask = (pi == i) & (qi == j)
            if mask.sum() >= min_count:
                grid[j, i] = val[mask].mean()
    return phi_edges, psi_edges, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='paper2_01b_tau_per_residue.png')
    ap.add_argument('--color_limit', type=float, default=2.0)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found"); sys.exit(1)

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['tau_deg', 'phi_deg', 'psi_deg', 'res_name'])
    df = df[df['res_name'].isin(_RES20)].copy()
    print(f"  {len(df):,} residues (18 standard types, GLY/PRO excluded)")

    # Per-residue median-normalize
    medians = df.groupby('res_name')['tau_deg'].median()
    df['delta_tau'] = df['tau_deg'] - df['res_name'].map(medians)

    # ── Table: Δτ per residue × region ────────────────────────────────────────
    print("\n" + "=" * 78)
    print("Δτ in canonical regions, per residue (vs that residue's median τ)")
    print("=" * 78)
    print(f"  {'res':<4} {'median':>8}  "
          f"{'α Δτ':>12} {'β Δτ':>12} {'PPII Δτ':>12} {'αL Δτ':>12}")
    print("  " + "-" * 76)

    results = {}  # res -> dict(region -> (mean, sem, n))
    for res in _RES20:
        sub = df[df['res_name'] == res]
        row = {}
        for reg in _REGIONS:
            row[reg] = region_mean(sub, reg)
        results[res] = row

        def _fmt(r):
            m, sem, n = r
            if n < 30:
                return f"   n={n:>4}   "
            return f"{m:+6.3f}±{sem:.3f}"

        print(f"  {res:<4} {medians[res]:>8.3f}  "
              f"{_fmt(row['α']):>12} {_fmt(row['β']):>12} "
              f"{_fmt(row['PPII']):>12} {_fmt(row['αL']):>12}")

    # ── Agreement summary ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SIGN AGREEMENT SUMMARY")
    print("=" * 78)
    # Expected signs from main map: α positive, β negative, PPII negative, αL positive
    expected = {'α': +1, 'β': -1, 'PPII': -1, 'αL': +1}

    for reg, exp in expected.items():
        agree = 0; disagree = 0; nil = 0
        disagree_list = []
        for res in _RES20:
            m, sem, n = results[res][reg]
            if n < 30 or np.isnan(m):
                nil += 1; continue
            if m * exp > 0:
                agree += 1
            else:
                disagree += 1
                disagree_list.append(f"{res}({m:+.2f})")
        sign_word = "expanded (+)" if exp > 0 else "compressed (−)"
        print(f"  {reg:<4} expected {sign_word:<15}  "
              f"agree: {agree:>2}/{len(_RES20)}, "
              f"disagree: {disagree:>2}, n_too_small: {nil}")
        if disagree_list and disagree <= 6:
            print(f"        disagreeing residues: {', '.join(disagree_list)}")

    # Overall verdict
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    agree_counts = []
    for reg, exp in expected.items():
        a = sum(1 for res in _RES20
                if results[res][reg][2] >= 30
                and not np.isnan(results[res][reg][0])
                and results[res][reg][0] * exp > 0)
        agree_counts.append(a)
    total_agree = sum(agree_counts)
    total_tests = sum(sum(1 for res in _RES20
                          if results[res][reg][2] >= 30
                          and not np.isnan(results[res][reg][0]))
                      for reg in expected)
    pct = 100 * total_agree / total_tests if total_tests else 0
    print(f"  Overall agreement: {total_agree}/{total_tests} tests "
          f"({pct:.1f}%)")
    if pct >= 85:
        print("  → STRONG: the τ deformation is a universal mechanical "
              "response, not a composition artifact.")
        print("    Paper 2's central claim is supported.")
    elif pct >= 70:
        print("  → MODERATE: signal is mostly universal but some residues "
              "behave differently.")
        print("    Worth investigating the outliers.")
    else:
        print("  → WEAK: signal is residue-specific. The global map may be "
              "dominated by composition.")
        print("    Paper 2 needs a different framing (maybe per-residue-class).")

    # ── Figure: 4×5 grid, one heatmap per residue type (+ 2 blank slots) ─────
    print(f"\nGenerating {args.out} ...")
    fig, axes = plt.subplots(4, 5, figsize=(18, 14),
                              gridspec_kw={'hspace': 0.35, 'wspace': 0.25})
    axes = axes.flatten()
    vlim = args.color_limit

    for k, res in enumerate(_RES20):
        ax = axes[k]
        sub = df[df['res_name'] == res]
        phi_e, psi_e, grid = compute_heatmap(sub)
        masked = np.ma.masked_invalid(grid)
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad(color='#dddddd')
        im = ax.pcolormesh(phi_e, psi_e, masked, cmap=cmap,
                            vmin=-vlim, vmax=vlim, shading='flat')
        ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
        ax.set_aspect('equal')
        ax.set_title(f'{res} (n={len(sub):,})', fontsize=9)
        if k % 5 == 0:
            ax.set_ylabel('ψ (°)', fontsize=8)
        if k >= 15:
            ax.set_xlabel('φ (°)', fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused panels (we have 18 residues in 20 slots)
    for k in range(len(_RES20), 20):
        axes[k].axis('off')

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
    cbar.set_label('Δτ (deg) = τ − median(τ | residue)')

    plt.suptitle('τ deformation across Ramachandran plane, per residue type',
                  fontsize=13, y=0.995)
    plt.savefig(args.out, dpi=180, bbox_inches='tight')
    print(f"Figure saved: {args.out}")


if __name__ == '__main__':
    main()