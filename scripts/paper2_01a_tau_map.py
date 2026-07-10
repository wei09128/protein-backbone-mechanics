"""
paper2_01_tau_map.py — Ramachandran τ-deformation heatmap
===========================================================

Paper 2 centerpiece figure: does τ (∠N-Cα-C) deform systematically as a
function of backbone dihedrals (φ, ψ)?  If yes, this is direct evidence
that the peptide backbone obeys a mechanical force field — bond angles
flex in response to dihedral strain.

Design
------
* Load features_p2.csv
* Exclude GLY, PRO, and residues with NaN in τ/φ/ψ
* Subtract each residue's type-median τ to remove composition bias
  (β-sheets are enriched in VAL/ILE/THR which have wider baseline τ —
   without normalization this would fake a signal)
* Bin into 10° × 10° Ramachandran cells
* Plot mean Δτ per cell with a diverging colormap centered on 0

Usage
-----
    python paper2_01_tau_map.py --csv /path/to/features_p2.csv
    python paper2_01_tau_map.py --csv features_p2.csv --min_bin_count 30
    python paper2_01_tau_map.py --csv features_p2.csv --out tau_map.png
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────
BIN_WIDTH       = 10.0        # degrees (Ramachandran resolution)
MIN_BIN_COUNT   = 30          # cells with fewer residues are masked (grey)
EXCLUDE_RES     = {'GLY', 'PRO'}
COLOR_LIMIT_DEG = 1.5         # colormap ±range in degrees (tune on first run)


def load_and_filter(csv_path):
    """Load features_p2.csv and return a clean DataFrame."""
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    n0 = len(df)
    print(f"  total rows: {n0:,}")

    # Need τ, φ, ψ, res_name
    required = ['tau_deg', 'phi_deg', 'psi_deg', 'res_name']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ERROR: CSV missing required columns: {missing}")
        print(f"  available: {list(df.columns)[:10]}...")
        sys.exit(1)

    # Drop NaN in required columns
    df = df.dropna(subset=required)
    print(f"  after NaN drop:      {len(df):,}  ({n0-len(df):,} dropped)")

    # Exclude GLY/PRO from main analysis
    df_main = df[~df['res_name'].isin(EXCLUDE_RES)].copy()
    df_gly  = df[df['res_name'] == 'GLY'].copy()
    df_pro  = df[df['res_name'] == 'PRO'].copy()
    print(f"  main (non-GLY/PRO):  {len(df_main):,}")
    print(f"  GLY separately:      {len(df_gly):,}")
    print(f"  PRO separately:      {len(df_pro):,}")

    return df_main, df_gly, df_pro


def normalize_by_res_median(df):
    """Subtract each residue type's median τ → Δτ column."""
    medians = df.groupby('res_name')['tau_deg'].median()
    print(f"\nResidue-type τ medians (deg):")
    for r in sorted(medians.index):
        n_r = (df['res_name'] == r).sum()
        print(f"  {r}: {medians[r]:6.3f}   (n = {n_r:,})")

    df = df.copy()
    df['delta_tau'] = df['tau_deg'] - df['res_name'].map(medians)
    return df, medians


def compute_heatmap(df, phi_col='phi_deg', psi_col='psi_deg',
                     value_col='delta_tau', bin_width=BIN_WIDTH):
    """
    Bin residues into (φ, ψ) cells and compute mean value_col per cell.
    Returns (phi_edges, psi_edges, mean_grid, count_grid).
    """
    phi_edges = np.arange(-180, 181, bin_width)
    psi_edges = np.arange(-180, 181, bin_width)

    phi = df[phi_col].values
    psi = df[psi_col].values
    val = df[value_col].values

    phi_idx = np.clip(np.digitize(phi, phi_edges) - 1, 0, len(phi_edges) - 2)
    psi_idx = np.clip(np.digitize(psi, psi_edges) - 1, 0, len(psi_edges) - 2)

    n_phi = len(phi_edges) - 1
    n_psi = len(psi_edges) - 1
    mean_grid  = np.full((n_psi, n_phi), np.nan)
    count_grid = np.zeros((n_psi, n_phi), dtype=int)

    for i in range(n_phi):
        for j in range(n_psi):
            mask = (phi_idx == i) & (psi_idx == j)
            c = int(mask.sum())
            count_grid[j, i] = c
            if c >= MIN_BIN_COUNT:
                mean_grid[j, i] = val[mask].mean()

    return phi_edges, psi_edges, mean_grid, count_grid


def plot_heatmap(phi_edges, psi_edges, mean_grid, count_grid,
                  title, ax, vmin=-COLOR_LIMIT_DEG, vmax=COLOR_LIMIT_DEG):
    """Plot a single τ-deformation heatmap on `ax`."""
    # Masked array: NaN cells show as grey
    masked = np.ma.masked_invalid(mean_grid)

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color='#dddddd')

    im = ax.pcolormesh(phi_edges, psi_edges, masked,
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        shading='flat')

    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_yticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_xlabel('φ (deg)')
    ax.set_ylabel('ψ (deg)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.axhline(0, color='k', lw=0.3, alpha=0.3)
    ax.axvline(0, color='k', lw=0.3, alpha=0.3)

    # Annotate canonical regions
    ax.text(-63, -42, 'α', fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='k', alpha=0.6))
    ax.text(-120, 130, 'β', fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='k', alpha=0.6))
    ax.text(-75, 145, 'PPII', fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='k', alpha=0.6))

    return im


def summary_stats(mean_grid, count_grid, medians, df_main):
    """Print key numbers the reviewer will want to see."""
    valid = ~np.isnan(mean_grid)
    n_valid = int(valid.sum())
    total_res = int(count_grid[valid].sum())

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"  Bins with ≥{MIN_BIN_COUNT} residues: {n_valid} / {count_grid.size}")
    print(f"  Total residues in valid bins:   {total_res:,}")
    print(f"  Median residues per valid bin:  "
          f"{int(np.median(count_grid[valid])):,}")

    print(f"\n  Global Δτ statistics (valid bins):")
    print(f"    mean:   {np.nanmean(mean_grid):+.4f}°")
    print(f"    median: {np.nanmedian(mean_grid):+.4f}°")
    print(f"    std:    {np.nanstd(mean_grid):.4f}°")
    print(f"    min:    {np.nanmin(mean_grid):+.4f}°  "
          f"(most compressed vs residue median)")
    print(f"    max:    {np.nanmax(mean_grid):+.4f}°  "
          f"(most expanded vs residue median)")
    print(f"    peak-to-peak range: "
          f"{np.nanmax(mean_grid) - np.nanmin(mean_grid):.4f}°")

    # Ramachandran region-specific means
    print(f"\n  Δτ in canonical regions (residue-median corrected):")
    regions = {
        'α-helix (φ∈[-80,-40], ψ∈[-60,-20])':
            (df_main['phi_deg'].between(-80, -40)
             & df_main['psi_deg'].between(-60, -20)),
        'β-sheet (φ∈[-150,-90], ψ∈[100,160])':
            (df_main['phi_deg'].between(-150, -90)
             & df_main['psi_deg'].between(100, 160)),
        'PPII    (φ∈[-90,-60],  ψ∈[120,160])':
            (df_main['phi_deg'].between(-90, -60)
             & df_main['psi_deg'].between(120, 160)),
        'αL      (φ∈[40,80],    ψ∈[20,80])':
            (df_main['phi_deg'].between(40, 80)
             & df_main['psi_deg'].between(20, 80)),
    }
    for name, mask in regions.items():
        n = int(mask.sum())
        if n > 0:
            mean = df_main.loc[mask, 'delta_tau'].mean()
            std  = df_main.loc[mask, 'delta_tau'].std()
            sem  = std / np.sqrt(n)
            print(f"    {name}: Δτ = {mean:+.4f}° ± {sem:.4f}° (n={n:,})")

    # Effect size relative to within-bin noise
    within_bin_std = df_main['delta_tau'].std()
    signal = np.nanmax(mean_grid) - np.nanmin(mean_grid)
    print(f"\n  Effect size:")
    print(f"    signal (peak-to-peak bin means): {signal:.3f}°")
    print(f"    noise  (within-bin residual std): {within_bin_std:.3f}°")
    print(f"    signal/noise: {signal/within_bin_std:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True,
                    help='Path to features_p2.csv')
    ap.add_argument('--min_bin_count', type=int, default=MIN_BIN_COUNT,
                    help='Minimum residues per bin for display')
    ap.add_argument('--color_limit', type=float, default=COLOR_LIMIT_DEG,
                    help='Colormap ±range in degrees')
    ap.add_argument('--out', default='paper2_01_tau_map.png',
                    help='Output figure path')
    args = ap.parse_args()

    # Rebind module-level names (no `global` needed since we only
    # assign at module scope via setattr on the module dict)
    import sys as _sys
    _mod = _sys.modules[__name__]
    _mod.MIN_BIN_COUNT   = args.min_bin_count
    _mod.COLOR_LIMIT_DEG = args.color_limit

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    # Load + filter
    df_main, df_gly, df_pro = load_and_filter(csv_path)

    # Normalize by residue-type median
    df_main, medians = normalize_by_res_median(df_main)

    # Compute main heatmap
    phi_e, psi_e, grid_main, counts_main = compute_heatmap(df_main)

    # Compute GLY and PRO heatmaps (use raw τ - global median since they're
    # single residue types, so normalization is just subtracting one number)
    if len(df_gly) > 0:
        df_gly = df_gly.copy()
        df_gly['delta_tau'] = df_gly['tau_deg'] - df_gly['tau_deg'].median()
        _, _, grid_gly, counts_gly = compute_heatmap(df_gly)
    else:
        grid_gly = counts_gly = None

    if len(df_pro) > 0:
        df_pro = df_pro.copy()
        df_pro['delta_tau'] = df_pro['tau_deg'] - df_pro['tau_deg'].median()
        _, _, grid_pro, counts_pro = compute_heatmap(df_pro)
    else:
        grid_pro = counts_pro = None

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 1], wspace=0.35)

    ax_main = fig.add_subplot(gs[0, 0])
    im_main = plot_heatmap(phi_e, psi_e, grid_main, counts_main,
                            f'Δτ(φ,ψ)  —  standard residues (n = {len(df_main):,})',
                            ax_main)
    cbar = plt.colorbar(im_main, ax=ax_main, fraction=0.046, pad=0.04)
    cbar.set_label('Δτ (deg)  [τ − median(τ | residue type)]')

    if grid_gly is not None:
        ax_gly = fig.add_subplot(gs[0, 1])
        plot_heatmap(phi_e, psi_e, grid_gly, counts_gly,
                      f'GLY only  (n = {len(df_gly):,})',
                      ax_gly, vmin=-COLOR_LIMIT_DEG*1.5, vmax=COLOR_LIMIT_DEG*1.5)

    if grid_pro is not None:
        ax_pro = fig.add_subplot(gs[0, 2])
        plot_heatmap(phi_e, psi_e, grid_pro, counts_pro,
                      f'PRO only  (n = {len(df_pro):,})',
                      ax_pro, vmin=-COLOR_LIMIT_DEG*1.5, vmax=COLOR_LIMIT_DEG*1.5)

    plt.suptitle('τ deformation across the Ramachandran plane',
                 fontsize=13, y=1.02)
    plt.savefig(args.out, dpi=200, bbox_inches='tight')
    print(f"\nFigure saved: {args.out}")

    # Summary
    summary_stats(grid_main, counts_main, medians, df_main)

    # ── Interpretation hint ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("  • If the α-helix region is BLUE (negative Δτ): τ is compressed")
    print("    there — consistent with the helix H-bond pulling N-Cα and")
    print("    Cα-C closer together.")
    print("  • If the β-sheet region is RED (positive Δτ): τ is expanded")
    print("    there — consistent with extended-chain geometry relaxing")
    print("    the sp³ constraint.")
    print("  • Signal/noise > 1 means the effect is real, not shot noise.")
    print("  • If the map is flat: either there's no mechanical effect,")
    print("    OR it's swamped by residue-composition drift — stratify")
    print("    by residue type (Option 1) as a sanity check.")


if __name__ == '__main__':
    main()