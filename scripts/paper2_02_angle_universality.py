"""
paper2_02_angle_universality.py — Universality test for any backbone angle
============================================================================

Parameterized version of paper2_01b_tau_by_residue.py. Runs the same
per-residue sign-agreement test on any angle column from features.csv.

The test logic:
  1. Load data, drop NaN, keep only 18 standard residues
  2. Compute each residue's median (or circular median for --circular)
  3. Define Δangle = angle − residue median
  4. Compute global Δangle in four Ramachandran regions (α, β, PPII, αL)
     — these become the "expected signs"
  5. For each of 18 residues, check whether they agree with the global signs
  6. Report per-residue table, sign-agreement counts, verdict, and figure

Usage
-----
    python paper2_02_angle_universality.py --csv features.csv \\
        --angle tau_deg

    python paper2_02_angle_universality.py --csv features.csv \\
        --angle omega_measured_deg --circular

    python paper2_02_angle_universality.py --csv features.csv \\
        --angle angle_N_CA_CB

    python paper2_02_angle_universality.py --csv features.csv \\
        --angle angle_C_CA_CB

Note
----
For ω, use --circular. For bond angles (τ, ∠N-Cα-Cβ, ∠C-Cα-Cβ), don't.
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


_RES20 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','HIS','ILE','LEU',
          'LYS','MET','PHE','SER','THR','TRP','TYR','VAL']

_REGIONS = {
    'α':    {'phi': (-80, -40),  'psi': (-60, -20)},
    'β':    {'phi': (-150, -90), 'psi': (100, 160)},
    'PPII': {'phi': (-90, -60),  'psi': (120, 160)},
    'αL':   {'phi': (40, 80),    'psi': (20, 80)},
}

BIN_WIDTH = 10.0
MIN_BIN_COUNT = 20


# ══════════════════════════════════════════════════════════════════════════════
# Circular statistics (for ω, which wraps at ±180°)
# ══════════════════════════════════════════════════════════════════════════════

def _circ_median_deg(values_deg):
    """
    Circular median in degrees. For values near ±180, this returns a value
    in [-180, 180]. Uses the 'antipode' method: find the median of values
    rotated to be near 0.
    """
    v = np.asarray(values_deg, dtype=float)
    # Rotate so values cluster near 0 (use circular mean as anchor)
    rad = np.radians(v)
    mean_ang = np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))
    mean_deg = np.degrees(mean_ang)
    # Wrap each value into [-180+mean, 180+mean], then ordinary median works
    shifted = ((v - mean_deg + 180) % 360) - 180
    median_shifted = np.median(shifted)
    return ((median_shifted + mean_deg + 180) % 360) - 180


def _circ_diff(a_deg, b_deg):
    """Shortest signed angular difference a - b in degrees, in [-180, 180]."""
    d = (a_deg - b_deg + 180) % 360 - 180
    return d


def _circ_mean_deg(values_deg):
    rad = np.radians(np.asarray(values_deg, dtype=float))
    return np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad))))


# ══════════════════════════════════════════════════════════════════════════════
# Region + heatmap helpers
# ══════════════════════════════════════════════════════════════════════════════

def region_mean(df, region, value_col, circular=False):
    """Return (mean, sem, n) of value_col in a Ramachandran region."""
    p, q = _REGIONS[region]['phi'], _REGIONS[region]['psi']
    m = (df['phi_deg'].between(p[0], p[1])
         & df['psi_deg'].between(q[0], q[1]))
    n = int(m.sum())
    if n == 0:
        return float('nan'), float('nan'), 0
    v = df.loc[m, value_col].values
    if circular:
        mean = _circ_mean_deg(v)
        # SEM for circular mean: use std of wrapped values
        diffs = _circ_diff(v, mean)
        std = float(np.std(diffs))
    else:
        mean = float(v.mean())
        std = float(v.std())
    sem = std / np.sqrt(n)
    return mean, sem, n


def compute_heatmap(df, value_col, bin_width=BIN_WIDTH,
                    min_count=MIN_BIN_COUNT, circular=False):
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
                if circular:
                    grid[j, i] = _circ_mean_deg(val[mask])
                else:
                    grid[j, i] = val[mask].mean()
    return phi_edges, psi_edges, grid


# ══════════════════════════════════════════════════════════════════════════════
# Main analysis
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--angle', required=True,
                    help='Column name in features.csv (tau_deg, '
                         'omega_measured_deg, angle_N_CA_CB, angle_C_CA_CB)')
    ap.add_argument('--circular', action='store_true',
                    help='Use circular statistics (required for ω)')
    ap.add_argument('--out', default=None,
                    help='Figure output path (default: paper2_02_<angle>.png)')
    ap.add_argument('--color_limit', type=float, default=None,
                    help='Colormap ±range (auto-detect if omitted)')
    args = ap.parse_args()

    if args.out is None:
        args.out = f'paper2_02_{args.angle}.png'

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found"); sys.exit(1)

    print(f"Loading {csv_path} ...")
    print(f"Angle: {args.angle}  (circular={args.circular})")
    df = pd.read_csv(csv_path)
    if args.angle not in df.columns:
        print(f"ERROR: column '{args.angle}' not in CSV")
        print(f"  available angle-ish columns: "
              f"{[c for c in df.columns if 'angle' in c or 'deg' in c]}")
        sys.exit(1)

    df = df.dropna(subset=[args.angle, 'phi_deg', 'psi_deg', 'res_name'])
    df = df[df['res_name'].isin(_RES20)].copy()
    print(f"  {len(df):,} residues (18 standard types, GLY/PRO excluded)")

    # ── Per-residue median-normalize (circular or linear) ────────────────────
    if args.circular:
        medians = df.groupby('res_name')[args.angle].apply(
            lambda g: _circ_median_deg(g.values))
        # Δangle = circular difference
        df['delta'] = _circ_diff(
            df[args.angle].values,
            df['res_name'].map(medians).values)
    else:
        medians = df.groupby('res_name')[args.angle].median()
        df['delta'] = df[args.angle] - df['res_name'].map(medians)

    # ── Global map: compute region means to determine expected signs ──────────
    print("\n" + "=" * 78)
    print(f"Global Δ{args.angle} in canonical regions (all 18 residues pooled)")
    print("=" * 78)
    global_signs = {}
    for reg in _REGIONS:
        m, sem, n = region_mean(df, reg, 'delta', circular=args.circular)
        sign_sym = '+' if m > 0 else '−'
        print(f"  {reg:<4}  Δ = {m:+7.3f}° ± {sem:.4f}°  (n={n:,})  → "
              f"sign = {sign_sym}")
        global_signs[reg] = +1 if m > 0 else -1

    # ── Per-residue table ─────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"Δ{args.angle} per residue × region "
          f"(vs that residue's {'circular ' if args.circular else ''}median)")
    print("=" * 78)
    print(f"  {'res':<4} {'median':>10}  "
          f"{'α':>13} {'β':>13} {'PPII':>13} {'αL':>13}")
    print("  " + "-" * 76)

    results = {}
    for res in _RES20:
        sub = df[df['res_name'] == res]
        row = {}
        for reg in _REGIONS:
            row[reg] = region_mean(sub, reg, 'delta', circular=args.circular)
        results[res] = row

        def _fmt(r):
            m, sem, n = r
            if n < 30:
                return f"   n={n:>4}   "
            return f"{m:+6.3f}±{sem:.3f}"

        print(f"  {res:<4} {medians[res]:>10.3f}  "
              f"{_fmt(row['α']):>13} {_fmt(row['β']):>13} "
              f"{_fmt(row['PPII']):>13} {_fmt(row['αL']):>13}")

    # ── Agreement summary ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SIGN AGREEMENT vs global signs")
    print("=" * 78)
    total_agree = 0
    total_tests = 0
    for reg, exp in global_signs.items():
        agree = disagree = nil = 0
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
        sign_word = '+' if exp > 0 else '−'
        print(f"  {reg:<4}  global sign = {sign_word}   "
              f"agree: {agree:>2}/18  disagree: {disagree:>2}  "
              f"n_too_small: {nil}")
        if disagree_list and disagree <= 6:
            print(f"        disagreeing: {', '.join(disagree_list)}")
        total_agree += agree
        total_tests += agree + disagree

    # ── Verdict ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    pct = 100 * total_agree / total_tests if total_tests else 0
    print(f"  Overall agreement: {total_agree}/{total_tests} ({pct:.1f}%)")
    if pct >= 85:
        print(f"  → STRONG: {args.angle} deformation is a universal "
              "mechanical response.")
    elif pct >= 70:
        print(f"  → MODERATE: signal is mostly universal but outliers exist.")
    else:
        print(f"  → WEAK: signal is residue-specific, not universal.")

    # ── Figure: 4×5 grid, one heatmap per residue ────────────────────────────
    print(f"\nGenerating {args.out} ...")

    # Auto-detect color limit if not given
    if args.color_limit is None:
        all_deltas = df['delta'].values
        p99 = np.percentile(np.abs(all_deltas), 99)
        vlim = float(np.ceil(p99 * 2) / 2)  # round up to nearest 0.5
        print(f"  Auto color limit: ±{vlim:.1f}°")
    else:
        vlim = args.color_limit

    fig, axes = plt.subplots(4, 5, figsize=(18, 14),
                              gridspec_kw={'hspace': 0.35, 'wspace': 0.25})
    axes = axes.flatten()

    for k, res in enumerate(_RES20):
        ax = axes[k]
        sub = df[df['res_name'] == res]
        phi_e, psi_e, grid = compute_heatmap(sub, 'delta',
                                              circular=args.circular)
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

    for k in range(len(_RES20), 20):
        axes[k].axis('off')

    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
    cbar.set_label(f'Δ{args.angle} (deg)  [vs residue median]')

    plt.suptitle(f'{args.angle} deformation across Ramachandran plane, '
                  f'per residue type', fontsize=13, y=0.995)
    plt.savefig(args.out, dpi=180, bbox_inches='tight')
    print(f"Figure saved: {args.out}")


if __name__ == '__main__':
    main()