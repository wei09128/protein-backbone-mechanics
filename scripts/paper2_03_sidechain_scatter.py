"""
paper2_03_sidechain_scatter.py — Is the ∠N-Cα-Cβ α-split sidechain-driven?
=============================================================================

In paper2_02_angle_universality.py the α-region showed a bimodal split for
∠N-Cα-Cβ: 9 residues positive, 9 negative. This looked like it might be
driven by sidechain size (bulky/β-branched pushing one way, small/flexible
pushing the other).

This script tests that hypothesis directly:
  For each of 18 residues, plot α-region Δ∠N-Cα-Cβ vs sidechain mass
  and vs number of heavy sidechain atoms.

If the relationship is cleanly monotonic → sidechain-size story is real.
If it's noisy or multimodal → the split has another driver (charge?
branching? H-bond capacity?).

Also plots the same scatter for the other three angles (τ, ω,
∠C-Cα-Cβ) for comparison. If only ∠N-Cα-Cβ shows mass dependence,
that's a specific result. If all four do, it's a general finding.

Usage
-----
    python paper2_03_sidechain_scatter.py --csv features.csv
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr


_RES20 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','HIS','ILE','LEU',
          'LYS','MET','PHE','SER','THR','TRP','TYR','VAL']

# Shared structural biology constants (from features_collector.py)
_SC_MASS = {
    'GLY':0,   'ALA':15,  'VAL':43,  'LEU':57,  'ILE':57, 'PRO':42,
    'PHE':91,  'TYR':107, 'TRP':130, 'SER':31,  'THR':45, 'CYS':47,
    'MET':75,  'ASP':58,  'ASN':58,  'GLU':72,  'GLN':72, 'LYS':72,
    'ARG':100, 'HIS':81,
}
_SC_N_HEAVY = {
    'GLY':0,'ALA':1,'VAL':3,'LEU':4,'ILE':4,'PRO':3,'PHE':7,'TYR':8,
    'TRP':10,'SER':2,'THR':3,'CYS':2,'MET':4,'ASP':4,'ASN':4,'GLU':5,
    'GLN':5,'LYS':5,'ARG':7,'HIS':6,
}
_SC_BRANCHED = {'VAL','ILE','THR'}  # β-branched

_REGIONS = {
    'α':    {'phi': (-80, -40),  'psi': (-60, -20)},
    'β':    {'phi': (-150, -90), 'psi': (100, 160)},
    'PPII': {'phi': (-90, -60),  'psi': (120, 160)},
    'αL':   {'phi': (40, 80),    'psi': (20, 80)},
}


def _circ_median_deg(v):
    v = np.asarray(v, dtype=float)
    rad = np.radians(v)
    mean = np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad))))
    shifted = ((v - mean + 180) % 360) - 180
    return ((np.median(shifted) + mean + 180) % 360) - 180


def _circ_diff(a, b):
    return (a - b + 180) % 360 - 180


def _circ_mean_deg(v):
    rad = np.radians(np.asarray(v, dtype=float))
    return np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad))))


def region_mean_per_residue(df, region, value_col, circular=False):
    """
    Returns dict res → mean Δ in region, and dict res → n.
    df must already have 'delta' column set.
    """
    p, q = _REGIONS[region]['phi'], _REGIONS[region]['psi']
    m = (df['phi_deg'].between(p[0], p[1])
         & df['psi_deg'].between(q[0], q[1]))
    sub = df[m]
    means, ns, sems = {}, {}, {}
    for res in _RES20:
        vals = sub.loc[sub['res_name'] == res, value_col].values
        ns[res] = len(vals)
        if len(vals) < 30:
            means[res] = float('nan')
            sems[res] = float('nan')
        else:
            if circular:
                mn = _circ_mean_deg(vals)
                std = float(np.std(_circ_diff(vals, mn)))
            else:
                mn = float(vals.mean())
                std = float(vals.std())
            means[res] = mn
            sems[res] = std / np.sqrt(len(vals))
    return means, sems, ns


def prepare_delta(df, angle, circular):
    """Add 'delta' column: angle − residue median (circular-aware)."""
    if circular:
        medians = df.groupby('res_name')[angle].apply(
            lambda g: _circ_median_deg(g.values))
        df['delta'] = _circ_diff(df[angle].values,
                                  df['res_name'].map(medians).values)
    else:
        medians = df.groupby('res_name')[angle].median()
        df['delta'] = df[angle] - df['res_name'].map(medians)
    return df


def scatter_panel(ax, xvals, yvals, res_labels, errs, xlabel, ylabel, title):
    """Plot a labeled scatter with error bars and correlation line."""
    ax.errorbar(xvals, yvals, yerr=errs, fmt='o', ms=7,
                 color='#2a5d9f', ecolor='#888', capsize=3, zorder=3)
    # Label each point with residue name
    for x, y, r in zip(xvals, yvals, res_labels):
        # β-branched residues in red for visual cue
        color = '#c0392b' if r in _SC_BRANCHED else '#222'
        ax.annotate(r, (x, y), xytext=(4, 4), textcoords='offset points',
                     fontsize=8, color=color)

    # Fit & annotate
    xa = np.asarray(xvals); ya = np.asarray(yvals)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if mask.sum() >= 3:
        r_p, p_p = pearsonr(xa[mask], ya[mask])
        r_s, _ = spearmanr(xa[mask], ya[mask])
        # Least-squares line
        z = np.polyfit(xa[mask], ya[mask], 1)
        xs = np.linspace(xa[mask].min(), xa[mask].max(), 50)
        ax.plot(xs, np.polyval(z, xs), '--', color='#888', lw=1, zorder=2)
        ax.text(0.03, 0.95,
                 f'Pearson r = {r_p:+.3f} (p={p_p:.2e})\n'
                 f'Spearman ρ = {r_s:+.3f}',
                 transform=ax.transAxes, fontsize=9, va='top',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white',
                           ec='#ccc', alpha=0.9))

    ax.axhline(0, color='k', lw=0.4, alpha=0.3, zorder=1)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='paper2_03_sidechain_scatter.png')
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found"); sys.exit(1)

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    df = df[df['res_name'].isin(_RES20)].copy()
    print(f"  {len(df):,} residues (18 standard types)")

    # We'll analyse 4 angles in the α basin
    angles = [
        ('tau_deg',            False, 'τ (N-Cα-C)'),
        ('omega_measured_deg', True,  'ω (peptide)'),
        ('angle_N_CA_CB',      False, '∠N-Cα-Cβ'),
        ('angle_C_CA_CB',      False, '∠C-Cα-Cβ'),
    ]

    # ── Build the per-residue α-region means for each angle ──────────────────
    print("\n" + "=" * 78)
    print("α-region Δ per residue, for each angle")
    print("=" * 78)

    all_data = {}  # angle -> (means dict, sems dict, n dict)
    for col, circ, label in angles:
        sub = df.dropna(subset=[col, 'phi_deg', 'psi_deg', 'res_name']).copy()
        sub = prepare_delta(sub, col, circ)
        means, sems, ns = region_mean_per_residue(sub, 'α', 'delta',
                                                    circular=circ)
        all_data[col] = (means, sems, ns, label)

    # ── Print table ──────────────────────────────────────────────────────────
    print(f"\n  {'res':<4} {'mass':>5} {'n_heavy':>7}  {'branched':>8}  "
          f"{'τ Δ':>10} {'ω Δ':>10} {'N-Cα-Cβ Δ':>11} {'C-Cα-Cβ Δ':>11}")
    print("  " + "-" * 92)
    for res in sorted(_RES20, key=lambda r: _SC_MASS[r]):
        m = _SC_MASS[res]
        nh = _SC_N_HEAVY[res]
        br = 'Y' if res in _SC_BRANCHED else '.'
        tau_m = all_data['tau_deg'][0][res]
        om_m  = all_data['omega_measured_deg'][0][res]
        n_m   = all_data['angle_N_CA_CB'][0][res]
        c_m   = all_data['angle_C_CA_CB'][0][res]
        print(f"  {res:<4} {m:>5} {nh:>7}  {br:>8}  "
              f"{tau_m:+10.3f} {om_m:+10.3f} {n_m:+11.3f} {c_m:+11.3f}")

    # ── Correlation summary ──────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("α-region Δ vs sidechain size — Pearson r (p-value)")
    print("=" * 78)
    print(f"  {'angle':<14}  {'vs sc_mass':>20}  {'vs sc_n_heavy':>22}")
    print("  " + "-" * 60)

    for col, circ, label in angles:
        means = all_data[col][0]
        masses = [_SC_MASS[r] for r in _RES20
                   if np.isfinite(means.get(r, float('nan')))]
        nhs    = [_SC_N_HEAVY[r] for r in _RES20
                   if np.isfinite(means.get(r, float('nan')))]
        ys     = [means[r] for r in _RES20
                   if np.isfinite(means.get(r, float('nan')))]
        if len(ys) < 3:
            continue
        r_m, p_m = pearsonr(masses, ys)
        r_n, p_n = pearsonr(nhs, ys)
        print(f"  {label:<14}  r={r_m:+.3f} (p={p_m:.2e})   "
              f"r={r_n:+.3f} (p={p_n:.2e})")

    # ── Figure: 2x2 grid of scatter plots (mass on x, Δ on y) ────────────────
    print(f"\nGenerating {args.out} ...")
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    axes = axes.flatten()

    for k, (col, circ, label) in enumerate(angles):
        ax = axes[k]
        means, sems, ns, _ = all_data[col]
        # Residues in order of mass
        residues = [r for r in sorted(_RES20, key=lambda x: _SC_MASS[x])
                    if np.isfinite(means.get(r, float('nan')))]
        xs = [_SC_MASS[r] for r in residues]
        ys = [means[r] for r in residues]
        es = [sems[r] for r in residues]
        scatter_panel(ax, xs, ys, residues, es,
                       'Sidechain mass (Da)',
                       f'α-region Δ (deg)',
                       f'{label}   —   α basin')

    plt.suptitle('α-basin deformation vs sidechain mass, per residue type\n'
                  '(red = β-branched: VAL, ILE, THR)',
                  fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {args.out}")


if __name__ == '__main__':
    main()