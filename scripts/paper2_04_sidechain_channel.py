"""
paper2_04_sidechain_channel.py — How does the sidechain modulate backbone angles?
==================================================================================

Move A of paper 2. The universality analysis (Move B) showed that τ, ω,
and ∠C-Cα-Cβ deform systematically across all residues, but ∠N-Cα-Cβ
showed a bimodal α-region split. The scatter diagnostic showed the split
is driven by β-branching at Cβ (VAL/ILE/THR), not sidechain mass.

This script tests the mechanical thesis:
   "The sidechain is a lever on Cα. Its substitution pattern at Cβ
    controls how much force it transmits. β-branched residues are
    stiffer levers."

Three tests:
   (1) Class-stratified Δ per angle × basin × class (G/A/U/B/Ar)
   (2) t-tests: β-branched vs unbranched, aromatic vs unbranched
   (3) χ1 rotamer effect: for each angle × basin, does Δ differ between
       gauche-, trans, gauche+ rotamer wells?

Outputs:
   paper2_04_class_bars.png — grouped bar chart, 4 panels (one per angle)
   paper2_04_rotamer_bars.png — χ1 rotamer effect, 4 panels

Usage
-----
    python paper2_04_sidechain_channel.py --csv features.csv
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_ind


# Sidechain classes
_CLASS = {
    'GLY': 'G',  # no Cβ, reference
    'ALA': 'A',  # methyl only
    'SER': 'U', 'CYS': 'U', 'LEU': 'U', 'MET': 'U',
    'ASP': 'U', 'ASN': 'U', 'GLU': 'U', 'GLN': 'U',
    'LYS': 'U', 'ARG': 'U',
    'VAL': 'B', 'ILE': 'B', 'THR': 'B',  # β-branched
    'PHE': 'Ar', 'TYR': 'Ar', 'TRP': 'Ar', 'HIS': 'Ar',
    'PRO': 'P',  # ring constraint — separate
}
_CLASS_ORDER = ['G', 'A', 'U', 'B', 'Ar']
_CLASS_LABEL = {
    'G':  'GLY\n(no Cβ)',
    'A':  'ALA\n(methyl)',
    'U':  'Unbranched\n(SER/CYS/LEU/MET/\nASP/ASN/GLU/GLN/LYS/ARG)',
    'B':  'β-branched\n(VAL/ILE/THR)',
    'Ar': 'Aromatic\n(PHE/TYR/TRP/HIS)',
}
_CLASS_COLOR = {'G': '#9b9b9b', 'A': '#7ba7d9', 'U': '#5aa369',
                'B': '#c0392b', 'Ar': '#d9a05b'}

_REGIONS = {
    'α':    {'phi': (-80, -40),  'psi': (-60, -20)},
    'β':    {'phi': (-150, -90), 'psi': (100, 160)},
    'PPII': {'phi': (-90, -60),  'psi': (120, 160)},
    'αL':   {'phi': (40, 80),    'psi': (20, 80)},
}
_REGION_ORDER = ['α', 'β', 'PPII', 'αL']

# χ1 rotamer bins in radians
_ROT_BINS = {
    'g⁻': (-np.pi * 5/6, -np.pi / 6),   # near −60° (−150° to −30°)
    'trans': (None, None),              # special case: wraps around ±π
    'g⁺': (np.pi / 6,  np.pi * 5/6),    # near +60° (+30° to +150°)
}

_ANGLES = [
    ('tau_deg',            False, 'τ (N-Cα-C)'),
    ('omega_measured_deg', True,  'ω (peptide)'),
    ('angle_N_CA_CB',      False, '∠N-Cα-Cβ'),
    ('angle_C_CA_CB',      False, '∠C-Cα-Cβ'),
]


# ── Circular helpers ─────────────────────────────────────────────────────────

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


def _rot_label(chi1_rad):
    """Classify a χ1 value (radians) into gauche-/trans/gauche+. NaN-safe."""
    if np.isnan(chi1_rad):
        return None
    # trans: |chi1| >= 5π/6 (±150° to ±180°)
    if abs(chi1_rad) >= np.pi * 5 / 6:
        return 'trans'
    if -np.pi * 5 / 6 < chi1_rad < -np.pi / 6:
        return 'g⁻'
    if np.pi / 6 < chi1_rad < np.pi * 5 / 6:
        return 'g⁺'
    return None  # eclipsed regions — rare, drop


def prepare_delta(df, angle, circular):
    df = df.dropna(subset=[angle, 'phi_deg', 'psi_deg', 'res_name']).copy()
    df = df[df['res_name'].isin(_CLASS)]
    if circular:
        medians = df.groupby('res_name')[angle].apply(
            lambda g: _circ_median_deg(g.values))
        df['delta'] = _circ_diff(df[angle].values,
                                  df['res_name'].map(medians).values)
    else:
        medians = df.groupby('res_name')[angle].median()
        df['delta'] = df[angle] - df['res_name'].map(medians)
    df['sc_class'] = df['res_name'].map(_CLASS)
    return df


def subset_region(df, region):
    p, q = _REGIONS[region]['phi'], _REGIONS[region]['psi']
    return df[df['phi_deg'].between(p[0], p[1])
              & df['psi_deg'].between(q[0], q[1])]


def mean_sem(v, circular=False):
    v = np.asarray(v)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return float('nan'), float('nan'), 0
    if circular:
        m = _circ_mean_deg(v)
        s = float(np.std(_circ_diff(v, m)))
    else:
        m = float(v.mean())
        s = float(v.std())
    return m, s / np.sqrt(len(v)), len(v)


# ── Main analyses ─────────────────────────────────────────────────────────────

def run_class_analysis(dfs_by_angle, circulars):
    """Compute Δ per (angle × basin × class) and run t-tests."""
    print("\n" + "=" * 88)
    print("CLASS-STRATIFIED Δ per angle × basin")
    print("=" * 88)

    rows = []  # for bar chart later
    for (col, circ, label), df in zip(_ANGLES, dfs_by_angle):
        print(f"\n  {label} (circular={circ}):")
        print(f"    {'basin':<6} " + ''.join(
            f'{c:>14}' for c in _CLASS_ORDER) + '    t(B vs U)   t(Ar vs U)')
        for reg in _REGION_ORDER:
            sub = subset_region(df, reg)
            cells = {}
            for cls in _CLASS_ORDER:
                vals = sub.loc[sub['sc_class'] == cls, 'delta'].values
                m, sem, n = mean_sem(vals, circular=circ)
                cells[cls] = (m, sem, n, vals)
                rows.append(dict(angle=label, basin=reg, cls=cls,
                                  mean=m, sem=sem, n=n))

            # t-tests: B vs U, Ar vs U
            def _t(a, b):
                a_vals = cells[a][3]; b_vals = cells[b][3]
                if len(a_vals) < 30 or len(b_vals) < 30:
                    return '   n/a   '
                t_stat, p_val = ttest_ind(a_vals, b_vals, equal_var=False)
                return f"t={t_stat:+6.1f} (p={p_val:.1e})"

            # Print line
            parts = [f"    {reg:<6}"]
            for cls in _CLASS_ORDER:
                m, sem, n, _ = cells[cls]
                if n < 30:
                    parts.append(f"{'n='+str(n):>14}")
                else:
                    parts.append(f"{m:+7.3f}±{sem:.3f}".rjust(14))
            parts.append(f"  {_t('B', 'U'):>18}")
            parts.append(f"  {_t('Ar', 'U'):>18}")
            print(''.join(parts))

    return pd.DataFrame(rows)


def run_rotamer_analysis(dfs_by_angle, circulars):
    """Compute Δ per (angle × basin × rotamer) for residues with χ1."""
    print("\n" + "=" * 88)
    print("χ1 ROTAMER EFFECT  —  Δ per basin × rotamer well")
    print("=" * 88)
    print("(pooled over all residues with valid χ1; proline and GLY/ALA excluded)")

    rows = []
    for (col, circ, label), df in zip(_ANGLES, dfs_by_angle):
        sub = df.dropna(subset=['chi1_rad']).copy()
        # Residues with meaningful χ1 — has_chi1 is 1 where Cγ-like atom exists
        if 'has_chi1' in sub.columns:
            sub = sub[sub['has_chi1'] == 1]
        # Exclude proline (ring) and anything classed as G or A
        sub = sub[~sub['sc_class'].isin(['G', 'A', 'P'])]
        sub['rot'] = sub['chi1_rad'].apply(_rot_label)
        sub = sub[sub['rot'].notna()]

        print(f"\n  {label}:")
        print(f"    {'basin':<6} {'g⁻ Δ':>16} {'trans Δ':>16} {'g⁺ Δ':>16}"
              f"    g⁻ vs g⁺ t")
        for reg in _REGION_ORDER:
            s_reg = subset_region(sub, reg)
            cells = {}
            for rot in ['g⁻', 'trans', 'g⁺']:
                vals = s_reg.loc[s_reg['rot'] == rot, 'delta'].values
                m, sem, n = mean_sem(vals, circular=circ)
                cells[rot] = (m, sem, n, vals)
                rows.append(dict(angle=label, basin=reg, rot=rot,
                                  mean=m, sem=sem, n=n))

            # t-test g- vs g+
            gm = cells['g⁻'][3]; gp = cells['g⁺'][3]
            if len(gm) >= 30 and len(gp) >= 30:
                t_stat, p_val = ttest_ind(gm, gp, equal_var=False)
                tstr = f"t={t_stat:+6.1f} (p={p_val:.1e})"
            else:
                tstr = 'n/a'

            def _fmt(c):
                m, sem, n, _ = c
                if n < 30:
                    return f"{'n='+str(n):>16}"
                return f"{m:+8.3f}±{sem:.3f}".rjust(16)

            print(f"    {reg:<6} {_fmt(cells['g⁻'])} "
                  f"{_fmt(cells['trans'])} {_fmt(cells['g⁺'])}"
                  f"    {tstr}")

    return pd.DataFrame(rows)


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_class_bars(df_class, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for k, (col, circ, label) in enumerate(_ANGLES):
        ax = axes[k]
        dsub = df_class[df_class['angle'] == label]
        n_basins = len(_REGION_ORDER)
        n_classes = len(_CLASS_ORDER)
        bar_w = 0.15
        x = np.arange(n_basins)

        for ci, cls in enumerate(_CLASS_ORDER):
            means = []; sems = []; ns = []
            for reg in _REGION_ORDER:
                row = dsub[(dsub['basin'] == reg) & (dsub['cls'] == cls)]
                if len(row) and row['n'].iloc[0] >= 30:
                    means.append(row['mean'].iloc[0])
                    sems.append(row['sem'].iloc[0])
                    ns.append(int(row['n'].iloc[0]))
                else:
                    means.append(0); sems.append(0); ns.append(0)
            offset = (ci - (n_classes - 1) / 2) * bar_w
            ax.bar(x + offset, means, width=bar_w, yerr=sems,
                    color=_CLASS_COLOR[cls], label=cls, capsize=2,
                    edgecolor='white', linewidth=0.6)

        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(_REGION_ORDER)
        ax.set_ylabel('Δ (deg)')
        ax.set_title(label, fontsize=12)
        if k == 0:
            ax.legend(fontsize=8, loc='upper left', ncol=5,
                      bbox_to_anchor=(0, 1.18))
        ax.grid(True, axis='y', alpha=0.25)

    plt.suptitle('Backbone angle deformation by sidechain class',
                  fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\nFigure saved: {out_path}")


def plot_rotamer_bars(df_rot, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    rot_colors = {'g⁻': '#8e44ad', 'trans': '#16a085', 'g⁺': '#e67e22'}

    for k, (col, circ, label) in enumerate(_ANGLES):
        ax = axes[k]
        dsub = df_rot[df_rot['angle'] == label]
        x = np.arange(len(_REGION_ORDER))
        bar_w = 0.26

        for ri, rot in enumerate(['g⁻', 'trans', 'g⁺']):
            means = []; sems = []
            for reg in _REGION_ORDER:
                row = dsub[(dsub['basin'] == reg) & (dsub['rot'] == rot)]
                if len(row) and row['n'].iloc[0] >= 30:
                    means.append(row['mean'].iloc[0])
                    sems.append(row['sem'].iloc[0])
                else:
                    means.append(0); sems.append(0)
            offset = (ri - 1) * bar_w
            ax.bar(x + offset, means, width=bar_w, yerr=sems,
                    color=rot_colors[rot], label=rot, capsize=3,
                    edgecolor='white', linewidth=0.6)

        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(_REGION_ORDER)
        ax.set_ylabel('Δ (deg)')
        ax.set_title(label, fontsize=12)
        if k == 0:
            ax.legend(fontsize=10, loc='upper left', title='χ1 rotamer')
        ax.grid(True, axis='y', alpha=0.25)

    plt.suptitle('Backbone angle deformation by χ1 rotamer well',
                  fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Figure saved: {out_path}")


# ── Orchestrator ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out_class',  default='paper2_04_class_bars.png')
    ap.add_argument('--out_rot',    default='paper2_04_rotamer_bars.png')
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found"); sys.exit(1)

    print(f"Loading {csv_path} ...")
    df_raw = pd.read_csv(csv_path)
    print(f"  {len(df_raw):,} total rows")

    # Prepare per-angle DataFrames
    dfs_by_angle = []
    circulars = []
    for col, circ, label in _ANGLES:
        dfa = prepare_delta(df_raw, col, circ)
        dfs_by_angle.append(dfa)
        circulars.append(circ)
        print(f"  {col:<22}: {len(dfa):>9,} residues")

    # Class analysis
    df_class = run_class_analysis(dfs_by_angle, circulars)
    plot_class_bars(df_class, args.out_class)

    # Rotamer analysis
    df_rot = run_rotamer_analysis(dfs_by_angle, circulars)
    plot_rotamer_bars(df_rot, args.out_rot)

    print("\n" + "=" * 88)
    print("INTERPRETATION GUIDE")
    print("=" * 88)
    print("  Class analysis:")
    print("   • If B (branched) bar differs significantly from U (unbranched)")
    print("     in most basins → β-branching is a major modulator.")
    print("   • If Ar bar differs too → aromatic sidechains act similarly.")
    print("   • If G bar (GLY, no Cβ) is flat → Cβ is necessary for the effect.")
    print("")
    print("  Rotamer analysis:")
    print("   • If g⁻, trans, g⁺ bars differ significantly within a basin")
    print("     → the sidechain is mechanically transmitting force through Cα.")
    print("   • If bars are identical → sidechain orientation doesn't matter,")
    print("     so the 'lever' story is too simple.")


if __name__ == '__main__':
    main()