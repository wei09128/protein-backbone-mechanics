"""
paper2_F2_assembly.py — Figure 2: the sidechain mechanical lever
==================================================================

Two-panel figure showing the sidechain channel:

  Panel A (top):    Class-stratified Δ per angle × basin.
                    5 classes × 4 basins × 4 angles = 80 bars.
                    Shows that β-branching amplifies (and sometimes
                    inverts) the backbone response.

  Panel B (bottom): Per-residue χ1 rotamer shift Δ(g⁺) − Δ(g⁻).
                    17 residues × 4 basins × 4 angles heatmap.
                    Shows universality across residues (16–17/17
                    agreement) with β-branched labels in red.

This script re-runs the analyses from paper2_04 and paper2_05 in one pass
so the figure is self-contained.

Usage
-----
    python paper2_F2_assembly.py --csv features.csv --out F2.png --dpi 300
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ── Residue classification (from paper2_04) ──────────────────────────────────
_CLASS = {
    'GLY': 'G', 'ALA': 'A',
    'SER': 'U', 'CYS': 'U', 'LEU': 'U', 'MET': 'U',
    'ASP': 'U', 'ASN': 'U', 'GLU': 'U', 'GLN': 'U',
    'LYS': 'U', 'ARG': 'U',
    'VAL': 'B', 'ILE': 'B', 'THR': 'B',
    'PHE': 'Ar', 'TYR': 'Ar', 'TRP': 'Ar', 'HIS': 'Ar',
    'PRO': 'P',
}
_CLASS_ORDER = ['G', 'A', 'U', 'B', 'Ar']
_CLASS_COLOR = {'G': '#9b9b9b', 'A': '#7ba7d9', 'U': '#5aa369',
                'B': '#c0392b', 'Ar': '#d9a05b'}

# Residues with valid χ1 (for Panel B)
_RES_CHI1 = ['ARG','ASN','ASP','CYS','GLN','GLU','HIS','ILE','LEU',
             'LYS','MET','PHE','SER','THR','TRP','TYR','VAL']
_BRANCHED = {'VAL', 'ILE', 'THR'}

_REGIONS = {
    'α':    {'phi': (-80, -40),  'psi': (-60, -20)},
    'β':    {'phi': (-150, -90), 'psi': (100, 160)},
    'PPII': {'phi': (-90, -60),  'psi': (120, 160)},
    'αL':   {'phi': (40, 80),    'psi': (20, 80)},
}
_REGION_ORDER = ['α', 'β', 'PPII', 'αL']

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
    if np.isnan(chi1_rad):
        return None
    if abs(chi1_rad) >= np.pi * 5 / 6:
        return 'trans'
    if -np.pi * 5 / 6 < chi1_rad < -np.pi / 6:
        return 'g-'
    if np.pi / 6 < chi1_rad < np.pi * 5 / 6:
        return 'g+'
    return None


def prepare_delta(df, angle, circular, res_filter=None):
    """Add 'delta' column: angle − per-residue median (circular-aware)."""
    df = df.dropna(subset=[angle, 'phi_deg', 'psi_deg', 'res_name']).copy()
    if res_filter is not None:
        df = df[df['res_name'].isin(res_filter)]
    else:
        df = df[df['res_name'].isin(_CLASS)]
    if circular:
        medians = df.groupby('res_name')[angle].apply(
            lambda g: _circ_median_deg(g.values))
        df['delta'] = _circ_diff(df[angle].values,
                                  df['res_name'].map(medians).values)
    else:
        medians = df.groupby('res_name')[angle].median()
        df['delta'] = df[angle] - df['res_name'].map(medians)
    return df


def subset_region(df, region):
    p, q = _REGIONS[region]['phi'], _REGIONS[region]['psi']
    return df[df['phi_deg'].between(p[0], p[1])
              & df['psi_deg'].between(q[0], q[1])]


def mean_sem(v, circular=False):
    v = np.asarray(v)
    v = v[np.isfinite(v)]
    if len(v) < 30:
        return float('nan'), float('nan'), len(v)
    if circular:
        m = _circ_mean_deg(v)
        s = float(np.std(_circ_diff(v, m)))
    else:
        m = float(v.mean())
        s = float(v.std())
    return m, s / np.sqrt(len(v)), len(v)


# ── Panel A data: class-stratified ───────────────────────────────────────────

def compute_class_data(df_raw):
    """Return dict: angle_label -> dict[(basin, class) -> (mean, sem, n)]."""
    out = {}
    for col, circ, label in _ANGLES:
        d = prepare_delta(df_raw, col, circ)
        d['sc_class'] = d['res_name'].map(_CLASS)
        cells = {}
        for reg in _REGION_ORDER:
            sub = subset_region(d, reg)
            for cls in _CLASS_ORDER:
                vals = sub.loc[sub['sc_class'] == cls, 'delta'].values
                cells[(reg, cls)] = mean_sem(vals, circular=circ)
        out[label] = cells
    return out


# ── Panel B data: per-residue rotamer shift ──────────────────────────────────

def compute_rotamer_data(df_raw):
    """Return dict: angle_label -> dict[(res, basin) -> (shift, sem, n)]."""
    out = {}
    for col, circ, label in _ANGLES:
        d = prepare_delta(df_raw, col, circ, res_filter=_RES_CHI1)
        d = d.dropna(subset=['chi1_rad'])
        if 'has_chi1' in d.columns:
            d = d[d['has_chi1'] == 1]
        d['rot'] = d['chi1_rad'].apply(_rot_label)
        d = d[d['rot'].notna()]

        cells = {}
        for res in _RES_CHI1:
            sub_res = d[d['res_name'] == res]
            for reg in _REGION_ORDER:
                sub_rg = subset_region(sub_res, reg)
                gm_vals = sub_rg.loc[sub_rg['rot'] == 'g-', 'delta'].values
                gp_vals = sub_rg.loc[sub_rg['rot'] == 'g+', 'delta'].values
                m_minus, sem_minus, n_minus = mean_sem(gm_vals, circ)
                m_plus,  sem_plus,  n_plus  = mean_sem(gp_vals, circ)
                if np.isnan(m_minus) or np.isnan(m_plus):
                    cells[(res, reg)] = (float('nan'), float('nan'),
                                          n_minus, n_plus)
                else:
                    if circ:
                        shift = _circ_diff(m_plus, m_minus)
                    else:
                        shift = m_plus - m_minus
                    sem = np.sqrt(sem_minus**2 + sem_plus**2)
                    cells[(res, reg)] = (shift, sem, n_minus, n_plus)
        out[label] = cells
    return out


# ── Plotting helpers ─────────────────────────────────────────────────────────

def plot_class_bars_panel(axes4, class_data):
    """Plot 4 sub-panels of class-stratified bars. axes4 is a 1D array of 4 axes."""
    for k, (col, circ, label) in enumerate(_ANGLES):
        ax = axes4[k]
        cells = class_data[label]
        n_basins = len(_REGION_ORDER)
        n_classes = len(_CLASS_ORDER)
        bar_w = 0.15
        x = np.arange(n_basins)

        for ci, cls in enumerate(_CLASS_ORDER):
            means, sems = [], []
            for reg in _REGION_ORDER:
                m, sem, n = cells[(reg, cls)]
                if n >= 30 and np.isfinite(m):
                    means.append(m); sems.append(sem)
                else:
                    means.append(0); sems.append(0)
            offset = (ci - (n_classes - 1) / 2) * bar_w
            ax.bar(x + offset, means, width=bar_w, yerr=sems,
                    color=_CLASS_COLOR[cls],
                    label=cls if k == 0 else None,
                    capsize=2, edgecolor='white', linewidth=0.6)

        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(_REGION_ORDER, fontsize=10)
        ax.set_ylabel('Δ (deg)', fontsize=10)
        ax.set_title(label, fontsize=11, pad=4)
        ax.grid(True, axis='y', alpha=0.25)
        ax.tick_params(axis='y', labelsize=9)

    # Legend on top-left panel only
    axes4[0].legend(fontsize=9, loc='upper left', ncol=5,
                     bbox_to_anchor=(0, 1.35), frameon=False,
                     handlelength=1.2, columnspacing=1.2)


def plot_rotamer_heatmap_panel(axes4, rot_data):
    """Plot 4 sub-panels of rotamer-shift heatmaps. axes4 is a 1D array of 4 axes."""
    for k, (col, circ, label) in enumerate(_ANGLES):
        ax = axes4[k]
        cells = rot_data[label]

        mat = np.full((len(_RES_CHI1), len(_REGION_ORDER)), np.nan)
        for i, res in enumerate(_RES_CHI1):
            for j, reg in enumerate(_REGION_ORDER):
                mat[i, j] = cells[(res, reg)][0]

        finite = mat[np.isfinite(mat)]
        if len(finite) > 0:
            vlim = np.ceil(np.percentile(np.abs(finite), 95) * 4) / 4
            vlim = max(vlim, 0.3)
        else:
            vlim = 1.0

        masked = np.ma.masked_invalid(mat)
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad(color='#dddddd')
        im = ax.imshow(masked, cmap=cmap, vmin=-vlim, vmax=vlim, aspect='auto')

        ax.set_xticks(range(len(_REGION_ORDER)))
        ax.set_xticklabels(_REGION_ORDER, fontsize=9)
        ax.set_yticks(range(len(_RES_CHI1)))
        ax.set_yticklabels(_RES_CHI1, fontsize=8)
        for tick, res in zip(ax.get_yticklabels(), _RES_CHI1):
            if res in _BRANCHED:
                tick.set_color('#c0392b')
                tick.set_fontweight('bold')

        ax.set_title(label, fontsize=11, pad=4)

        # Annotate each cell with its value (compact, small font)
        for i in range(len(_RES_CHI1)):
            for j in range(len(_REGION_ORDER)):
                v = mat[i, j]
                if np.isnan(v):
                    ax.text(j, i, 'n/a', ha='center', va='center',
                             fontsize=6, color='#888')
                else:
                    color = 'white' if abs(v) > vlim * 0.6 else 'black'
                    ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                             fontsize=6.5, color=color)

        cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cbar.set_label('Δ(g⁺)−Δ(g⁻) (deg)', fontsize=8)
        cbar.ax.tick_params(labelsize=7)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='paper2_F2_sidechain_lever.png')
    ap.add_argument('--dpi', type=int, default=220)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found"); sys.exit(1)

    print(f"Loading {csv_path} ...")
    df_raw = pd.read_csv(csv_path)
    print(f"  {len(df_raw):,} total rows")

    print("Computing class-stratified data (Panel A) ...")
    class_data = compute_class_data(df_raw)

    print("Computing per-residue rotamer shifts (Panel B) ...")
    rot_data = compute_rotamer_data(df_raw)

    print("Assembling figure ...")
    # Layout: 2 main rows (A on top, B on bottom). Each main row has 4 sub-panels.
    # Give Panel B more vertical space because heatmaps are denser.
    fig = plt.figure(figsize=(18, 10))
    gs_main = fig.add_gridspec(
        2, 1,
        height_ratios=[1.0, 1.8],
        hspace=0.25,
        left=0.07, right=0.96, top=0.93, bottom=0.05,
    )

    # ── Panel A: 1×4 subgrid of class bars ───────────────────────────────────
    gs_A = gs_main[0].subgridspec(1, 4, wspace=0.30)
    ax_A_parent = fig.add_subplot(gs_main[0])
    ax_A_parent.set_xticks([]); ax_A_parent.set_yticks([])
    for s in ax_A_parent.spines.values():
        s.set_visible(False)
    ax_A_parent.set_title('A', fontsize=16, fontweight='bold', loc='left', pad=10)
    ax_A_parent.patch.set_alpha(0)

    axes_A = [fig.add_subplot(gs_A[0, i]) for i in range(4)]
    plot_class_bars_panel(axes_A, class_data)

    # ── Panel B: 1×4 subgrid of rotamer heatmaps ─────────────────────────────
    gs_B = gs_main[1].subgridspec(1, 4, wspace=0.55)
    ax_B_parent = fig.add_subplot(gs_main[1])
    ax_B_parent.set_xticks([]); ax_B_parent.set_yticks([])
    for s in ax_B_parent.spines.values():
        s.set_visible(False)
    ax_B_parent.set_title('B', fontsize=16, fontweight='bold', loc='left', pad=10)
    ax_B_parent.patch.set_alpha(0)

    axes_B = [fig.add_subplot(gs_B[0, i]) for i in range(4)]
    plot_rotamer_heatmap_panel(axes_B, rot_data)

    # ── Super-title ──────────────────────────────────────────────────────────
    fig.suptitle(
        'Figure 2.  The sidechain mechanical lever: '
        'class-stratified and χ1-rotamer effects on backbone angles',
        fontsize=13, fontweight='bold', y=0.975)

    plt.savefig(args.out, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved: {args.out}  ({args.dpi} dpi)")


if __name__ == '__main__':
    main()