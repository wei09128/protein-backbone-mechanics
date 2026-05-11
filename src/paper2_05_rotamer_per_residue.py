"""
paper2_05_rotamer_per_residue.py — χ1 rotamer effect, per residue
====================================================================

Move A analysis 2 pooled over all residues with valid χ1, and found a
massive rotamer effect on backbone angles (g⁻ vs g⁺ differences up to
1.15°, t-stats up to −129). But the pooled analysis leaves open whether:

  (a) The effect is universal: every residue shows the same direction
      (e.g., g⁺ > g⁻ for every AA in every basin) → clean lever story.

  (b) The effect is driven by a subset: β-branched residues carry the
      signal, others are flat → rotamer effect is a branching effect.

  (c) It's a composition artifact: different residues populate different
      rotamers with different preferences, and (φ,ψ) preferences vary too,
      so the pooled signal is an illusion.

This script tests by computing Δ_{g⁺} − Δ_{g⁻} for each residue × basin
× angle individually. If the sign is consistent across residues, the
effect is universal. If only a few residues drive it, we know who.

Output
------
  - Per-residue table: Δ_{g⁺} − Δ_{g⁻} for each basin × angle
  - Sign-agreement summary (like Move B, but now over residues for
    the rotamer effect instead of the basin effect)
  - Heatmap figure: 4 panels (one per angle), rows = residues,
    cols = basins, cell = g⁺ − g⁻ shift

Usage
-----
    python paper2_05_rotamer_per_residue.py --csv features.csv
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Residues with meaningful χ1 (exclude GLY, ALA — no χ1; PRO — ring)
_RES_CHI1 = ['ARG','ASN','ASP','CYS','GLN','GLU','HIS','ILE','LEU',
             'LYS','MET','PHE','SER','THR','TRP','TYR','VAL']

# β-branched residues (highlighted in figure)
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
        return 'g⁻'
    if np.pi / 6 < chi1_rad < np.pi * 5 / 6:
        return 'g⁺'
    return None


def prepare_delta(df, angle, circular):
    df = df.dropna(subset=[angle, 'phi_deg', 'psi_deg', 'res_name',
                             'chi1_rad']).copy()
    df = df[df['res_name'].isin(_RES_CHI1)]
    # Keep only residues with valid chi1 (has_chi1 = 1 if col present)
    if 'has_chi1' in df.columns:
        df = df[df['has_chi1'] == 1]
    # Per-residue median subtraction (so the rotamer effect is on top of baseline)
    if circular:
        medians = df.groupby('res_name')[angle].apply(
            lambda g: _circ_median_deg(g.values))
        df['delta'] = _circ_diff(df[angle].values,
                                  df['res_name'].map(medians).values)
    else:
        medians = df.groupby('res_name')[angle].median()
        df['delta'] = df[angle] - df['res_name'].map(medians)
    df['rot'] = df['chi1_rad'].apply(_rot_label)
    df = df[df['rot'].notna()]
    return df


def mean_sem(vals, circular):
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 30:
        return float('nan'), float('nan'), len(vals)
    if circular:
        m = _circ_mean_deg(vals)
        s = float(np.std(_circ_diff(vals, m)))
    else:
        m = float(vals.mean())
        s = float(vals.std())
    return m, s / np.sqrt(len(vals)), len(vals)


def per_residue_rotamer_shift(df, circular):
    """
    For each residue × basin, compute (mean_gplus − mean_gminus) and its SEM.
    Returns a dict: {(res, basin): (shift, sem, n_g_minus, n_g_plus)}
    """
    result = {}
    for res in _RES_CHI1:
        sub_res = df[df['res_name'] == res]
        for reg in _REGION_ORDER:
            p, q = _REGIONS[reg]['phi'], _REGIONS[reg]['psi']
            mask = (sub_res['phi_deg'].between(p[0], p[1])
                    & sub_res['psi_deg'].between(q[0], q[1]))
            sub_rg = sub_res[mask]
            gm_vals = sub_rg.loc[sub_rg['rot'] == 'g⁻', 'delta'].values
            gp_vals = sub_rg.loc[sub_rg['rot'] == 'g⁺', 'delta'].values
            m_minus, sem_minus, n_minus = mean_sem(gm_vals, circular)
            m_plus,  sem_plus,  n_plus  = mean_sem(gp_vals, circular)
            if np.isnan(m_minus) or np.isnan(m_plus):
                shift = float('nan')
                shift_sem = float('nan')
            else:
                if circular:
                    shift = _circ_diff(m_plus, m_minus)
                else:
                    shift = m_plus - m_minus
                shift_sem = np.sqrt(sem_minus ** 2 + sem_plus ** 2)
            result[(res, reg)] = (shift, shift_sem, n_minus, n_plus)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='paper2_05_rotamer_per_residue.png')
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found"); sys.exit(1)

    print(f"Loading {csv_path} ...")
    df_raw = pd.read_csv(csv_path)
    print(f"  {len(df_raw):,} total rows")

    # Compute per-residue shifts for each angle
    all_shifts = {}  # angle_label -> dict[(res, reg) -> (shift, sem, nm, np)]
    for col, circ, label in _ANGLES:
        d = prepare_delta(df_raw, col, circ)
        all_shifts[label] = per_residue_rotamer_shift(d, circ)

    # ── Table per angle ──────────────────────────────────────────────────────
    for col, circ, label in _ANGLES:
        print("\n" + "=" * 86)
        print(f"Δ(g⁺) − Δ(g⁻) per residue × basin  —  {label}")
        print("=" * 86)
        print(f"  {'res':<4} {'branched':>8}  "
              f"{'α':>14} {'β':>14} {'PPII':>14} {'αL':>14}")
        print("  " + "-" * 84)

        shifts = all_shifts[label]
        for res in _RES_CHI1:
            br = 'Y' if res in _BRANCHED else '.'
            parts = [f"  {res:<4} {br:>8}"]
            for reg in _REGION_ORDER:
                shift, sem, nm, np_ = shifts[(res, reg)]
                if np.isnan(shift):
                    parts.append(f"{'n='+str(min(nm,np_)):>14}")
                else:
                    parts.append(f"{shift:+8.3f}±{sem:.3f}".rjust(14))
            print(''.join(parts))

    # ── Sign agreement summary ───────────────────────────────────────────────
    print("\n" + "=" * 86)
    print("SIGN AGREEMENT — does g⁺ > g⁻ (or g⁺ < g⁻) hold universally?")
    print("=" * 86)
    print("For each angle × basin we ask: of residues with valid data,")
    print("how many agree with the pooled-analysis sign?")
    print("")

    # Pooled signs from Move A analysis 2 (hardcoded from the output you saw):
    # We'll compute them from this data to stay honest.
    pooled_signs = {}
    for col, circ, label in _ANGLES:
        d = prepare_delta(df_raw, col, circ)
        pooled_signs[label] = {}
        for reg in _REGION_ORDER:
            p, q = _REGIONS[reg]['phi'], _REGIONS[reg]['psi']
            mask = d['phi_deg'].between(p[0], p[1]) & d['psi_deg'].between(q[0], q[1])
            sub = d[mask]
            gm = sub.loc[sub['rot'] == 'g⁻', 'delta'].values
            gp = sub.loc[sub['rot'] == 'g⁺', 'delta'].values
            if len(gm) >= 30 and len(gp) >= 30:
                if circ:
                    diff = _circ_diff(_circ_mean_deg(gp), _circ_mean_deg(gm))
                else:
                    diff = float(gp.mean() - gm.mean())
                pooled_signs[label][reg] = +1 if diff > 0 else -1
            else:
                pooled_signs[label][reg] = 0

    print(f"  {'angle':<16} {'basin':<6}  {'pooled':>7}  "
          f"{'agree':>5} {'disagree':>8} {'n<30':>5}  disagreeing residues")
    print("  " + "-" * 84)

    for col, circ, label in _ANGLES:
        for reg in _REGION_ORDER:
            exp = pooled_signs[label][reg]
            agree = disagree = nil = 0
            disagreeing = []
            for res in _RES_CHI1:
                shift, sem, nm, np_ = all_shifts[label][(res, reg)]
                if np.isnan(shift):
                    nil += 1; continue
                if shift * exp > 0:
                    agree += 1
                else:
                    disagree += 1
                    disagreeing.append(f"{res}({shift:+.2f})")
            sign_sym = '+' if exp > 0 else ('−' if exp < 0 else '0')
            dlist = ', '.join(disagreeing[:5])
            if len(disagreeing) > 5:
                dlist += f' + {len(disagreeing)-5} more'
            print(f"  {label:<16} {reg:<6}  {sign_sym:>7}  "
                  f"{agree:>5} {disagree:>8} {nil:>5}  {dlist}")

    # ── Figure: 4-panel heatmap, rows=residues, cols=basins ──────────────────
    print(f"\nGenerating {args.out} ...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Global color limit: use 95th pct of absolute shifts
    all_vals = []
    for label in [l for _, _, l in _ANGLES]:
        for res in _RES_CHI1:
            for reg in _REGION_ORDER:
                s = all_shifts[label][(res, reg)][0]
                if not np.isnan(s):
                    all_vals.append(abs(s))

    for k, (col, circ, label) in enumerate(_ANGLES):
        ax = axes[k]
        shifts = all_shifts[label]
        # Build matrix
        mat = np.full((len(_RES_CHI1), len(_REGION_ORDER)), np.nan)
        for i, res in enumerate(_RES_CHI1):
            for j, reg in enumerate(_REGION_ORDER):
                mat[i, j] = shifts[(res, reg)][0]
        # Auto color limit per panel
        finite = mat[np.isfinite(mat)]
        if len(finite) > 0:
            vlim = np.ceil(np.percentile(np.abs(finite), 95) * 4) / 4
            vlim = max(vlim, 0.3)
        else:
            vlim = 1.0

        masked = np.ma.masked_invalid(mat)
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad(color='#dddddd')
        im = ax.imshow(masked, cmap=cmap, vmin=-vlim, vmax=vlim,
                        aspect='auto')

        ax.set_xticks(range(len(_REGION_ORDER)))
        ax.set_xticklabels(_REGION_ORDER)
        ax.set_yticks(range(len(_RES_CHI1)))
        # Colour β-branched residue labels red
        ax.set_yticklabels(_RES_CHI1, fontsize=9)
        for tick, res in zip(ax.get_yticklabels(), _RES_CHI1):
            if res in _BRANCHED:
                tick.set_color('#c0392b')
                tick.set_fontweight('bold')

        ax.set_title(label, fontsize=12)
        ax.set_xlabel('Ramachandran basin')

        # Annotate each cell with the value
        for i in range(len(_RES_CHI1)):
            for j in range(len(_REGION_ORDER)):
                v = mat[i, j]
                if np.isnan(v):
                    ax.text(j, i, 'n/a', ha='center', va='center',
                             fontsize=7, color='#888')
                else:
                    color = 'white' if abs(v) > vlim * 0.6 else 'black'
                    ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                             fontsize=7, color=color)

        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        cbar.set_label('Δ(g⁺) − Δ(g⁻) (deg)', fontsize=9)

    plt.suptitle('χ1 rotamer shift per residue × basin '
                  '(red labels = β-branched)',
                  fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches='tight')
    print(f"Figure saved: {args.out}")

    # ── Verdict ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 86)
    print("INTERPRETATION")
    print("=" * 86)
    print("• Rows with the same color everywhere → effect is uniform across")
    print("  basins for that residue.")
    print("• Columns with the same color everywhere → effect is uniform across")
    print("  residues for that basin (the universality test).")
    print("• If β-branched residues (red labels) stand out from the rest,")
    print("  the rotamer effect is branching-driven, not universal.")
    print("• If the patterns are identical across all residues, the lever")
    print("  story is a clean universal mechanical coupling.")


if __name__ == '__main__':
    main()