"""
subgroup_projection_analysis.py  (FIXED — αR included)
========================================================
Fix: αR basin previously had hat=[0,0] because its restoring direction
pointed toward itself. The fix uses per-residue displacement from the
αR *basin centre* as the projection axis for αR residues, so we test
whether forces oppose displacement within the well (Hooke's law), rather
than a direction that is geometrically degenerate.

For all other basins the logic is unchanged: projection is onto the
unit vector from basin centre → αR reference.
"""

import argparse
import csv as _csv_mod
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import sys
_csv_mod.field_size_limit(sys.maxsize)
warnings.filterwarnings('ignore')

REF_PHI = -63.0
REF_PSI = -43.0

BASIN_NAMES   = {0: 'αR', 1: 'β', 2: 'PPII', 3: '3₁₀', 4: 'loop', 5: 'αL'}
BASIN_CENTRES = {
    0: (-63,  -43),
    1: (-120, 128),
    2: (-72,  146),
    3: (-52,  -32),
    4: (-95,   10),
    5: (  60,  40),
}
BASIN_COLORS  = {
    0: '#1D9E75',
    1: '#378ADD',
    2: '#BA7517',
    3: '#D4537E',
    4: '#888780',
    5: '#9B59B6',
}

AA_GROUPS = {
    'Gly':      {'GLY'},
    'Pro':      {'PRO'},
    'Branched': {'VAL', 'ILE', 'THR'},
    'Aromatic': {'PHE', 'TYR', 'TRP'},
    'Charged':  {'LYS', 'ARG', 'ASP', 'GLU', 'HIS'},
    'Polar':    {'SER', 'ASN', 'GLN', 'CYS'},
    'Aliphatic':{'ALA', 'LEU', 'MET'},
}

def get_aa_groups(res):
    groups = [grp for grp, aas in AA_GROUPS.items() if res in aas]
    return groups if groups else ['Other']

def wrap(a: float) -> float:
    return ((a + 180.0) % 360.0) - 180.0

def _ss_bin(phi: float, psi: float) -> int:
    p, q = phi, psi
    if p > 0 and -20 <= q <= 80:              return 5
    if -100 <= p <= -40 and -60 <= q <= 20:   return 0
    if p <= -90 and q >= 90:                   return 1
    if -90 <= p <= -50 and q >= 120:           return 2
    if -80 <= p <= -30 and -40 <= q <= 0:      return 3
    return 4


def _f(row, key, default=0.0):
    try:
        v = row.get(key, '')
        return float(v) if v not in ('', None) else default
    except (ValueError, TypeError):
        return default


def load_data(csv_path, max_rows=None):
    rows = []
    with open(csv_path, newline='') as fh:
        sample = fh.read(4096); fh.seek(0)
        try:
            dialect = _csv_mod.Sniffer().sniff(sample, delimiters='\t,')
            delim = dialect.delimiter
        except Exception:
            delim = ','
        reader = _csv_mod.DictReader(fh, delimiter=delim)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            rows.append(row)

    print(f"  {len(rows):,} rows loaded")

    records = []
    for row in rows:
        phi = _f(row, 'phi_deg')
        psi = _f(row, 'psi_deg')
        if abs(phi) < 0.5 or abs(psi) < 0.5: continue

        res   = row.get('res_name', 'ALA').strip().upper()
        tau_phi = _f(row, 'tau_phi_correct')
        tau_psi = _f(row, 'tau_psi_correct')

        basin = _ss_bin(phi, psi)
        phi_c, psi_c = BASIN_CENTRES[basin]

        # Displacement of this residue from its basin centre
        d_phi = wrap(phi - phi_c)
        d_psi = wrap(psi - psi_c)
        displacement = float(np.sqrt(d_phi**2 + d_psi**2))

        # ── KEY FIX ──────────────────────────────────────────────────────────
        # For αR (basin 0): the restoring direction is *away from* the basin
        # centre toward the periphery — we want to test whether forces oppose
        # displacement, i.e. projection onto the unit vector pointing FROM
        # the basin centre TO the residue.  A negative projection means the
        # force points back toward centre = restoring.
        #
        # For all other basins: restoring direction = basin centre → αR ref,
        # unchanged from original logic.
        # ─────────────────────────────────────────────────────────────────────
        if basin == 0:
            # Use per-residue displacement direction (toward residue from centre)
            disp_vec = np.array([d_phi, d_psi])
            dv_norm  = float(np.linalg.norm(disp_vec))
            if dv_norm > 1.0:
                hat = disp_vec / dv_norm
            else:
                hat = np.array([0.0, 0.0])   # residue is exactly at centre
            # Convention: negative projection = force opposes displacement = restoring
            # We flip sign so that "positive = restoring" is consistent across basins
            projection = -float(tau_phi * hat[0] + tau_psi * hat[1])
        else:
            dv = np.array([wrap(REF_PHI - phi_c), wrap(REF_PSI - psi_c)])
            dv_norm = float(np.linalg.norm(dv))
            hat = dv / dv_norm if dv_norm > 1.0 else np.array([0.0, 0.0])
            projection = float(tau_phi * hat[0] + tau_psi * hat[1])

        tau_mag = float(np.sqrt(tau_phi**2 + tau_psi**2))

        records.append({
            'phi':       phi,
            'psi':       psi,
            'res_name':  res,
            'aa_groups': get_aa_groups(res),
            'basin':     basin,
            'tau_phi':   tau_phi,
            'tau_psi':   tau_psi,
            'tau_mag':   tau_mag,
            'projection': projection,
            'disp':      displacement,
            'hat':       hat,
        })

    print(f"  {len(records):,} residues parsed")
    for b in sorted(BASIN_NAMES):
        n = sum(1 for r in records if r['basin'] == b)
        print(f"    {BASIN_NAMES[b]}: {n:,}")
    return records


def filter_valid_groups(records, min_n=10):
    counts = defaultdict(lambda: defaultdict(int))
    for r in records:
        for grp in r['aa_groups']:
            counts[r['basin']][grp] += 1
    valid = {
        basin: {g for g, c in grp_counts.items() if c >= min_n}
        for basin, grp_counts in counts.items()
    }
    filtered = []
    for r in records:
        valid_groups = [g for g in r['aa_groups']
                        if any(grp in valid[r['basin']] for grp in r['aa_groups'])]
        if valid_groups:
            r['aa_groups'] = valid_groups
            filtered.append(r)
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# Analysis i
# ══════════════════════════════════════════════════════════════════════════════

def analyse_by_aa(records):
    from scipy import stats
    data = defaultdict(lambda: defaultdict(list))
    for r in records:
        for grp in r['aa_groups']:
            data[r['basin']][grp].append(r['projection'])

    results = {}
    print(f"\n{'─'*70}")
    print(f"  Analysis i — Per-AA-group restoring projection")
    print(f"  (αR: positive = forces oppose displacement from centre = restoring)")
    print(f"  (others: positive = forces point toward αR)")
    print(f"{'─'*70}")
    print(f"  Basin     AA-group    n       mean±SE           t-stat    p-val")
    print(f"  {'─'*65}")

    for basin in sorted(BASIN_NAMES.keys()):
        results[basin] = {}
        for grp in list(AA_GROUPS.keys()) + ['Other']:
            vals = np.array(data[basin][grp])
            if len(vals) < 10:
                continue
            mean = float(np.mean(vals))
            se   = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            t, p = stats.ttest_1samp(vals, 0.0)
            results[basin][grp] = {
                'mean': mean, 'se': se, 'n': len(vals),
                'projections': vals, 't': t, 'p': p,
            }
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            print(f"  {BASIN_NAMES[basin]:<6}  {grp:<12}  {len(vals):>5}  "
                  f"{mean:>+8.4f} ± {se:.4f}   {t:>+8.3f}   {p:.2e} {sig}")
        print()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Analysis ii
# ══════════════════════════════════════════════════════════════════════════════

def analyse_sign_consistency(records):
    from scipy import stats
    print(f"\n{'─'*70}")
    print(f"  Analysis ii — Sign consistency of restoring projection")
    print(f"  (αR: positive = opposes displacement, i.e. restoring within well)")
    print(f"{'─'*70}")
    print(f"  Basin     AA-group       n    %>0    %<0   binom-p  interpretation")
    print(f"  {'─'*66}")

    results = defaultdict(dict)
    for basin in sorted(BASIN_NAMES.keys()):
        basin_recs = [r for r in records if r['basin'] == basin]
        if not basin_recs:
            continue
        projs_all = np.array([r['projection'] for r in basin_recs])
        _print_sign_row(BASIN_NAMES[basin], 'ALL', projs_all, results[basin], stats)
        for grp in list(AA_GROUPS.keys()) + ['Other']:
            vals = np.array([r['projection'] for r in basin_recs
                             if grp in r['aa_groups']])
            if len(vals) >= 10:
                _print_sign_row(BASIN_NAMES[basin], grp, vals,
                                results[basin], stats, indent=True)
        print()
    return results


def _print_sign_row(basin_name, grp_name, projs, store, stats, indent=False):
    n      = len(projs)
    n_pos  = int(np.sum(projs > 0))
    n_neg  = int(np.sum(projs < 0))
    pct_p  = 100.0 * n_pos / n
    pct_n  = 100.0 * n_neg / n
    binom  = stats.binomtest(n_pos, n, p=0.5)
    bp     = binom.pvalue
    sig    = '***' if bp < 0.001 else ('**' if bp < 0.01 else ('*' if bp < 0.05 else ''))
    if pct_p > 65:   interp = 'coherent restoring'
    elif pct_p < 35: interp = 'coherent driving'
    else:            interp = 'ambivalent'
    prefix = '  ' if indent else ''
    label  = f"  {prefix}{basin_name:<6}  {grp_name:<14}"
    print(f"{label}  {n:>5}  {pct_p:>5.1f}  {pct_n:>5.1f}  "
          f"{bp:.2e}  {interp} {sig}")
    store[grp_name] = {
        'n': n, 'pct_pos': pct_p, 'pct_neg': pct_n,
        'binom_p': bp, 'interp': interp,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Analysis iii
# ══════════════════════════════════════════════════════════════════════════════

def analyse_torque_vs_displacement(records):
    from scipy import stats
    print(f"\n{'─'*70}")
    print(f"  Analysis iii — Torque magnitude vs displacement from basin centre")
    print(f"  (αR: restoring = negative slope in proj vs disp, shown flipped)")
    print(f"{'─'*70}")
    print(f"  Basin   AA-group       n    r(|τ|,disp)  r(proj,disp)  slope(k)  p-val")
    print(f"  {'─'*68}")

    results = defaultdict(dict)
    for basin in sorted(BASIN_NAMES.keys()):
        basin_recs = [r for r in records if r['basin'] == basin]
        if len(basin_recs) < 20:
            continue
        _print_disp_row(BASIN_NAMES[basin], 'ALL', basin_recs, results[basin], stats)
        for grp in list(AA_GROUPS.keys()) + ['Other']:
            sub = [r for r in basin_recs if grp in r['aa_groups']]
            if len(sub) < 15:
                continue
            _print_disp_row(BASIN_NAMES[basin], grp, sub,
                            results[basin], stats, indent=True)
        print()
    return results


def _print_disp_row(basin_name, grp_name, recs, store, stats, indent=False):
    disp  = np.array([r['disp']       for r in recs])
    tmag  = np.array([r['tau_mag']    for r in recs])
    proj  = np.array([r['projection'] for r in recs])
    r_mag,  p_mag  = stats.pearsonr(disp, tmag)
    r_proj, p_proj = stats.pearsonr(disp, proj)
    slope, intercept, _, p_slope, _ = stats.linregress(disp, proj)
    sig = '***' if p_slope < 0.001 else ('**' if p_slope < 0.01 else
          ('*' if p_slope < 0.05 else ''))
    prefix = '  ' if indent else ''
    label  = f"  {prefix}{basin_name:<5}  {grp_name:<14}"
    print(f"{label}  {len(recs):>5}  {r_mag:>+.4f}       {r_proj:>+.4f}   "
          f"{slope:>+8.4f}  {p_slope:.2e} {sig}")
    store[grp_name] = {
        'n': len(recs),
        'r_mag': r_mag,   'p_mag': p_mag,
        'r_proj': r_proj, 'p_proj': p_proj,
        'slope': slope,   'p_slope': p_slope,
        'disp': disp, 'tmag': tmag, 'proj': proj,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plots — ALL BASINS including αR
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(records, aa_results, sign_results, disp_results, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy import stats

    out = Path(out_dir)

    # ── Figure 1: Per-AA-group projection — all 6 basins ─────────────────────
    all_basins = sorted(BASIN_NAMES.keys())   # 0–5 inclusive
    fig, axes = plt.subplots(2, 3, figsize=(21, 11))
    axes_flat = axes.flat
    fig.suptitle(
        'Restoring projection per AA group × basin\n'
        'αR: positive = forces oppose displacement from centre (within-well restoring)\n'
        'Others: positive = forces point toward αR reference',
        fontsize=13, fontweight='bold'
    )

    for ax, basin in zip(axes_flat, all_basins):
        bd = aa_results.get(basin, {})
        if not bd:
            ax.set_visible(False)
            continue

        groups = sorted(bd.keys())
        means  = [bd[g]['mean'] for g in groups]
        ses    = [bd[g]['se']   for g in groups]
        ns     = [bd[g]['n']    for g in groups]
        ps     = [bd[g]['p']    for g in groups]
        y      = np.arange(len(groups))

        bar_colors = [('#E24B4A' if m < 0 else '#1D9E75') for m in means]
        ax.barh(y, means, xerr=ses, color=bar_colors, alpha=0.75,
                capsize=4, height=0.6)
        ax.axvline(0, color='black', lw=1.0, ls='--')

        for yi, (m, p) in enumerate(zip(means, ps)):
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else
                  ('*' if p < 0.05 else ''))
            if sig:
                offset = (ses[yi] + 0.005) * (1 if m >= 0 else -1)
                ax.text(m + offset, yi, sig, va='center', fontsize=9)

        ax.set_yticks(y)
        ax.set_yticklabels([f"{g} (n={ns[i]:,})" for i, g in enumerate(groups)],
                           fontsize=8)
        ax.set_xlabel('Mean restoring projection', fontsize=9)

        # Label for αR clarifies the different convention
        if basin == 0:
            subtitle = '(within-well: + = opposes displacement)'
        else:
            subtitle = '(toward αR: + = restoring)'
        ax.set_title(f'{BASIN_NAMES[basin]}\n{subtitle}',
                     color=BASIN_COLORS[basin], fontweight='bold', fontsize=11)
        ax.invert_yaxis()

    plt.tight_layout()
    p = out / 'proj_by_aa_group.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig 1 → {p}")

    # ── Figure 2: Sign consistency — all basins ───────────────────────────────
    basins_to_plot = [b for b in all_basins
                      if b in sign_results and 'ALL' in sign_results[b]]
    n_basins = len(basins_to_plot)

    fig, axes = plt.subplots(2, 3, figsize=(21, 11))
    axes_flat  = axes.flat
    fig.suptitle(
        'Sign consistency of restoring projection\n'
        'Green = restoring (>0) | Red = driving (<0)\n'
        'αR: restoring = forces oppose displacement within the well',
        fontsize=12, fontweight='bold'
    )

    for ax, basin in zip(axes_flat, basins_to_plot):
        bd     = sign_results[basin]
        groups = [g for g in (['ALL'] + list(AA_GROUPS.keys()) + ['Other'])
                  if g in bd]
        pos_vals = [bd[g]['pct_pos'] for g in groups]
        neg_vals = [bd[g]['pct_neg'] for g in groups]
        y = np.arange(len(groups))

        ax.barh(y,  pos_vals,           color='#1D9E75', alpha=0.80, height=0.55)
        ax.barh(y, [-v for v in neg_vals], color='#E24B4A', alpha=0.80, height=0.55)
        ax.axvline(0,   color='black',   lw=0.8)
        ax.axvline(50,  color='#1D9E75', lw=0.8, ls=':')
        ax.axvline(-50, color='#E24B4A', lw=0.8, ls=':')

        for yi, g in enumerate(groups):
            bp  = bd[g]['binom_p']
            sig = '***' if bp < 0.001 else ('**' if bp < 0.01 else
                  ('*' if bp < 0.05 else ''))
            if sig:
                ax.text(52, yi, sig, va='center', fontsize=8)

        ax.set_yticks(y)
        ax.set_yticklabels([f"{g} (n={bd[g]['n']:,})" for g in groups], fontsize=8)
        ax.set_xlabel('% residues', fontsize=9)
        if basin == 0:
            subtitle = '(+ = opposes displacement)'
        else:
            subtitle = '(+ = toward αR)'
        ax.set_title(f'{BASIN_NAMES[basin]}\n{subtitle}',
                     color=BASIN_COLORS[basin], fontweight='bold', fontsize=11)
        ax.set_xlim(-80, 80)
        ax.invert_yaxis()

    # Hide unused axes
    for ax in list(axes_flat)[len(basins_to_plot):]:
        ax.set_visible(False)

    handles = [
        mpatches.Patch(color='#1D9E75', alpha=0.80, label='>0 (restoring)'),
        mpatches.Patch(color='#E24B4A', alpha=0.80, label='<0 (driving)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    plt.tight_layout()
    p = out / 'sign_consistency.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig 2 → {p}")

    # ── Figure 3: Torque vs displacement — all basins ─────────────────────────
    basins_to_plot = all_basins
    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 12))
    axes_flat = axes.flat

    fig.suptitle(
        'Torque vs displacement from basin centre\n'
        'αR: negative OLS slope = forces oppose displacement = Hooke-like restoring\n'
        'Others: positive slope = forces point toward αR with increasing distance',
        fontsize=12, fontweight='bold'
    )

    for i, basin in enumerate(basins_to_plot):
        ax = axes_flat[i]
        bd = disp_results.get(basin, {}).get('ALL')
        if bd is None:
            ax.set_facecolor('#ffeeee')
            ax.set_title(f"{BASIN_NAMES[basin]}\n(no data)", color='red')
            continue

        disp = bd['disp']
        proj = bd['proj']

        n_scatter = min(len(disp), 3000)
        idx = np.random.choice(len(disp), n_scatter, replace=False)
        ax.scatter(disp[idx], proj[idx], s=2, alpha=0.15,
                   color=BASIN_COLORS[basin])

        # Binned mean ± SE
        n_bins = 10
        bin_edges = np.percentile(disp, np.linspace(0, 100, n_bins + 1))
        bin_centres, bin_means, bin_ses = [], [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (disp >= lo) & (disp < hi)
            if mask.sum() < 5:
                continue
            v = proj[mask]
            bin_centres.append(float(np.mean(disp[mask])))
            bin_means.append(float(np.mean(v)))
            bin_ses.append(float(np.std(v, ddof=1) / np.sqrt(len(v))))
        ax.errorbar(bin_centres, bin_means, yerr=bin_ses,
                    fmt='o-', color='black', lw=2, ms=5, capsize=3, zorder=5)

        slope   = bd['slope']
        p_sl    = bd['p_slope']
        xr      = np.linspace(disp.min(), disp.max(), 100)
        intercept = np.mean(proj) - slope * np.mean(disp)
        ax.plot(xr, slope * xr + intercept, 'r--', lw=1.5, alpha=0.7,
                label=f'OLS k={slope:+.4f}  p={p_sl:.1e}')

        ax.axhline(0, color='gray', lw=0.8, ls=':')
        ax.set_xlabel('Displacement from basin centre (°)', fontsize=9)
        if basin == 0:
            ylabel = 'Restoring projection\n(−τ·d̂, + = opposes displacement)'
        else:
            ylabel = 'Restoring projection\n(+ = toward αR)'
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(
            f'{BASIN_NAMES[basin]}\n'
            f'r(|τ|,d)={bd["r_mag"]:+.3f}  '
            f'r(proj,d)={bd["r_proj"]:+.3f}',
            color=BASIN_COLORS[basin], fontweight='bold', fontsize=10
        )
        ax.legend(fontsize=7, loc='best')

    # Hide unused axes
    for ax in list(axes_flat)[len(basins_to_plot):]:
        ax.set_visible(False)

    plt.tight_layout()
    p = out / 'torque_vs_displacement.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig 3 → {p}")

    # ── Figure 4: αL deep-dive — Gly vs non-Gly ──────────────────────────────
    al_recs   = [r for r in records if r['basin'] == 5]
    if len(al_recs) > 20:
        gly_proj  = np.array([r['projection'] for r in al_recs
                               if r['res_name'] == 'GLY'])
        ngly_proj = np.array([r['projection'] for r in al_recs
                               if r['res_name'] != 'GLY'])

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('αL deep-dive: Glycine vs non-Glycine\n'
                     'Hypothesis: Cβ removal liberates electrostatic stabilisation',
                     fontsize=12, fontweight='bold')

        ax = axes[0]
        if len(gly_proj) > 0:
            ax.hist(gly_proj,  bins=40, alpha=0.7, color='#9B59B6',
                    density=True, label=f'Gly (n={len(gly_proj):,})')
        if len(ngly_proj) > 0:
            ax.hist(ngly_proj, bins=40, alpha=0.7, color='#888780',
                    density=True, label=f'non-Gly (n={len(ngly_proj):,})')
        ax.axvline(0, color='black', lw=1, ls='--')
        for arr, color, label in [(gly_proj, '#9B59B6', 'Gly'),
                                   (ngly_proj, '#888780', 'non-Gly')]:
            if len(arr) > 0:
                ax.axvline(np.mean(arr), color=color, lw=2,
                           label=f'{label} mean={np.mean(arr):+.3f}')
        ax.set_xlabel('Restoring projection'); ax.set_ylabel('Density')
        ax.set_title('Projection distribution'); ax.legend(fontsize=8)

        ax = axes[1]
        for i, (label, arr, color) in enumerate([
            ('Gly', gly_proj, '#9B59B6'), ('non-Gly', ngly_proj, '#888780')
        ]):
            if len(arr) == 0:
                continue
            pct_pos = 100.0 * np.mean(arr > 0)
            pct_neg = 100.0 * np.mean(arr < 0)
            ax.bar(i - 0.2, pct_pos, 0.35, color=color, alpha=0.8)
            ax.bar(i + 0.2, pct_neg, 0.35, color=color, alpha=0.4)
            ax.text(i - 0.2, pct_pos + 1, f'{pct_pos:.0f}%', ha='center', fontsize=9)
            ax.text(i + 0.2, pct_neg + 1, f'{pct_neg:.0f}%', ha='center', fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Gly', 'non-Gly'])
        ax.set_ylabel('% residues')
        ax.set_title('Sign consistency\nDark=restoring, Light=driving')
        ax.axhline(50, color='gray', ls=':', lw=0.8)

        ax = axes[2]
        for label, sub_recs, color in [
            ('Gly',     [r for r in al_recs if r['res_name'] == 'GLY'],  '#9B59B6'),
            ('non-Gly', [r for r in al_recs if r['res_name'] != 'GLY'],  '#888780'),
        ]:
            if len(sub_recs) < 10:
                continue
            d  = np.array([r['disp']       for r in sub_recs])
            pr = np.array([r['projection'] for r in sub_recs])
            sl, ic, _, pv, _ = stats.linregress(d, pr)
            xr = np.linspace(d.min(), d.max(), 100)
            ax.scatter(d[::max(1, len(d)//500)], pr[::max(1, len(pr)//500)],
                       s=3, alpha=0.2, color=color)
            ax.plot(xr, sl * xr + ic, color=color, lw=2,
                    label=f'{label}  k={sl:+.4f}  p={pv:.1e}')
        ax.axhline(0, color='gray', ls=':', lw=0.8)
        ax.set_xlabel('Displacement from αL centre (°)')
        ax.set_ylabel('Restoring projection')
        ax.set_title('Torque vs displacement\nGly vs non-Gly')
        ax.legend(fontsize=8)

        plt.tight_layout()
        p = out / 'aL_gly_vs_nongly.png'
        plt.savefig(p, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Fig 4 → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# Text summary
# ══════════════════════════════════════════════════════════════════════════════

def write_summary(aa_results, sign_results, disp_results, out_dir):
    lines = [
        "Subgroup Projection Analysis — Summary (αR INCLUDED)",
        "=" * 60,
        "",
        "NOTE: αR projection convention differs from other basins.",
        "  αR:    positive = forces OPPOSE displacement from centre (restoring within well)",
        "  others: positive = forces point TOWARD αR reference",
        "",
        "ANALYSIS i: Mean restoring projection by AA group",
        "─" * 60,
    ]
    for basin in sorted(BASIN_NAMES.keys()):
        bd = aa_results.get(basin, {})
        if not bd:
            continue
        lines.append(f"  {BASIN_NAMES[basin]}:")
        for g, d in sorted(bd.items(), key=lambda x: x[1]['mean']):
            sig = '***' if d['p'] < 0.001 else ('**' if d['p'] < 0.01 else
                  ('*' if d['p'] < 0.05 else 'ns'))
            lines.append(f"    {g:<12}  n={d['n']:>5}  "
                         f"mean={d['mean']:>+.4f} ± {d['se']:.4f}  {sig}")
        lines.append("")

    lines += ["", "ANALYSIS ii: Sign consistency", "─" * 60]
    for basin in sorted(BASIN_NAMES.keys()):
        bd = sign_results.get(basin, {})
        if 'ALL' not in bd:
            continue
        d = bd['ALL']
        lines.append(f"  {BASIN_NAMES[basin]}:  {d['pct_pos']:.1f}% restoring  "
                     f"{d['pct_neg']:.1f}% driving  "
                     f"binom-p={d['binom_p']:.2e}  → {d['interp']}")
    lines.append("")

    lines += ["", "ANALYSIS iii: Torque vs displacement", "─" * 60]
    for basin in sorted(BASIN_NAMES.keys()):
        bd = disp_results.get(basin, {}).get('ALL')
        if bd is None:
            continue
        sig = '***' if bd['p_slope'] < 0.001 else ('**' if bd['p_slope'] < 0.01 else
              ('*' if bd['p_slope'] < 0.05 else 'ns'))
        lines.append(f"  {BASIN_NAMES[basin]}:  r(|τ|,d)={bd['r_mag']:+.3f}  "
                     f"r(proj,d)={bd['r_proj']:+.3f}  "
                     f"k={bd['slope']:+.4f}  p={bd['p_slope']:.2e} {sig}")

    p = Path(out_dir) / 'subgroup_summary.txt'
    p.write_text('\n'.join(lines))
    print(f"\n  Summary → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',      required=True)
    ap.add_argument('--out_dir',  default='./proj_results')
    ap.add_argument('--max_rows', type=int, default=None)
    ap.add_argument('--seed',     type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    try:
        from scipy import stats  # noqa
    except ImportError:
        print("pip install scipy matplotlib"); import sys; sys.exit(1)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Loading {args.csv} ...")
    records = load_data(args.csv, max_rows=args.max_rows)
    records = filter_valid_groups(records, min_n=10)

    aa_results   = analyse_by_aa(records)
    sign_results = analyse_sign_consistency(records)
    disp_results = analyse_torque_vs_displacement(records)

    print(f"\nGenerating plots ...")
    plot_all(records, aa_results, sign_results, disp_results, args.out_dir)
    write_summary(aa_results, sign_results, disp_results, args.out_dir)

    print(f"\nDone. Outputs in {args.out_dir}/")


if __name__ == '__main__':
    main()