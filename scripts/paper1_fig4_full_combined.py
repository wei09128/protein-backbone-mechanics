"""
paper1_fig4_full_combined.py — Figure 4, all panels A-F, ONE assembled figure
================================================================================
Builds the complete Figure 4 as a single matplotlib figure (GridSpec), not
stitched PNGs, so sizing/fonts/dpi stay consistent throughout:

  Row 1: A/B/C (torque vectors, restoring projection, per-source decomposition)
         -- ~15% smaller row height than baseline, fonts increased
  Row 2: D (sign consistency) -- 5 basins in one row (alphaR/beta/PPII/loop/alphaL)
  Row 3: F (restoring projection per AA group) -- slightly smaller than D/E
  Row 4: E (torque vs displacement) -- 5 basins in one row

3_10 is excluded from rows D/E/F (very low n, ~18-530 residues depending on
basin) but still included in row A/B/C since it's naturally part of that
panel's existing basin_data computation.

Uses pandas throughout (not the original scripts' slow row-by-row
csv.DictReader loops) for speed on the full 1.7M-row dataset.

Usage:
  python paper1_fig4_full_combined.py --csv features_lj_v3_FINAL_CLEAN.csv --out_dir ./figures/paper1
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

REF_PHI, REF_PSI = -63.0, -43.0
BASIN_NAMES   = {0: 'alphaR', 1: 'beta', 2: 'PPII', 3: '3_10', 4: 'loop', 5: 'alphaL'}
BASIN_CENTRES = {0: (-63, -43), 1: (-120, 128), 2: (-72, 146),
                 3: (-52, -32), 4: (-95, 10), 5: (60, 40)}
BASIN_COLORS  = {0: '#1D9E75', 1: '#378ADD', 2: '#BA7517',
                 3: '#D4537E', 4: '#888780', 5: '#9B59B6'}
ROW_BASINS = [0, 1, 2, 4, 5]  # D/E/F rows exclude 3_10

AA_GROUPS = {
    'Gly': {'GLY'}, 'Pro': {'PRO'}, 'Branched': {'VAL', 'ILE', 'THR'},
    'Aromatic': {'PHE', 'TYR', 'TRP'}, 'Charged': {'LYS', 'ARG', 'ASP', 'GLU', 'HIS'},
    'Polar': {'SER', 'ASN', 'GLN', 'CYS'}, 'Aliphatic': {'ALA', 'LEU', 'MET'},
}


def wrap(a):
    return ((a + 180.0) % 360.0) - 180.0


def ss_bin(phi, psi):
    if phi > 0 and -20 <= psi <= 80:            return 5
    if -100 <= phi <= -40 and -60 <= psi <= 20:  return 0
    if phi <= -90 and psi >= 90:                 return 1
    if -90 <= phi <= -50 and psi >= 120:         return 2
    if -80 <= phi <= -30 and -40 <= psi <= 0:    return 3
    return 4


def aa_group(res):
    for g, aas in AA_GROUPS.items():
        if res in aas:
            return g
    return 'Other'


def fmt_n(n):
    return f"{n/1000:.0f}k" if n >= 1000 else str(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out_dir', default='.')
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import matplotlib.gridspec as gridspec

    print("Loading CSV (pandas)...")
    df = pd.read_csv(args.csv, low_memory=False)
    df = df.dropna(subset=['phi_deg', 'psi_deg', 'tau_phi_correct', 'tau_psi_correct'])
    df = df[(df['phi_deg'].abs() >= 0.5) & (df['psi_deg'].abs() >= 0.5)]
    df['basin'] = [ss_bin(p, q) for p, q in zip(df['phi_deg'], df['psi_deg'])]
    if 'res_name' not in df.columns:
        df['res_name'] = 'ALA'
    df['res_name'] = df['res_name'].astype(str).str.strip().str.upper()
    df['aa_group'] = df['res_name'].map(aa_group)
    print(f"  {len(df):,} residues")

    phi_all = df['phi_deg'].values
    psi_all = df['psi_deg'].values

    d_phi_c = df['basin'].map(lambda b: BASIN_CENTRES[b][0])
    d_psi_c = df['basin'].map(lambda b: BASIN_CENTRES[b][1])
    d_phi = np.array([wrap(p - c) for p, c in zip(df['phi_deg'], d_phi_c)])
    d_psi = np.array([wrap(p - c) for p, c in zip(df['psi_deg'], d_psi_c)])
    df['disp'] = np.sqrt(d_phi**2 + d_psi**2)
    df['tau_mag'] = np.sqrt(df['tau_phi_correct']**2 + df['tau_psi_correct']**2)

    projection = np.zeros(len(df))
    is_alphaR = (df['basin'] == 0).values
    dphi_aR = d_phi[is_alphaR]; dpsi_aR = d_psi[is_alphaR]
    norm_aR = np.sqrt(dphi_aR**2 + dpsi_aR**2)
    safe_norm = np.where(norm_aR > 1.0, norm_aR, 1.0)
    hat_phi_aR = np.where(norm_aR > 1.0, dphi_aR/safe_norm, 0.0)
    hat_psi_aR = np.where(norm_aR > 1.0, dpsi_aR/safe_norm, 0.0)
    projection[is_alphaR] = -(df.loc[is_alphaR, 'tau_phi_correct'].values * hat_phi_aR +
                              df.loc[is_alphaR, 'tau_psi_correct'].values * hat_psi_aR)
    for b in [1, 2, 3, 4, 5]:
        mask = (df['basin'] == b).values
        if mask.sum() == 0:
            continue
        phi_c, psi_c = BASIN_CENTRES[b]
        dv_phi = wrap(REF_PHI - phi_c); dv_psi = wrap(REF_PSI - psi_c)
        dv_norm = np.sqrt(dv_phi**2 + dv_psi**2)
        hat_phi, hat_psi = (dv_phi/dv_norm, dv_psi/dv_norm) if dv_norm > 1.0 else (0.0, 0.0)
        projection[mask] = (df.loc[mask, 'tau_phi_correct'].values * hat_phi +
                            df.loc[mask, 'tau_psi_correct'].values * hat_psi)
    df['projection'] = projection

    source_keys = {
        'bb_donor':  ('tau_phi_bb_donor',  'tau_psi_bb_donor'),
        'bb_acc':    ('tau_phi_bb_acc',    'tau_psi_bb_acc'),
        'sc_hb':     ('tau_phi_sc_hb',     'tau_psi_sc_hb'),
        'steric':    ('tau_phi_steric',    'tau_psi_steric'),
        'elec':      ('tau_phi_elec_corr', 'tau_psi_elec_corr'),
    }
    basin_data = {}
    for b, grp in df.groupby('basin'):
        basin_data[b] = {'tau_phi': grp['tau_phi_correct'].values,
                         'tau_psi': grp['tau_psi_correct'].values}
        for src, (pk, qk) in source_keys.items():
            if pk in grp.columns:
                basin_data[b][f'tau_phi_{src}'] = grp[pk].values
                basin_data[b][f'tau_psi_{src}'] = grp[qk].values
            else:
                basin_data[b][f'tau_phi_{src}'] = np.zeros(len(grp))
                basin_data[b][f'tau_psi_{src}'] = np.zeros(len(grp))

    restoring_data = {}
    for b in sorted(basin_data.keys()):
        d = basin_data[b]
        phi_c, psi_c = BASIN_CENTRES[b]
        dv = np.array([wrap(REF_PHI - phi_c), wrap(REF_PSI - psi_c)])
        dv_norm = np.linalg.norm(dv)
        rhat = np.array([0.0, 0.0]) if dv_norm < 1.0 else dv / dv_norm
        tp, tq = d['tau_phi'], d['tau_psi']
        mean_proj = float(np.mean(tp * rhat[0] + tq * rhat[1]))
        restoring_data[b] = {'hat': rhat, 'mean_proj': mean_proj,
                             'mean_tau': (float(np.mean(tp)), float(np.mean(tq)))}

    print("Computing per-group statistics for D/E/F...")
    aa_results, sign_results, disp_results = {}, {}, {}
    for b in ROW_BASINS:
        bdf = df[df['basin'] == b]
        aa_results[b] = {}
        sign_results[b] = {}
        disp_results[b] = {}

        proj_all = bdf['projection'].values
        n_all = len(proj_all)
        n_pos = int((proj_all > 0).sum())
        if n_all > 0:
            binom_all = stats.binomtest(n_pos, n_all, p=0.5).pvalue
            sign_results[b]['ALL'] = {'n': n_all, 'pct_pos': 100*n_pos/n_all,
                                      'pct_neg': 100*(1-n_pos/n_all), 'binom_p': binom_all}
        if n_all >= 20:
            r_mag, _ = stats.pearsonr(bdf['disp'].values, bdf['tau_mag'].values)
            r_proj, _ = stats.pearsonr(bdf['disp'].values, proj_all)
            slope, intercept, _, p_slope, _ = stats.linregress(bdf['disp'].values, proj_all)
            disp_results[b]['ALL'] = {'disp': bdf['disp'].values, 'proj': proj_all,
                                      'r_mag': r_mag, 'r_proj': r_proj,
                                      'slope': slope, 'p_slope': p_slope}

        for g in list(AA_GROUPS.keys()) + ['Other']:
            gdf = bdf[bdf['aa_group'] == g]
            vals = gdf['projection'].values
            if len(vals) >= 10:
                mean = float(np.mean(vals)); se = float(np.std(vals, ddof=1)/np.sqrt(len(vals)))
                t, p = stats.ttest_1samp(vals, 0.0)
                aa_results[b][g] = {'mean': mean, 'se': se, 'n': len(vals), 'p': p}
                n_pos_g = int((vals > 0).sum())
                bp = stats.binomtest(n_pos_g, len(vals), p=0.5).pvalue
                sign_results[b][g] = {'n': len(vals), 'pct_pos': 100*n_pos_g/len(vals),
                                      'pct_neg': 100*(1-n_pos_g/len(vals)), 'binom_p': bp}

    print("Drawing combined figure...")
    fig = plt.figure(figsize=(24, 20))
    gs = gridspec.GridSpec(3, 15, figure=fig,
                       height_ratios=[3.0, 3.2, 3.2],
                       hspace=0.45, wspace=0.6)

    def _draw_bg(ax):
        H, xe, ye = np.histogram2d(phi_all, psi_all, bins=72,
                                    range=[[-180, 180], [-180, 180]])
        ax.imshow(H.T,
                origin='lower',
                aspect='equal',
                extent=[-180,180,-180,180],
                cmap='Greys',
                norm=LogNorm(vmin=0.5),
                alpha=0.55)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
        ax.axhline(0, color='gray', lw=0.4, alpha=0.4)
        ax.axvline(0, color='gray', lw=0.4, alpha=0.4)
        ax.set_xlabel('phi (deg)', fontsize=13)
        ax.set_ylabel('psi (deg)', fontsize=13)
        ax.tick_params(labelsize=11)
        ax.scatter([REF_PHI], [REF_PSI], s=140, marker='*', color='black', zorder=6)

    ax_a = fig.add_subplot(gs[0, 0:5])
    _draw_bg(ax_a)
    ax_a.set_title('(A) Environmental torque (= -tau_resistance)\n'
                   'Non-zero because backbone spring\n'
                   '+ solvent balance these at equilibrium', fontsize=13)
    all_net_mags = []
    for b in BASIN_COLORS:
        d = basin_data.get(b, {})
        if 'tau_phi' not in d or len(d['tau_phi']) < 10: continue
        mt, mq = restoring_data[b]['mean_tau']
        all_net_mags.append(max(abs(mt), abs(mq)))
    net_scale = 30.0 / (max(all_net_mags) + 1e-8 if all_net_mags else 1.0)
    for b, color in BASIN_COLORS.items():
        d = basin_data.get(b, {})
        if 'tau_phi' not in d or len(d['tau_phi']) < 10: continue
        mt, mq = restoring_data[b]['mean_tau']
        phi_c, psi_c = BASIN_CENTRES[b]
        ax_a.annotate('', xy=(phi_c + mt*net_scale, psi_c + mq*net_scale), xytext=(phi_c, psi_c),
                     arrowprops=dict(arrowstyle='->', color=color, lw=2.8))
        ax_a.scatter([phi_c], [psi_c], s=90, color=color, zorder=5,
                    label=f"{BASIN_NAMES[b]} n={fmt_n(len(d['tau_phi']))}")
    ax_a.legend(fontsize=10, loc='lower right')

    ax_b = fig.add_subplot(gs[0, 5:10])
    _draw_bg(ax_b)
    ax_b.set_title('(B) Restoring projection\nGreen -> toward alphaR (restoring)\n'
                   'Red -> away from alphaR (driving into basin)', fontsize=13)
    for b, color in BASIN_COLORS.items():
        if b == 0: continue
        phi_c, psi_c = BASIN_CENTRES[b]
        ax_b.plot([phi_c, REF_PHI], [psi_c, REF_PSI], '--', color=color, lw=0.9, alpha=0.3, zorder=2)
    for b, color in BASIN_COLORS.items():
        d = basin_data.get(b, {})
        if 'tau_phi' not in d or len(d['tau_phi']) < 10: continue
        phi_c, psi_c = BASIN_CENTRES[b]
        if b == 0:
            ax_b.scatter([phi_c], [psi_c], s=110, color=color, zorder=5, label=f"{BASIN_NAMES[b]} (ref)")
            continue
        hat = restoring_data[b]['hat']; proj = restoring_data[b]['mean_proj']
        dx, dy = proj*hat[0]*120.0, proj*hat[1]*120.0
        alen = np.sqrt(dx**2+dy**2)
        if alen > 0.01:
            tlen = np.clip(alen, 12.0, 40.0)
            dx *= tlen/alen; dy *= tlen/alen
        acolor = '#1D9E75' if proj > 0 else '#E24B4A'
        ax_b.annotate('', xy=(phi_c+dx, psi_c+dy), xytext=(phi_c, psi_c),
                     arrowprops=dict(arrowstyle='->', color=acolor, lw=3.2))
        ax_b.scatter([phi_c], [psi_c], s=90, color=color, zorder=5)
        ax_b.text(phi_c+5, psi_c-10, f"{BASIN_NAMES[b]}\nproj={proj:+.3f}",
                  fontsize=9, color=color, fontweight='bold', zorder=7)
    ax_b.legend(fontsize=10, loc='lower right')

    ax_c = fig.add_subplot(gs[0, 10:15])
    _draw_bg(ax_c)
    ax_c.set_title('(C) Per-source torque decomposition\nWhich forces dominate in each basin?', fontsize=13)
    source_colors = {'bb_donor': '#1D9E75', 'bb_acc': '#378ADD', 'sc_hb': '#BA7517',
                     'steric': '#D4537E', 'elec': '#9B59B6'}
    source_labels = {'bb_donor': 'bb N-H donor', 'bb_acc': 'bb C=O acceptor', 'sc_hb': 'SC H-bond',
                     'steric': 'SC steric (Cg)', 'elec': 'electrostatic'}
    all_mags = []
    for b in BASIN_COLORS:
        d = basin_data.get(b, {})
        for src in source_colors:
            k1, k2 = f'tau_phi_{src}', f'tau_psi_{src}'
            if k1 in d and len(d[k1]) > 0:
                all_mags.append(max(abs(np.mean(d[k1])), abs(np.mean(d[k2]))))
    src_scale = 22.0 / (max(all_mags) + 1e-8 if all_mags else 1.0)
    legend_added = set()
    for b in BASIN_COLORS:
        d = basin_data.get(b, {})
        if 'tau_phi_bb_donor' not in d or len(d['tau_phi_bb_donor']) < 10: continue
        phi_c, psi_c = BASIN_CENTRES[b]
        ax_c.scatter([phi_c], [psi_c], s=65, color=BASIN_COLORS[b], zorder=5,
                    edgecolors='black', linewidth=0.5)
        ax_c.text(phi_c+4, psi_c-12, BASIN_NAMES[b], fontsize=9,
                  color=BASIN_COLORS[b], fontweight='bold', zorder=7)
        offsets = [(-4,4),(4,4),(-4,-4),(4,-4),(0,7)]
        for si, (src, scolor) in enumerate(source_colors.items()):
            k1, k2 = f'tau_phi_{src}', f'tau_psi_{src}'
            mt, mq = float(np.mean(d[k1])), float(np.mean(d[k2]))
            if abs(mt) < 1e-6 and abs(mq) < 1e-6: continue
            ox, oy = offsets[si % len(offsets)]
            ax_c.annotate('', xy=(phi_c+ox+mt*src_scale, psi_c+oy+mq*src_scale),
                         xytext=(phi_c+ox, psi_c+oy),
                         arrowprops=dict(arrowstyle='->', color=scolor, lw=2.0, alpha=0.85))
            if src not in legend_added:
                ax_c.plot([], [], color=scolor, lw=2, label=source_labels[src])
                legend_added.add(src)
    ax_c.legend(fontsize=10, loc='lower right')

    for i, b in enumerate(ROW_BASINS):
        ax = fig.add_subplot(gs[1, i*3:(i+1)*3])
        # shrink width by ~12% while keeping center fixed
        pos = ax.get_position()
        shrink = 0.92        # try 0.90 or 0.92 if desired
        new_w = pos.width * shrink
        ax.set_position([
            pos.x0 + (pos.width - new_w)/2,
            pos.y0,
            new_w,
            pos.height
        ])
        bd = sign_results.get(b, {})
        groups = [g for g in (['ALL'] + list(AA_GROUPS.keys()) + ['Other']) if g in bd]
        if not groups:
            ax.set_visible(False); continue
        pos_vals = [bd[g]['pct_pos'] for g in groups]
        neg_vals = [bd[g]['pct_neg'] for g in groups]
        y = np.arange(len(groups))
        ax.barh(y, pos_vals, color='#1D9E75', alpha=0.80, height=0.55)
        ax.barh(y, [-v for v in neg_vals], color='#E24B4A', alpha=0.80, height=0.55)
        ax.axvline(0, color='black', lw=0.8)
        ax.axvline(50, color='#1D9E75', lw=0.8, ls=':')
        ax.axvline(-50, color='#E24B4A', lw=0.8, ls=':')

        aa_bd = aa_results.get(b, {})
        mean_groups = [g for g in groups if g in aa_bd]
        if mean_groups:
            ax2 = ax.twiny()
            mean_y = [groups.index(g) for g in mean_groups]
            mean_vals = [aa_bd[g]['mean'] for g in mean_groups]
            ax2.scatter(mean_vals, mean_y, color='black', marker='D', s=32,
                       zorder=6, edgecolor='white', linewidth=0.5)
            ax2.axvline(0, color='black', lw=0.5, alpha=0.3)
            ax2.set_xlabel('Mean projected torque (kcal/mol/rad)', fontsize=10, color='#333333')
            ax2.tick_params(axis='x', labelsize=10, colors='#333333')

        for yi, g in enumerate(groups):
            bp = bd[g]['binom_p']
            sig = '***' if bp < 0.001 else ('**' if bp < 0.01 else ('*' if bp < 0.05 else ''))
            if sig:
                ax.text(52, yi, sig, va='center', fontsize=11)
        ax.set_yticks(y)
        labels = [f"{g}\n({fmt_n(bd[g]['n'])})" for g in groups]
        ax.set_yticklabels(labels, fontsize=10)
        ax.tick_params(axis='y', pad=1, labelsize=10)
        ax.set_xlabel('% residues', fontsize=13)
        ax.tick_params(axis='x', labelsize=11)
        subtitle = '(+ = opposes displacement)' if b == 0 else '(+ = toward alphaR)'
        ax.set_title(f'{BASIN_NAMES[b]}\n{subtitle}', color=BASIN_COLORS[b], fontweight='bold', fontsize=14)
        ax.set_xlim(-80, 80)
        ax.invert_yaxis()
        if i == 0:
            ax.legend(handles=[
                plt.Rectangle((0,0),1,1, color='#1D9E75', alpha=0.8, label='>0 (restoring)'),
                plt.Rectangle((0,0),1,1, color='#E24B4A', alpha=0.8, label='<0 (driving)'),
            ], fontsize=10, loc='upper left', bbox_to_anchor=(0, -0.15))

    for i, b in enumerate(ROW_BASINS):
        ax = fig.add_subplot(gs[2, i*3:(i+1)*3])
        bd = disp_results.get(b, {}).get('ALL')
        if bd is None:
            ax.set_facecolor('#ffeeee')
            ax.set_title(f"{BASIN_NAMES[b]}\n(no data)", color='red', fontsize=12)
            continue
        disp = bd['disp']; proj = bd['proj']
        n_scatter = min(len(disp), 3000)
        idx = np.random.choice(len(disp), n_scatter, replace=False)
        ax.scatter(disp[idx], proj[idx], s=2, alpha=0.15, color=BASIN_COLORS[b])
        n_bins = 10
        bin_edges = np.percentile(disp, np.linspace(0, 100, n_bins+1))
        bc, bm, bs = [], [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (disp >= lo) & (disp < hi)
            if mask.sum() < 5: continue
            v = proj[mask]
            bc.append(float(np.mean(disp[mask]))); bm.append(float(np.mean(v)))
            bs.append(float(np.std(v, ddof=1)/np.sqrt(len(v))))
        ax.errorbar(bc, bm, yerr=bs, fmt='o-', color='black', lw=2, ms=5, capsize=3, zorder=5)
        slope, p_sl = bd['slope'], bd['p_slope']
        xr = np.linspace(disp.min(), disp.max(), 100)
        intercept = np.mean(proj) - slope*np.mean(disp)
        ax.plot(xr, slope*xr+intercept, 'r--', lw=1.5, alpha=0.7, label=f'OLS k={slope:+.4f} p={p_sl:.1e}')
        ax.axhline(0, color='gray', lw=0.8, ls=':')
        ax.set_xlabel('Displacement from basin centre (deg)', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        ylabel = 'Restoring projection\n(-tau.d, + = opposes displacement)' if b == 0 else 'Restoring projection\n(+ = toward alphaR)'
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{BASIN_NAMES[b]}\nr(|tau|,d)={bd["r_mag"]:+.3f}  r(proj,d)={bd["r_proj"]:+.3f}',
                    color=BASIN_COLORS[b], fontweight='bold', fontsize=13)
        ax.legend(fontsize=10, loc='best')

    out_path = out / 'figure4_combined_full.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
