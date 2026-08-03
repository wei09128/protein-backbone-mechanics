"""
paper1_fig4abc_fast.py — Figure 4 panels A/B/C, standalone and fast
================================================================================
paper1_2_backbone_analysis.py's torque_vectors.png (panels A/B/C) only needs
CSV loading + basin classification -- it never needed the slow model-fitting
stages (Parts 3+) that make the full script take ~2 days. This script
reproduces the EXACT plotting code for these 3 panels (copied verbatim from
that script's Fig 2 section) but loads data via pandas (vectorized, much
faster than the original's row-by-row csv.DictReader loop) and skips
everything else entirely.

Usage:
  python paper1_fig4abc_fast.py --csv features_lj_v3_FINAL_CLEAN.csv --out_dir ./figures/paper1
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

REF_PHI = -63.0
REF_PSI = -43.0

BASIN_NAMES   = {0: 'alphaR', 1: 'beta', 2: 'PPII', 3: '3_10', 4: 'loop', 5: 'alphaL'}
BASIN_CENTRES = {0: (-63, -43), 1: (-120, 128), 2: (-72, 146),
                 3: (-52, -32), 4: (-95, 10), 5: (60, 40)}
BASIN_COLORS  = {0: '#1D9E75', 1: '#378ADD', 2: '#BA7517',
                 3: '#D4537E', 4: '#888780', 5: '#9B59B6'}


def wrap(a):
    return ((a + 180.0) % 360.0) - 180.0


def ss_bin(phi_deg, psi_deg):
    p, q = phi_deg, psi_deg
    if p > 0 and -20 <= q <= 80:            return 5
    if -100 <= p <= -40 and -60 <= q <= 20:  return 0
    if p <= -90 and q >= 90:                 return 1
    if -90 <= p <= -50 and q >= 120:         return 2
    if -80 <= p <= -30 and -40 <= q <= 0:    return 3
    return 4


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

    print("Loading CSV (pandas, vectorized)...")
    df = pd.read_csv(args.csv, low_memory=False)
    df = df.dropna(subset=['phi_deg', 'psi_deg'])
    df = df[(df['phi_deg'].abs() >= 0.5) & (df['psi_deg'].abs() >= 0.5)]
    print(f"  {len(df):,} residues")

    df['ss_bin'] = [ss_bin(p, q) for p, q in zip(df['phi_deg'], df['psi_deg'])]
    phi_all = df['phi_deg'].values
    psi_all = df['psi_deg'].values

    source_keys = {
        'bb_donor':  ('tau_phi_bb_donor',  'tau_psi_bb_donor'),
        'bb_acc':    ('tau_phi_bb_acc',    'tau_psi_bb_acc'),
        'sc_hb':     ('tau_phi_sc_hb',     'tau_psi_sc_hb'),
        'steric':    ('tau_phi_steric',    'tau_psi_steric'),
        'elec':      ('tau_phi_elec_corr', 'tau_psi_elec_corr'),
    }

    print("Aggregating per-basin torques...")
    basin_data = defaultdict(dict)
    for b, grp in df.groupby('ss_bin'):
        basin_data[b]['tau_phi'] = grp['tau_phi_correct'].values
        basin_data[b]['tau_psi'] = grp['tau_psi_correct'].values
        for src, (pk, qk) in source_keys.items():
            if pk in grp.columns and qk in grp.columns:
                basin_data[b][f'tau_phi_{src}'] = grp[pk].values
                basin_data[b][f'tau_psi_{src}'] = grp[qk].values
            else:
                basin_data[b][f'tau_phi_{src}'] = np.zeros(len(grp))
                basin_data[b][f'tau_psi_{src}'] = np.zeros(len(grp))

    print(f"\n  Net torque per basin:")
    print(f"  {'Basin':8}  {'n':>8}  {'<tau_phi>':>10}  {'<tau_psi>':>10}")
    for basin in sorted(basin_data.keys()):
        d = basin_data[basin]
        print(f"  {BASIN_NAMES[basin]:8}  {len(d['tau_phi']):>8,}  "
              f"{np.mean(d['tau_phi']):>+10.4f}  {np.mean(d['tau_psi']):>+10.4f}")

    print(f"\n  Restoring projection:")
    restoring_data = {}
    for basin in sorted(basin_data.keys()):
        d = basin_data[basin]
        phi_c, psi_c = BASIN_CENTRES[basin]
        dv = np.array([wrap(REF_PHI - phi_c), wrap(REF_PSI - psi_c)])
        dv_norm = np.linalg.norm(dv)
        restoring_hat = np.array([0.0, 0.0]) if dv_norm < 1.0 else dv / dv_norm

        tp = d['tau_phi']; tq = d['tau_psi']
        projections = tp * restoring_hat[0] + tq * restoring_hat[1]
        mean_proj = float(np.mean(projections))
        restoring_data[basin] = {
            'hat': restoring_hat, 'mean_proj': mean_proj,
            'mean_tau': (float(np.mean(tp)), float(np.mean(tq))),
        }
        print(f"  {BASIN_NAMES[basin]:8}  proj={mean_proj:+.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(26, 8))

    def _draw_bg(ax):
        H, xe, ye = np.histogram2d(phi_all, psi_all, bins=72,
                                    range=[[-180, 180], [-180, 180]])
        ax.imshow(H.T, origin='lower', aspect='auto',
                  extent=[-180, 180, -180, 180],
                  cmap='Greys', norm=LogNorm(vmin=0.5), alpha=0.55)
        ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
        ax.axhline(0, color='gray', lw=0.4, alpha=0.4)
        ax.axvline(0, color='gray', lw=0.4, alpha=0.4)
        ax.set_xlabel('phi (deg)'); ax.set_ylabel('psi (deg)')
        ax.scatter([REF_PHI], [REF_PSI], s=130, marker='*', color='black', zorder=6)

    ax = axes[0]
    _draw_bg(ax)
    ax.set_title('(A) Environmental torque (= -tau_resistance)\n'
                 'Non-zero because backbone spring\n'
                 '+ solvent balance these at equilibrium', fontsize=10)
    all_net_mags = []
    for basin in BASIN_COLORS:
        d = basin_data.get(basin, {})
        if 'tau_phi' not in d or len(d['tau_phi']) < 10: continue
        rd = restoring_data[basin]
        mt, mq = rd['mean_tau']
        all_net_mags.append(max(abs(mt), abs(mq)))
    global_net_max = max(all_net_mags) if all_net_mags else 1.0
    net_scale = 30.0 / (global_net_max + 1e-8)
    for basin, color in BASIN_COLORS.items():
        d = basin_data.get(basin, {})
        if 'tau_phi' not in d or len(d['tau_phi']) < 10: continue
        rd = restoring_data[basin]
        mean_tp, mean_tq = rd['mean_tau']
        phi_c, psi_c = BASIN_CENTRES[basin]
        ax.annotate('', xy=(phi_c + mean_tp*net_scale, psi_c + mean_tq*net_scale),
                    xytext=(phi_c, psi_c),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
        ax.scatter([phi_c], [psi_c], s=80, color=color, zorder=5,
                   label=f"{BASIN_NAMES[basin]} n={len(d['tau_phi']):,}")
    ax.legend(fontsize=7, loc='lower right')

    ax = axes[1]
    _draw_bg(ax)
    ax.set_title('(B) Restoring projection\n'
                 'Green -> toward alphaR (restoring)\n'
                 'Red -> away from alphaR (driving into basin)', fontsize=10)
    for basin, color in BASIN_COLORS.items():
        if basin == 0: continue
        phi_c, psi_c = BASIN_CENTRES[basin]
        ax.plot([phi_c, REF_PHI], [psi_c, REF_PSI], '--', color=color, lw=0.8, alpha=0.3, zorder=2)
    for basin, color in BASIN_COLORS.items():
        d = basin_data.get(basin, {})
        if 'tau_phi' not in d or len(d['tau_phi']) < 10: continue
        rd = restoring_data[basin]
        phi_c, psi_c = BASIN_CENTRES[basin]
        if basin == 0:
            ax.scatter([phi_c], [psi_c], s=100, color=color, zorder=5,
                       label=f"{BASIN_NAMES[basin]} (ref)")
            continue
        hat = rd['hat']; proj = rd['mean_proj']
        arrow_scale = 120.0
        dx = proj * hat[0] * arrow_scale
        dy = proj * hat[1] * arrow_scale
        arrow_len = np.sqrt(dx**2 + dy**2)
        if arrow_len > 0.01:
            target_len = np.clip(arrow_len, 12.0, 40.0)
            dx *= target_len / arrow_len
            dy *= target_len / arrow_len
        arrow_color = '#1D9E75' if proj > 0 else '#E24B4A'
        ax.annotate('', xy=(phi_c + dx, psi_c + dy), xytext=(phi_c, psi_c),
                    arrowprops=dict(arrowstyle='->', color=arrow_color, lw=3.0))
        ax.scatter([phi_c], [psi_c], s=80, color=color, zorder=5)
        ax.text(phi_c + 5, psi_c - 10, f"{BASIN_NAMES[basin]}\nproj={proj:+.3f}",
                fontsize=7, color=color, fontweight='bold', zorder=7)
    ax.legend(fontsize=7, loc='lower right')

    ax = axes[2]
    _draw_bg(ax)
    ax.set_title('(C) Per-source torque decomposition\n'
                 'Which forces dominate in each basin?', fontsize=10)
    source_colors = {'bb_donor': '#1D9E75', 'bb_acc': '#378ADD', 'sc_hb': '#BA7517',
                     'steric': '#D4537E', 'elec': '#9B59B6'}
    source_labels = {'bb_donor': 'bb N-H donor', 'bb_acc': 'bb C=O acceptor',
                     'sc_hb': 'SC H-bond', 'steric': 'SC steric (Cg)', 'elec': 'electrostatic'}
    all_mags = []
    for basin in BASIN_COLORS:
        d = basin_data.get(basin, {})
        for src in source_colors:
            k1 = f'tau_phi_{src}'; k2 = f'tau_psi_{src}'
            if k1 in d and len(d[k1]) > 0:
                all_mags.append(max(abs(np.mean(d[k1])), abs(np.mean(d[k2]))))
    global_max = max(all_mags) if all_mags else 1.0
    src_scale = 22.0 / (global_max + 1e-8)
    legend_added = set()
    for basin in BASIN_COLORS:
        d = basin_data.get(basin, {})
        if 'tau_phi_bb_donor' not in d or len(d['tau_phi_bb_donor']) < 10:
            continue
        phi_c, psi_c = BASIN_CENTRES[basin]
        ax.scatter([phi_c], [psi_c], s=60, color=BASIN_COLORS[basin],
                   zorder=5, edgecolors='black', linewidth=0.5)
        ax.text(phi_c + 4, psi_c - 12, BASIN_NAMES[basin],
                fontsize=7, color=BASIN_COLORS[basin], fontweight='bold', zorder=7)
        offsets = [(-4, 4), (4, 4), (-4, -4), (4, -4), (0, 7)]
        for si, (src, scolor) in enumerate(source_colors.items()):
            k1 = f'tau_phi_{src}'; k2 = f'tau_psi_{src}'
            mt = float(np.mean(d[k1])); mq = float(np.mean(d[k2]))
            if abs(mt) < 1e-6 and abs(mq) < 1e-6: continue
            ox, oy = offsets[si % len(offsets)]
            ax.annotate('', xy=(phi_c + ox + mt*src_scale, psi_c + oy + mq*src_scale),
                        xytext=(phi_c + ox, psi_c + oy),
                        arrowprops=dict(arrowstyle='->', color=scolor, lw=1.8, alpha=0.85))
            if src not in legend_added:
                ax.plot([], [], color=scolor, lw=2, label=source_labels[src])
                legend_added.add(src)
    ax.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    p = out / 'torque_vectors.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {p}")


if __name__ == '__main__':
    main()
