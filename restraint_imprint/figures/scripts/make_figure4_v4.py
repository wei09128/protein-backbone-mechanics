"""Figure 4 v3: judging a learned library by the paper's three rules (home advantage, forecast, independent referee).
Usage: python3 make_figure4_v3.py <scratchpad> <out>. Swap panel is a labelled placeholder until stage 2 returns."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, GRID = '#1f1e1c', '#5c5b56', '#ecebe7'
PH, RF = '#2a78d6', '#d0662f'
HI, GR1, GR2, GR3 = '#1b7f5f', '#8a8983', '#b9b8b2', '#d6d5cf'
ANG = {'ACN': '#6a51a3', 'OCN': '#c2410c'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.edgecolor': GR2, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 8.5, 'axes.titlecolor': INK, 'axes.titlelocation': 'left', 'legend.frameon': False,
                     'svg.fonttype': 'none'})
SRC = {}


def lab(ax, s, src):
    ax.text(-0.2, 1.07, s, transform=ax.transAxes, fontsize=11, fontweight='bold', color=INK); SRC[s] = src


def grid(ax, axis='x'):
    ax.grid(axis=axis, color=GRID, lw=0.6); ax.set_axisbelow(True)


def ci_point(ax, y, r, color, ms=4, marker='o'):
    ax.errorbar(r['estimate'], y, xerr=[[r['estimate'] - r['ci95'][0]], [r['ci95'][1] - r['estimate']]], fmt=marker,
                ms=ms, color=color, lw=1.1, capsize=0)


ha = json.loads((S / 'ha_ret/pbm_home_advantage.SO1x48/results/CONTRASTS.json').read_text())
ham = pd.read_csv(S / 'ha_ret/pbm_home_advantage.SO1x48/results/STRATUM_MAE.tsv', sep='\t')
tm = json.loads((S / 'tmix/pbm_training_mix.ryUSuc/results/CONTRASTS.json').read_text())
tshift = pd.read_csv(S / 'tmix/pbm_training_mix.ryUSuc/results/SURFACE_SHIFT.tsv', sep='\t')
ch = json.loads((S / 'chi_ret/pbm_chi1.TrfkcA/results/CONTRASTS.json').read_text())
qm = json.loads((S / 'qm_rescore_out/QM_RESCORE.json').read_text())

fig, axs = plt.subplots(2, 4, figsize=(15, 6.8)); axs = axs.ravel()

# A: amplitude of learned surfaces by training group (corrected E1; descriptive, no intervals)
e1 = json.loads((S / 'ms11/PBM_MANUSCRIPT_UPDATE_v1_1/supporting_values/signal_vs_noise_summary.json').read_text())
ax = axs[0]
TT = [('angle_CaCN', 'Cα–C–N'), ('angle_CA_C_O', 'Cα–C–O'), ('angle_CNCa', 'C–N–Cα'), ('tau_deg', 'τ')]
GROUPS = [('program=PHENIX_pre_default', 'PHENIX pre-CDL', '#9db8d8'), ('program=PHENIX_post_default', 'PHENIX post-CDL', PH), ('res_tier=<=1.0', '≤1.0 Å (all programs)', '#0b3a66')]
for j, (gk, gl, col) in enumerate(GROUPS):
    ax.bar(np.arange(4) + (j - 1) * 0.26, [e1[t]['group_surfaces'][gk]['slope_vs_REFMAC'] for t, _ in TT], 0.26, color=col, label=gl)
ax.axhline(1, ls='--', color=GR2, lw=0.8); ax.plot([], [], ls='--', color=GR2, lw=0.8, label='REFMAC-trained = 1')
ax.set_xticks(range(4)); ax.set_xticklabels([t[1] for t in TT]); ax.set_ylim(0, 2.05)
ax.set_ylabel('surface amplitude vs REFMAC-trained'); ax.set_title('Training data carry their refinement amplitude'); ax.legend(fontsize=6.3, loc='upper right'); grid(ax, 'y')
lab(ax, 'A', 'signal_vs_noise_summary.json group_surfaces (corrected E1; ledger P1b; no intervals)')

# B: surface shift
ax = axs[1]
pairs = ['F3_REFMAC_only vs F4_PHENIX_only', 'F2_balanced_PR vs F1_natural_PR']; pl = ['REFMAC-only\nvs PHENIX-only', 'balanced\nvs natural']
for j, t in enumerate(('ACN', 'OCN')):
    r = tshift[tshift.term == t].set_index('pair').loc[pairs]
    ax.bar(np.arange(2) + (j - 0.5) * 0.36, r.mean_abs_shift, 0.36, color=ANG[t], label={'ACN': 'Cα–C–N', 'OCN': 'O–C–N'}[t])
    ax.errorbar(np.arange(2) + (j - 0.5) * 0.36, r.mean_abs_shift, yerr=[np.zeros(2), r.p95_abs_shift - r.mean_abs_shift], fmt='none', color=INK2, lw=0.8)
ax.set_xticks(range(2)); ax.set_xticklabels(pl); ax.set_ylabel('surface shift (°): mean, whisker to 95th pct')
ax.set_title('Program, not weighting, moves the surface'); ax.legend(fontsize=6.5); grid(ax, 'y')
lab(ax, 'B', 'tmix SURFACE_SHIFT.tsv; ledger M1')

# C: training mix, single-program fits vs natural mix
ax = axs[2]
items = [('PHENIX_only_vs_natural', 'PHENIX-only fit'), ('REFMAC_only_vs_natural', 'REFMAC-only fit'), ('Delta_bal', 'balanced 50/50')]
for i, (k, name) in enumerate(items):
    for j, (s_, col) in enumerate((('PHENIX_post', PH), ('REFMAC', RF))):
        for m, t in enumerate(('ACN', 'OCN')):
            ci_point(ax, i + (j - 0.5) * 0.36 + (m - 0.5) * 0.12, tm[f'{k}|{s_}'][t], col, ms=3.2, marker='o' if t == 'ACN' else 's')
ax.axvline(0, color=GR2, lw=0.8); ax.set_yticks(range(3)); ax.set_yticklabels([n for _, n in items]); ax.invert_yaxis()
ax.set_xlabel('held-out MAE minus natural-mix fit (°)'); ax.set_title('A learned library inherits its training mix'); grid(ax)
ax.plot([], [], 'o', color=PH, label='PHENIX deposits'); ax.plot([], [], 'o', color=RF, label='REFMAC deposits')
ax.plot([], [], 'o', color=GR1, label='Cα–C–N'); ax.plot([], [], 's', color=GR1, label='O–C–N'); ax.legend(fontsize=6.2, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)
lab(ax, 'C', 'tmix CONTRASTS.json; ledger M2, M3')

# D: home advantage shown directly: CDL vs CCP4 2018, on each library's home deposits, both terms
ax = axs[3]
grp = [('ACN', 'PHENIX_post'), ('ACN', 'REFMAC_new'), ('OCN', 'PHENIX_post'), ('OCN', 'REFMAC_new')]
xl = ['Cα–C–N\nPHENIX\npost-2015', 'Cα–C–N\nREFMAC\npost-2018', 'O–C–N\nPHENIX\npost-2015', 'O–C–N\nREFMAC\npost-2018']
xs = np.array([0, 1, 2.4, 3.4])
for j, (p_, col) in enumerate((('CDL', PH), ('CCP4new', RF))):
    v = [ham[(ham.predictor == p_) & (ham.stratum == s_) & (ham.term == t)].mae_group_weighted.iloc[0] for t, s_ in grp]
    ax.bar(xs + (j - 0.5) * 0.36, v, 0.36, color=col, label={'CDL': 'CDL', 'CCP4new': 'CCP4 2018 targets'}[p_])
ax.set_xticks(xs); ax.set_xticklabels(xl, fontsize=6.3); ax.set_ylabel('MAE vs deposits (°)')
ax.text(0.5, 1.76, 'DiD −0.24°', ha='center', fontsize=6.6, color=INK); ax.text(2.9, 1.76, 'DiD −0.69°; winner flips', ha='center', fontsize=6.6, color=INK)
ax.set_ylim(0, 1.9); ax.set_title('Each library scores relatively better at home'); ax.legend(fontsize=6.5, loc='upper right', bbox_to_anchor=(1, 0.88)); grid(ax, 'y')
lab(ax, 'D', 'ha_ret STRATUM_MAE.tsv (group-weighted); DiD from CONTRASTS.json C1 (ledger H1)')

# E: chi1 adds information for C-beta angles, not tau
ax = axs[4]
terms = [('NCB', 'N–Cα–Cβ'), ('CCB', 'C–Cα–Cβ'), ('TAU', 'τ')]
for i, (t, name) in enumerate(terms):
    for j, (s, col) in enumerate((('REFMAC', RF), ('PHENIX_post', PH))):
        ci_point(ax, i + (j - 0.5) * 0.3, ch[f'Delta_chi|{s}'][t], col)
ax.axvline(0, color=GR2, lw=0.8); ax.set_yticks(range(3)); ax.set_yticklabels([n for _, n in terms]); ax.invert_yaxis()
ax.set_xlabel('MAE with χ1 minus without (°)'); ax.set_title('Rotamer adds information beyond φ/ψ'); grid(ax)
ax.plot([], [], 'o', color=RF, label='REFMAC deposits (neutral)'); ax.plot([], [], 'o', color=PH, label='PHENIX deposits'); ax.legend(fontsize=6.3, loc='lower right')
lab(ax, 'E', 'chi_ret CONTRASTS.json; ledger X1, X2')

# F: prospective swap (registered)
ax = axs[5]
sw = pd.read_csv(sorted((S / 'swap3').glob('pbm_swap_stage2.*/results/STRUCTURE_RESULTS.tsv'))[0], sep='\t')
summ = json.loads(sorted((S / 'swap3').glob('pbm_swap_stage2.*/results/SUMMARY.json'))[0].read_text())
BAND = {'le1.0': '#0b3a66', 'gt1.0_le1.5': '#3d84bf', 'gt1.5_le2.0': '#8fc0e3'}
for _, r in sw.iterrows():
    ax.plot([0, 1], [r.mae_copy, r.mae_site_matrix], '-o', ms=3, lw=0.9, color=BAND[r.band], alpha=0.9)
for b, lbl in (('le1.0', '≤1.0 Å'), ('gt1.0_le1.5', '1.0–1.5 Å'), ('gt1.5_le2.0', '1.5–2.0 Å')):
    ax.plot([], [], '-o', ms=3, color=BAND[b], label=lbl)
ax.set_xticks([0, 1]); ax.set_xticklabels(['targets copied', 'measured response']); ax.set_xlim(-0.3, 1.3)
ax.set_ylabel('error predicting the CDL → MechLib swap (°)')
pr = summ['primary']
ax.text(0.98, 0.97, f"registered before refinement\ncopy − response {pr['mean_D_copy_minus_site_matrix']:+.3f}°\n[{pr['ci95'][0]:.3f}, {pr['ci95'][1]:.3f}]; {pr['structures_matrix_better']}/19",
        transform=ax.transAxes, ha='right', va='top', fontsize=6.6, color=INK)
ax.set_title('A library swap, forecast before it was run'); ax.legend(fontsize=6.3, loc='center right'); grid(ax, 'y')
lab(ax, 'F', 'swap3 STRUCTURE_RESULTS.tsv, SUMMARY.json; ledger S0, S1')

# G: QM per-term MAE ratio, reported MechLib / AMBER
ax = axs[6]
T = qm['primary_reported_mechlib']['terms']
order = ['tau_deg', 'angle_N_CA_CB', 'angle_C_CA_CB', 'angle_CaCN', 'angle_CNCa', 'angle_CA_C_O', 'bond_CA_CB', 'bond_CA_C', 'bond_N_CA', 'bond_C_O', 'bond_C_N_next']
names = {'tau_deg': 'τ', 'angle_N_CA_CB': 'N–Cα–Cβ', 'angle_C_CA_CB': 'C–Cα–Cβ', 'angle_CaCN': 'Cα–C–N', 'angle_CNCa': 'C–N–Cα',
         'angle_CA_C_O': 'Cα–C–O', 'bond_CA_CB': 'Cα–Cβ', 'bond_CA_C': 'Cα–C', 'bond_N_CA': 'N–Cα', 'bond_C_O': 'C=O', 'bond_C_N_next': 'C–N(next)'}
ratio = [T[t]['model_mae'] / T[t]['amber_mae'] for t in order]
cols = [RF if T[t]['holm_significant_worse'] else HI for t in order]
ax.barh(range(len(order)), ratio, color=cols); ax.axvline(1, color=INK2, lw=0.8)
ax.set_yticks(range(len(order))); ax.set_yticklabels([names[t] for t in order]); ax.invert_yaxis()
ax.set_xlabel('MAE vs constrained DFT: MechLib / fixed AMBER'); ax.set_title('QM referee: angles closer, three bonds not'); grid(ax)
lab(ax, 'G', 'qm_rescore_out QM_RESCORE.json (80 cases, registered rule); ledger Q1R')

# H: QM primary endpoint, three libraries
ax = axs[7]
vers = [('reference_earlier_q6_library', 'earlier library\n(pre-correction)'), ('secondary_plain_v2', 'MechLib\nwithout χ1'), ('primary_reported_mechlib', 'MechLib\n(with χ1)')]
for i, (k, name) in enumerate(vers):
    r = qm[k]; ci_point(ax, i, {'estimate': r['mean_delta'], 'ci95': r['cluster_ci95']}, HI if k.startswith('primary') else GR1, ms=5)
    ax.text(r['cluster_ci95'][1] + 0.0008, i, f"{r['positive_residues']}/20 · {r['status']}", va='center', fontsize=6.5, color=INK2)
ax.axvline(0, color=GR2, lw=0.8); ax.set_yticks(range(3)); ax.set_yticklabels([n for _, n in vers]); ax.invert_yaxis()
ax.set_xlabel('mean weighted error, AMBER minus MechLib\n(positive favours MechLib; residue-clustered 95% CI)')
ax.set_title('Registered QM rule: three bonds fail for every version'); grid(ax); ax.set_xlim(right=max(r['cluster_ci95'][1] for r in qm.values() if isinstance(r, dict) and 'cluster_ci95' in r) + 0.012)
lab(ax, 'H', 'qm_rescore_out QM_RESCORE.json; ledger Q1R')

fig.tight_layout(h_pad=2.2, w_pad=2.0)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'Figure4_three_rules_v4.{ext}', dpi=300 if ext == 'png' else None, bbox_inches='tight')
(OUT / 'FIGURE4_SOURCES_v4.json').write_text(json.dumps(SRC, indent=2))
print('done')
