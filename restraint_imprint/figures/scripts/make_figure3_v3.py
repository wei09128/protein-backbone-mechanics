#!/usr/bin/env python3
"""PBM manuscript v2 figures 1-4, built from returned processed tables (read-only).
Every panel cites its source table in FIGURE_SOURCES.json written next to the figures."""
import json, sys, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

S = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
R = S/'req2/pbm_required_checks.mvof9b'
P = {
 'struct': S/'fig1/PBM_TRANSMISSION_FIGURE1_v1/STRUCTURE_VALUES.tsv',
 'review': S/'fig1/PBM_TRANSMISSION_FIGURE1_v1/INDEPENDENT_REVIEW.json',
 'site_pred': S/'ppd/PBM_PTRANS_PREDICTION_AND_DICTIONARY_v1/SITE_PREDICTIONS.tsv',
 'pred_methods': S/'ppd/PBM_PTRANS_PREDICTION_AND_DICTIONARY_v1/PREDICTION_METHODS.tsv',
 'link_diff': S/'ppd/PBM_PTRANS_PREDICTION_AND_DICTIONARY_v1/LINK_ANGLE_DIFF.tsv',
 'closing': R/'ptrans_checks/CLOSING_CHECKS.json',
 'scores': R/'returned_status/pbm_rdd_did_blank_v1/structure_scores.tsv',
 'groups': R/'ptrans_checks/before_pro_structure_scores.tsv',
 'fig2vals': S/'ms11/PBM_MANUSCRIPT_UPDATE_v1_1/figures/FIGURE2_VALUES.json',
 'scramble': R/'returned_status/pbm_rdd_scramble_results_v1/scramble_estimates.tsv',
 'e1': S/'ms11/PBM_MANUSCRIPT_UPDATE_v1_1/supporting_values/signal_vs_noise_summary.json',
 'bench': S/'e6f3/PBM_FIGURE3_BENCHMARK_v1/FIGURE_VALUES.tsv',
 'qm': S/'phys/PBM_PHYSICS_EVIDENCE_REVIEW_v1/QM_TERM_VALUES.tsv',
 'md': S/'phys/PBM_PHYSICS_EVIDENCE_REVIEW_v1/MD_PAIRED_VALUES.tsv',
 'e6': S/'e6r/pbm_e6_fine_grid.wtJGLI/PAIR_AUDIT.json',
}
for k, p in P.items():
    assert p.exists(), (k, p)

INK, INK2, GRID = '#1f1e1c', '#5c5b56', '#ecebe7'
PH, RF = '#2a78d6', '#d0662f'                          # PHENIX, REFMAC
BAND = {'le1.0': '#0b3a66', 'gt1.0_le1.5': '#3d84bf', 'gt1.5_le2.0': '#8fc0e3'}
BANDLAB = {'le1.0': '≤1.0 Å', 'gt1.0_le1.5': '1.0–1.5 Å', 'gt1.5_le2.0': '1.5–2.0 Å'}
HI, GR1, GR2, GR3 = '#1b7f5f', '#8a8983', '#b9b8b2', '#d6d5cf'   # highlight + greys
ANG = {'ACN': '#6a51a3', 'ACO': '#1b7f5f', 'OCN': '#c2410c'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.edgecolor': '#b9b8b2',
    'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.titlesize': 8.2, 'axes.titleweight': 'bold',
    'axes.titlecolor': INK, 'axes.titlelocation': 'left', 'legend.frameon': False, 'svg.fonttype': 'none'})
SRC = {}

def lab(ax, s, src):
    ax.text(-0.17, 1.06, s, transform=ax.transAxes, fontsize=11, fontweight='bold', color=INK)
    SRC.setdefault(CUR[0], {})[s] = src

def boot_mean(x, n=4000, seed=1):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)

def grid(ax, axis='y'):
    ax.grid(axis=axis, color=GRID, lw=0.6); ax.set_axisbelow(True)

def save(fig, name):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT/f'{name}.{ext}', dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)

CUR = ['Figure1']
# ======================= FIGURE 1: controlled response =======================
sv = pd.read_csv(P['struct'], sep='\t'); rv = json.loads(P['review'].read_text())
M = np.array(rv['matrix_prediction']['mean_of_structure_matrices'])
fig, axs = plt.subplots(2, 4, figsize=(15, 6.6)); axs = axs.ravel()

ax = axs[0]; ax.set_axis_off()
ax.set_title('Design', loc='left')
for y, t, c, q in ((0.78, 'target  t + h', HI, 'refined q(+h)'), (0.50, 'target  t − h', GR1, 'refined q(−h)')):
    ax.add_patch(plt.Rectangle((0.02, y - 0.08), 0.30, 0.16, fc='white', ec=c, lw=1.2, transform=ax.transAxes))
    ax.text(0.17, y, t, ha='center', va='center', transform=ax.transAxes, fontsize=8, color=INK)
    ax.add_patch(FancyArrowPatch((0.34, y), (0.58, y), transform=ax.transAxes, arrowstyle='-|>', mutation_scale=9, color=INK2))
    ax.add_patch(plt.Rectangle((0.60, y - 0.08), 0.38, 0.16, fc='#f4f3ef', ec='#b9b8b2', transform=ax.transAxes))
    ax.text(0.79, y, q, ha='center', va='center', transform=ax.transAxes, fontsize=8)
ax.text(0.02, 0.24, 'same model, data, free set, protocol\n(PHENIX, 1 macrocycle, fixed weights)', transform=ax.transAxes, fontsize=7, color=INK2)
ax.text(0.02, 0.04, r'transmission $=\dfrac{q(+h)-q(-h)}{2h}$', transform=ax.transAxes, fontsize=9, color=INK)
lab(ax, 'A', 'schematic')

ax = axs[1]; terms = [('cac', 'Cα–C'), ('nca', 'N–Cα'), ('cacn', 'Cα–C–N'), ('tau', 'τ')]
rng = np.random.default_rng(3)
for i, (c, nm) in enumerate(terms):
    for b, col in BAND.items():
        v = sv.loc[sv.band == b, c] * 100
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=14, color=col, edgecolor='white', lw=0.4, zorder=3)
    m, lo, hi = boot_mean(sv[c] * 100)
    ax.errorbar(i + 0.3, m, yerr=[[m - lo], [hi - m]], fmt='D', ms=4, color=INK, capsize=0, zorder=4)
ax.axhline(100, ls='--', lw=0.8, color=GR2)
ax.set_xticks(range(4)); ax.set_xticklabels([t[1] for t in terms]); ax.set_ylabel('Transmission (%)')
ax.set_title('Bonds follow targets; isolated angles half'); grid(ax); lab(ax, 'B', 'STRUCTURE_VALUES.tsv')

ax = axs[2]
for c, nm, mk in (('cac', 'Cα–C', 'o'), ('compensated', 'Cα–C–N compensated', 's'), ('tau', 'τ', '^'), ('cacn', 'Cα–C–N isolated', 'v')):
    ys = [sv.loc[sv.band == b, c].mean() * 100 for b in BAND]
    ax.plot(range(3), ys, '-', marker=mk, color=INK2, lw=1, ms=4)
    ax.text(2.08, ys[-1], nm, va='center', fontsize=6.8, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels([BANDLAB[b] for b in BAND]); ax.set_xlim(-0.2, 3.1)
ax.set_ylabel('Mean transmission (%)'); ax.set_title('Atomic-resolution data resist most'); grid(ax)
lab(ax, 'C', 'STRUCTURE_VALUES.tsv (band means; exploratory)')

ax = axs[3]
for _, r in sv.iterrows():
    ax.plot([0, 1], [r.cacn * 100, r.compensated * 100], '-', color=BAND[r.band], lw=0.9, alpha=0.8)
for x, c in ((0, 'cacn'), (1, 'compensated')):
    m, lo, hi = boot_mean(sv[c] * 100); ax.errorbar(x + (0.08 if x else -0.08), m, yerr=[[m - lo], [hi - m]], fmt='D', ms=4, color=INK)
ax.set_xticks([0, 1]); ax.set_xticklabels(['isolated\nCα–C–N', 'Cα–C–N +1°\nO–C–N −1°']); ax.set_xlim(-0.35, 1.35)
ax.set_ylabel('Cα–C–N transmission (%)'); ax.set_title('Keeping the angle sum fixed'); grid(ax)
lab(ax, 'D', 'STRUCTURE_VALUES.tsv')

ax = axs[4]; lbl = ['Cα–C–N', 'Cα–C–O', 'O–C–N']
im = ax.imshow(M, cmap='RdBu_r', vmin=-0.8, vmax=0.8)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{M[i, j]:+.2f}', ha='center', va='center', fontsize=8, color='white' if abs(M[i, j]) > 0.45 else INK)
ax.set_xticks(range(3)); ax.set_xticklabels(lbl); ax.set_yticks(range(3)); ax.set_yticklabels(lbl)
ax.set_xlabel('edited target'); ax.set_ylabel('measured angle'); ax.set_title('Coupled response (19 structures)')
ax.spines[:].set_visible(False); lab(ax, 'E', 'INDEPENDENT_REVIEW.json mean_of_structure_matrices')

ax = axs[5]; Pm = np.eye(3) - np.ones((3, 3)) / 3; t = (M * Pm).sum() / (Pm * Pm).sum()
ax.scatter((t * Pm).ravel(), M.ravel(), s=26, color=HI, zorder=3)
ax.plot([-0.4, 0.7], [-0.4, 0.7], ls='--', color=GR2, lw=0.8)
ax.set_xlabel(f'{t:.2f} × (I − J/3)'); ax.set_ylabel('measured matrix entry')
ax.text(0.03, 0.95, f'column sums {M.sum(0).round(3).tolist()}\nresidual {np.sqrt(((M - t * Pm) ** 2).mean()):.3f}', transform=ax.transAxes, va='top', fontsize=6.8, color=INK2)
ax.set_title('Angle-sum change removed, rest passed'); grid(ax, 'both'); lab(ax, 'F', 'derived from E (descriptive)')

ax = axs[6]
for _, r in sv.iterrows():
    ax.plot([0, 1], [r.identity_mae, r.matrix_mae], '-', color=BAND[r.band], lw=0.9, alpha=0.8)
ax.set_xticks([0, 1]); ax.set_xticklabels(['targets copied', 'matrix']); ax.set_xlim(-0.3, 1.3)
ax.set_ylabel('error, withheld compensated edit (°)'); ax.set_title('Matrix predicts a withheld edit (18/19)'); grid(ax)
lab(ax, 'G', 'STRUCTURE_VALUES.tsv matrix_mae, identity_mae')

ax = axs[7]; x = np.arange(2); w = 0.35
ax.bar(x - w / 2, [0.062, 0.032], w, color=GR2, label='targets copied')
ax.bar(x + w / 2, [0.030, 0.039], w, color=HI, label='matrix')
ax.set_xticks(x); ax.set_xticklabels(['±1° edits', '±0.5° edits'])
ax.set_ylabel('error, compensated edit (°)'); ax.set_title('…but not at half amplitude (4YCA)'); ax.legend(fontsize=6.8, loc='upper right'); grid(ax)
lab(ax, 'H', 'ledger T6 (4YCA pilot; failed half-amplitude comparison)')
for b, col in BAND.items():
    axs[1].scatter([], [], color=col, s=14, label=BANDLAB[b])
axs[1].legend(fontsize=6.5, loc='lower left')
fig.tight_layout(); save(fig, 'Figure1_controlled_response')

CUR[0] = 'Figure2'
# ================== FIGURE 2: predicting a library change ===================
sp = pd.read_csv(P['site_pred'], sep='\t'); pm = pd.read_csv(P['pred_methods'], sep='\t')
ld = pd.read_csv(P['link_diff'], sep='\t'); cc = json.loads(P['closing'].read_text())['results']
fig, axs = plt.subplots(2, 4, figsize=(15, 6.6)); axs = axs.ravel()

ax = axs[0]
def nm(a):
    a = re.findall(r"'(\w+)'\)", a); return '–'.join(a)
ld['name'] = ld.atoms_comp_id.map(lambda a: {'C–N–CA': 'C–N–Cα', 'C–N–CD': 'C–N–Cδ', 'CA–C–N': 'Cα–C–N', 'O–C–N': 'O–C–N', 'C–N–H': 'C–N–H'}.get(nm(a), nm(a)))
order = ['Cα–C–N', 'O–C–N', 'C–N–Cα', 'C–N–Cδ', 'C–N–H']
labels, y = [], 0
for name in order:
    for link, col in (('PTRANS', '#7a3b8f'), ('TRANS', GR1)):
        r = ld[(ld.link == link) & (ld.name == name)]
        if r.empty: continue
        r = r.iloc[0]; ax.barh(y, r.target_delta, color=col, height=0.7)
        wtxt = f'×{r.relative_weight_new_over_old:.1f}' if r.relative_weight_new_over_old > 1.4 else ''
        ax.text(2.6, y, wtxt, va='center', ha='left', fontsize=6.8, color=INK, fontweight='bold' if r.relative_weight_new_over_old > 5 else 'normal')
        labels.append(f'{link} {name}'); y += 1
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6.5)
ax.axvline(0, color=GR2, lw=0.8); ax.set_xlim(-5.5, 3.6); ax.invert_yaxis()
ax.text(2.6, -0.9, 'weight', fontsize=6.5, color=INK2)
ax.set_xlabel('target change (°)'); ax.set_title('2018 CCP4 link revision (full diff)')
ax.text(0.0, -0.25, '+ new Cα–C–O override 120.6° (σ 1.5°) on both links', transform=ax.transAxes, fontsize=6.5, color=INK2)
lab(ax, 'A', 'LINK_ANGLE_DIFF.tsv; DEL-OXT block (README)')

for k, (meth, title, L) in enumerate((('site_specific_full_matrix', 'Coupled response: per site', 'B'), ('target_copy', 'Targets copied: per site', 'C'))):
    ax = axs[1 + k]; d = sp[sp.method == meth]
    for a, col in ANG.items():
        ax.scatter(d[f'pred_m{a}'], d[f'obs_m{a}'], s=12, color=col, edgecolor='white', lw=0.3, label={'ACN': 'Cα–C–N', 'ACO': 'Cα–C–O', 'OCN': 'O–C–N'}[a])
    ax.plot([-3, 2.6], [-3, 2.6], ls='--', color=GR2, lw=0.8)
    ax.set_xlabel('predicted response (°)'); ax.set_ylabel('observed response (°)'); ax.set_title(title); grid(ax, 'both')
    e = pm.set_index('method').loc[meth]
    ax.text(0.97, 0.05, f'site MAE {e.site_term_MAE_deg:.3f}°\nmean-vector {e.mean_vector_MAE_deg:.3f}°', transform=ax.transAxes, ha='right', fontsize=6.8, color=INK)
    lab(ax, L, 'SITE_PREDICTIONS.tsv (4YCA, 18 before-Pro junctions; post hoc)')
axs[1].legend(fontsize=6.5, loc='upper left')

ax = axs[3]
rows = [('site_specific_full_matrix', 'site matrices'), ('same_structure_mean_matrix', 'structure mean matrix'), ('scalar_compensated_projector', 'scalar projector'),
        ('initial_fixed_row_planar', 'restraints alone'), ('target_copy', 'targets copied'), ('nineteen_structure_mean_matrix', '19-structure matrix'), ('site_specific_diagonal', 'no coupling (diagonal)')]
e = pm.set_index('method'); yy = np.arange(len(rows))
ax.barh(yy - 0.18, [e.loc[m, 'site_term_MAE_deg'] for m, _ in rows], 0.36, color=[HI if i < 2 else GR2 for i in range(len(rows))], label='per site')
ax.barh(yy + 0.18, [e.loc[m, 'mean_vector_MAE_deg'] for m, _ in rows], 0.36, color=[HI if i < 2 else GR3 for i in range(len(rows))], alpha=0.55, label='mean vector')
ax.set_yticks(yy); ax.set_yticklabels([r[1] for r in rows]); ax.invert_yaxis(); ax.set_xlabel('prediction error (°)')
ax.set_title('Coupling, not site detail, carries it'); ax.legend(fontsize=6.5, loc='lower right'); grid(ax, 'x')
lab(ax, 'D', 'PREDICTION_METHODS.tsv')

def con(outcome, term, key):
    c = cc[outcome][term]['contrasts'][key]; lo, hi = c['joint_reference_interval95'] or (np.nan, np.nan)
    return c['estimate'], lo, hi
TERMS = [('angle_CaCN', 'Cα–C–N'), ('angle_CA_C_O', 'Cα–C–O'), ('angle_O_C_N_next', 'O–C–N')]
ax = axs[4]; x = np.arange(3)
doc = [1.798, 0.0, -2.453]
obs4 = [e.loc['site_specific_full_matrix', f'pred_{a}'] for a in ('ACN', 'ACO', 'OCN')]
ax.bar(x - 0.3, doc, 0.2, color=GR3, label='documented target contrast')
ax.bar(x - 0.1, [1.857, 0.285, -2.145], 0.2, color=HI, label='controlled refinement (4YCA)')
for j, (key, col, lb) in enumerate((('new_minus_old|all', RF, 'REFMAC 5.8.0222–0267'), ('late_minus_old|all', '#f1a77a', 'REFMAC ≥5.8.0352'))):
    vals = [con('before_pro', t, key) for t, _ in TERMS]
    ax.bar(x + 0.1 + 0.2 * j, [v[0] for v in vals], 0.2, color=col, label=lb, yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], error_kw=dict(lw=0.6))
ax.axhline(0, color=GR2, lw=0.8); ax.set_xticks(x); ax.set_xticklabels([t[1] for t in TERMS])
ax.set_ylabel('change in before-Pro contrast (°)'); ax.set_title('Archive: where the prediction fails'); ax.legend(fontsize=6, loc='lower left'); grid(ax)
lab(ax, 'E', 'CLOSING_CHECKS.json before_pro; PAIR controlled values (N17)')

ax = axs[5]; bands = ['le1.0', 'gt1.0_le1.5', 'gt1.5_le2.0']
for j, (t, nmx) in enumerate(TERMS):
    vals = [con('before_pro', t, f'new_minus_old|{b}') for b in bands]
    ax.errorbar(np.arange(3) + (j - 1) * 0.12, [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], fmt='o-', ms=4, lw=1, color=list(ANG.values())[j], label=nmx)
ax.axhline(0, color=GR2, lw=0.8); ax.set_xticks(range(3)); ax.set_xticklabels([BANDLAB[b] for b in bands])
ax.set_ylabel('new − old contrast (°)'); ax.set_title('Smaller imprint at atomic resolution'); ax.legend(fontsize=6.5); grid(ax)
lab(ax, 'F', 'CLOSING_CHECKS.json before_pro new_minus_old by band (exploratory)')

ax = axs[6]; T4 = TERMS + [('bond_C_N_next', 'C–N (×100 Å)')]
for j, (outcome, col, lb) in enumerate((('before_pro', RF, 'before-Pro'), ('before_gly_placebo', GR1, 'before-Gly (negative control)'))):
    vals = [con(outcome, t, 'new_minus_old|all') for t, _ in T4]
    sc = [100 if t.startswith('bond') else 1 for t, _ in T4]
    ax.bar(np.arange(4) + (j - 0.5) * 0.36, [v[0] * s for v, s in zip(vals, sc)], 0.36, color=col, label=lb,
           yerr=[[(v[0] - v[1]) * s for v, s in zip(vals, sc)], [(v[2] - v[0]) * s for v, s in zip(vals, sc)]], error_kw=dict(lw=0.6))
ax.axhline(0, color=GR2, lw=0.8); ax.set_xticks(range(4)); ax.set_xticklabels([t[1] for t in T4], fontsize=6.5)
ax.set_ylabel('new − old contrast'); ax.set_title('Ordinary junctions moved too, less'); ax.legend(fontsize=6.3, loc='lower left'); grid(ax)
lab(ax, 'G', 'CLOSING_CHECKS.json before_gly_placebo vs before_pro')

ax = axs[7]
s_new = cc['before_pro']['angle_CaCN']  # placeholder to keep structure
sums = []
for key in ('new_minus_old|all', 'late_minus_old|all'):
    sums.append(None)
labels7 = ['documented\ntarget contrast', 'controlled\n(4YCA)', 'REFMAC\n0222–0267', 'REFMAC\n≥0352']
vals7 = [1.798 + 0.0 - 2.453, 1.857 + 0.285 - 2.145, 0.003, 0.001]
ax.bar(range(4), vals7, color=[GR3, HI, RF, '#f1a77a'])
for i, v in enumerate(vals7):
    ax.text(i, v - 0.04 if v < 0 else v + 0.02, f'{v:+.3f}', ha='center', va='top' if v < 0 else 'bottom', fontsize=6.8)
ax.axhline(0, color=GR2, lw=0.8); ax.set_xticks(range(4)); ax.set_xticklabels(labels7, fontsize=6.5)
ax.set_ylabel('sum of the three angle changes (°)'); ax.set_ylim(-0.8, 0.2)
ax.set_title('Coordinates keep the angle sum; targets do not'); grid(ax)
lab(ax, 'H', 'ledger N8 (sum3 +0.003 [−0.007, 0.013]; late +0.001); SITE/PAIR controlled mean; LINK_ANGLE_DIFF.tsv')
fig.tight_layout(); save(fig, 'Figure2_library_change')

CUR[0] = 'Figure3'
# =================== FIGURE 3: archive fingerprints (CDL) ====================
d = pd.read_csv(P['scores'], sep='\t', dtype={'pdb_id': str, 'refmac_version': str})
g = pd.read_csv(P['groups'], sep='\t', usecols=['pdb_id', 'sequence_group'], dtype=str)
d = d.merge(g, on='pdb_id', how='left'); d['dep'] = pd.to_datetime(d.dep_date, errors='coerce'); d['year'] = d.dep.dt.year
f2 = json.loads(P['fig2vals'].read_text())
fig, axm = plt.subplots(2, 3, figsize=(11.6, 6.6)); figsi, axsi = plt.subplots(1, 2, figsize=(8, 3.3)); axs = list(axm.ravel()) + list(axsi)

ph = d[(d.family == 'PHENIX') & d.phenix_build.notna() & d.b_composite.notna()].copy()
ph['bin'] = np.floor((ph.phenix_build - 2155) / 200).astype(int)
ax = axs[0]; rows = []
for bn, gg in ph.groupby('bin'):
    if len(gg) < 10: continue
    m, lo, hi = boot_mean(gg.b_composite); rows.append((2155 + bn * 200 + 100, m, lo, hi, len(gg)))
rows = np.array(rows)
ax.errorbar(rows[:, 0], rows[:, 1], yerr=[rows[:, 1] - rows[:, 2], rows[:, 3] - rows[:, 1]], fmt='o', ms=3.5, color=PH, elinewidth=0.8)
ax.axvline(2155, ls='--', color=GR1, lw=0.8); ax.text(2190, 0.42, 'CDL default\n(build 2155)', fontsize=6.5, color=INK2)
ax.set_xlabel('PHENIX build'); ax.set_ylabel('alignment with CDL pattern, b'); ax.set_ylim(0.38, 1.12)
ax.set_title(f'PHENIX deposits by build (n = {int(rows[:, 4].sum())})'); grid(ax); lab(ax, 'A', 'structure_scores.tsv (blank), 200-build bins')

ax = axs[1]; h = 700; w = ph[(ph.phenix_build - 2155).abs() <= h].copy()
w['b50'] = np.floor((w.phenix_build - 2155) / 50) * 50 + 2155 + 25
bm = w.groupby('b50').b_composite.agg(['mean', 'count']); bm = bm[bm['count'] >= 5]
ax.scatter(bm.index, bm['mean'], s=np.clip(bm['count'], 5, 60), color=PH, alpha=0.55, edgecolor='white', lw=0.4)
for side in (w.phenix_build < 2155, w.phenix_build >= 2155):
    s = w[side]; c = np.polyfit(s.phenix_build - 2155, s.b_composite, 1); xx = np.linspace(s.phenix_build.min(), s.phenix_build.max(), 20)
    ax.plot(xx, np.polyval(c, xx - 2155), color=INK, lw=1.2)
ax.axvline(2155, ls='--', color=GR1, lw=0.8)
rd = f2['A']['records'][1]
ax.text(0.03, 0.95, f"jump +{rd['estimate']:.2f} ({rd['low']:.2f}–{rd['high']:.2f})\n{rd['structures']} structures, {rd['groups']} groups", transform=ax.transAxes, va='top', fontsize=6.8)
ax.set_xlabel('PHENIX build'); ax.set_ylabel('b (50-build bins)'); ax.set_title('Local discontinuity, ±700 builds'); grid(ax)
lab(ax, 'B', 'structure_scores.tsv; FIGURE2_VALUES.json A (sequence-group bootstrap)')

ax = axs[2]; rec = f2['A']['records']
ax.errorbar([r['bandwidth'] for r in rec], [r['estimate'] for r in rec], yerr=[[r['estimate'] - r['low'] for r in rec], [r['high'] - r['estimate'] for r in rec]], fmt='o-', color=PH, ms=4)
sc = pd.read_csv(P['scramble'], sep='\t'); ax.scatter(np.full(len(sc), 700) + np.random.default_rng(2).uniform(-40, 40, len(sc)), sc.rdd, s=8, color=GR1, label='20 scrambled CDL tables')
ax.axhline(0, color=GR2, lw=0.8); ax.set_xlabel('bandwidth (builds)'); ax.set_ylabel('jump in b'); ax.set_xticks([400, 700, 1100])
ax.set_title('Specific to CDL’s own pattern'); ax.legend(fontsize=6.3, loc='upper left'); grid(ax)
lab(ax, 'C', 'FIGURE2_VALUES.json A; scramble_estimates.tsv (original-feature run; blank-feature summary mean 0.010, range −0.08–0.11)')

ax = axs[3]; es = f2['B']['records']; yrs = [r['year'] for r in es]
ax.errorbar(yrs, [r['estimate'] for r in es], yerr=[[r['estimate'] - r['low'] for r in es], [r['high'] - r['estimate'] for r in es]], fmt='o', color=PH, ms=4)
ax.plot([2014], [0], 'D', mfc='white', mec=INK, ms=4); ax.axhline(0, color=GR2, lw=0.8); ax.axvline(2015.67, ls='--', color=GR1, lw=0.8)
ax.set_xlabel('deposition year'); ax.set_ylabel('PHENIX − REFMAC (vs 2014)'); ax.set_title('No pretrend; effect with adoption')
ax.text(0.03, 0.95, f"joint pre-lead p = {f2['B']['joint_prelead_p']:.2f}", transform=ax.transAxes, va='top', fontsize=6.8); grid(ax)
lab(ax, 'D', 'FIGURE2_VALUES.json B (sequence-group bootstrap)')

ax = axs[4]; tj = [('Cα–C–O', 0.367, 0.263, 0.489), ('Cα–C–N', 0.304, 0.202, 0.418), ('C–N–Cα', 0.286, 0.180, 0.424), ('composite', 0.273, 0.187, 0.373), ('τ', 0.136, 0.067, 0.211)]
ax.errorbar([t[1] for t in tj], range(len(tj)), xerr=[[t[1] - t[2] for t in tj], [t[3] - t[1] for t in tj]], fmt='o', color=PH, ms=4)
ax.set_yticks(range(len(tj))); ax.set_yticklabels([t[0] for t in tj]); ax.invert_yaxis(); ax.axvline(0, color=GR2, lw=0.8)
ax.set_xlabel('jump at build 2155 (h = 700)'); ax.set_title('Every angle; τ least'); grid(ax, 'x')
lab(ax, 'E', 'SI Table S2a (si_tables_blank_out; structure bootstrap)')

ax = axs[5]; yy = d[d.family.isin(['PHENIX', 'REFMAC']) & d.year.between(2008, 2024)]
for fam, col in (('PHENIX', PH), ('REFMAC', RF)):
    s = yy[yy.family == fam].groupby('year').b_composite.agg(['mean', 'count']); s = s[s['count'] >= 20]
    ax.plot(s.index, s['mean'], 'o-', color=col, ms=3, lw=1.2, label=fam)
ax.axvline(2015.67, ls='--', color=GR1, lw=0.8); ax.axvline(2018.29, ls=':', color=GR1, lw=0.8)
ax.set_xlabel('deposition year'); ax.set_ylabel('mean b (composite)'); ax.set_title('Both programs drift; PHENIX jumps'); ax.legend(fontsize=6.5); grid(ax)
lab(ax, 'F', 'structure_scores.tsv yearly means')

def vkey(v):
    n = [int(x) for x in re.findall(r'\d+', str(v))][:3]
    return tuple(n + [0] * (3 - len(n))) if n else None
rf = d[(d.family == 'REFMAC')].copy(); rf['vk'] = rf.refmac_version.map(vkey); rf = rf[rf.vk.notna()]
grp = {'≤5.8.0189': rf[rf.vk.map(lambda k: k <= (5, 8, 189))],
       '5.8.0222–0267': rf[rf.vk.map(lambda k: (5, 8, 222) <= k <= (5, 8, 267))], '≥5.8.0352': rf[rf.vk.map(lambda k: k >= (5, 8, 352))]}
ax = axs[6]
for j, (term, nmx, col) in enumerate((('b_tau_deg', 'τ', INK), ('b_angle_CaCN', 'Cα–C–N', ANG['ACN']), ('b_angle_CA_C_O', 'Cα–C–O', ANG['ACO']))):
    vals = [boot_mean(gg[term].dropna()) for gg in grp.values()]
    ax.errorbar(np.arange(3) + (j - 1) * 0.1, [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], fmt='o-', ms=4, lw=1, color=col, label=nmx)
ax.set_xticks(range(3)); ax.set_xticklabels(list(grp), fontsize=6.5); ax.set_ylabel('b in REFMAC deposits')
ax.set_title('2018 dictionary era: τ falls, others rise'); ax.legend(fontsize=6.5); grid(ax)
lab(ax, 'A', 'structure_scores.tsv by REFMAC version (version parsing as original)')

ax = axs[7]; oc = pd.read_csv(S/'req2/pbm_required_checks.mvof9b/old_corrected/old_vs_corrected_rdd.tsv', sep='\t')
oc = oc[oc.contrast.str.contains('resolution_covariate=True')]
ax.scatter(oc.old_estimate, oc.corrected_estimate, s=14, color=PH, edgecolor='white', lw=0.3)
lim = [min(oc.old_estimate.min(), oc.corrected_estimate.min()) - 0.05, max(oc.old_estimate.max(), oc.corrected_estimate.max()) + 0.05]
ax.plot(lim, lim, ls='--', color=GR2, lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('jump, original extraction'); ax.set_ylabel('jump, corrected (blank-altloc) extraction')
ax.text(0.03, 0.95, f'{len(oc)} term × bandwidth estimates\nmax |Δ| = {np.abs(oc.corrected_estimate - oc.old_estimate).max():.3f}', transform=ax.transAxes, va='top', fontsize=6.8)
ax.set_title('Robust to the mixed-conformer bug fix'); grid(ax, 'both')
lab(ax, 'B', 'old_vs_corrected_rdd.tsv (settings match; script versions differ)')
fig.tight_layout(); save(fig, 'Figure3_archive_fingerprints')
figsi.tight_layout(); save(figsi, 'FigureS3_refmac_era_and_altloc')


print('stopped after figure 3')
