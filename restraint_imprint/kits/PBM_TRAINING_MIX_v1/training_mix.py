#!/usr/bin/env python3
"""Training-mix sensitivity (spec v1.0): refit Ca-C-N / O-C-N surfaces with v2's exact rules on
different program mixes and score them on the frozen held-out deposits. No library is promoted.
  python training_mix.py --root /mnt/c/Wei/PBM --out RESULT_DIR
  python training_mix.py --selftest
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

KIT = Path(__file__).resolve().parent
SPEC_SHA = '22447a12c8e0cc85a4183f09793a9c45e1b6e9986b2c741c5d1becf9cb47f66b'
H = {'train_geometry': 'a2929836e8f23fc6d5109b1423baea24952ae9cb98e492bc64a8c25c3765300f',
     'targets': 'ef7b86b359c84492a0be791de52a1e64112a0153142b9f9f5a0417448d2731b1',
     'training_chains': '35f69bd270b9f6ca274af6c1dfedcc83adc97aa38f82e6104140d99ee535e494',
     'headers': '25f466e255d6742ff9cfd83eecd92d5769f7bce0ce7e254c02025cb8f71f40d4',
     'junctions': '8b3ccb9ee7481288c336bae9aafff8411f207a87c6cb3ecba45bcad30f110cf6',
     'metadata': '2e1a820d7e6165d4c838e77d5a999b7205bf71814115065a04e61df055cbfa92',
     'strata': '34525a769eb6b273f3bb1a0efcf964b9ceaaba61fa372dee29d08fc6b1c6bd18'}
PATHS = {'train_geometry': 'pbm_altloc_training_audit_results_v1/blank_only_geometry.csv.gz',
         'targets': 'PBM_CLEAN_REBUILD_v1/results/training_targets.npz',
         'junctions': 'pbm_v2_geometry_qc_v1/junctions.tsv',
         'metadata': 'pbm_v2_geometry_metadata_v1/junction_metadata.tsv',
         'strata': 'PBM_V2_HELDOUT_SCORE/refinement_strata.tsv'}
NONPGIV = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'LEU', 'LYS', 'MET', 'PHE', 'SER', 'THR', 'TRP', 'TYR'}
RES20 = sorted(NONPGIV | {'ILE', 'VAL', 'GLY', 'PRO'})
AA = set(RES20)
B, MIN_COUNT, SEED, DRAWS, MIN_J = 36, 10, 20261114, 10000, 5
TERMS = {'ACN': ('angle_CaCN', 'cacn_deg'), 'OCN': ('angle_O_C_N_next', 'ocn_deg')}
FITS = ['F0_v2', 'F1_natural_PR', 'F2_balanced_PR', 'F3_REFMAC_only', 'F4_PHENIX_only']
STRATA = {'REFMAC': 'REFMAC', 'PHENIX_post': 'PHENIX_post_default'}
KEY = ['pdb_id', 'component', 'chain', 'seq_i', 'seq_next']


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def cell_index(x):
    return (np.floor((np.asarray(x, float) + 180.0) / 10.0).astype(int)) % B


def accumulate(rows, term, w):
    ri = rows['res_name'].map({r: k for k, r in enumerate(RES20)}).to_numpy(float)
    v = rows[term].to_numpy(float)
    ok = ~np.isnan(ri) & np.isfinite(v)
    idx = (ri[ok].astype(int), cell_index(rows['phi_deg'])[ok], cell_index(rows['psi_deg'])[ok])
    S, W, N = (np.zeros((len(RES20), B, B)) for _ in range(3))
    np.add.at(S, idx, (w[ok] * v[ok])); np.add.at(W, idx, w[ok]); np.add.at(N, idx, 1)
    return S, W, N


def grids(term, rows, pro, w):
    """v2 rules (clean pooled rule), weighted means; >= 10 rule on unweighted counts."""
    s, wt, n = accumulate(rows, term, w)
    ps, pw, pn = s.sum(0), wt.sum(0), n.sum(0)
    gmean = float(s.sum() / wt.sum())
    pooled = np.where(pn >= MIN_COUNT, ps / np.maximum(pw, 1e-300), gmean)
    unsplit = np.where(n >= MIN_COUNT, s / np.maximum(wt, 1e-300), pooled)
    split, counts = [], []
    for flag in (False, True):
        m = (pro == flag).to_numpy()
        ss, ww, nn = accumulate(rows.loc[m], term, w[m])
        split.append(np.where(nn >= MIN_COUNT, ss / np.maximum(ww, 1e-300), unsplit)); counts.append(nn.astype(np.int64))
    return unsplit, np.stack(split), n.astype(np.int64), np.stack(counts)


def eligibility(t):
    t = t.sort_values(['pdb_id', 'chain', 'res_idx']).reset_index(drop=True)
    nxt = t.groupby(['pdb_id', 'chain'], sort=False)[['res_idx', 'res_name']].shift(-1)
    cn, om = t['bond_C_N_next'], t['omega_measured_deg'].abs()
    pro = nxt['res_name'].eq('PRO')
    elig = (nxt['res_idx'].eq(t['res_idx'] + 1) & nxt['res_name'].isin(AA) & np.isfinite(cn) & cn.gt(0) & cn.le(1.8)
            & t['res_name'].isin(AA) & np.isfinite(t['phi_deg']) & np.isfinite(t['psi_deg'])
            & om.gt(150) & om.le(180.001) & t['is_pro_np1'].isin([0, 1]) & t['is_pro_np1'].eq(pro.astype(int)))
    return t, pro, elig


def catmull(G, g, x, y, x0=-175.0):
    u, v = (np.asarray(x, float) - x0) / 10.0, (np.asarray(y, float) - x0) / 10.0
    i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
    tx, ty = u - i0, v - j0

    def w(t):
        t2, t3 = t * t, t * t * t
        return [0.5 * (-t3 + 2 * t2 - t), 0.5 * (3 * t3 - 5 * t2 + 2), 0.5 * (-3 * t3 + 4 * t2 + t), 0.5 * (t3 - t2)]
    wx, wy = w(tx), w(ty)
    out = np.zeros(len(u))
    for a in range(4):
        for b in range(4):
            out += wx[a] * wy[b] * G[g, (i0 + a - 1) % B, (j0 + b - 1) % B]
    return out


def fit_all(t, pro, elig, family, frozen=None):
    fits, gate = {}, {}
    fam = t['pdb_id'].map(family)
    for key, (term, _) in TERMS.items():
        v = t[term]
        ok = elig & np.isfinite(v) & v.gt(0) & v.lt(180)
        for name in FITS:
            if name == 'F0_v2':
                m = ok
                w = np.ones(int(m.sum()))
            else:
                prog = {'F1_natural_PR': ['PHENIX', 'REFMAC'], 'F2_balanced_PR': ['PHENIX', 'REFMAC'],
                        'F3_REFMAC_only': ['REFMAC'], 'F4_PHENIX_only': ['PHENIX']}[name]
                m = ok & fam.isin(prog)
                w = np.ones(int(m.sum()))
                if name == 'F2_balanced_PR':
                    f = fam[m].to_numpy()
                    nP, nR = (f == 'PHENIX').sum(), (f == 'REFMAC').sum()
                    w = np.where(f == 'PHENIX', 0.5 / nP, 0.5 / nR) * len(f)
            arr = grids(term, t.loc[m], pro[m], w)
            fits[(name, key)] = arr
            if name == 'F0_v2' and frozen is not None:
                same = [np.array_equal(a, frozen[f'{term}__{s}']) if s.startswith('n_') else bool(np.all(np.abs(a - frozen[f'{term}__{s}']) <= 1e-12))
                        for a, s in zip(arr, ('unsplit', 'v2', 'n_unsplit', 'n_split'))]
                gate[term] = dict(rows=int(m.sum()), reproduced=all(same))
                if not all(same):
                    sys.exit(f'STOP: F0 does not reproduce the frozen v2 arrays for {term}')
    return fits, gate


def surface_shift(fits):
    rows = []
    for key in TERMS:
        for a, b in (('F2_balanced_PR', 'F1_natural_PR'), ('F3_REFMAC_only', 'F4_PHENIX_only')):
            A, NA = fits[(a, key)][1], fits[(a, key)][3]
            Bv, NB = fits[(b, key)][1], fits[(b, key)][3]
            sup = (NA >= MIN_COUNT) & (NB >= MIN_COUNT)
            d = np.abs(A - Bv)[sup]
            rows.append(dict(term=key, pair=f'{a} vs {b}', supported_cells=int(sup.sum()),
                             mean_abs_shift=float(d.mean()), p95_abs_shift=float(np.quantile(d, .95))))
    return pd.DataFrame(rows)


def score(d, fits):
    out = []
    ri = d['res_i'].map({r: k for k, r in enumerate(RES20)})
    ok0 = ri.notna() & d['res_next'].isin(AA) & d['phi_i'].notna() & d['psi_i'].notna()
    d = d[ok0]
    si = (ri[ok0].astype(int) + 20 * d['res_next'].eq('PRO').astype(int)).to_numpy()
    for key, (term, col) in TERMS.items():
        obs = d[col].to_numpy(float)
        res = {}
        for name in FITS:
            res[name] = np.abs(catmull(fits[(name, key)][1].reshape(40, B, B), si, d['phi_i'], d['psi_i']) - obs)
        keep = np.isfinite(obs) & np.all([np.isfinite(r) for r in res.values()], axis=0)
        frame = d.loc[keep, ['pdb_id', 'component', 'refinement_stratum']].copy()
        frame['term'] = key
        for name in FITS:
            frame[name] = res[name][keep]
        out.append(frame)
    j = pd.concat(out)
    s = j.groupby(['term', 'refinement_stratum', 'pdb_id', 'component']).agg(
        **{n: (n, 'mean') for n in FITS}, junctions=('F0_v2', 'size')).reset_index()
    return s[s['junctions'] >= MIN_J]


def boot(s, term, a, b, strata, seed=SEED, draws=DRAWS):
    """Component-weighted mean of (a - b) in one stratum, or the gap between two strata (shared weights)."""
    x = s[(s['term'] == term) & s['refinement_stratum'].isin(strata)].copy()
    x['D'] = x[a] - x[b]
    comps = pd.Index(sorted(x['component'].unique()))
    cells = []
    for st in strata:
        m = x[x['refinement_stratum'] == st].groupby('component')['D'].mean().reindex(comps)
        cells.append((m.notna().to_numpy(float), m.fillna(0).to_numpy()))

    def est(W):
        vals = []
        for pres, val in cells:
            den = W @ pres
            vals.append(np.where(den > 0, (W @ val) / np.where(den > 0, den, 1), np.nan))
        return vals[0] if len(vals) == 1 else vals[0] - vals[1]
    point = float(est(np.ones((1, len(comps))))[0])
    rng = np.random.default_rng(seed)
    smp = np.concatenate([est(rng.multinomial(len(comps), np.full(len(comps), 1 / len(comps)), size=min(500, draws - k)).astype(float))
                          for k in range(0, draws, 500)])
    smp = smp[np.isfinite(smp)]
    p = min(1.0, 2 * min((smp <= 0).mean(), (smp >= 0).mean()))
    return dict(estimate=point, ci95=np.quantile(smp, [.025, .975]).tolist(), p_boot=float(p),
                components={st: int(x.loc[x['refinement_stratum'] == st, 'component'].nunique()) for st in strata})


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj, run = [None] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj


def contrasts(s):
    res = {}
    for st_label, st in STRATA.items():
        for name, (a, b) in {'Delta_bal': ('F2_balanced_PR', 'F1_natural_PR'),
                             'REFMAC_only_vs_natural': ('F3_REFMAC_only', 'F1_natural_PR'),
                             'PHENIX_only_vs_natural': ('F4_PHENIX_only', 'F1_natural_PR'),
                             'natural_PR_vs_v2': ('F1_natural_PR', 'F0_v2')}.items():
            r = {k: boot(s, k, a, b, [st]) for k in TERMS}
            for k, pa in zip(TERMS, holm([r[k]['p_boot'] for k in TERMS])):
                r[k]['p_holm'] = pa
            res[f'{name}|{st_label}'] = r
    r = {k: boot(s, k, 'F2_balanced_PR', 'F1_natural_PR', [STRATA['REFMAC'], STRATA['PHENIX_post']]) for k in TERMS}
    for k, pa in zip(TERMS, holm([r[k]['p_boot'] for k in TERMS])):
        r[k]['p_holm'] = pa
    res['Gap_change_G'] = r
    return res


def figure(res, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    rows = [(k, t) for k in res for t in TERMS]
    for i, (k, t) in enumerate(rows):
        e = res[k][t]
        ax.errorbar(e['estimate'], i, xerr=[[e['estimate'] - e['ci95'][0]], [e['ci95'][1] - e['estimate']]],
                    fmt='o', ms=3.5, lw=1, color='#2b6cb0' if 'REFMAC' in k else '#c05621' if 'PHENIX' in k else '#444')
    ax.axvline(0, color='#555', lw=0.8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f'{k}  {t}' for k, t in rows], fontsize=6.5); ax.invert_yaxis()
    ax.set_xlabel('MAE difference (deg)'); ax.set_title('Training-mix sensitivity', fontsize=9, loc='left')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'TrainingMix.{ext}', dpi=200)


def main(a):
    if sha(KIT / 'TRAINING_MIX_SPEC_20261112.md') != SPEC_SHA:
        sys.exit('STOP: spec hash differs')
    root = Path(a.root)
    P = {k: root / v for k, v in PATHS.items()}
    P['training_chains'] = KIT / 'inputs' / 'training_chains.tsv'
    P['headers'] = KIT / 'inputs' / 'headers.tsv'
    hashes = {}
    for k, p in P.items():
        hashes[k] = sha(p)
        if hashes[k] != H[k]:
            sys.exit(f'STOP: {k} hash differs')
    if a.out.exists():
        sys.exit(f'STOP: {a.out} exists')
    cols = ['pdb_id', 'chain', 'res_idx', 'res_name', 'phi_deg', 'psi_deg', 'omega_measured_deg', 'is_pro_np1',
            'bond_C_N_next', 'angle_CaCN', 'angle_O_C_N_next']
    t = pd.read_csv(P['train_geometry'], usecols=cols, keep_default_na=False, low_memory=False, dtype={'pdb_id': str, 'chain': str})
    t['pdb_id'] = t['pdb_id'].str.strip().str.upper()
    for c in set(cols) - {'pdb_id', 'chain', 'res_name'}:
        t[c] = pd.to_numeric(t[c], errors='coerce')
    keys = pd.read_csv(P['training_chains'], sep='\t', dtype=str, keep_default_na=False)
    keys['pdb_id'] = keys['pdb_id'].str.upper()
    t = t.merge(keys[['pdb_id', 'chain']], on=['pdb_id', 'chain'])
    if len(t) != 1224328:
        sys.exit('STOP: training row count differs from 1,224,328')
    hd = pd.read_csv(P['headers'], sep='\t', dtype=str).set_index('pdb_id')
    family = hd['family'].to_dict()
    ids = keys['pdb_id'].unique()
    lab = pd.Series([family.get(i) for i in ids])
    cov = dict(training_structures=len(ids), labelled=float(lab.notna().mean()),
               PHENIX=int((lab == 'PHENIX').sum()), REFMAC=int((lab == 'REFMAC').sum()),
               other_or_unlabelled=int((~lab.isin(['PHENIX', 'REFMAC'])).sum()))
    if cov['labelled'] < 0.90:
        sys.exit(f'STOP: label coverage {cov["labelled"]:.3f} < 0.90')
    with np.load(P['targets'], allow_pickle=False) as z:
        frozen = {k: z[k] for k in z.files}
    t, pro, elig = eligibility(t)
    fits, gate = fit_all(t, pro, elig, family, frozen)
    a.out.mkdir(parents=True)
    (a.out / 'REBUILD_GATE.json').write_text(json.dumps(gate, indent=2) + '\n')
    (a.out / 'LABEL_COVERAGE.json').write_text(json.dumps(cov, indent=2) + '\n')
    surface_shift(fits).to_csv(a.out / 'SURFACE_SHIFT.tsv', sep='\t', index=False)
    dt = {n: str for n in KEY[:3]}
    rows = pd.read_csv(P['junctions'], sep='\t', dtype=dt)
    meta = pd.read_csv(P['metadata'], sep='\t', dtype=dt, usecols=KEY + ['B30_window_pass'])
    strata = pd.read_csv(P['strata'], sep='\t', dtype=str, usecols=['pdb_id', 'component', 'refinement_stratum'])
    d = rows.merge(meta, on=KEY, validate='one_to_one').merge(strata, on=KEY[:2], validate='many_to_one')
    if len(d) != 618881:
        sys.exit('STOP: held-out join count differs')
    d = d[d['B30_window_pass'].eq('pass') & d['refinement_stratum'].isin(STRATA.values())]
    s = score(d, fits)
    s.to_csv(a.out / 'STRUCTURE_MAE.tsv', sep='\t', index=False)
    res = contrasts(s)
    (a.out / 'CONTRASTS.json').write_text(json.dumps(res, indent=2) + '\n')
    figure(res, a.out)
    mae = {f'{k}|{st}': {n: float(s[(s.term == k) & (s.refinement_stratum == st)].groupby('component')[n].mean().mean())
                         for n in FITS} for k in TERMS for st in STRATA.values()}
    (a.out / 'RUN_LOCK.json').write_text(json.dumps(dict(spec_sha256=SPEC_SHA, hashes=hashes, code_sha256=sha(Path(__file__)),
                                                         seed=SEED, draws=DRAWS, label_coverage=cov, group_weighted_mae=mae), indent=2) + '\n')
    print('Label coverage', cov)
    print(surface_shift(fits).to_string(index=False))
    for k, v in res.items():
        for term, r in v.items():
            print(f"{k:34s}{term}: {r['estimate']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] p_holm={r['p_holm']:.3g}")
    print('Return the whole folder:', a.out)


def selftest():
    rng = np.random.default_rng(5)
    n = 60000
    t = pd.DataFrame({'pdb_id': [f'S{i % 300}' for i in range(n)], 'res_name': rng.choice(['ALA', 'LEU'], n),
                      'phi_deg': rng.uniform(-80, -40, n), 'psi_deg': rng.uniform(-60, -20, n)})
    fam = {f'S{i}': ('PHENIX' if i < 100 else 'REFMAC') for i in range(300)}
    prog = t['pdb_id'].map(fam)
    t['angle_CaCN'] = 116.5 + np.where(prog == 'PHENIX', 0.6, 0.0) + rng.normal(0, .3, n)
    t['angle_O_C_N_next'] = 122.8 + rng.normal(0, .3, n)
    pro = pd.Series(np.zeros(n, bool))
    elig = pd.Series(np.ones(n, bool))
    fits, _ = fit_all(t, pro, elig, fam)
    v = {k: float(np.median(fits[(k, 'ACN')][1][0, 0])) for k in FITS}  # ALA, non-Pro, all cells (fallbacks fill most)
    sel = (fits[('F1_natural_PR', 'ACN')][3][0, 0] >= MIN_COUNT)
    m = {k: float(fits[(k, 'ACN')][1][0, 0][sel].mean()) for k in FITS}
    print({k: round(x, 3) for k, x in m.items()})
    assert abs(m['F3_REFMAC_only'] - 116.5) < 0.05 and abs(m['F4_PHENIX_only'] - 117.1) < 0.05
    assert abs(m['F2_balanced_PR'] - 116.8) < 0.05            # equal programme weights
    assert abs(m['F1_natural_PR'] - (116.5 + 0.6 / 3)) < 0.05  # natural 1:2 mix
    print('PASS selftest: weighting reproduces the intended programme mixes')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/mnt/c/Wei/PBM')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        if not a.out:
            ap.error('--out required')
        main(a)
