#!/usr/bin/env python3
"""Build MechLib v3 (spec MECHLIB_V3_SPEC.md v1.1): v2 refitted on training junctions corrected for
measured refinement transmission in Ca-C-N, O-C-N and Ca-C-O. All other terms are copied from v2.
  python build_v3.py --root /mnt/c/Wei/PBM --cdl /path/cdl_database.py --deimprint /path/pbm_deimprint.5tttMa/results --out DIR
  python build_v3.py --selftest
"""
import argparse, hashlib, json, re, runpy, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

KIT = Path(__file__).resolve().parent
warnings.filterwarnings('ignore', message='Mean of empty slice')
SPEC = KIT / 'MECHLIB_V3_SPEC.md'
H = {'train_geometry': 'a2929836e8f23fc6d5109b1423baea24952ae9cb98e492bc64a8c25c3765300f',
     'targets': 'ef7b86b359c84492a0be791de52a1e64112a0153142b9f9f5a0417448d2731b1',
     'training_chains': '35f69bd270b9f6ca274af6c1dfedcc83adc97aa38f82e6104140d99ee535e494',
     'headers': '25f466e255d6742ff9cfd83eecd92d5769f7bce0ce7e254c02025cb8f71f40d4',
     'cdl': '79c197e5692986cde26db9aa492cecfb5b8b7d5b1f3cba4dc222bb2e04d23c68',
     'parameters': '970aa2c9991edb80a1479a32efcd32f68ac2df8b4959633897630920ad9b2f45',
     'bands': '10dfec0c19848bd734288becc3ddda02dcdb195ce0e4aa71be9612c2e7bc201b',
     'groups': '33299405ec11d22d0366d58c88324a7f2124313a6b6cfe8700c0896a9efc4348'}
BOOT_SEED, BOOT_DRAWS = 20261117, 200
PATHS = {'train_geometry': 'pbm_altloc_training_audit_results_v1/blank_only_geometry.csv.gz',
         'targets': 'PBM_CLEAN_REBUILD_v1/results/training_targets.npz'}
NONPGIV = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'LEU', 'LYS', 'MET', 'PHE', 'SER', 'THR', 'TRP', 'TYR'}
RES20 = sorted(NONPGIV | {'ILE', 'VAL', 'GLY', 'PRO'})
AA = set(RES20)
B, MIN_COUNT, MAD_CUT = 36, 10, 8.0
N_TRAIN = 1224328
# term key -> (feature/npz column, CDL v1.2 column index)
TERMS = {'ACN': ('angle_CaCN', 12), 'OCN': ('angle_O_C_N_next', 14), 'ACO': ('angle_CA_C_O', 10)}
USED = ('PHENIX_post', 'REFMAC')
NEW_MIN, OLD_MAX = (5, 8, 222), (5, 8, 189)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ v2 rules (as training_mix.py, which reproduced v2 exactly)
def cell_index(x):
    return (np.floor((np.asarray(x, float) + 180.0) / 10.0).astype(int)) % B


def accumulate(res, phi, psi, v):
    ri = pd.Series(res).map({r: k for k, r in enumerate(RES20)}).to_numpy(float)
    v = np.asarray(v, float)
    ok = ~np.isnan(ri) & np.isfinite(v)
    idx = (ri[ok].astype(int), cell_index(phi)[ok], cell_index(psi)[ok])
    S, N = np.zeros((len(RES20), B, B)), np.zeros((len(RES20), B, B))
    np.add.at(S, idx, v[ok]); np.add.at(N, idx, 1)
    return S, N


def grids(res, phi, psi, v, pro):
    res, phi, psi, v, pro = (np.asarray(a) for a in (res, phi, psi, v, pro))
    s, n = accumulate(res, phi, psi, v)
    ps, pn = s.sum(0), n.sum(0)
    gmean = float(s.sum() / n.sum())
    pooled = np.where(pn >= MIN_COUNT, ps / np.maximum(pn, 1e-300), gmean)
    unsplit = np.where(n >= MIN_COUNT, s / np.maximum(n, 1e-300), pooled)
    split, counts = [], []
    for flag in (False, True):
        m = pro == flag
        ss, nn = accumulate(res[m], phi[m], psi[m], v[m])
        split.append(np.where(nn >= MIN_COUNT, ss / np.maximum(nn, 1e-300), unsplit)); counts.append(nn.astype(np.int64))
    return unsplit, np.stack(split), n.astype(np.int64), np.stack(counts)


def eligibility(t):
    t = t.sort_values(['pdb_id', 'chain', 'res_idx']).reset_index(drop=True)
    nxt = t.groupby(['pdb_id', 'chain'], sort=False)[['res_idx', 'res_name']].shift(-1)
    cn, om = t['bond_C_N_next'], t['omega_measured_deg'].abs()
    pro = nxt['res_name'].eq('PRO')
    elig = (nxt['res_idx'].eq(t['res_idx'] + 1) & nxt['res_name'].isin(AA) & np.isfinite(cn) & cn.gt(0) & cn.le(1.8)
            & t['res_name'].isin(AA) & np.isfinite(t['phi_deg']) & np.isfinite(t['psi_deg'])
            & om.gt(150) & om.le(180.001) & t['is_pro_np1'].isin([0, 1]) & t['is_pro_np1'].eq(pro.astype(int)))
    t['next_name'] = nxt['res_name']
    return t, pro, elig


# ------------------------------------------------------------------ CDL (as the de-imprint kit)
def catmull(G, g, x, y, x0=-180.0):
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


def cdl_group(res, nxt):
    g = 'NonPGIV' if res in NONPGIV else 'IleVal' if res in ('ILE', 'VAL') else 'Gly' if res == 'GLY' else 'Pro' if res == 'PRO' else None
    return None if g is None else g + ('_xpro' if nxt == 'PRO' else '_nonxpro')


def cdl_arrays(path):
    db = runpy.run_path(str(path))['cdl_database']
    if getattr(db, 'version', None) != 'CDL v1.2':
        sys.exit('STOP: unexpected CDL version')
    groups = sorted(db)
    arrays = {}
    for term, (_, col) in TERMS.items():
        grid = np.full((len(groups), B, B), np.nan)
        for i, gname in enumerate(groups):
            for (phi, psi), row in db[gname].items():
                if row[col] > 0:
                    grid[i, (phi + 180) // 10, (psi + 180) // 10] = row[col]
        arrays[term] = grid
    return groups, arrays


# ------------------------------------------------------------------ strata
def parse_version(v):
    v = str(v).strip()
    if not re.fullmatch(r'\d+(\.\d+){2,}', v):
        return None
    return tuple(int(x) for x in v.split('.'))


def strata(H):
    build = pd.to_numeric(H['phenix_build'], errors='coerce')
    lab = pd.Series(None, index=H.index, dtype=object); era = pd.Series(None, index=H.index, dtype=object)
    post = H['family'].eq('PHENIX') & build.ge(2155)
    lab[post] = 'PHENIX_post'; era[post] = 'cdl'
    lab[H['family'].eq('PHENIX') & build.lt(2155)] = 'PHENIX_pre'
    ref = H['family'].eq('REFMAC')
    lab[ref] = 'REFMAC'
    vers = H['refmac_version'].map(parse_version)
    era[ref & vers.map(lambda x: x is not None and x >= NEW_MIN)] = 'new'
    before = pd.to_datetime(H['dep_date'], errors='coerce') < pd.Timestamp('2018-04-16')  # revised dictionary did not yet exist
    era[ref & (vers.map(lambda x: x is not None and x <= OLD_MAX) | before)] = 'old'
    return lab, era


# ------------------------------------------------------------------ correction
def correct(q, L, T, unit):
    """d = mu_u + [(q - mu_u) - T (L - Lbar_u)] / (1 - T); q, L, T arrays; unit: hashable labels."""
    df = pd.DataFrame({'q': q, 'L': L, 'T': T, 'u': unit})
    mu = df.groupby('u')['q'].transform('mean'); Lb = df.groupby('u')['L'].transform('mean')
    return (mu + ((df['q'] - mu) - df['T'] * (df['L'] - Lb)) / (1.0 - df['T'])).to_numpy()


def guard(v, pro, ok):
    v = v.copy(); removed = 0
    for flag in (False, True):
        m = ok & (pro == flag)
        med = np.nanmedian(v[m]); mad = np.nanmedian(np.abs(v[m] - med)) * 1.4826
        bad = m & (np.abs(v - med) > MAD_CUT * mad)
        removed += int(bad.sum()); v[bad] = np.nan
    return v, removed


def corrected_values(t, pro, elig, lab, era, res_band, cdl_pred, key, Tset, include_pre=None, include_others=False):
    """Corrected values d (NaN where unused), use mask and diagnostics. Tset: stratum -> T or band dict."""
    col = TERMS[key][0]
    v = t[col].to_numpy(float).copy()
    ok = (elig & np.isfinite(t[col]) & t[col].gt(0) & t[col].lt(180)).to_numpy().copy()
    pro_a = pro.to_numpy()
    v, removed = guard(v, pro_a, ok)
    ok &= np.isfinite(v)
    L = np.where(lab.eq('PHENIX_post').to_numpy(), cdl_pred, 0.0)
    T = np.full(len(t), np.nan)
    for s in USED:
        m = lab.eq(s).to_numpy()
        if isinstance(Tset[s], dict):
            T[m] = np.where(res_band[m] <= 1.5, Tset[s]['le1.5'], Tset[s]['gt1.5'])
        else:
            T[m] = Tset[s]
    use = ok & lab.isin(USED).to_numpy() & era.notna().to_numpy() & np.isfinite(L) & np.isfinite(T)
    if include_pre is not None:
        mpre = ok & lab.eq('PHENIX_pre').to_numpy()
        T[mpre] = include_pre; use |= mpre
    unit = pd.Series(list(zip(lab.fillna('x'), era.fillna('pre'), t['res_name'], pro_a))).astype(str).to_numpy()
    d = np.full(len(t), np.nan)
    d[use] = correct(v[use], L[use], T[use], unit[use])
    if include_others:
        mo = ok & ~lab.isin(['PHENIX_post', 'REFMAC', 'PHENIX_pre']).to_numpy()
        d[mo] = v[mo]; use |= mo
    by = pd.Series(lab.fillna('other').to_numpy()[use]).astype(str) + '|' + pd.Series(era.fillna('-').to_numpy()[use]).astype(str)
    diag = dict(rows=int(use.sum()), eligible_rows=int(ok.sum()), guard_removed=removed, rows_by_stratum_era=by.value_counts().to_dict())
    return d, use, v, diag


def fit(t, pro, d, use):
    return grids(t['res_name'].to_numpy()[use], t['phi_deg'].to_numpy()[use], t['psi_deg'].to_numpy()[use], d[use], pro.to_numpy()[use])


def build_term(t, pro, elig, lab, era, res_band, cdl_pred, key, **kw):
    d, use, _, diag = corrected_values(t, pro, elig, lab, era, res_band, cdl_pred, key, **kw)
    return fit(t, pro, d, use), diag


# ------------------------------------------------------------------ added reports (spec v1.1): stability, dominance, support
def wgrids(ri, ci, pro, v, w):
    """v2 rules with integer resampling multiplicities w (count rule on resampled counts); split grid only."""
    nR = len(RES20); size = nR * B * B
    def acc(m):
        S = np.bincount(ci[m], weights=w[m] * v[m], minlength=size).reshape(nR, B, B)
        N = np.bincount(ci[m], weights=w[m], minlength=size).reshape(nR, B, B)
        return S, N
    allm = np.ones(len(v), bool)
    s, n = acc(allm)
    ps, pn = s.sum(0), n.sum(0)
    g = s.sum() / n.sum()
    pooled = np.where(pn >= MIN_COUNT, ps / np.maximum(pn, 1e-300), g)
    unsplit = np.where(n >= MIN_COUNT, s / np.maximum(n, 1e-300), pooled)
    out = []
    for flag in (False, True):
        ss, nn = acc(pro == flag)
        out.append(np.where(nn >= MIN_COUNT, ss / np.maximum(nn, 1e-300), unsplit))
    return np.stack(out)


def flat_index(t, use):
    ri = t['res_name'].map({r: k for k, r in enumerate(RES20)}).to_numpy()[use].astype(int)
    return ri * B * B + cell_index(t['phi_deg'].to_numpy()[use]) * B + cell_index(t['psi_deg'].to_numpy()[use])


def bootstrap(t, pro, grp, cases, draws=BOOT_DRAWS, seed=BOOT_SEED):
    """cases: name -> (values, use). Sequence-group resampling; same draws for every case. Returns SE arrays and paired SE of (case - ref)."""
    codes, uniq = pd.factorize(grp)
    rng = np.random.default_rng(seed)
    prep = {k: (flat_index(t, u), pro.to_numpy()[u], v[u], codes[u]) for k, (v, u) in cases.items()}
    sums = {k: 0.0 for k in cases}; sq = {k: 0.0 for k in cases}; dsum = {}; dsq = {}
    names = list(cases)
    for _ in range(draws):
        mult = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq))).astype(float)
        est = {}
        for k, (ci, pr, v, gc) in prep.items():
            est[k] = wgrids(None, ci, pr, v, mult[gc])
            sums[k] = sums[k] + est[k]; sq[k] = sq[k] + est[k] ** 2
        for k in names[1:]:
            dd = est[k] - est[names[0]]
            dsum[k] = dsum.get(k, 0.0) + dd; dsq[k] = dsq.get(k, 0.0) + dd ** 2
    se = {k: np.sqrt(np.maximum(sq[k] / draws - (sums[k] / draws) ** 2, 0)) for k in names}
    sed = {k: np.sqrt(np.maximum(dsq[k] / draws - (dsum[k] / draws) ** 2, 0)) for k in names[1:]}
    return se, sed


def dominance(t, pro, use):
    df = pd.DataFrame({'pdb': t['pdb_id'].to_numpy()[use], 'pro': pro.to_numpy()[use].astype(int),
                       'ci': flat_index(t, use)})
    c = df.groupby(['pro', 'ci', 'pdb']).size().rename('n').reset_index()
    g = c.groupby(['pro', 'ci'])['n'].agg(['sum', 'max', 'size'])
    nstr = np.zeros((2, len(RES20) * B * B)); share = np.zeros_like(nstr)
    nstr[g.index.get_level_values(0), g.index.get_level_values(1)] = g['size']
    share[g.index.get_level_values(0), g.index.get_level_values(1)] = g['max'] / g['sum']
    return nstr.reshape(2, len(RES20), B, B), share.reshape(2, len(RES20), B, B)


def level(n_split, n_unsplit):
    pooled_n = n_unsplit.sum(0)
    lv = np.full(n_split.shape, 'global', dtype=object)
    for p in (0, 1):
        lv[p][np.broadcast_to(pooled_n >= MIN_COUNT, lv[p].shape)] = 'pooled'
        lv[p][n_unsplit >= MIN_COUNT] = 'unsplit'
        lv[p][n_split[p] >= MIN_COUNT] = 'own'
    return lv


def sum_project(dd, qq, uses):
    """Sensitivity: shift the three corrected angles equally so their junction sum equals the deposited sum."""
    both = uses[0] & uses[1] & uses[2]
    delta = (sum(d for d in dd) - sum(q for q in qq)) / 3.0
    return [np.where(both, d - delta, np.nan) for d in dd], both


def compare(a, b, na, nb):
    """mean/p95 |a-b| and slope of centred b on centred a over cells supported (>= MIN_COUNT) in both split grids."""
    sup = (na >= MIN_COUNT) & (nb >= MIN_COUNT)
    d = np.abs(a - b)[sup]
    ac = a - np.nanmean(np.where(sup, a, np.nan), axis=(2, 3), keepdims=True)
    bc = b - np.nanmean(np.where(sup, b, np.nan), axis=(2, 3), keepdims=True)
    x, y = ac[sup], bc[sup]
    return dict(supported_cells=int(sup.sum()), mean_abs=float(d.mean()), p95_abs=float(np.quantile(d, .95)),
                slope_on_v2=float((x * y).sum() / (x * x).sum()), corr=float(np.corrcoef(x, y)[0, 1]))


def sum_sd(arrs, ns):
    sup = np.logical_and.reduce([n >= MIN_COUNT for n in ns])
    s = sum(arrs)
    sc = s - np.nanmean(np.where(sup, s, np.nan), axis=(2, 3), keepdims=True)
    return float(np.sqrt(np.mean(sc[sup] ** 2))), int(sup.sum())


# ------------------------------------------------------------------ main
def main(a):
    root = Path(a.root)
    P = {k: root / v for k, v in PATHS.items()}
    P['training_chains'] = KIT / 'inputs' / 'training_chains.tsv'
    P['headers'] = KIT / 'inputs' / 'headers.tsv'
    P['groups'] = KIT / 'inputs' / 'structure_sequence_groups.tsv'
    P['cdl'] = Path(a.cdl); P['parameters'] = Path(a.deimprint) / 'PARAMETERS.json'; P['bands'] = Path(a.deimprint) / 'BANDS.json'
    hashes = {}
    for k, p in P.items():
        hashes[k] = sha(p)
        if hashes[k] != H[k]:
            sys.exit(f'STOP: {k} hash {hashes[k][:8]} differs from {H[k][:8]}')
    hashes['spec'] = sha(SPEC); hashes['code'] = sha(Path(__file__))
    out = Path(a.out)
    if out.exists():
        sys.exit(f'STOP: {out} exists')
    par = json.loads(P['parameters'].read_text()); bands = json.loads(P['bands'].read_text())
    cols = ['pdb_id', 'chain', 'res_idx', 'res_name', 'phi_deg', 'psi_deg', 'omega_measured_deg', 'is_pro_np1',
            'bond_C_N_next'] + [c for c, _ in TERMS.values()]
    head = list(pd.read_csv(P['train_geometry'], nrows=0).columns)
    miss = [c for c in cols if c not in head]
    if miss:
        sys.exit(f'STOP: training geometry lacks {miss}')
    t = pd.read_csv(P['train_geometry'], usecols=cols, keep_default_na=False, low_memory=False, dtype={'pdb_id': str, 'chain': str})
    t['pdb_id'] = t['pdb_id'].str.strip().str.upper()
    for c in set(cols) - {'pdb_id', 'chain', 'res_name'}:
        t[c] = pd.to_numeric(t[c], errors='coerce')
    keys = pd.read_csv(P['training_chains'], sep='\t', dtype=str, keep_default_na=False)
    keys['pdb_id'] = keys['pdb_id'].str.upper()
    t = t.merge(keys[['pdb_id', 'chain']], on=['pdb_id', 'chain'])
    if len(t) != N_TRAIN:
        sys.exit('STOP: training row count differs from 1,224,328')
    t, pro, elig = eligibility(t)
    with np.load(P['targets'], allow_pickle=False) as z:
        v2 = {k: z[k] for k in z.files}
    # ---- gate 2: identity (T = 0, no guard, all strata)
    gate = {}
    for key, (col, _) in TERMS.items():
        ok = (elig & np.isfinite(t[col]) & t[col].gt(0) & t[col].lt(180)).to_numpy()
        arr = grids(t['res_name'].to_numpy()[ok], t['phi_deg'].to_numpy()[ok], t['psi_deg'].to_numpy()[ok],
                    t[col].to_numpy()[ok], pro.to_numpy()[ok])
        same = [np.array_equal(x, v2[f'{col}__{s}']) if s.startswith('n_') else bool(np.all(np.abs(x - v2[f'{col}__{s}']) <= 1e-12))
                for x, s in zip(arr, ('unsplit', 'v2', 'n_unsplit', 'n_split'))]
        gate[key] = dict(rows=int(ok.sum()), reproduced=all(same))
        if not all(same):
            sys.exit(f'STOP: identity gate failed for {col}: the refit does not reproduce v2')
    print('Gate 2 identity PASS:', gate, flush=True)
    # ---- strata
    hd = pd.read_csv(P['headers'], sep='\t', dtype=str).set_index('pdb_id')
    lab_s, era_s = strata(hd)
    lab = t['pdb_id'].map(lab_s); era = t['pdb_id'].map(era_s)
    resol = pd.to_numeric(t['pdb_id'].map(hd['resolution']), errors='coerce').to_numpy()
    ids = keys['pdb_id'].unique()
    cov = dict(training_structures=len(ids), labelled=float(pd.Series([hd['family'].get(i) for i in ids]).notna().mean()),
               by_stratum=pd.Series([lab_s.get(i) for i in ids]).fillna('other/unlabelled').value_counts().to_dict(),
               refmac_era=pd.Series([era_s.get(i) for i in ids if lab_s.get(i) == 'REFMAC']).fillna('unplaced').value_counts().to_dict())
    if cov['labelled'] < 0.90:
        sys.exit(f'STOP: label coverage {cov["labelled"]:.3f} < 0.90')
    groups, cdl = cdl_arrays(P['cdl'])
    gmap = {g: i for i, g in enumerate(groups)}
    gi = np.array([gmap.get(cdl_group(r, n), -1) for r, n in zip(t['res_name'], t['next_name'])])
    T = {k: {'PHENIX_post': par[k]['T_P'], 'REFMAC': par[k]['T_R']} for k in TERMS}
    Tb = {k: {'PHENIX_post': {'le1.5': bands['bands'][k]['le1.5']['T_P'], 'gt1.5': bands['bands'][k]['gt1.5_le2.0']['T_P']},
              'REFMAC': {'le1.5': bands['bands'][k]['le1.5']['T_R'], 'gt1.5': bands['bands'][k]['gt1.5_le2.0']['T_R']}} for k in TERMS}
    Tpre = {k: 1 - bands['phenix_pre'][k]['alpha_pre_over_alpha_R'] * par[k]['alpha_R'] for k in TERMS}
    amp = {k: {s: 1 / (1 - T[k][s]) for s in USED} | {'PHENIX_pre': 1 / (1 - Tpre[k])} for k in TERMS}
    if max(x for d in amp.values() for x in d.values()) > 5:
        sys.exit('STOP: amplification > 5')
    Tlo = {k: {'PHENIX_post': par[k]['T_P_ci95'][0], 'REFMAC': par[k]['T_R_ci95'][0]} for k in TERMS}
    Thi = {k: {'PHENIX_post': par[k]['T_P_ci95'][1], 'REFMAC': par[k]['T_R_ci95'][1]} for k in TERMS}
    libs = {'V3': {}, 'CONTROL_uncorrected_PR': {}, 'V3_band': {}, 'V3_all': {}, 'V3_Tlow': {}, 'V3_Thigh': {}, 'V3_sumproj': {}}
    diags = {n: {} for n in libs}
    zero = {s: 0.0 for s in USED}
    keep = {}
    for key, (col, ci) in TERMS.items():
        cp = np.full(len(t), np.nan); okg = gi >= 0
        cp[okg] = catmull(cdl[key], gi[okg], t['phi_deg'].to_numpy()[okg], t['psi_deg'].to_numpy()[okg])
        for name, kw in (('V3', dict(Tset=T[key])), ('CONTROL_uncorrected_PR', dict(Tset=zero)),
                         ('V3_band', dict(Tset=Tb[key])), ('V3_all', dict(Tset=T[key], include_pre=Tpre[key], include_others=True)),
                         ('V3_Tlow', dict(Tset=Tlo[key])), ('V3_Thigh', dict(Tset=Thi[key]))):
            d, use, q, dg = corrected_values(t, pro, elig, lab, era, resol, cp, key, **kw)
            libs[name][key] = fit(t, pro, d, use); diags[name][key] = dg
            if name in ('V3', 'CONTROL_uncorrected_PR'):
                keep[(name, key)] = (d, use, q)
            if name == 'V3' and dg['rows'] < 0.5 * gate[key]['rows']:
                sys.exit(f'STOP: v3 keeps fewer than 50% of v2 rows for {key}')
    # sensitivity: angle-sum-preserving correction (equal shift of the three corrected angles)
    dd = [keep[('V3', k)][0] for k in TERMS]; qq = [keep[('V3', k)][2] for k in TERMS]; uu = [keep[('V3', k)][1] for k in TERMS]
    sp, both = sum_project(dd, qq, uu)
    for k, dk in zip(TERMS, sp):
        libs['V3_sumproj'][k] = fit(t, pro, dk, both); diags['V3_sumproj'][k] = dict(rows=int(both.sum()))
        print(f'{key}: v3 rows {diags["V3"][key]["rows"]:,} of v2 {gate[key]["rows"]:,}; guard removed {diags["V3"][key]["guard_removed"]}', flush=True)
    # ---- diagnostics
    rep = {}
    for name in libs:
        rep[name] = {}
        for key, (col, _) in TERMS.items():
            arr = libs[name][key]
            rep[name][key] = compare(v2[f'{col}__v2'], arr[1], v2[f'{col}__n_split'], arr[3])
    sums = {'v2': sum_sd([v2[f'{c}__v2'] for c, _ in TERMS.values()], [v2[f'{c}__n_split'] for c, _ in TERMS.values()])}
    for name in libs:
        sums[name] = sum_sd([libs[name][k][1] for k in TERMS], [libs[name][k][3] for k in TERMS])
    for name in ('V3_Tlow', 'V3_Thigh', 'V3_sumproj', 'V3_band'):
        rep[name + '_vs_V3'] = {k: compare(libs['V3'][k][1], libs[name][k][1], libs['V3'][k][3], libs[name][k][3]) for k in TERMS}
    # ---- stability (sequence-group bootstrap), dominance and support (spec v1.1)
    grp = t['pdb_id'].map(pd.read_csv(P['groups'], sep='\t', dtype=str).set_index('pdb_id')['sequence_group'])
    grp = grp.fillna('S_' + t['pdb_id']).to_numpy()
    rows_out, stab = [], {}
    for key, (col, _) in TERMS.items():
        okv2 = (elig & np.isfinite(t[col]) & t[col].gt(0) & t[col].lt(180)).to_numpy()
        d, use, _ = keep[('V3', key)]
        print(f'bootstrap {key} ({BOOT_DRAWS} sequence-group draws)...', flush=True)
        se, sed = bootstrap(t, pro, grp, {'v2': (t[col].to_numpy(float), okv2), 'V3': (d, use)})
        nstr, share = dominance(t, pro, use)
        v3a = libs['V3'][key]; lv2 = level(v2[f'{col}__n_split'], v2[f'{col}__n_unsplit']); lv3 = level(v3a[3], v3a[2])
        sup = (v2[f'{col}__n_split'] >= MIN_COUNT) & (v3a[3] >= MIN_COUNT)
        diff = np.abs(v3a[1] - v2[f'{col}__v2'])
        stab[key] = dict(cells_supported_both=int(sup.sum()), median_se_v2=float(np.median(se['v2'][sup])),
                         median_se_v3=float(np.median(se['V3'][sup])), se_ratio_median=float(np.median(se['V3'][sup] / np.maximum(se['v2'][sup], 1e-9))),
                         frac_cells_change_gt_2se=float(np.mean(diff[sup] > 2 * sed['V3'][sup])),
                         frac_cells_one_structure_gt_half=float(np.mean(share[sup] > 0.5)), median_structures_per_cell=float(np.median(nstr[sup])),
                         fallback_levels_v2=pd.Series(lv2.ravel()).value_counts().to_dict(), fallback_levels_v3=pd.Series(lv3.ravel()).value_counts().to_dict())
        P_, R_, F_, S_ = np.meshgrid(range(2), range(len(RES20)), range(B), range(B), indexing='ij')
        rows_out.append(pd.DataFrame(dict(term=col, next_pro=P_.ravel(), residue=np.array(RES20)[R_.ravel()], phi=-175 + 10 * F_.ravel(),
                                          psi=-175 + 10 * S_.ravel(), n_v2=v2[f'{col}__n_split'].ravel(), n_v3=v3a[3].ravel(),
                                          structures_v3=nstr.ravel().astype(int), max_structure_share_v3=share.ravel().round(3),
                                          level_v2=lv2.ravel(), level_v3=lv3.ravel(), v2=v2[f'{col}__v2'].ravel().round(4), v3=v3a[1].ravel().round(4),
                                          se_v2=se['v2'].ravel().round(4), se_v3=se['V3'].ravel().round(4), se_v3_minus_v2=sed['V3'].ravel().round(4))))
    # ---- write
    out.mkdir(parents=True)
    pd.concat(rows_out).to_csv(out / 'V3_SUPPORT_MAP.tsv.gz', sep='\t', index=False)

    def save(name, arrs, fname):
        d = dict(v2)
        for key, (col, _) in TERMS.items():
            u, s2, nu, ns = arrs[key]
            d[f'{col}__unsplit'], d[f'{col}__v2'], d[f'{col}__n_unsplit'], d[f'{col}__n_split'] = u, s2, nu, ns
        for k in list(d):
            if k.endswith('__v2'):
                d[k[:-4] + '__v3'] = d[k]
        d['library_version'] = np.array(name)
        d['corrected_terms'] = np.array([c for c, _ in TERMS.values()])
        np.savez(out / fname, **d)
        return sha(out / fname)
    files = {'V3': 'MECHLIB_V3_TARGETS.npz', 'CONTROL_uncorrected_PR': 'CONTROL_uncorrected_PR.npz'}
    files.update({n: f'SENSITIVITY_{n}.npz' for n in libs if n not in files})
    fh = {n: save('MechLib v3' if n == 'V3' else 'control (uncorrected PHENIX_post+REFMAC)' if n.startswith('CONTROL')
                  else f'sensitivity {n} (not a referee library)', libs[n], f) for n, f in files.items()}
    report = dict(spec='MECHLIB_V3_SPEC.md v1.1', hashes=hashes, identity_gate=gate, label_coverage=cov,
                  transmission=T, transmission_band=Tb, T_pre=Tpre, amplification=amp, rows=diags,
                  transmission_ci_bounds=dict(low=Tlo, high=Thi), vs_v2=rep, three_angle_sum_rms_pattern=sums,
                  stability_support=stab, files={files[n]: fh[n] for n in files})
    (out / 'V3_BUILD_REPORT.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    (out / 'V3_FREEZE.json').write_text(json.dumps(dict(library='MechLib v3', file='MECHLIB_V3_TARGETS.npz', sha256=fh['V3'],
                                                        built=time.strftime('%Y-%m-%dT%H:%M:%S'), spec_sha256=hashes['spec'],
                                                        code_sha256=hashes['code']), indent=2) + '\n')
    print('\nv3 vs v2 (split grids, cells supported in both):')
    for name in libs:
        for key in TERMS:
            r = rep[name][key]
            print(f"  {name:24s} {key}: mean|d| {r['mean_abs']:.3f}  p95 {r['p95_abs']:.3f}  slope {r['slope_on_v2']:.3f}  r {r['corr']:.3f}  cells {r['supported_cells']}")
    print('three-angle sum, rms pattern (deg):', {k: round(v[0], 4) for k, v in sums.items()})
    for k in TERMS:
        for name in ('V3_Tlow', 'V3_Thigh', 'V3_sumproj'):
            r = rep[name + '_vs_V3'][k]; print(f"  {name} vs V3 {k}: mean|d| {r['mean_abs']:.3f}  p95 {r['p95_abs']:.3f}")
    print('stability/support:', json.dumps(stab, indent=1))
    print(f"\nMechLib v3 FROZEN: MECHLIB_V3_TARGETS.npz sha256 {fh['V3']}")


# ------------------------------------------------------------------ self-test
def selftest():
    rng = np.random.default_rng(11)
    n = 200000
    phi, psi = rng.uniform(-150, -40, n), rng.uniform(-60, 160, n)
    D = 1.2 * np.sin(np.radians(phi)) + 0.8 * np.cos(np.radians(psi))          # data pattern
    Lc = 1.0 * np.sin(np.radians(phi + 30)) + 0.6 * np.cos(np.radians(psi))     # CDL-like library pattern
    prog = rng.choice(['P', 'R'], n, p=[0.35, 0.65]); era = np.where(prog == 'R', rng.choice(['old', 'new'], n), 'cdl')
    TP, TR = 0.52, 0.39
    lev = 116.5 + np.where(era == 'new', 0.4, 0.0) + np.where(prog == 'P', 0.3, 0.0)  # unit levels differ
    q = np.where(prog == 'P', lev + (1 - TP) * D + TP * Lc, lev + (1 - TR) * D) + rng.normal(0, 0.8, n)
    L = np.where(prog == 'P', Lc, 0.0); T = np.where(prog == 'P', TP, TR)
    unit = np.char.add(prog.astype(str), era.astype(str))
    d = correct(q, L, T, unit)
    # recovered pattern within each unit should equal D (up to unit means)
    for u in np.unique(unit):
        m = unit == u
        x, y = D[m] - D[m].mean(), d[m] - d[m].mean()
        slope = (x * y).sum() / (x * x).sum()
        assert abs(slope - 1) < 0.03, (u, slope)
        assert abs(d[m].mean() - q[m].mean()) < 1e-9          # levels preserved
    # uncorrected REFMAC is flattened to (1 - TR); PHENIX carries library
    m = unit == 'Rold'; x = D[m] - D[m].mean(); y = q[m] - q[m].mean()
    assert abs((x * y).sum() / (x * x).sum() - (1 - TR)) < 0.03
    # T = 0 leaves values unchanged
    assert np.allclose(correct(q, L, np.zeros(n), unit), q)
    # v2-rule grids: fallback order
    res = np.array(['ALA'] * 25 + ['GLY'] * 3); ph = np.full(28, -65.0); ps = np.full(28, -40.0); val = np.r_[np.full(25, 1.0), np.full(3, 4.0)]
    u, s, nu, ns = grids(res, ph, ps, val, np.zeros(28, bool))
    gi = RES20.index('GLY'); ai = RES20.index('ALA'); c = (cell_index([-65.0])[0], cell_index([-40.0])[0])
    assert u[ai][c] == 1.0 and abs(u[gi][c] - (25 + 12) / 28) < 1e-12 and s[1][gi][c] == u[gi][c]
    assert parse_version('5.8.0222') == (5, 8, 222) and parse_version('5.8') is None
    # weighted (bootstrap) grids equal v2 rules at unit weights; sum projection keeps the deposited sum
    r2 = rng.choice(RES20, 5000); p2 = rng.uniform(-180, 180, 5000); s2 = rng.uniform(-180, 180, 5000); v2v = rng.normal(116, 1, 5000)
    pr2 = rng.random(5000) < 0.2
    ref = grids(r2, p2, s2, v2v, pr2)[1]
    ci = np.array([RES20.index(x) for x in r2]) * B * B + cell_index(p2) * B + cell_index(s2)
    assert np.allclose(wgrids(None, ci, pr2, v2v, np.ones(5000)), ref, atol=1e-9)
    dd = [rng.normal(0, 1, 50) for _ in range(3)]; qq = [rng.normal(0, 1, 50) for _ in range(3)]
    sp, both = sum_project(dd, qq, [np.ones(50, bool)] * 3)
    assert np.allclose(sum(sp), sum(qq))
    print('SELFTEST PASS: correction recovers the data pattern (slope 1 +/- 0.03) in every unit, preserves levels, '
          'is the identity at T=0; v2 fallback rules; strict version parsing; bootstrap grids; sum projection')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/mnt/c/Wei/PBM'); ap.add_argument('--cdl'); ap.add_argument('--deimprint'); ap.add_argument('--out')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        if not (a.cdl and a.deimprint and a.out):
            ap.error('--cdl, --deimprint and --out are required')
        main(a)
