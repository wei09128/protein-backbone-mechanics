#!/usr/bin/env python3
"""Atomic-resolution referee (spec ATOMIC_RES_REFEREE_SPEC_20261116.md v1.0).
  python atomic_referee.py --root /mnt/c/Wei/PBM --cdl CDL.py --v3 MECHLIB_V3_TARGETS.npz [--v4 V4.npz --v4-sha HASH] --out DIR
  python atomic_referee.py --selftest
"""
import argparse, hashlib, json, runpy, sys, time
from pathlib import Path
import numpy as np, pandas as pd

KIT = Path(__file__).resolve().parent
H = {'features': 'e905f8b8', 'headers': '25f466e2', 'groups': '33299405', 'chains': '35f69bd2', 'cdl': '79c197e5',
     'v2': 'ef7b86b3', 'v3': '8404d010'}
NONPGIV = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'LEU', 'LYS', 'MET', 'PHE', 'SER', 'THR', 'TRP', 'TYR'}
RES20 = sorted(NONPGIV | {'ILE', 'VAL', 'GLY', 'PRO'})
B, MAD_CUT, SEED, DRAWS = 36, 8.0, 20261118, 10000
TERMS = {'ACN': ('angle_CaCN', 12), 'OCN': ('angle_O_C_N_next', 14), 'ACO': ('angle_CA_C_O', 10)}
FLAT = ('REFMAC', 'SHELX', 'PHENIX_pre')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def catmull(G, g, x, y, x0):
    u, v = (np.asarray(x, float) - x0) / 10.0, (np.asarray(y, float) - x0) / 10.0
    i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int); tx, ty = u - i0, v - j0
    def w(t):
        t2, t3 = t * t, t * t * t
        return [0.5 * (-t3 + 2 * t2 - t), 0.5 * (3 * t3 - 5 * t2 + 2), 0.5 * (-3 * t3 + 4 * t2 + t), 0.5 * (t3 - t2)]
    wx, wy = w(tx), w(ty); out = np.zeros(len(u))
    for a in range(4):
        for b in range(4):
            out += wx[a] * wy[b] * G[g, (i0 + a - 1) % B, (j0 + b - 1) % B]
    return out


def cdl_group(res, nxt):
    g = 'NonPGIV' if res in NONPGIV else 'IleVal' if res in ('ILE', 'VAL') else 'Gly' if res == 'GLY' else 'Pro' if res == 'PRO' else None
    return None if g is None else g + ('_xpro' if nxt == 'PRO' else '_nonxpro')


def cdl_arrays(path):
    db = runpy.run_path(str(path))['cdl_database']
    if getattr(db, 'version', None) != 'CDL v1.2': sys.exit('STOP: unexpected CDL version')
    groups = sorted(db); arrays = {}
    for term, (_, col) in TERMS.items():
        grid = np.full((len(groups), B, B), np.nan)
        for i, gname in enumerate(groups):
            for (phi, psi), row in db[gname].items():
                if row[col] > 0: grid[i, (phi + 180) // 10, (psi + 180) // 10] = row[col]
        arrays[term] = grid
    return groups, arrays


def load_features(path):
    cols = ['pdb_id', 'chain', 'res_idx', 'res_name', 'phi_deg', 'psi_deg', 'omega_measured_deg', 'is_pro_np1', 'bond_C_N_next'] + [t[0] for t in TERMS.values()]
    df = pd.read_csv(path, usecols=cols, low_memory=False, dtype={'pdb_id': str, 'chain': str, 'res_name': str})
    df['pdb_id'] = df['pdb_id'].str.strip().str.upper(); df['chain'] = df['chain'].fillna(''); df['res_name'] = df['res_name'].str.strip().str.upper()
    for c in cols[2:]:
        if c not in ('res_name', 'chain'): df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df[~df.duplicated(['pdb_id', 'chain', 'res_idx'], keep=False)]
    s = df[['pdb_id', 'chain', 'res_idx', 'res_name']].sort_values(['pdb_id', 'chain', 'res_idx'])
    nxt = s.groupby(['pdb_id', 'chain'])[['res_idx', 'res_name']].shift(-1)
    s['next_name'] = nxt['res_name'].where(nxt['res_idx'].eq(s['res_idx'] + 1)); df = df.join(s['next_name'])
    keep = (df['omega_measured_deg'].abs() > 150) & df['is_pro_np1'].isin([0, 1]) & np.isfinite(df['phi_deg']) & np.isfinite(df['psi_deg'])
    keep &= df['res_name'].isin(RES20) & df['next_name'].isin(RES20) & df['bond_C_N_next'].between(0, 1.8, inclusive='right')
    keep &= df['is_pro_np1'].eq(df['next_name'].eq('PRO').astype(int))
    df = df[keep].copy()
    for term, (col, _) in TERMS.items():
        for flag in (0, 1):
            m = df['is_pro_np1'] == flag
            med = df.loc[m, col].median(); mad = (df.loc[m, col] - med).abs().median() * 1.4826
            df.loc[m & ((df[col] - med).abs() > MAD_CUT * mad), col] = np.nan
    return df


def stratum(H):
    b = pd.to_numeric(H['phenix_build'], errors='coerce'); lab = pd.Series('other', index=H.index)
    lab[H['family'].eq('REFMAC')] = 'REFMAC'; lab[H['family'].eq('SHELX')] = 'SHELX'
    lab[H['family'].eq('PHENIX') & b.lt(2155)] = 'PHENIX_pre'; lab[H['family'].eq('PHENIX') & b.ge(2155)] = 'PHENIX_post'
    return lab


def lib_pred(arr, res, pro, phi, psi):
    ri = np.array([RES20.index(r) for r in res]); return catmull(arr.reshape(40, B, B), ri + 20 * pro.astype(int), phi, psi, -175.0)


def amplitude(d, ycol, xcol, grp, draws=DRAWS, seed=SEED):
    """slope through origin of centred y on centred x (centring within unit fixed); group bootstrap via sufficient statistics."""
    x, y = d[xcol].to_numpy(), d[ycol].to_numpy()
    g = pd.factorize(d[grp])[0]; G = g.max() + 1
    Sxy, Sxx, Syy = (np.bincount(g, weights=v, minlength=G) for v in (x * y, x * x, y * y))
    point = Sxy.sum() / Sxx.sum(); r = Sxy.sum() / np.sqrt(Sxx.sum() * Syy.sum())
    rng = np.random.default_rng(seed); W = rng.multinomial(G, np.full(G, 1 / G), size=draws).astype(float)
    bs = (W @ Sxy) / (W @ Sxx)
    return dict(slope=float(point), ci95=np.quantile(bs, [.025, .975]).tolist(), r=float(r), groups=int(G), junctions=int(len(x))), bs


def analyse(df, preds, grp_col='grp'):
    """df has y_<term>, pred_<lib>_<term>, unit, grp. Returns per term: A_flat on CDL and library amplitudes (library on CDL)."""
    out = {}
    for t in TERMS:
        d = df[np.isfinite(df[f'y_{t}'])].copy()
        cols = [f'y_{t}'] + [f'pred_{L}_{t}' for L in preds]
        d = d[np.all(np.isfinite(d[cols]), axis=1)]
        for c in cols: d[c + '_c'] = d[c] - d.groupby('unit')[c].transform('mean')
        A, bsA = amplitude(d, f'y_{t}_c', f'pred_CDL_{t}_c', grp_col)
        res = {'A_flat_on_CDL': A, 'library_amplitude_on_CDL': {}, 'r_deposit_vs_library': {}}
        for L in preds:
            if L == 'CDL': continue
            la, _ = amplitude(d, f'pred_{L}_{t}_c', f'pred_CDL_{t}_c', grp_col, draws=1)
            res['library_amplitude_on_CDL'][L] = la['slope']
            x = d[f'pred_{L}_{t}_c'].to_numpy(); y = d[f'y_{t}_c'].to_numpy()
            res['r_deposit_vs_library'][L] = float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))
        res['_bs'] = bsA
        out[t] = res
    # decisions
    p_too_weak = {t: float((out[t]['_bs'] <= out[t]['library_amplitude_on_CDL']['v2']).mean()) for t in TERMS}
    order = sorted(TERMS, key=lambda t: p_too_weak[t]); adj, run = {}, 0.0
    for k, t in enumerate(order):
        run = max(run, min(1.0, (len(order) - k) * p_too_weak[t])); adj[t] = run
    for t in TERMS:
        r = out[t]; lo, hi = r['A_flat_on_CDL']['ci95']
        r['v2_too_weak'] = dict(p_one_sided=p_too_weak[t], p_holm=adj[t], verdict=bool(adj[t] < 0.05 and lo > r['library_amplitude_on_CDL']['v2']))
        for L in [k for k in r['library_amplitude_on_CDL'] if k != 'v2']:
            r[f'consistent_with_{L}'] = bool(hi >= 0.9 * r['library_amplitude_on_CDL'][L])
        del r['_bs']
    return out


def main(a):
    if sha(KIT / 'ATOMIC_RES_REFEREE_SPEC_20261116.md')[:8] != a.spec_sha:
        sys.exit('STOP: spec hash differs from the frozen one given with --spec-sha')
    root = Path(a.root)
    P = {'features': root / 'pbm_altloc_full_rdd_audit_v1/blank_only_geometry.csv.gz', 'headers': KIT / 'inputs/headers.tsv',
         'groups': KIT / 'inputs/structure_sequence_groups.tsv', 'chains': KIT / 'inputs/training_chains.tsv', 'cdl': Path(a.cdl),
         'v2': root / 'PBM_CLEAN_REBUILD_v1/results/training_targets.npz', 'v3': Path(a.v3)}
    hashes = {k: sha(p) for k, p in P.items()}
    for k, v in H.items():
        if not hashes[k].startswith(v): sys.exit(f'STOP: {k} hash {hashes[k][:8]} != {v}')
    libs = {'v2': np.load(P['v2']), 'v3': np.load(P['v3'])}
    if a.v4:
        if sha(a.v4) != a.v4_sha: sys.exit('STOP: v4 hash differs from the frozen one')
        libs['v4'] = np.load(a.v4); hashes['v4'] = a.v4_sha
    out = Path(a.out)
    if out.exists(): sys.exit(f'STOP: {out} exists')
    hd = pd.read_csv(P['headers'], sep='\t', dtype=str).set_index('pdb_id')
    resol = pd.to_numeric(hd['resolution'], errors='coerce'); lab = stratum(hd)
    ids = resol.index[resol <= 1.0]
    df = load_features(P['features']); df = df[df['pdb_id'].isin(ids)].copy()
    df['stratum'] = df['pdb_id'].map(lab)
    grp = pd.read_csv(P['groups'], sep='\t', dtype=str).set_index('pdb_id')['sequence_group']
    df['grp'] = df['pdb_id'].map(grp).fillna('S_' + df['pdb_id'])
    tr = set(pd.read_csv(P['chains'], sep='\t', dtype=str)['pdb_id'].str.upper()); trg = set(grp.reindex(list(tr)).dropna())
    df['held_out'] = ~df['pdb_id'].isin(tr) & ~df['grp'].isin(trg)
    groups, cdl = cdl_arrays(P['cdl']); gmap = {g: i for i, g in enumerate(groups)}
    gi = np.array([gmap.get(cdl_group(r, n), -1) for r, n in zip(df['res_name'], df['next_name'])]); df = df[gi >= 0].copy(); gi = gi[gi >= 0]
    pro = df['is_pro_np1'].to_numpy().astype(bool)
    for t, (col, _) in TERMS.items():
        df[f'y_{t}'] = df[col]
        df[f'pred_CDL_{t}'] = catmull(cdl[t], gi, df['phi_deg'], df['psi_deg'], -180.0)
        for L, z in libs.items():
            df[f'pred_{L}_{t}'] = lib_pred(z[f'{col}__v2'], df['res_name'], pro, df['phi_deg'].to_numpy(), df['psi_deg'].to_numpy())
    df['unit'] = df['stratum'] + '|' + df['res_name'] + '|' + df['is_pro_np1'].astype(str)
    preds = ['CDL'] + list(libs)
    flat = df[df['stratum'].isin(FLAT)]
    n_str, n_grp = flat['pdb_id'].nunique(), flat['grp'].nunique()
    res = dict(spec_sha=a.spec_sha, hashes=hashes, primary_structures=int(n_str), primary_groups=int(n_grp),
               training_overlap_structures=int(flat.loc[~flat['held_out'], 'pdb_id'].nunique()),
               by_stratum={s: int(df.loc[df.stratum == s, 'pdb_id'].nunique()) for s in df['stratum'].unique()})
    if n_str < 100 or n_grp < 50:
        res['verdict'] = 'UNDERPOWERED (primary stratum below 100 structures / 50 groups): decides nothing'
    res['primary_flat'] = analyse(flat, preds)
    res['secondary'] = {s: analyse(df[df['stratum'] == s], preds) for s in ('PHENIX_post',) if df['stratum'].eq(s).sum() > 500}
    ho = flat[flat['held_out']]
    if ho['pdb_id'].nunique() >= 5: res['sensitivity_held_out_only'] = analyse(ho, preds)
    out.mkdir(parents=True)
    (out / 'ATOMIC_REFEREE_RESULT.json').write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ('primary_structures', 'primary_groups', 'training_overlap_structures', 'by_stratum')}, indent=1))
    for t in TERMS:
        r = res['primary_flat'][t]; A = r['A_flat_on_CDL']
        print(f"{t}: A_flat {A['slope']:.3f} [{A['ci95'][0]:.3f}, {A['ci95'][1]:.3f}] r {A['r']:.2f} | libraries " +
              ', '.join(f"{L} {v:.2f}" for L, v in r['library_amplitude_on_CDL'].items()) +
              f" | v2 too weak: {r['v2_too_weak']['verdict']} (Holm p {r['v2_too_weak']['p_holm']:.3g}) | " +
              ', '.join(f"{k}: {v}" for k, v in r.items() if k.startswith('consistent_with')))
    print('wrote', out)


def selftest():
    rng = np.random.default_rng(1); n = 40000
    groups = rng.integers(0, 300, n); x = rng.normal(0, 1, n)
    y = 0.8 * x + rng.normal(0, 1.0, n)
    d = pd.DataFrame({'x': x, 'y': y, 'g': groups})
    A, bs = amplitude(d, 'y', 'x', 'g', draws=2000)
    assert abs(A['slope'] - 0.8) < 0.03 and A['ci95'][0] < 0.8 < A['ci95'][1], A
    # analyse(): deposit = 0.8 x CDL pattern; v2 = 0.57 x CDL; v3 = 0.95 x CDL
    rows = pd.DataFrame({'grp': groups, 'unit': rng.choice(['REFMAC|ALA|0', 'SHELX|LEU|0'], n)})
    for t in TERMS:
        c = rng.normal(0, 1, n)
        rows[f'pred_CDL_{t}'] = c; rows[f'pred_v2_{t}'] = 0.57 * c; rows[f'pred_v3_{t}'] = 0.95 * c
        rows[f'y_{t}'] = 0.9 * c + rng.normal(0, 1, n)
    r = analyse(rows.copy(), ['CDL', 'v2', 'v3'])
    for t in TERMS:
        assert abs(r[t]['library_amplitude_on_CDL']['v2'] - 0.57) < 1e-9 and r[t]['v2_too_weak']['verdict']
        assert r[t]['consistent_with_v3'] is True
    for t in TERMS:   # deposit at v2 strength: v2 not too weak, v3 not consistent
        rows[f'y_{t}'] = 0.57 * rows[f'pred_CDL_{t}'] + rng.normal(0, 1, n)
    r = analyse(rows.copy(), ['CDL', 'v2', 'v3'])
    for t in TERMS:
        assert not r[t]['v2_too_weak']['verdict'] and r[t]['consistent_with_v3'] is False
    print('SELFTEST PASS: amplitude estimate and group bootstrap; library amplitudes; v2-too-weak and consistency rules both ways')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for k in ('root', 'cdl', 'v3', 'v4', 'v4_sha', 'out', 'spec_sha'): ap.add_argument('--' + k.replace('_', '-'))
    ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
    if a.selftest: selftest()
    else:
        if a.root is None: a.root = '/mnt/c/Wei/PBM'
        main(a)
