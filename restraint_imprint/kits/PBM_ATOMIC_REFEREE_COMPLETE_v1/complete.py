#!/usr/bin/env python3
"""Completeness outputs for the atomic-resolution referee (spec 82f68380), from the frozen kit's own code, unchanged.
Nothing here can change a registered verdict. Outputs, as listed in the spec but not emitted by the kit:
  1. bootstrap-paired shape comparison r(deposit, v2) - r(deposit, v3), per angle (primary flat stratum);
  2. the 'other programs' stratum (secondary, as PHENIX_post);
  3. the transmission-corrected amplitude A_flat / (1 - T), T = <= 1.5 A de-imprint band (0.31 / 0.30 / 0.34), model-dependent.
Gate: the primary A_flat slopes are recomputed and must equal the frozen result (|diff| < 1e-9), else STOP.
Usage: python complete.py --kit PBM_ATOMIC_REFEREE_v1 --result pbm_atomic_referee.20261117_1008/ATOMIC_REFEREE_RESULT.json
                          --root /mnt/c/Wei/PBM --cdl CDL.py --v3 MECHLIB_V3_TARGETS.npz --out DIR
       python complete.py --selftest
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd

T_BAND = {'ACN': 0.31, 'OCN': 0.30, 'ACO': 0.34}   # spec: <= 1.5 A band T from the de-imprint run (D3)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def paired_r(d, grp, a, b, y, draws, seed):
    """r(y, a) - r(y, b) with sequence-group bootstrap (same multinomial draws as the referee's amplitude())."""
    g = pd.factorize(d[grp])[0]; G = g.max() + 1
    Y, A, Bv = d[y].to_numpy(), d[a].to_numpy(), d[b].to_numpy()
    S = {k: np.bincount(g, weights=v, minlength=G) for k, v in
         dict(ya=Y * A, yb=Y * Bv, yy=Y * Y, aa=A * A, bb=Bv * Bv).items()}
    def r_of(w):
        sy = w @ S['yy']
        return (w @ S['ya']) / np.sqrt(sy * (w @ S['aa'])), (w @ S['yb']) / np.sqrt(sy * (w @ S['bb']))
    ra, rb = r_of(np.ones(G))
    rng = np.random.default_rng(seed); W = rng.multinomial(G, np.full(G, 1 / G), size=draws).astype(float)
    bra, brb = r_of(W); diff = bra - brb
    lo, hi = np.quantile(diff, [.025, .975])
    return dict(r_a=float(ra), r_b=float(rb), diff=float(ra - rb), ci95=[float(lo), float(hi)],
                worse_shape_b=bool((ra - rb) > 0.05 and lo > 0), worse_shape_a=bool((rb - ra) > 0.05 and hi < 0))


def main(a):
    kit = Path(a.kit).resolve()
    if not sha(kit / 'ATOMIC_RES_REFEREE_SPEC_20261116.md').startswith('82f68380'): sys.exit('STOP: not the frozen referee kit')
    import subprocess
    if subprocess.run(['sha256sum', '-c', '--quiet', 'SHA256SUMS.txt'], cwd=kit).returncode: sys.exit('STOP: referee kit files changed')
    sys.path.insert(0, str(kit)); import atomic_referee as AR
    frozen = json.loads(Path(a.result).read_text())
    root = Path(a.root)
    P = {'features': root / 'pbm_altloc_full_rdd_audit_v1/blank_only_geometry.csv.gz', 'headers': kit / 'inputs/headers.tsv',
         'groups': kit / 'inputs/structure_sequence_groups.tsv', 'chains': kit / 'inputs/training_chains.tsv', 'cdl': Path(a.cdl),
         'v2': root / 'PBM_CLEAN_REBUILD_v1/results/training_targets.npz', 'v3': Path(a.v3)}
    hashes = {k: sha(p) for k, p in P.items()}
    for k, v in AR.H.items():
        if not hashes[k].startswith(v): sys.exit(f'STOP: {k} hash {hashes[k][:8]} != {v}')
    # ---- data preparation: the referee's main(), verbatim in substance
    libs = {'v2': np.load(P['v2']), 'v3': np.load(P['v3'])}
    hd = pd.read_csv(P['headers'], sep='\t', dtype=str).set_index('pdb_id')
    resol = pd.to_numeric(hd['resolution'], errors='coerce'); lab = AR.stratum(hd)
    ids = resol.index[resol <= 1.0]
    df = AR.load_features(P['features']); df = df[df['pdb_id'].isin(ids)].copy()
    df['stratum'] = df['pdb_id'].map(lab)
    grp = pd.read_csv(P['groups'], sep='\t', dtype=str).set_index('pdb_id')['sequence_group']
    df['grp'] = df['pdb_id'].map(grp).fillna('S_' + df['pdb_id'])
    groups, cdl = AR.cdl_arrays(P['cdl']); gmap = {g: i for i, g in enumerate(groups)}
    gi = np.array([gmap.get(AR.cdl_group(r, n), -1) for r, n in zip(df['res_name'], df['next_name'])]); df = df[gi >= 0].copy(); gi = gi[gi >= 0]
    pro = df['is_pro_np1'].to_numpy().astype(bool)
    for t, (col, _) in AR.TERMS.items():
        df[f'y_{t}'] = df[col]
        df[f'pred_CDL_{t}'] = AR.catmull(cdl[t], gi, df['phi_deg'], df['psi_deg'], -180.0)
        for L, z in libs.items():
            df[f'pred_{L}_{t}'] = AR.lib_pred(z[f'{col}__v2'], df['res_name'], pro, df['phi_deg'].to_numpy(), df['psi_deg'].to_numpy())
    df['unit'] = df['stratum'] + '|' + df['res_name'] + '|' + df['is_pro_np1'].astype(str)
    preds = ['CDL', 'v2', 'v3']
    flat = df[df['stratum'].isin(AR.FLAT)]
    # ---- gate: reproduce the registered primary
    prim = AR.analyse(flat, preds)
    gate = {t: abs(prim[t]['A_flat_on_CDL']['slope'] - frozen['primary_flat'][t]['A_flat_on_CDL']['slope']) for t in AR.TERMS}
    if max(gate.values()) > 1e-9: sys.exit(f'STOP: primary not reproduced {gate}')
    print('gate PASS: primary A_flat reproduced', {t: round(prim[t]['A_flat_on_CDL']['slope'], 4) for t in AR.TERMS})
    res = dict(referee_result_sha=sha(a.result), hashes=hashes, gate_max_abs_diff=max(gate.values()), shape={}, corrected={}, other=None)
    for t in AR.TERMS:
        d = flat[np.isfinite(flat[f'y_{t}'])].copy()
        cols = [f'y_{t}'] + [f'pred_{L}_{t}' for L in preds]
        d = d[np.all(np.isfinite(d[cols]), axis=1)]
        for c in cols: d[c + '_c'] = d[c] - d.groupby('unit')[c].transform('mean')
        res['shape'][t] = paired_r(d, 'grp', f'pred_v2_{t}_c', f'pred_v3_{t}_c', f'y_{t}_c', AR.DRAWS, AR.SEED)
        A, bs = AR.amplitude(d, f'y_{t}_c', f'pred_CDL_{t}_c', 'grp')
        c = bs / (1 - T_BAND[t])
        res['corrected'][t] = dict(T=T_BAND[t], point=A['slope'] / (1 - T_BAND[t]), ci95=np.quantile(c, [.025, .975]).tolist(),
                                   note='model-dependent; the <= 1.5 A T likely overstates transmission at <= 1.0 A; never decisive')
    oth = df[df['stratum'] == 'other']
    res['other'] = dict(structures=int(oth['pdb_id'].nunique()), groups=int(oth['grp'].nunique()), junctions=int(len(oth)),
                        result=(AR.analyse(oth, preds) if len(oth) > 500 else 'fewer than 500 junctions: not analysed (kit rule)'))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    (out / 'ATOMIC_REFEREE_COMPLETENESS.json').write_text(json.dumps(res, indent=2, default=float))
    for t in AR.TERMS:
        s, c = res['shape'][t], res['corrected'][t]
        print(f"{t}: r v2 {s['r_a']:.3f} vs v3 {s['r_b']:.3f}, diff {s['diff']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}] "
              f"worse(v3) {s['worse_shape_b']} worse(v2) {s['worse_shape_a']} | corrected A {c['point']:.2f} [{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]")
    o = res['other']
    print(f"other stratum: {o['structures']} structures, {o['groups']} groups, {o['junctions']} junctions")
    if isinstance(o['result'], dict):
        for t in AR.TERMS:
            r = o['result'][t]; A = r['A_flat_on_CDL']
            print(f"  {t}: A {A['slope']:.3f} [{A['ci95'][0]:.3f}, {A['ci95'][1]:.3f}]; v2 {r['library_amplitude_on_CDL']['v2']:.2f}, v3 {r['library_amplitude_on_CDL']['v3']:.2f}")
    print('wrote', out)


def selftest():
    rng = np.random.default_rng(0); n = 4000
    d = pd.DataFrame(dict(grp=rng.integers(0, 200, n), a=rng.normal(size=n)))
    d['b'] = d['a'] + rng.normal(0, 1.0, n); d['y'] = d['a'] + rng.normal(0, 0.5, n)
    r = paired_r(d, 'grp', 'a', 'b', 'y', 2000, 1)
    assert r['diff'] > 0.05 and r['ci95'][0] > 0 and r['worse_shape_b'] and not r['worse_shape_a'], r
    d['b'] = d['a'] * 2.0; r = paired_r(d, 'grp', 'a', 'b', 'y', 500, 1)
    assert abs(r['diff']) < 1e-12 and not r['worse_shape_b'], r
    print('SELFTEST PASS: paired shape test')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for k in ('kit', 'result', 'root', 'cdl', 'v3', 'out'): ap.add_argument('--' + k)
    ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
    selftest() if a.selftest else main(a)
