#!/usr/bin/env python3
"""Descriptive comparison of MechLib v2 / v3 (and v3 sensitivity libraries) with CDL v1.2 on the three carbonyl angles.
Not a referee: CDL is itself crystallographic. Expectation stated before running (ledger D4): the de-imprinted pattern has
CDL's amplitude (slope 0.93-1.01) and REFMAC deposits carry ~0.6 of it, so v3's slope on CDL should be near 1 and v2's lower.
  python v3_vs_cdl.py --cdl /path/cdl_database.py --v3dir /path/pbm_mechlib_v3.s37m8h --out RESULT.json
"""
import argparse, hashlib, json, runpy, sys
from pathlib import Path
import numpy as np

CDL_SHA = '79c197e5692986cde26db9aa492cecfb5b8b7d5b1f3cba4dc222bb2e04d23c68'
V3_SHA = '8404d010e9117b078524a781c9b73073b7116200b980b237dc7b0570831cd7f3'
NONPGIV = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'LEU', 'LYS', 'MET', 'PHE', 'SER', 'THR', 'TRP', 'TYR'}
RES20 = sorted(NONPGIV | {'ILE', 'VAL', 'GLY', 'PRO'})
TERMS = {'ACN': ('angle_CaCN', 12, (114.0, 119.0)), 'OCN': ('angle_O_C_N_next', 14, (121.0, 125.0)), 'ACO': ('angle_CA_C_O', 10, (119.0, 122.5))}
B, MIN = 36, 10
LIBS = {'v3': 'MECHLIB_V3_TARGETS.npz', 'control: uncorrected PHENIX+REFMAC': 'CONTROL_uncorrected_PR.npz',
        'v3 T low': 'SENSITIVITY_V3_Tlow.npz', 'v3 T high': 'SENSITIVITY_V3_Thigh.npz', 'v3 sum-preserving': 'SENSITIVITY_V3_sumproj.npz',
        'v3-all': 'SENSITIVITY_V3_all.npz'}


def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


def catmull(G, x, y, x0=-180.0):
    u, v = (np.asarray(x, float) - x0) / 10.0, (np.asarray(y, float) - x0) / 10.0
    i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int); tx, ty = u - i0, v - j0
    def w(t):
        t2, t3 = t * t, t * t * t
        return [0.5 * (-t3 + 2 * t2 - t), 0.5 * (3 * t3 - 5 * t2 + 2), 0.5 * (-3 * t3 + 4 * t2 + t), 0.5 * (t3 - t2)]
    wx, wy = w(tx), w(ty); out = np.zeros(np.shape(u))
    for a in range(4):
        for b in range(4):
            out = out + wx[a] * wy[b] * G[(i0 + a - 1) % B, (j0 + b - 1) % B]
    return out


def cdl_grids(path):
    db = runpy.run_path(str(path))['cdl_database']
    if getattr(db, 'version', None) != 'CDL v1.2':
        sys.exit('STOP: unexpected CDL version')
    out = {}
    for term, (_, col, (lo, hi)) in TERMS.items():
        for g in db:
            G = np.full((B, B), np.nan)
            for (phi, psi), row in db[g].items():
                if row[col] > 0:
                    G[(phi + 180) // 10, (psi + 180) // 10] = row[col]
            out[(term, g)] = G
        med = np.nanmedian([out[(term, g)] for g in db])
        if not lo <= med <= hi:
            sys.exit(f'STOP: CDL column {col} median {med:.2f} outside {lo}-{hi}')
    return out


def group(res, nextpro):
    g = 'NonPGIV' if res in NONPGIV else 'IleVal' if res in ('ILE', 'VAL') else 'Gly' if res == 'GLY' else 'Pro'
    return g + ('_xpro' if nextpro else '_nonxpro')


def cdl_on_library_grid(cdl, term):
    """CDL evaluated at the library's bin centres (-175..175), shape (2, 20, 36, 36)."""
    c = np.arange(-175, 180, 10.0); X, Y = np.meshgrid(c, c, indexing='ij')
    out = np.full((2, len(RES20), B, B), np.nan)
    for p in (0, 1):
        for i, r in enumerate(RES20):
            out[p, i] = catmull(cdl[(term, group(r, p))], X, Y)
    return out


def compare(lib, n_lib, n_ref, C, weight=None):
    sup = (n_lib >= MIN) & (n_ref >= MIN) & np.isfinite(C)
    rows = []
    xs, ys, ws, off = [], [], [], []
    for p in (0, 1):
        for i in range(len(RES20)):
            m = sup[p, i]
            if m.sum() < 5:
                continue
            x, y = C[p, i][m], lib[p, i][m]
            w = np.ones(m.sum()) if weight is None else weight[p, i][m].astype(float)
            xs.append(x - np.average(x, weights=w)); ys.append(y - np.average(y, weights=w)); ws.append(w)
            off.append((np.average(y - x, weights=w), w.sum()))
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    slope = float((w * x * y).sum() / (w * x * x).sum())
    r = float(np.sum(w * x * y) / np.sqrt(np.sum(w * x * x) * np.sum(w * y * y)))
    resid = float(np.sqrt(np.average((y - slope * x) ** 2, weights=w)))
    level = float(sum(o * s for o, s in off) / sum(s for _, s in off))
    return dict(cells=int(len(x)), slope_on_cdl=slope, r=r, rms_pattern_difference=float(np.sqrt(np.average((y - x) ** 2, weights=w))),
                residual_after_scaling=resid, mean_level_minus_cdl=level)


def main(a):
    if sha(a.cdl) != CDL_SHA:
        sys.exit('STOP: cdl_database.py hash differs')
    v3dir = Path(a.v3dir)
    if sha(v3dir / 'MECHLIB_V3_TARGETS.npz') != V3_SHA:
        sys.exit('STOP: v3 file hash differs from the frozen hash')
    cdl = cdl_grids(a.cdl)
    v3 = np.load(v3dir / 'MECHLIB_V3_TARGETS.npz')
    res = {'note': 'descriptive; not a referee. Cells: own-supported (>=10) in both the library and v3, patterns centred within residue x next-Pro.'}
    for key, (col, _, _) in TERMS.items():
        C = cdl_on_library_grid(cdl, key)
        nref = v3[f'{col}__n_split']
        res[key] = {}
        for name, f in LIBS.items():
            p = v3dir / f
            if not p.exists():
                continue
            z = np.load(p)
            res[key][name] = {'unweighted': compare(z[f'{col}__v2'], z[f'{col}__n_split'], nref, C),
                              'count_weighted': compare(z[f'{col}__v2'], z[f'{col}__n_split'], nref, C, weight=nref)}
        if a.v2:
            if sha(a.v2) != 'ef7b86b359c84492a0be791de52a1e64112a0153142b9f9f5a0417448d2731b1':
                sys.exit('STOP: v2 file hash differs')
            z = np.load(a.v2)
            res[key]['v2 (frozen)'] = {'unweighted': compare(z[f'{col}__v2'], z[f'{col}__n_split'], nref, C),
                                       'count_weighted': compare(z[f'{col}__v2'], z[f'{col}__n_split'], nref, C, weight=nref)}
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(res['note'])
    print(f"{'library':38s} {'term':4s} {'slope':>6s} {'r':>6s} {'rms diff':>8s} {'level-CDL':>9s}")
    for key in TERMS:
        for name, d in res[key].items():
            u = d['unweighted']
            print(f"{name:38s} {key:4s} {u['slope_on_cdl']:6.3f} {u['r']:6.3f} {u['rms_pattern_difference']:8.3f} {u['mean_level_minus_cdl']:+9.3f}")
    print('wrote', a.out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--cdl', required=True); ap.add_argument('--v3dir', required=True)
    ap.add_argument('--v2', help='frozen v2 training_targets.npz (ef7b86b3...)'); ap.add_argument('--out', default='V3_VS_CDL.json')
    main(ap.parse_args())
