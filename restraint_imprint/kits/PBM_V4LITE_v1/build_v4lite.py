#!/usr/bin/env python3
"""Build v4-lite, the intrinsic-calibrated library (spec MECHLIB_V4LITE_SPEC.md):
v4_c = v2_c + lambda_c (v3_c - v2_c) for the three carbonyl angles, lambda_c fitted on the pilot water QM (S1).
  python build_v4lite.py --pilot ~/pbm_qm_pilot --v2 V2.npz --v3dir pbm_mechlib_v3.s37m8h --out DIR
  python build_v4lite.py --selftest
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

KIT = Path(__file__).resolve().parent
V2_SHA, V3_SHA, CASES_SHA = 'ef7b86b3', '8404d010', '160e3672'
RES20 = sorted(['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'LEU', 'LYS', 'MET', 'PHE', 'SER', 'THR', 'TRP', 'TYR', 'ILE', 'VAL', 'GLY', 'PRO'])
B, R_GATE = 36, 0.5
TERMS = {'ACN': ('angle_CaCN', 'angle_CaCN'), 'OCN': ('angle_O_C_N_next', 'angle_O_C_N'), 'ACO': ('angle_CA_C_O', 'angle_CA_C_O')}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def cm(G, x, y, x0=-175.0):
    u, v = (x - x0) / 10, (y - x0) / 10; i0, j0 = int(np.floor(u)), int(np.floor(v)); tx, ty = u - i0, v - j0
    w = lambda t: [0.5 * (-t ** 3 + 2 * t * t - t), 0.5 * (3 * t ** 3 - 5 * t * t + 2), 0.5 * (-3 * t ** 3 + 4 * t * t + t), 0.5 * (t ** 3 - t * t)]
    wx, wy = w(tx), w(ty)
    return sum(wx[a] * wy[b] * G[(i0 + a - 1) % B, (j0 + b - 1) % B] for a in range(4) for b in range(4))


def fit_lambda(q, a2, a3, res):
    """centre within residue; least squares of q on a2 + lam (a3 - a2); clip [0, 1]; shape gate r(q, a3) < 0.5 -> 0."""
    d = pd.DataFrame({'q': q, 'a2': a2, 'a3': a3, 'res': res})
    for c in ('q', 'a2', 'a3'): d[c] = d[c] - d.groupby('res')[c].transform('mean')
    D = d.a3 - d.a2; e = d.q - d.a2
    lam_raw = float((D * e).sum() / (D * D).sum()) if (D * D).sum() > 0 else 0.0
    r = float(np.corrcoef(d.q, d.a3)[0, 1]) if d.q.std() > 0 else 0.0
    lam = 0.0 if r < R_GATE else float(np.clip(lam_raw, 0, 1))
    return dict(lam=lam, lam_unclipped=lam_raw, r_qm_v3=r, gate='v2 kept (r < 0.5)' if r < R_GATE else 'passed', n=int(len(d)))


def jackknife(q, a2, a3, res):
    out = {}
    for r_ in sorted(set(res)):
        m = np.array([x != r_ for x in res])
        if len(set(np.array(res)[m])) >= 1 and m.sum() >= 4:
            out[r_] = fit_lambda(q[m], a2[m], a3[m], list(np.array(res)[m]))['lam']
    return out


def main(a):
    pilot = Path(a.pilot).expanduser(); v3dir = Path(a.v3dir)
    if not sha(KIT / 'inputs/cases_confirmatory.tsv').startswith(CASES_SHA): sys.exit('STOP: cases hash')
    if not sha(a.v2).startswith(V2_SHA): sys.exit('STOP: v2 hash')
    if not sha(v3dir / 'MECHLIB_V3_TARGETS.npz').startswith(V3_SHA): sys.exit('STOP: v3 hash')
    st = [json.loads(p.read_text()) for p in (pilot / 'S1').glob('C-*/status.json')]
    acc = [s for s in st if s.get('status') in ('PASS', 'FAIL')]
    if len({s['case_id'] for s in acc}) < 12 and not a.allow_incomplete:
        sys.exit(f'STOP: pilot water arm has {len(acc)} of 12 cases accounted; the spec requires all 12 before freezing')
    out = Path(a.out)
    if out.exists(): sys.exit(f'STOP: {out} exists')
    cases = pd.read_csv(KIT / 'inputs/cases_confirmatory.tsv', sep='\t').set_index('case_id')
    v2, v3 = np.load(a.v2), np.load(v3dir / 'MECHLIB_V3_TARGETS.npz')
    sm = pd.read_csv(v3dir / 'V3_SUPPORT_MAP.tsv.gz', sep='\t')
    passed = [s for s in acc if s['status'] == 'PASS']
    calib, arrays = {}, {k: v3[k] for k in v3.files}
    for key, (col, qterm) in TERMS.items():
        q, a2, a3, res = [], [], [], []
        for s in passed:
            cid = s['case_id']; r_ = cid.split('-')[1]; ri = RES20.index(r_)
            phi, psi = float(cases.loc[cid, 'phi_deg']), float(cases.loc[cid, 'psi_deg'])
            q.append(s['terms'][qterm]); res.append(r_)
            a2.append(cm(v2[f'{col}__v2'][0, ri], phi, psi)); a3.append(cm(v3[f'{col}__v2'][0, ri], phi, psi))
        q, a2, a3 = map(np.array, (q, a2, a3))
        f = fit_lambda(q, a2, a3, res); f['jackknife_leave_one_residue_out'] = jackknife(q, a2, a3, res)
        f['training_cases'] = sorted(s['case_id'] for s in passed); calib[key] = f
        lam = f['lam']
        for suf in ('v2', 'unsplit'):
            arrays[f'{col}__{suf}'] = v2[f'{col}__{suf}'] + lam * (v3[f'{col}__{suf}'] - v2[f'{col}__{suf}'])
        arrays[f'{col}__v4'] = arrays[f'{col}__v2']
        x = sm[sm.term == col]
        var2, var3, vard = (x[c].to_numpy().reshape(2, 20, B, B) ** 2 for c in ('se_v2', 'se_v3', 'se_v3_minus_v2'))
        cov = (var3 - var2 - vard) / 2
        arrays[f'{col}__sigma'] = np.sqrt(np.maximum(var2 + lam ** 2 * vard + 2 * lam * cov, 0))
        arrays[f'{col}__support_level'] = x['level_v3' if lam > 0 else 'level_v2'].to_numpy().reshape(2, 20, B, B).astype('<U8')
    arrays['library_version'] = np.array('v4-lite (intrinsic-calibrated)')
    arrays['lambda'] = np.array([calib[k]['lam'] for k in TERMS]); arrays['lambda_terms'] = np.array([TERMS[k][0] for k in TERMS])
    out.mkdir(parents=True)
    np.savez(out / 'MECHLIB_V4LITE_TARGETS.npz', **arrays)
    h = sha(out / 'MECHLIB_V4LITE_TARGETS.npz')
    rep = dict(spec_sha256=sha(KIT / 'MECHLIB_V4LITE_SPEC.md'), code_sha256=sha(Path(__file__)), v2=sha(a.v2), v3=sha(v3dir / 'MECHLIB_V3_TARGETS.npz'),
               support_map=sha(v3dir / 'V3_SUPPORT_MAP.tsv.gz'), pilot_water_cases_accounted=len(acc), pilot_water_pass=len(passed),
               incomplete_override=bool(a.allow_incomplete), calibration=calib)
    (out / 'V4_CALIBRATION.json').write_text(json.dumps(rep, indent=2))
    (out / 'V4_FREEZE.json').write_text(json.dumps(dict(library='v4-lite', file='MECHLIB_V4LITE_TARGETS.npz', sha256=h,
                                                        built=time.strftime('%Y-%m-%dT%H:%M:%S'), spec_sha256=rep['spec_sha256']), indent=2))
    for k in TERMS:
        c = calib[k]; print(f"{k}: lambda {c['lam']:.3f} (unclipped {c['lam_unclipped']:.3f}; r(QM, v3) {c['r_qm_v3']:.2f}; {c['gate']}; n {c['n']}); jackknife {', '.join(f'{r}:{v:.2f}' for r, v in c['jackknife_leave_one_residue_out'].items())}")
    print(f'v4-lite FROZEN: MECHLIB_V4LITE_TARGETS.npz sha256 {h}')


def selftest():
    rng = np.random.default_rng(3)
    res = ['ALA'] * 4 + ['GLY'] * 4 + ['SER'] * 4
    a2 = rng.normal(0, 0.5, 12); a3 = 1.6 * a2
    q = a2 + 0.3 * (a3 - a2) + rng.normal(0, 0.02, 12)
    f = fit_lambda(q, a2, a3, res); assert abs(f['lam'] - 0.3) < 0.05 and f['gate'] == 'passed', f
    f = fit_lambda(-a3, a2, a3, res); assert f['lam'] == 0.0 and f['gate'].startswith('v2'), f        # opposite shape -> v2
    f = fit_lambda(3 * a3, a2, a3, res); assert f['lam'] == 1.0, f                                      # clipped at v3
    print('SELFTEST PASS: lambda recovery, shape gate, clipping')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for k in ('pilot', 'v2', 'v3dir', 'out'): ap.add_argument('--' + k)
    ap.add_argument('--allow-incomplete', action='store_true', help='dry run only; never for the frozen build')
    ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
    selftest() if a.selftest else main(a)
