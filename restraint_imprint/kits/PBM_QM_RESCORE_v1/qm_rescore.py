"""QM re-score of the reported MechLib on the frozen Q6 panel (spec QM_RESCORE_SPEC_20261112.md v1.0)."""
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

SPEC_SHA = 'ddfd59f1a227d4356bfa95d3b397de3be0ef8b2eccd81877e4fea6aaf7ff2679'
H = {'detail': '3af8c2fee6113422004b1a2a9f36d9d8c0887e7c41077bbd8dda0b2e7c3170b1',
     'cases': '160e367216a7c1205d5d069c98a0addf8cbcb4cf74ed6fd771c7efe0491552c1',
     'targets': 'ef7b86b359c84492a0be791de52a1e64112a0153142b9f9f5a0417448d2731b1',
     'chi1': '6210897c9c29ed2896a9ddbf60c32f5c8b541794833759027ae357fb5e50160c'}
NONPGIV = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','HIS','LEU','LYS','MET','PHE','SER','THR','TRP','TYR'}
RES20 = sorted(NONPGIV | {'ILE','VAL','GLY','PRO'})
B = 36
CHI1_TERMS = {'angle_N_CA_CB', 'angle_C_CA_CB'}
WELLS = np.array([-60.0, 180.0, 60.0])

def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()

def cr(G, g, x, y, x0=-175.0):
    u, v = (x - x0) / 10, (y - x0) / 10; i0, j0 = math.floor(u), math.floor(v); tx, ty = u - i0, v - j0
    w = lambda t: [0.5*(-t**3+2*t*t-t), 0.5*(3*t**3-5*t*t+2), 0.5*(-3*t**3+4*t*t+t), 0.5*(t**3-t*t)]
    wx, wy = w(tx), w(ty)
    return float(sum(wx[a]*wy[b]*G[g, (i0+a-1) % B, (j0+b-1) % B] for a in range(4) for b in range(4)))

def chi_class(c):
    if not np.isfinite(c): return -1
    d = np.abs(((c - WELLS) + 180) % 360 - 180); k = int(np.argmin(d)); return k if d[k] <= 40 else -1

def weighted(row, value):
    diff = abs(value - row.qm_value)
    if row.kind == 'angle':
        diff = abs(((value - row.qm_value) + 180) % 360 - 180) * math.pi / 180
    return math.sqrt(row.force_constant) * diff

def evaluate(d, col):
    d = d.copy()
    d['e_model'] = [weighted(r, v) for r, v in zip(d.itertuples(), d[col])]
    d['dl'] = d.amber_weighted_error - d.e_model
    cs = d.groupby(['case_id', 'residue'])['dl'].mean().reset_index()
    res = sorted(cs.residue.unique()); by = {r: cs[cs.residue == r].dl.to_numpy() for r in res}
    rng = np.random.default_rng(20260915)
    s = [np.concatenate([by[res[i]] for i in rng.choice(len(res), len(res), replace=True)]).mean() for _ in range(10000)]
    ci = np.quantile(s, [.025, .975]).tolist()
    pos = int((cs.groupby('residue').dl.mean() > 0).sum())
    terms = {}
    for t, x in d.groupby('term'):
        terms[t] = dict(n=len(x), amber_mae=float((x.amber_value - x.qm_value).abs().mean()), model_mae=float((x[col] - x.qm_value).abs().mean()),
                        model_signed_bias=float((x[col] - x.qm_value).mean()), mean_weighted_delta=float(x.dl.mean()),
                        p_worse=float(stats.ttest_1samp(x.dl, 0, alternative='less').pvalue))
    names = sorted(terms); ps = [terms[t]['p_worse'] for t in names]
    order = sorted(range(len(ps)), key=lambda i: ps[i]); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); terms[names[i]]['holm_p_worse'] = run
        terms[names[i]]['holm_significant_worse'] = run < 0.05
    worse = [t for t in names if terms[t]['holm_significant_worse']]
    crit = {'cluster_ci_lower_gt_zero': ci[0] > 0, 'at_least_12_positive_residues': pos >= 12,
            'no_holm_significant_worse_term': not worse, 'at_least_68_qc_pass_cases': d.case_id.nunique() >= 68}
    return dict(mean_delta=float(cs.dl.mean()), cluster_ci95=ci, positive_residues=pos, cases=int(d.case_id.nunique()),
                worse_terms=worse, criteria=crit, status='PASS' if all(crit.values()) else 'FAIL', terms=terms)

def main(a):
    if sha(a.spec) != SPEC_SHA: sys.exit('STOP: spec hash')
    for k in H:
        if sha(getattr(a, k)) != H[k]: sys.exit(f'STOP: {k} hash')
    d = pd.read_csv(a.detail, sep='\t'); cases = pd.read_csv(a.cases, sep='\t').set_index('case_id')
    assert len(d) == 868 and d.case_id.nunique() == 80
    with np.load(a.targets) as z: v2 = {t: z[t + '__v2'].reshape(40, B, B) for t in d.term.unique()}
    with np.load(a.chi1) as z: chi = {t: z[t] for t in CHI1_TERMS}
    plain, reported = [], []
    for r in d.itertuples():
        c = cases.loc[r.case_id]; g = RES20.index(r.residue)  # next residue NME -> non-Pro row
        phi, psi = float(c.phi_deg), float(c.psi_deg)
        p = cr(v2[r.term], g, phi, psi); plain.append(p)
        k = chi_class(float(c.initial_chi1_deg)) if str(c.initial_chi1_deg) != 'nan' else -1
        if r.term in CHI1_TERMS and k >= 0 and r.residue not in ('GLY', 'ALA', 'PRO'):
            reported.append(cr(chi[r.term][k], g, phi, psi))
        else:
            reported.append(p)
    d['v2_plain'] = plain; d['mechlib_reported'] = reported
    out = {'spec_sha256': SPEC_SHA, 'inputs': H,
           'primary_reported_mechlib': evaluate(d, 'mechlib_reported'),
           'secondary_plain_v2': evaluate(d, 'v2_plain'),
           'reference_earlier_q6_library': evaluate(d, 'mechlib_value')}
    a.out.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.out / 'QM_RESCORE_TERM_DETAIL.tsv', sep='\t', index=False)
    (a.out / 'QM_RESCORE.json').write_text(json.dumps(out, indent=2, default=bool) + '\n')
    for k in ('primary_reported_mechlib', 'secondary_plain_v2', 'reference_earlier_q6_library'):
        r = out[k]; print(f"{k:32s} {r['status']}: mean {r['mean_delta']:+.4f} CI {np.round(r['cluster_ci95'],4)} residues+ {r['positive_residues']}/20 worse={r['worse_terms']}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('spec', 'detail', 'cases', 'targets', 'chi1'): p.add_argument('--' + k, type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); main(p.parse_args())
