"""Stage 2 scoring: audit the stock and swap arms and score the registered predictions.
Run with phenix.python (needs iotbx for the MTZ parity check)."""
import argparse, ast, csv, json, math, sys
from decimal import Decimal
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT / 'batchcode'))
sys.path.insert(0, str(KIT))
from audit_base import geometry, coordinates, angle, final_r, array_map  # noqa: E402
from matrix_helpers import identify  # noqa: E402
import swapcore as S  # noqa: E402

SEED, DRAWS, EDIT_TOL = 20261113, 10000, 0.006
def num(x):
    """Read a stage-1 number; stage 1 wrote numpy floats as 'np.float64(v)' (value exact, wrapper only)."""
    x = str(x)
    if x.startswith('np.float64(') and x.endswith(')'):
        x = x[len('np.float64('):-1]
    return float(x)


PRED = {'site_matrix': 'pred_site_', 'mean_matrix': 'pred_mean_', 'copy': 'e_', 'zero': None}


def mtz_support(path):
    from iotbx import mtz
    m = mtz.object(file_name=str(path))
    return set(array_map(m, 'F-obs-filtered')), array_map(m, 'R-free-flags')


def audit_pair(run, pid, preds):
    st, sw = run / 'stock', run / 'swap'
    g0, g1 = geometry(st / 'initial.geo'), geometry(sw / 'initial.geo')
    if g0.keys() != g1.keys():
        raise ValueError('Restraint identities differ between stock and swap')
    changed = {}
    for k, v in g0.items():
        nv = g1[k]
        if v[1:] != nv[1:]:
            raise ValueError(f'Initial model value, sigma or weight differs: {k}')
        if nv[0] != v[0]:
            if identify(k) not in S.TERMS:
                raise ValueError(f'Unexpected changed ideal: {k}')
            changed[k] = float(nv[0] - v[0])
    for arm in (st, sw):
        meta = json.loads((arm / 'ARM.json').read_text())
        if meta.get('live_calls', 0) <= 0:
            raise ValueError(f'No live lookup trace in {arm}')
    if json.loads((sw / 'ARM.json').read_text()).get('edited_lookups', 0) <= 0:
        raise ValueError('Swap arm edited no lookup')
    f0, r0 = mtz_support(st / 'final.mtz')
    f1, r1 = mtz_support(sw / 'final.mtz')
    if f0 != f1 or r0 != r1:
        raise ValueError('Observation or free-flag support differs between arms')
    c0, c1 = coordinates(st / 'final.pdb'), coordinates(sw / 'final.pdb')
    if c0.keys() != c1.keys():
        raise ValueError('Final atom sets differ')
    rows, bad = [], 0
    for p in preds:
        keys = ast.literal_eval(p['keys'])
        e = np.array([num(p['e_' + t]) for t in S.TERMS])
        realised = []
        for t, atoms in zip(S.TERMS, keys):
            k = ('angle', tuple(atoms))
            realised.append(changed.get(k, 0.0))
        if np.any(np.abs(np.array(realised) - e) > EDIT_TOL):
            bad += 1
            continue
        obs = np.array([angle(c1, tuple(a)) - angle(c0, tuple(a)) for a in keys])
        r = dict(pdb_id=pid, band=p['band'], carbonyl=p['carbonyl'], max_abs_e=float(np.abs(e).max()),
                 e_sum=float(e.sum()), obs_sum=float(obs.sum()))
        for j, t in enumerate(S.TERMS):
            r['e_' + t] = e[j]; r['obs_' + t] = obs[j]
            for name, col in PRED.items():
                pv = 0.0 if col is None else num(p[col + t])
                r[f'{name}_{t}'] = pv
        rows.append(r)
    try:
        R = {'stock': list(map(float, final_r(st / 'RUN.log'))), 'swap': list(map(float, final_r(sw / 'RUN.log')))}
    except Exception as e:
        R = {'unavailable': repr(e)}
    return rows, dict(pdb_id=pid, predicted_sites=len(preds), scored_sites=len(rows), realised_edit_mismatch=bad,
                      changed_ideals=len(changed), R_work_R_free=R)


def structure_mae(rows, name):
    err = [abs(r['obs_' + t] - r[f'{name}_{t}']) for r in rows for t in S.TERMS]
    return float(np.mean(err)) if err else float('nan')


def band_bootstrap(values, bands, seed=SEED, draws=DRAWS):
    values, bands = np.asarray(values, float), np.asarray(bands)
    rng = np.random.default_rng(seed)
    groups = [np.where(bands == b)[0] for b in sorted(set(bands))]
    samples = []
    for _ in range(draws):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        samples.append(values[idx].mean())
    return float(values.mean()), np.quantile(samples, [.025, .975]).tolist()


def main(a):
    preds_path = a.stage1 / 'PREDICTIONS.tsv'
    if S.sha(preds_path) != a.registered:
        sys.exit('STOP: PREDICTIONS.tsv does not match the registered hash')
    with preds_path.open() as f:
        preds = list(csv.DictReader(f, delimiter='\t'))
    by = {}
    for p in preds:
        by.setdefault(p['pdb_id'], []).append(p)
    allrows, audits, struct = [], [], []
    for pid, plist in sorted(by.items()):
        try:
            rows, info = audit_pair(a.runs / pid.lower(), pid, plist)
            info['status'] = 'PASS' if rows and info['realised_edit_mismatch'] <= 0.1 * len(plist) else 'FAIL_AUDIT'
        except Exception as e:
            rows, info = [], dict(pdb_id=pid, status='FAIL_AUDIT', error=repr(e))
        audits.append(info)
        print(json.dumps(info), flush=True)
        if info['status'] != 'PASS':
            continue
        allrows += rows
        s = dict(pdb_id=pid, band=rows[0]['band'], sites=len(rows),
                 mean_abs_e=float(np.mean([abs(r['e_' + t]) for r in rows for t in S.TERMS])))
        for name in PRED:
            s['mae_' + name] = structure_mae(rows, name)
        s['D_copy_minus_matrix'] = s['mae_copy'] - s['mae_site_matrix']
        s['target_sum_change'] = float(np.mean([r['e_sum'] for r in rows]))
        s['observed_sum_change'] = float(np.mean([r['obs_sum'] for r in rows]))
        for lab, sel in (('large_e', lambda r: r['max_abs_e'] >= 1.0), ('small_e', lambda r: r['max_abs_e'] < 1.0)):
            sub = [r for r in rows if sel(r)]
            s[f'{lab}_sites'] = len(sub)
            for name in ('site_matrix', 'copy'):
                s[f'{lab}_mae_{name}'] = structure_mae(sub, name)
        struct.append(s)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'AUDITS.json').write_text(json.dumps(audits, indent=2) + '\n')
    for path, rows in (('SITE_RESULTS.tsv', allrows), ('STRUCTURE_RESULTS.tsv', struct)):
        if rows:
            with (a.out / path).open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t'); w.writeheader(); w.writerows(rows)
    if not struct:
        sys.exit('STOP: no structure passed the audits')
    D = [s['D_copy_minus_matrix'] for s in struct]
    est, ci = band_bootstrap(D, [s['band'] for s in struct])
    summary = dict(structures_scored=len(struct), structures_failed=[x['pdb_id'] for x in audits if x['status'] != 'PASS'],
                   sites=len(allrows), primary=dict(mean_D_copy_minus_site_matrix=est, ci95=ci,
                   structures_matrix_better=int(sum(d > 0 for d in D)),
                   reading='The measured response predicted the library swap' if ci[0] > 0 else 'not shown'),
                   mean_mae={n: float(np.mean([s['mae_' + n] for s in struct])) for n in PRED},
                   mean_target_sum_change=float(np.mean([s['target_sum_change'] for s in struct])),
                   mean_observed_sum_change=float(np.mean([s['observed_sum_change'] for s in struct])),
                   by_band={b: dict(n=sum(s['band'] == b for s in struct),
                                    mean_D=float(np.mean([s['D_copy_minus_matrix'] for s in struct if s['band'] == b])))
                            for b in sorted({s['band'] for s in struct})},
                   edit_size={lab: {n: float(np.nanmean([s[f'{lab}_mae_{n}'] for s in struct if s[f'{lab}_sites']]))
                                    for n in ('site_matrix', 'copy')} for lab in ('large_e', 'small_e')})
    (a.out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    (a.out / 'RUN_LOCK.json').write_text(json.dumps(dict(registered_predictions_sha256=a.registered,
                                         code_sha256={p.name: S.sha(p) for p in KIT.glob('*.py')}), indent=2) + '\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4))
        for s in struct:
            ax[0].plot([0, 1], [s['mae_copy'], s['mae_site_matrix']], '-o', ms=3, lw=0.9, color='#2b6cb0')
        ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(['targets copied', 'measured response'])
        ax[0].set_ylabel('error, CDL -> v2 swap (deg)'); ax[0].set_title('Per structure', fontsize=9, loc='left')
        col = {'mACN': '#2b6cb0', 'mACO': '#c05621', 'mOCN': '#2f855a'}
        for t in S.TERMS:
            ax[1].scatter([r['site_matrix_' + t] for r in allrows], [r['obs_' + t] for r in allrows], s=3, alpha=.4,
                          color=col[t], label={'mACN': 'Ca-C-N', 'mACO': 'Ca-C-O', 'mOCN': 'O-C-N'}[t])
        lim = [min(ax[1].get_xlim()[0], ax[1].get_ylim()[0]), max(ax[1].get_xlim()[1], ax[1].get_ylim()[1])]
        ax[1].plot(lim, lim, '--', color='#777', lw=0.8); ax[1].legend(fontsize=7, frameon=False)
        ax[1].set_xlabel('predicted (deg)'); ax[1].set_ylabel('observed (deg)'); ax[1].set_title('Per site', fontsize=9, loc='left')
        for x in ax:
            for sp in ('top', 'right'):
                x.spines[sp].set_visible(False)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(a.out / f'Swap.{ext}', dpi=200)
    except Exception as e:
        print('figure skipped:', e)
    print(json.dumps(summary['primary'], indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage1', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--registered', required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
