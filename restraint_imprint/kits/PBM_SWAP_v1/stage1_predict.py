"""Stage 1: write predictions for the CDL -> MechLib v2 carbonyl-angle swap. No refinement.

For each of the 19 batch structures: site matrices from the frozen isolated cacn / aco / ocn
arms (same logic as the batch analysis), the planned edit at each site's CDL lookup node,
and the predicted coordinate response. Prints the SHA256 of PREDICTIONS.tsv for registration.
"""
import argparse, csv, json, math, runpy, sys
from decimal import Decimal
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT / 'batchcode'))
sys.path.insert(0, str(KIT))
from audit_base import geometry, coordinates, angle  # noqa: E402  (batch code, unchanged)
from matrix_helpers import identify, triplets  # noqa: E402
import swapcore as S  # noqa: E402

SPEC_SHA = '92e2b8a466f4d277f9816055d41a1047986ad30fb126165ce6ebe93f193c9b83'
BATCHCODE_SHA = {'audit_base.py': 'f91bdcc0c3c26eebdc5c0e17cc90b0fc6ca5a67c31f37630070249d10c5f98ef',
                 'matrix_helpers.py': '63f0158418a8000dcfae1bcaa18a857f277f01c0f12a5fa0972a493b77747b84'}
STANDARD = set(S.RES20)
ISOLATED = {'cacn': 'mACN', 'aco': 'mACO', 'ocn': 'mOCN'}
KEY_TOL = Decimal('0.006')
MAX_SITE_EXCLUSION = 0.10


def arm_files(folder):
    geos = [p for p in folder.glob('*.geo') if not p.name.endswith('_final.geo')]
    pdbs = list(folder.glob('*.pdb'))
    if len(geos) != 1 or len(pdbs) != 1:
        raise ValueError(f'{folder}: expected one initial .geo and one .pdb, found {len(geos)} / {len(pdbs)}')
    state = json.loads((folder / 'STATE.json').read_text())
    for name, h in state.get('files_sha256', {}).items():
        if S.sha(folder / name) != h:
            raise ValueError(f'{folder}/{name} changed since the batch run')
    return geos[0], pdbs[0]


def primary(key):
    return all(not a[1] and a[2] in STANDARD for a in key[1])


def changed_keys(stock, arm, term, sign):
    out = set()
    if stock.keys() != arm.keys():
        raise ValueError('Restraint identities differ between stock and arm')
    for k, v in stock.items():
        d = arm[k][0] - v[0]
        if d:
            if identify(k) != term or d != Decimal(sign):
                raise ValueError(f'Unexpected edit {k} {d}')
            out.add(k)
    return out


def model_residues(path):
    """Ordered residues per chain from the input model: {chain: [(resseq, icode, resname, {atom: xyz})]}."""
    chains, index = {}, {}
    for line in Path(path).read_text().splitlines():
        if not line.startswith(('ATOM  ', 'HETATM')) or line[16] not in (' ', 'A'):
            continue
        ch, seq, ic, rn, name = line[21], line[22:26].strip(), line[26].strip(), line[17:20].strip(), line[12:16].strip()
        key = (ch, seq, ic)
        if key not in index:
            index[key] = len(chains.setdefault(ch, []))
            chains[ch].append((seq, ic, rn, {}))
        atoms = chains[ch][index[key]][3]
        if name not in atoms or line[16] == ' ':
            atoms[name] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return chains, index


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v, w = b0 - np.dot(b0, b1n) * b1n, b2 - np.dot(b2, b1n) * b1n
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w))))


def structure(case, batch_root, cdl, v2):
    pid = case['pdb_id']
    root = batch_root / 'structures' / pid.lower()
    if S.sha(case['model']) != case['coordinate_sha256'] or S.sha(case['common']) != case['common_mtz_sha256']:
        raise ValueError(f'{pid}: input model or MTZ hash differs from BATCH_CONFIG')
    arms = ['stock'] + [f'{n}_{s}' for n in ISOLATED for s in ('plus', 'minus')]
    files = {a: arm_files(root / a) for a in arms}
    geo = {a: geometry(files[a][0]) for a in arms}
    xyz = {a: coordinates(files[a][1]) for a in arms}
    eligible = {}
    for name, term in ISOLATED.items():
        plus = changed_keys(geo['stock'], geo[name + '_plus'], term, 1)
        minus = changed_keys(geo['stock'], geo[name + '_minus'], term, -1)
        if plus != minus:
            raise ValueError(f'{pid}: signed masks differ for {name}')
        eligible[term] = {k for k in plus if primary(k)}
    sites = triplets(geo['stock'], eligible)
    chains, index = model_residues(case['model'])
    rows, excluded = [], {'no_phi_psi': 0, 'not_standard_or_no_group': 0, 'cdl_key_mismatch': 0}
    for carbon, keys in sites:
        _, _, rn, ch, seq, ic = carbon
        acn = keys[0][1]
        nxt_n = next(a for a in acn if a[0] == 'N')
        res_next = nxt_n[2]
        k = (ch, seq, ic)
        if k not in index or rn not in S.RES20 or res_next not in S.RES20:
            excluded['not_standard_or_no_group'] += 1; continue
        i = index[k]
        chain = chains[ch]
        cur = chain[i][3]
        prev = chain[i - 1][3] if i > 0 else {}
        nxt = chain[i + 1][3] if i + 1 < len(chain) else {}
        need = ('C' in prev and {'N', 'CA', 'C'} <= cur.keys() and 'N' in nxt
                and np.linalg.norm(prev['C'] - cur['N']) <= 1.8 and np.linalg.norm(cur['C'] - nxt['N']) <= 1.8)
        if not need:
            excluded['no_phi_psi'] += 1; continue
        phi = dihedral(prev['C'], cur['N'], cur['CA'], cur['C'])
        psi = dihedral(cur['N'], cur['CA'], cur['C'], nxt['N'])
        node = (S.round_to_ten(phi), S.round_to_ten(psi))
        group = S.cdl_group(rn, res_next)
        row_cdl = cdl.get(group, {}).get(node)
        stock_ideal = [geo['stock'][kk][0] for kk in keys]
        if row_cdl is None or any(abs(Decimal(str(row_cdl[S.COLS[t]])) - si) > KEY_TOL for t, si in zip(S.TERMS, stock_ideal)):
            excluded['cdl_key_mismatch'] += 1; continue
        cdl_t = np.array([row_cdl[S.COLS[t]] for t in S.TERMS], float)
        v2_t = np.array([S.v2_node(v2, t, rn, res_next, node) for t in S.TERMS], float)
        e = v2_t - cdl_t
        vals = {a: np.array([angle(xyz[a], kk[1]) for kk in keys]) for a in arms}
        T = np.column_stack([(vals[n + '_plus'] - vals[n + '_minus']) / 2 for n in ISOLATED])
        row = dict(pdb_id=pid, band=case['band'], carbonyl=repr(carbon), keys=repr([kk[1] for kk in keys]),
                   res_i=rn, res_next=res_next, phi=phi, psi=psi, node_phi=node[0], node_psi=node[1], group=group)
        for j, t in enumerate(S.TERMS):
            row[f'cdl_{t}'] = cdl_t[j]; row[f'v2_{t}'] = v2_t[j]; row[f'e_{t}'] = e[j]
            for kcol, n in enumerate(ISOLATED):
                row[f'T_{t}_{n}'] = T[j, kcol]
        pred = T @ e
        for j, t in enumerate(S.TERMS):
            row[f'pred_site_{t}'] = pred[j]
        rows.append(row)
    n_sites = len(sites)
    frac = 1 - len(rows) / n_sites if n_sites else 1.0
    status = 'included' if frac <= MAX_SITE_EXCLUSION and rows else 'excluded_stage1_key_gate'
    if rows:
        Tmean = np.mean([[[r[f'T_{t}_{n}'] for n in ISOLATED] for t in S.TERMS] for r in rows], axis=0)
        for r in rows:
            p = Tmean @ np.array([r[f'e_{t}'] for t in S.TERMS])
            for j, t in enumerate(S.TERMS):
                r[f'pred_mean_{t}'] = p[j]
    return rows, dict(pdb_id=pid, band=case['band'], eligible_sites=n_sites, predicted_sites=len(rows),
                      excluded=excluded, excluded_fraction=frac, status=status)


def main(a):
    if S.sha(KIT / 'SWAP_PREDICTION_SPEC_20261112.md') != SPEC_SHA:
        sys.exit('STOP: spec hash differs from frozen v1.0')
    for name, h in BATCHCODE_SHA.items():
        if S.sha(KIT / 'batchcode' / name) != h:
            sys.exit(f'STOP: bundled batch code {name} changed')
    if S.sha(a.cdl) != S.CDL_SHA:
        sys.exit('STOP: CDL database hash differs')
    db = runpy.run_path(str(a.cdl))['cdl_database']
    cdl = {g: dict(grid) for g, grid in db.items()}
    v2 = S.load_v2(a.targets)
    cases = json.loads((KIT / 'inputs' / 'cases.json').read_text())
    if a.out.exists():
        sys.exit(f'STOP: {a.out} exists')
    a.out.mkdir(parents=True)
    allrows, summary = [], []
    for case in cases:
        try:
            rows, info = structure(case, a.batch, cdl, v2)
        except Exception as e:  # reported, never silently replaced
            rows, info = [], dict(pdb_id=case['pdb_id'], band=case['band'], status='stage1_error', error=repr(e))
        print(json.dumps(info), flush=True)
        summary.append(info)
        if info.get('status') == 'included':
            allrows += rows
    if not allrows:
        sys.exit('STOP: no predictions')
    cols = list(allrows[0])
    with (a.out / 'PREDICTIONS.tsv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter='\t', lineterminator='\n')
        w.writeheader()
        for r in allrows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    e = np.array([[r[f'e_{t}'] for t in S.TERMS] for r in allrows])
    lock = dict(spec='SWAP_PREDICTION_SPEC_20261112.md v1.0', spec_sha256=SPEC_SHA, cdl_sha256=S.CDL_SHA,
                targets_sha256=S.TARGETS_SHA, code_sha256={p.name: S.sha(p) for p in KIT.glob('*.py')},
                structures=summary, sites=len(allrows),
                planned_edit_summary={t: dict(mean=float(e[:, j].mean()), mean_abs=float(np.abs(e[:, j]).mean()),
                                              p95_abs=float(np.quantile(np.abs(e[:, j]), .95))) for j, t in enumerate(S.TERMS)},
                planned_sum_change_mean=float(e.sum(1).mean()),
                predictions_sha256=S.sha(a.out / 'PREDICTIONS.tsv'))
    (a.out / 'PREDICTION_LOCK.json').write_text(json.dumps(lock, indent=2) + '\n')
    print('\nPREDICTIONS.tsv SHA256 (send this to be registered before stage 2):')
    print(lock['predictions_sha256'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--cdl', type=Path, required=True)
    p.add_argument('--targets', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
