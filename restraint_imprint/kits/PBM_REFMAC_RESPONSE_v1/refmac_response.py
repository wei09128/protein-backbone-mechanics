"""Controlled restraint response in REFMAC (spec REFMAC_RESPONSE_SPEC_20261113.md v1.0).

  ccp4-python refmac_response.py --selftest
  ccp4-python refmac_response.py --root /mnt/c/Wei/PBM --out RUN_DIR [--jobs 4]
Needs refmac5 on PATH and CLIBD_MON set (source ccp4.setup-sh), numpy and gemmi (both ship with ccp4-python).
"""
import argparse, hashlib, json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

KIT = Path(__file__).resolve().parent
SPEC = KIT / 'REFMAC_RESPONSE_SPEC_20261113.md'
SPEC_SHA = '532843f7eee7ebc2a4b8e7c1f53148781c89e0055621653364cc92f83bcb79e6'
CASES_SHA = 'd3269d703b466f7733d0468145917924157524909a1fb38bb993bb2876e6fbbc'
BANDS = ['le1.0', 'gt1.0_le1.5', 'gt1.5_le2.0']
ARMS = {'ACN+': {'ACN': +1}, 'ACN-': {'ACN': -1}, 'OCN+': {'OCN': +1}, 'OCN-': {'OCN': -1},
        'COMP+': {'ACN': +1, 'OCN': -1}, 'COMP-': {'ACN': -1, 'OCN': +1}}
PAIRS = {'ACN': ('ACN+', 'ACN-'), 'OCN': ('OCN+', 'OCN-'), 'COMP': ('COMP+', 'COMP-')}
STD = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO',
       'SER', 'THR', 'TRP', 'TYR', 'VAL'}
NCYC, SEED, DRAWS, PER_BAND = 10, 20261117, 10000, 2
GATE_MIN = 0.10


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ dictionary edit
def cif_blocks(text):
    """Split CIF text into {block_name: block_text}."""
    out, name, buf = {}, None, []
    for line in text.splitlines(keepends=True):
        if line.startswith('data_'):
            if name is not None:
                out[name] = ''.join(buf)
            name, buf = line.strip()[5:], [line]
        elif name is not None:
            buf.append(line)
    if name is not None:
        out[name] = ''.join(buf)
    return out


def edit_link_block(block, shifts):
    """Shift CA-C-N and/or O-C-N angle targets in a link block. Returns (new_block, list of edits)."""
    lines = block.splitlines(keepends=True)
    out, edits, i = [], [], 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == 'loop_' and i + 1 < len(lines) and lines[i + 1].strip().startswith('_chem_link_angle.'):
            out.append(line); i += 1
            cols = []
            while i < len(lines) and lines[i].strip().startswith('_chem_link_angle.'):
                cols.append(lines[i].strip().split('.', 1)[1]); out.append(lines[i]); i += 1
            c = {k: n for n, k in enumerate(cols)}
            need = ['atom_1_comp_id', 'atom_id_1', 'atom_2_comp_id', 'atom_id_2', 'atom_3_comp_id', 'atom_id_3', 'value_angle']
            if any(k not in c for k in need):
                raise ValueError(f'unexpected link angle columns {cols}')
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(('loop_', '_', 'data_', '#')):
                tok = lines[i].split()
                if len(tok) == len(cols):
                    a = [(tok[c['atom_1_comp_id']], tok[c['atom_id_1']]), (tok[c['atom_2_comp_id']], tok[c['atom_id_2']]),
                         (tok[c['atom_3_comp_id']], tok[c['atom_id_3']])]
                    ends = {a[0], a[2]}
                    term = None
                    if a[1] == ('1', 'C') and ends == {('1', 'CA'), ('2', 'N')}:
                        term = 'ACN'
                    elif a[1] == ('1', 'C') and ends == {('1', 'O'), ('2', 'N')}:
                        term = 'OCN'
                    if term and term in shifts:
                        old = float(tok[c['value_angle']]); new = old + shifts[term]
                        tok[c['value_angle']] = f'{new:.3f}'
                        edits.append(dict(term=term, old=old, new=new))
                        lines[i] = ' ' + ' '.join(tok) + '\n'
                out.append(lines[i]); i += 1
            continue
        out.append(line); i += 1
    return ''.join(out), edits


def make_libin(mon_lib_list, shifts, path):
    blocks = cif_blocks(Path(mon_lib_list).read_text())
    parts, record = [], {}
    for link in ('TRANS', 'PTRANS'):
        key = f'link_{link}'
        if key not in blocks:
            raise ValueError(f'{key} not in {mon_lib_list}')
        new, edits = edit_link_block(blocks[key], shifts)
        got = sorted(e['term'] for e in edits)
        if got != sorted(shifts):
            raise ValueError(f'{link}: expected edits {sorted(shifts)}, made {got}')
        parts.append(new); record[link] = edits
    # declare the two links (copied rows from the installed link list)
    lst = blocks.get('link_list', '')
    rows = [l for l in lst.splitlines() if l.split()[:1] in (['TRANS'], ['PTRANS'])]
    head = [l for l in lst.splitlines() if l.strip().startswith(('_chem_link.', 'loop_'))]
    if rows and head:
        parts.insert(0, 'data_link_list\n' + '\n'.join(head) + '\n' + '\n'.join(rows) + '\n')
    Path(path).write_text('global_\n' + '\n'.join(parts))
    return record


# ------------------------------------------------------------------ mtz and refmac
def mtz_labels(path):
    import gemmi
    m = gemmi.read_mtz_file(str(path))
    cols = [(c.label, c.type) for c in m.columns]
    F = [l for l, t in cols if t == 'F']
    Q = [l for l, t in cols if t == 'Q']
    I = [l for l, t in cols if t == 'I' and 'free' in l.lower()] or [l for l, t in cols if t == 'I']
    if not (F and Q and I):
        raise ValueError(f'cannot find F/SIGF/FREE in {cols}')
    f = F[0]
    sig = next((q for q in Q if f.lower() in q.lower()), Q[0])
    fl = np.array(m.column_with_label(I[0]), float); fl = fl[np.isfinite(fl)].astype(int)
    vals, counts = np.unique(fl, return_counts=True)
    test = int(vals[np.argmin(counts)]) if set(vals) <= {0, 1} else 0
    return dict(F=f, SIGF=sig, FREE=I[0], test=test, flag_values=vals.tolist(), flag_counts=counts.tolist(),
                test_fraction=float(counts[list(vals).index(test)] / counts.sum()))


def refmac(model, mtz, labels, outdir, weight, libin=None):
    outdir.mkdir(parents=True, exist_ok=True)
    kw = [f"LABIN FP={labels['F']} SIGFP={labels['SIGF']} FREE={labels['FREE']}", f"FREE {labels['test']}",
          'REFI TYPE REST', f'NCYC {NCYC}', 'MAKE HOUT NO', 'MAKE NEWLIGAND CONTINUE',
          'WEIGHT AUTO' if weight is None else f'WEIGHT MATRIX {weight}', 'END']
    cmd = ['refmac5', 'HKLIN', str(mtz), 'XYZIN', str(model), 'HKLOUT', str(outdir / 'out.mtz'), 'XYZOUT', str(outdir / 'out.pdb')]
    if libin:
        cmd += ['LIBIN', str(libin)]
    (outdir / 'keywords.txt').write_text('\n'.join(kw) + '\n')
    with open(outdir / 'refmac.log', 'w') as log:
        p = subprocess.run(cmd, input='\n'.join(kw) + '\n', text=True, stdout=log, stderr=subprocess.STDOUT, cwd=outdir)
    text = (outdir / 'refmac.log').read_text(errors='replace')
    ok = p.returncode == 0 and (outdir / 'out.pdb').exists()
    rf = re.findall(r'^\s*R free\s+([0-9.]+)\s+([0-9.]+)', text, re.M)
    w = re.findall(r'[Ww]eight matrix\s*[:=]?\s*([0-9]*\.[0-9]+)', text)
    return dict(ok=ok, returncode=p.returncode, rfree=[float(x) for x in rf[-1]] if rf else None,
                weight=float(w[-1]) if w else None)


# ------------------------------------------------------------------ geometry
def angle(a, b, c):
    u, v = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - (b0 @ b1) * b1; w = b2 - (b2 @ b1) * b1
    return float(np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w)))


def junctions(pdb):
    """{(chain, resseq+icode): (ACN, ACO, OCN)} for blank-altloc standard trans junctions (first model)."""
    res, order = {}, []
    for line in Path(pdb).read_text(errors='replace').splitlines():
        if line.startswith('ENDMDL'):
            break
        if not line.startswith(('ATOM', 'HETATM')) or line[16] not in ' ':
            continue
        name, rn, ch, rs = line[12:16].strip(), line[17:20].strip(), line[21], line[22:27]
        if rn not in STD:
            continue
        key = (ch, rs)
        if key not in res:
            res[key] = {}; order.append(key)
        res[key][name] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    out = {}
    for k1, k2 in zip(order, order[1:]):
        if k1[0] != k2[0]:
            continue
        a, b = res[k1], res[k2]
        if not all(x in a for x in ('CA', 'C', 'O')) or not all(x in b for x in ('N', 'CA')):
            continue
        if np.linalg.norm(a['C'] - b['N']) > 1.8 or abs(dihedral(a['CA'], a['C'], b['N'], b['CA'])) < 150:
            continue
        out[k1] = (angle(a['CA'], a['C'], b['N']), angle(a['CA'], a['C'], a['O']), angle(a['O'], a['C'], b['N']))
    return out


def responses(run):
    """Per-structure mean response per degree for each edit pair; junctions common to all arms."""
    geo = {arm: junctions(run / arm / 'out.pdb') for arm in ARMS}
    common = set.intersection(*(set(g) for g in geo.values()))
    res = {'n_junctions': len(common)}
    for name, (p, m) in PAIRS.items():
        d = np.array([[(geo[p][k][i] - geo[m][k][i]) / 2 for i in range(3)] for k in sorted(common)])
        res[name] = dict(ACN=float(d[:, 0].mean()), ACO=float(d[:, 1].mean()), OCN=float(d[:, 2].mean()),
                         SUM=float(d.sum(1).mean()))
    return res


# ------------------------------------------------------------------ main
def boot_mean(x, seed=SEED, draws=DRAWS):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (draws, len(x)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def run_structure(case, root, out, mon_lib, labels_cache, log):
    lid = case['pdb_id'].lower(); d = out / 'runs' / lid
    model, mtz = Path(case['model']), Path(case['common'])
    if sha(model) != case['coordinate_sha256'] or sha(mtz) != case['common_mtz_sha256']:
        return dict(pdb_id=case['pdb_id'], status='input hash differs')
    lab = mtz_labels(mtz); labels_cache[case['pdb_id']] = lab
    auto = refmac(model, mtz, lab, d / 'stock_auto', None)
    if not auto['ok'] or auto['weight'] is None:
        return dict(pdb_id=case['pdb_id'], status='stock failed or weight not parsed', auto=auto, labels=lab)
    w = auto['weight']
    stock = refmac(model, mtz, lab, d / 'stock', w)
    if not stock['ok'] or stock['rfree'] is None or abs(stock['rfree'][1] - stock['rfree'][0]) > 0.02:
        return dict(pdb_id=case['pdb_id'], status='stock gate failed', stock=stock, labels=lab, weight=w)
    log(f"{case['pdb_id']}: stock ok, weight {w}, R-free {stock['rfree']}")
    arms = {}
    for arm, shifts in ARMS.items():
        lib = d / f'{arm}.cif'
        rec = make_libin(mon_lib, shifts, lib)
        arms[arm] = dict(refmac(model, mtz, lab, d / arm, w, lib), edits=rec)
        if not arms[arm]['ok']:
            return dict(pdb_id=case['pdb_id'], status=f'arm {arm} failed', arms=arms)
    r = responses(d)
    return dict(pdb_id=case['pdb_id'], band=case['band'], status='complete', weight=w, labels=lab,
                stock_rfree=stock['rfree'], arm_rfree={a: v['rfree'] for a, v in arms.items()}, response=r)


def main(a):
    if sha(SPEC) != SPEC_SHA:
        sys.exit('STOP: spec hash differs')
    if sha(KIT / 'inputs' / 'cases.json') != CASES_SHA:
        sys.exit('STOP: case list hash differs')
    mon = Path(os.environ.get('CLIBD_MON', '')) / 'list' / 'mon_lib_list.cif'
    if not mon.exists():
        sys.exit('STOP: CLIBD_MON/list/mon_lib_list.cif not found (source ccp4.setup-sh)')
    ver = subprocess.run(['refmac5', '-i'], input='END\n', capture_output=True, text=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    (out / 'REFMAC_VERSION.txt').write_text(ver.stdout[-3000:] + ver.stderr[-2000:])
    # dictionary check before any refinement
    probe = make_libin(mon, {'ACN': +1, 'OCN': -1}, out / 'probe_COMP+.cif')
    (out / 'LINK_EDITS.json').write_text(json.dumps(probe, indent=2))
    print('Link targets found and edited:', json.dumps(probe), flush=True)
    cases = json.loads((KIT / 'inputs' / 'cases.json').read_text())
    logs = []
    log = lambda s: (print(s, flush=True), logs.append(s))
    labels, results, used = {}, [], {b: [] for b in BANDS}
    # gate: first structure of the first band, ACN pair only
    queue = {b: sorted([c for c in cases if c['band'] == b], key=lambda c: c['pdb_id']) for b in BANDS}
    first = queue['le1.0'][0]
    gate_dir = out / 'gate' / first['pdb_id'].lower()
    lab = mtz_labels(Path(first['model']).parent / 'common.mtz')
    auto = refmac(Path(first['model']), Path(first['common']), lab, gate_dir / 'stock_auto', None)
    if not auto['ok'] or auto['weight'] is None:
        sys.exit(f'STOP at gate: stock REFMAC failed or weight not parsed ({auto}); see {gate_dir}/stock_auto/refmac.log')
    for arm in ('ACN+', 'ACN-'):
        lib = gate_dir / f'{arm}.cif'; make_libin(mon, ARMS[arm], lib)
        r = refmac(Path(first['model']), Path(first['common']), lab, gate_dir / arm, auto['weight'], lib)
        if not r['ok']:
            sys.exit(f'STOP at gate: {arm} failed; see {gate_dir}/{arm}/refmac.log')
    gp, gm = junctions(gate_dir / 'ACN+' / 'out.pdb'), junctions(gate_dir / 'ACN-' / 'out.pdb')
    ks = sorted(set(gp) & set(gm))
    r = np.array([[(gp[k][i] - gm[k][i]) / 2 for i in range(3)] for k in ks]).mean(0)
    ok = bool(r[0] > 0.005 and r[1] < 0 and r[2] < 0)
    (out / 'GATE.json').write_text(json.dumps(dict(pdb_id=first['pdb_id'], response_ACN_ACO_OCN=r.tolist(), angle_sum=float(r.sum()), n=len(ks),
                                                   weight=auto['weight'], criterion='amendment 1: ACN > 0.005 and ACO, OCN < 0', passed=ok), indent=2))
    log(f'Gate (amendment 1): isolated edit response in {first["pdb_id"]} ACN {r[0]:+.4f}, ACO {r[1]:+.4f}, OCN {r[2]:+.4f} per degree; sum {r.sum():+.4f}')
    if not ok:
        sys.exit('STOP: no coupling signature; link override not active; no further refinement')

    def band_worker(b):
        got = []
        for c in queue[b]:
            if len(got) == PER_BAND:
                break
            r = run_structure(c, a.root, out, mon, labels, log)
            results.append(r); log(f"{c['pdb_id']} ({b}): {r['status']}")
            if r['status'] == 'complete':
                got.append(c['pdb_id'])
        used[b] = got
    with ThreadPoolExecutor(max_workers=min(a.jobs, 3)) as ex:
        list(ex.map(band_worker, BANDS))
    done = [r for r in results if r['status'] == 'complete']
    ph = {}
    for line in (KIT / 'inputs' / 'PHENIX_STRUCTURE_VALUES.tsv').read_text().splitlines()[1:]:
        t = line.split('\t'); ph[t[0]] = dict(cacn=float(t[4]), compensated=float(t[6]))
    rows = []
    for r in done:
        x = r['response']
        rows.append(dict(pdb_id=r['pdb_id'], band=r['band'], n=x['n_junctions'],
                         iso_ACN=x['ACN']['ACN'], comp_ACN=x['COMP']['ACN'], E1=x['COMP']['ACN'] - x['ACN']['ACN'],
                         sum_ACN=x['ACN']['SUM'], sum_OCN=x['OCN']['SUM'], sum_COMP=x['COMP']['SUM'],
                         col_ACN=[x['ACN'][k] for k in ('ACN', 'ACO', 'OCN')], col_OCN=[x['OCN'][k] for k in ('ACN', 'ACO', 'OCN')],
                         phenix_iso_ACN=ph[r['pdb_id']]['cacn'], phenix_comp_ACN=ph[r['pdb_id']]['compensated']))
    summ = {}
    if rows:
        E1 = [r['E1'] for r in rows]
        summ['E1_comp_minus_iso'] = dict(mean=float(np.mean(E1)), ci95=boot_mean(E1), positive=f"{sum(e > 0 for e in E1)}/{len(E1)}")
        for k in ('sum_ACN', 'sum_OCN', 'sum_COMP'):
            v = [r[k] for r in rows]
            summ['E2_' + k] = dict(mean=float(np.mean(v)), max_abs=float(np.max(np.abs(v))), conserved=bool(np.max(np.abs(v)) <= 0.02))
        for k, p in (('iso_ACN', 'phenix_iso_ACN'), ('comp_ACN', 'phenix_comp_ACN')):
            dlt = [r[k] - r[p] for r in rows]
            summ['E3_' + k + '_refmac_minus_phenix'] = dict(mean=float(np.mean(dlt)), ci95=boot_mean(dlt), negative=f"{sum(x < 0 for x in dlt)}/{len(dlt)}",
                                                             refmac_mean=float(np.mean([r[k] for r in rows])), phenix_mean=float(np.mean([r[p] for r in rows])))
        summ['E4_mean_columns'] = dict(ACN_edit=np.mean([r['col_ACN'] for r in rows], 0).tolist(), OCN_edit=np.mean([r['col_OCN'] for r in rows], 0).tolist(),
                                       phenix_mean_matrix_columns=dict(ACN_edit=[0.48, -0.24, -0.23], OCN_edit=[-0.28, -0.28, 0.56]))
    (out / 'STRUCTURE_RESULTS.json').write_text(json.dumps(rows, indent=2))
    (out / 'ALL_ATTEMPTS.json').write_text(json.dumps(results, indent=2, default=str))
    (out / 'SUMMARY.json').write_text(json.dumps(dict(spec_sha256=SPEC_SHA, code_sha256=sha(Path(__file__)), used=used, summary=summ), indent=2))
    print(json.dumps(summ, indent=1))


def selftest():
    blk = """data_link_TRANS
loop_
_chem_link_angle.link_id
_chem_link_angle.atom_1_comp_id
_chem_link_angle.atom_id_1
_chem_link_angle.atom_2_comp_id
_chem_link_angle.atom_id_2
_chem_link_angle.atom_3_comp_id
_chem_link_angle.atom_id_3
_chem_link_angle.value_angle
_chem_link_angle.value_angle_esd
 TRANS 1 CA 1 C 2 N 116.200 2.000
 TRANS 1 O 1 C 2 N 123.000 1.600
 TRANS 1 C 2 N 2 CA 121.700 1.800
"""
    new, e = edit_link_block(blk, {'ACN': 1, 'OCN': -1})
    assert sorted(x['term'] for x in e) == ['ACN', 'OCN'] and '117.200' in new and '122.000' in new and '121.700' in new
    rev = blk.replace(' TRANS 1 CA 1 C 2 N 116.200', ' TRANS 2 N 1 C 1 CA 116.200')
    new2, e2 = edit_link_block(rev, {'ACN': -1})
    assert e2 and e2[0]['new'] == 115.2
    # geometry on a tiny ideal dipeptide
    pdb = '\n'.join([
        'ATOM      1  N   ALA A   1       0.000   1.458   0.000  1.00 10.00           N',
        'ATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C',
        'ATOM      3  C   ALA A   1       1.525   0.000   0.000  1.00 10.00           C',
        'ATOM      4  O   ALA A   1       2.150  -1.061   0.000  1.00 10.00           O',
        'ATOM      5  N   ALA A   2       2.155   1.172   0.000  1.00 10.00           N',
        'ATOM      6  CA  ALA A   2       3.604   1.292   0.000  1.00 10.00           C'])
    tmp = KIT / '_selftest.pdb'; tmp.write_text(pdb + '\n')
    j = junctions(tmp); tmp.unlink()
    (acn, aco, ocn), = j.values()
    assert abs(acn + aco + ocn - 360) < 0.5, (acn, aco, ocn)
    print('SELFTEST PASS: link edits (both atom orders) and junction geometry')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/mnt/c/Wei/PBM')
    ap.add_argument('--out')
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    selftest() if a.selftest else main(a)
