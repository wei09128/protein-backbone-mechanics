#!/usr/bin/env python3
"""COD probe (counts only). Finds peptide junctions C(=O)-N-CA-C(=O)-N with both amides trans in a local COD
mirror and counts them by residue class and phi/psi region. It does NOT compute or report the carbonyl angles
(Ca-C-N, O-C-N, Ca-C-O): seeing counts is not seeing outcomes. Decides, by the rule fixed in COD_ROLE_RULE.md,
whether COD is the training source for v4 or a referee.
  python cod_probe.py --cod ~/cod/cif --out ~/cod_probe --workers 4
  python cod_probe.py --selftest
"""
import argparse, json, math, os, re, sys, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

RADII = {'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05, 'H': 0.31}
R_MAX = 0.05
FORM = re.compile(r"_chemical_formula_sum\s+['\"]?([^'\"\n]+)")
RFAC = re.compile(r"_refine_ls_R_factor_gt\s+([0-9.]+)")


def formula_ok(text):
    m = FORM.search(text)
    if not m:
        return False
    el = dict((e, float(n) if n else 1.0) for e, n in re.findall(r"([A-Z][a-z]?)\s*([0-9.]*)", m.group(1)))
    allowed = {'C', 'H', 'N', 'O', 'S', 'Cl', 'Br', 'F', 'I', 'P', 'Na', 'K', 'Li', 'Ca', 'Mg', 'Zn', 'Cu', 'Ni', 'Co', 'Fe', 'B', 'Si', 'Se'}
    return el.get('C', 0) >= 5 and el.get('N', 0) >= 2 and el.get('O', 0) >= 2 and set(el) <= allowed


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v, w = b0 - (b0 @ b1n) * b1n, b2 - (b2 @ b1n) * b1n
    return math.degrees(math.atan2(np.cross(b1n, v) @ w, v @ w))


def region(phi, psi):
    if phi > 0:
        return 'left'
    if -100 <= psi < 50:
        return 'alphaR'
    return 'beta' if phi < -90 else 'ppII'


def molecules_from_block(block):
    """Heavy atoms of the asymmetric unit (occupancy 1, no disorder group) in Cartesian coordinates."""
    import gemmi
    ss = gemmi.make_small_structure_from_block(block)
    dg = list(block.find_values('_atom_site_disorder_group')) if block.find_values('_atom_site_disorder_group') else []
    els, xyz, flag = [], [], []
    for i, s in enumerate(ss.sites):
        e = s.element.name
        if e == 'H' or e == 'D':
            continue
        els.append(e); xyz.append(ss.cell.orthogonalize(s.fract).tolist())
        bad = s.occ < 0.999 or (i < len(dg) and dg[i] not in ('.', '?', ''))
        flag.append(bad)
    return els, np.array(xyz, float), np.array(flag, bool)


def junctions(els, X, bad):
    n = len(els)
    if n < 6:
        return []
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    R = np.array([RADII.get(e, 0.0) for e in els])
    adj = (D < (R[:, None] + R[None] + 0.45)) & (D > 0.6) & (R[:, None] > 0) & (R[None] > 0)
    nb = [set(np.where(adj[i])[0]) for i in range(n)]
    isC = [e == 'C' for e in els]; isN = [e == 'N' for e in els]; isO = [e == 'O' for e in els]

    def carbonyl(c):  # amide carbonyl carbon: C with a terminal O at < 1.30 A and an N neighbour
        if not isC[c]:
            return None
        o = [j for j in nb[c] if isO[j] and len(nb[j]) == 1 and D[c, j] < 1.30]
        return o[0] if len(o) == 1 else None
    out = []
    for c3 in range(n):
        o3 = carbonyl(c3)
        if o3 is None:
            continue
        for ca in nb[c3]:
            if not isC[ca] or ca == o3:
                continue
            for n2 in nb[ca]:
                if not isN[n2]:
                    continue
                c1s = [j for j in nb[n2] if j != ca and carbonyl(j) is not None]
                n4s = [j for j in nb[c3] if isN[j]]
                for c1 in c1s:
                    for n4 in n4s:
                        o1 = carbonyl(c1)
                        x1 = [j for j in nb[c1] if j not in (n2, o1)]
                        x5 = [j for j in nb[n4] if j != c3]
                        if not x1 or not x5:
                            continue
                        w1 = dihedral(X[x1[0]], X[c1], X[n2], X[ca])  # omega(i-1)
                        w2 = dihedral(X[ca], X[c3], X[n4], X[x5[0]])  # omega(i)
                        if abs(w1) < 150 or abs(w2) < 150:
                            continue
                        heavy_ca = [j for j in nb[ca] if j not in (n2, c3)]
                        n2_extra = [j for j in nb[n2] if j not in (c1, ca)]
                        n4_extra = [j for j in nb[n4] if j != c3]
                        if len(heavy_ca) == 0:
                            cls = 'Gly'
                        elif len(heavy_ca) == 2:
                            cls = 'Calpha_disubstituted'
                        elif n2_extra:
                            ring = any(k in nb[cb] for k in n2_extra for cb in heavy_ca for _ in [0]) or \
                                   any(len(nb[k] & nb[cb]) > 0 for k in n2_extra for cb in heavy_ca)
                            cls = 'Pro' if ring else 'N_alkyl'
                        else:
                            cls = 'general'
                        phi = dihedral(X[c1], X[n2], X[ca], X[c3]); psi = dihedral(X[n2], X[ca], X[c3], X[n4])
                        if cls in ('general', 'Pro') and heavy_ca:
                            # chirality: (N-CA)x(C-CA).(CB-CA) > 0 for L (checked on ideal L-Ala, CB at phi - 120)
                            cb = heavy_ca[0]
                            chi = np.cross(X[n2] - X[ca], X[c3] - X[ca]) @ (X[cb] - X[ca])
                            if chi < 0:  # D configuration: mirror onto L
                                phi, psi = -phi, -psi
                        nextpro = len(n4_extra) >= 2
                        atoms = [c1, o1, n2, ca, c3, o3, n4]
                        out.append(dict(cls=cls, next_pro=bool(nextpro), region=region(phi, psi),
                                        clean=not bool(bad[atoms].any()), terminal_next=len(n4_extra) <= 1 and all(len(nb[k]) <= 1 for k in n4_extra)))
    return out


def scan(path):
    import gemmi
    try:
        text = Path(path).read_text(errors='replace')
    except Exception:
        return path, 'unreadable', []
    if not formula_ok(text):
        return path, 'formula', []
    m = RFAC.search(text)
    if not m or float(m.group(1)) > R_MAX:
        return path, 'rfactor', []
    try:
        doc = gemmi.cif.read_string(text)
        res = []
        for b in doc:
            els, X, bad = molecules_from_block(b)
            res += junctions(els, X, bad)
        return path, 'parsed', res
    except Exception:
        return path, 'parse_error', []


def main(a):
    files = sorted(str(p) for p in Path(a.cod).expanduser().rglob('*.cif'))
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    print(f'{len(files):,} CIF files', flush=True)
    st, J, entries = Counter(), Counter(), {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for k, (p, status, res) in enumerate(pool.map(scan, files, chunksize=200)):
            st[status] += 1
            for r in res:
                if not r['clean']:
                    J['excluded_disorder'] += 1; continue
                key = (r['cls'], 'xpro' if r['next_pro'] else 'nonxpro', r['region'])
                J[key] += 1; entries.setdefault(key, set()).add(Path(p).stem)
            if k % 50000 == 0 and k:
                print(f'  {k:,} files, {time.time() - t0:.0f} s', flush=True)
    rows = [dict(cls=c, next=x, region=g, junctions=n, entries=len(entries[(c, x, g)])) for (c, x, g), n in
            ((k, v) for k, v in J.items() if isinstance(k, tuple))]
    gen = {g: sum(r['junctions'] for r in rows if r['cls'] == 'general' and r['region'] == g) for g in ('alphaR', 'beta', 'ppII', 'left')}
    gen_e = {g: len(set().union(*[entries[k] for k in entries if k[0] == 'general' and k[2] == g])) if any(k[0] == 'general' and k[2] == g for k in entries) else 0
             for g in ('alphaR', 'beta', 'ppII', 'left')}
    rule = json.loads((Path(__file__).parent / 'COD_ROLE_RULE.json').read_text())
    ok = all(gen[g] >= rule['min_junctions'] and gen_e[g] >= rule['min_entries'] for g in rule['regions'])
    res = dict(files=len(files), file_status=dict(st), excluded_disorder=J['excluded_disorder'], counts=rows,
               general_by_region_junctions=gen, general_by_region_entries=gen_e, rule=rule,
               COD_ROLE='TRAINING_FOR_V4' if ok else 'REFEREE', note='counts only; no carbonyl angle was computed')
    (out / 'COD_PROBE.json').write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ('files', 'file_status', 'general_by_region_junctions', 'general_by_region_entries', 'COD_ROLE')}, indent=1))
    print('wrote', out / 'COD_PROBE.json')


def selftest():
    # Ac-Ala-NHMe heavy atoms built with ideal internal geometry; phi/psi set; check detection, region and mirror
    def place(a, b, c, bond, angle, tors):
        bc = c - b; bc /= np.linalg.norm(bc); nrm = np.cross(b - a, bc); nrm /= np.linalg.norm(nrm)
        m = np.array([bc, np.cross(nrm, bc), nrm]).T
        ang, t = math.radians(angle), math.radians(tors)
        d2 = np.array([-bond * math.cos(ang), bond * math.sin(ang) * math.cos(t), bond * math.sin(ang) * math.sin(t)])
        return c + m @ d2

    def build(phi, psi):
        CH3 = np.zeros(3); C1 = np.array([1.52, 0, 0]); ref = np.array([2.1, 1.05, 0])
        N2 = place(ref, CH3, C1, 1.33, 116, 180)
        CA = place(CH3, C1, N2, 1.46, 121, 180)
        O1 = place(CA, N2, C1, 1.23, 123, 0)
        C3 = place(C1, N2, CA, 1.52, 111, phi)
        CB = place(C1, N2, CA, 1.53, 110, phi - 120)  # L configuration
        N4 = place(N2, CA, C3, 1.33, 116, psi)
        O3 = place(N2, CA, C3, 1.23, 121, psi + 180)
        C5 = place(CA, C3, N4, 1.46, 121, 180)
        X = np.array([CH3, C1, O1, N2, CA, C3, CB, N4, O3, C5])
        assert abs(dihedral(C1, N2, CA, C3) - phi) < 1e-6 and abs(dihedral(N2, CA, C3, N4) - psi) < 1e-6
        return ['C', 'C', 'O', 'N', 'C', 'C', 'C', 'N', 'O', 'C'], X
    els, X = build(-60, -40)
    j = junctions(els, X, np.zeros(len(els), bool))
    assert len(j) == 1 and j[0]['cls'] == 'general' and not j[0]['next_pro'], j
    assert j[0]['region'] == 'alphaR', j
    jm = junctions(els, X * np.array([1, 1, -1]), np.zeros(len(els), bool))  # mirror image = D residue at (60, 40)
    assert jm[0]['region'] == 'alphaR', jm                                    # mapped back onto L
    els2, X2 = build(-120, 130); assert junctions(els2, X2, np.zeros(10, bool))[0]['region'] == 'beta'
    els3, X3 = build(-75, 145); assert junctions(els3, X3, np.zeros(10, bool))[0]['region'] == 'ppII'
    assert region(-60, -40) == 'alphaR' and region(-120, 130) == 'beta' and region(-75, 145) == 'ppII' and region(60, 40) == 'left'
    assert formula_ok("_chemical_formula_sum 'C6 H12 N2 O2'\n") and not formula_ok("_chemical_formula_sum 'C6 H12 N O2'\n")
    print('SELFTEST PASS: junction detection, trans-amide filter, D->L mirror, regions, formula prefilter')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--cod', default='~/cod/cif'); ap.add_argument('--out', default='~/cod_probe')
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    selftest() if a.selftest else main(a)
