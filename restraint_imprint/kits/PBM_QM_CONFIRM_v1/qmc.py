#!/usr/bin/env python
"""PBM_QM_CONFIRM_v1 (v1.1: secondary comparators CDL, 2018 CCP4 TRANS, AMBER ff14SB and all-term absolute errors; run code unchanged from v1.0.2) -- confirmatory QM referee, molecule B (ACE-X-Ala-NME), spec QM_CONFIRMATORY_SPEC_v1_0.md.

Commands (run through run.sh, which sets the environment):
  selftest                  pure-python checks (no QM)
  preflight                 tools, ddX options, chi1 start rotamers, pilot basis rule; writes PREFLIGHT.json
  build                     tleap geometries for all 160 frozen points (cheap); writes build/<case>/
  run --set timing          the 4 timing cases (ALA points 1-4) at the rule basis (and at def2-SVP too if the rule says TZVP)
  design --workers W        applies the frozen feasibility rule to the timing results; writes DESIGN.json (once)
  run --set production      every case of DESIGN.json (refuses without it)
  status                    progress
  analyse                   primary and secondary endpoints of the frozen spec (refuses without DESIGN.json)
"""
import argparse, fcntl, hashlib, json, os, re, shutil, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parent
SPEC = KIT / 'QM_CONFIRMATORY_SPEC_v1_0.md'
SPEC_SHA = '31edca32bea46f2d456c767296a65b242ea6123f59e6be1f259ed9217b7d0968'
POINTS = KIT / 'inputs' / 'QM_POINTS_v1_0.tsv'
POINTS_SHA = 'b8e2cedd27eba2a0f21b1799be3a997c1662d22c18f112bf87eefd870df54186'
V3_SHA, CHI1_SHA = '8404d010', '6210897c'
OUT_DEFAULT = Path.home() / 'pbm_qm_confirm'
PILOT_DEFAULT = Path.home() / 'pbm_qm_pilot'
ANGLES = {'angle_CaCN': 'angle_CaCN', 'angle_O_C_N': 'angle_O_C_N_next', 'angle_CA_C_O': 'angle_CA_C_O'}  # QM term -> library term
PILOT_POINTS = {'alpha': (-60, -40), 'beta': (-120, 130), 'ppii': (-75, 145), 'left': (60, 40)}
TWELVE = ['ALA', 'GLY', 'SER', 'LEU', 'VAL', 'ILE', 'THR', 'ASP', 'GLU', 'LYS', 'PHE', 'PRO']
LEAP_NAME = {'HIS': 'HIE'}            # neutral His, N-epsilon-H; Asp/Glu/Lys/Arg charged; Cys neutral
CHARGE = {'ASP': -1, 'GLU': -1, 'LYS': 1, 'ARG': 1}
CHI1_ATOM = {'SER': 'OG', 'THR': 'OG1', 'CYS': 'SG', 'VAL': 'CG1', 'ILE': 'CG1'}
NO_CHI1 = {'ALA', 'GLY', 'PRO'}
WELLS = {'g-': -60.0, 't': 180.0, 'g+': 60.0}
ALA_NEXT = (-120.0, 130.0)
BUDGET_DAYS, KMAX, KMIN = 28.0, 8, 4
DDX_BASE = {'ddx': 'true', 'ddx_model': 'pcm', 'ddx_solvent': 'water', 'ddx_radii_set': 'uff'}
DDX_TIGHT = [{'ddx_n_lebedev': '590', 'ddx_solvation_convergence': '1e-10'}, {'ddx_solvation_convergence': '1e-10'}, {}]
BOOT, SEED = 10000, 20261119
V2_SHA, CDL_SHA = 'ef7b86b3', '79c197e5'
LIB_TERM = {'angle_CaCN': 'angle_CaCN', 'angle_O_C_N': 'angle_O_C_N_next', 'angle_CA_C_O': 'angle_CA_C_O', 'tau_deg': 'tau_deg',
            'angle_CNCa': 'angle_CNCa', 'angle_N_CA_CB': 'angle_N_CA_CB', 'angle_C_CA_CB': 'angle_C_CA_CB', 'bond_N_CA': 'bond_N_CA',
            'bond_CA_C': 'bond_CA_C', 'bond_C_O': 'bond_C_O', 'bond_C_N_next': 'bond_C_N_next', 'bond_CA_CB': 'bond_CA_CB'}
CDL_COL = {'angle_CaCN': 12, 'angle_O_C_N': 14, 'angle_CA_C_O': 10}   # CDL v1.2 columns, as in the atomic-resolution referee
CCP4_TRANS_2018 = {'angle_CaCN': 115.917, 'angle_O_C_N': 123.469, 'angle_CA_C_O': 120.614}  # ledger N10/N10b (TRANS link; DEL-OXT Ca-C-O override)
AMBER_TYPES = {'angle_CaCN': ('CX', 'C', 'N'), 'angle_O_C_N': ('O', 'C', 'N'), 'angle_CA_C_O': ('CX', 'C', 'O')}



# ----------------------------------------------------------------------------- small helpers
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def wrap(d): return (d + 180.0) % 360.0 - 180.0


def dihedral(x, i, j, k, l):
    b0, b1, b2 = x[i] - x[j], x[k] - x[j], x[l] - x[k]
    b1n = b1 / np.linalg.norm(b1)
    v, w = b0 - np.dot(b0, b1n) * b1n, b2 - np.dot(b2, b1n) * b1n
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w))))


def angle(x, i, j, k):
    a, b = x[i] - x[j], x[k] - x[j]
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def dist(x, i, j): return float(np.linalg.norm(x[i] - x[j]))


def read_tsv(p):
    lines = Path(p).read_text().splitlines()
    head = lines[0].split('\t')
    return [dict(zip(head, l.split('\t'))) for l in lines[1:] if l.strip()]


def read_xyz(p):
    L = Path(p).read_text().splitlines(); n = int(L[0])
    sym = [l.split()[0] for l in L[2:2 + n]]
    x = np.array([[float(v) for v in l.split()[1:4]] for l in L[2:2 + n]])
    return sym, x, L[1]


def write_xyz(p, sym, x, comment=''):
    Path(p).write_text(f"{len(sym)}\n{comment}\n" + "\n".join(f"{s} {a:.10f} {b:.10f} {c:.10f}" for s, (a, b, c) in zip(sym, x)) + "\n")


def last_frame(traj, out):
    L = Path(traj).read_text().splitlines(); n = int(L[0]); size = n + 2
    if len(L) % size: raise ValueError('malformed trajectory')
    Path(out).write_text("\n".join(L[-size:]) + "\n"); return len(L) // size


def run_cmd(cmd, cwd, log, env, timeout=None):
    t = time.time()
    with open(log, 'w') as h:
        try:
            code = subprocess.run(cmd, cwd=cwd, stdout=h, stderr=subprocess.STDOUT, text=True, env=env, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = -9
    return code, time.time() - t


def thread_env(threads):
    e = os.environ.copy(); e.update({k: str(threads) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    return e


def case_id(res, order, basis): return f'{res}-{int(order)}-{basis}'


def load_points():
    if sha(POINTS) != POINTS_SHA: sys.exit('STOP: frozen points file changed')
    return read_tsv(POINTS)


def check_spec():
    if sha(SPEC) != SPEC_SHA: sys.exit('STOP: frozen spec changed')


# ----------------------------------------------------------------------------- periodic Catmull-Rom (library evaluation off-node)
def cr1(p0, p1, p2, p3, t):
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)


def cr2(grid, centers, phi, psi):
    """Tensor periodic Catmull-Rom on a 36 x 36 grid with node centres -175..175 (step 10); equals the cell value at a node."""
    def locate(a):
        u = (wrap(a) - centers[0]) / 10.0
        i = int(np.floor(u)); return i, u - i
    i, ti = locate(phi); j, tj = locate(psi)
    rows = []
    for di in (-1, 0, 1, 2):
        r = grid[(i + di) % 36]
        rows.append(cr1(r[(j - 1) % 36], r[j % 36], r[(j + 1) % 36], r[(j + 2) % 36], tj))
    return float(cr1(*rows, ti))


# ----------------------------------------------------------------------------- geometry building (tleap)
def leap_script(res, phi, psi, chi1):
    """tleap impose resolves "C" "N" "CA" "C" as C of the named residue + N, CA, C of the next one, so the phi of
    residue i is imposed on residue i-1; "N" "CA" "C" "N" on residue i is its psi. Upstream first."""
    name = LEAP_NAME.get(res, res)
    s = ['source leaprc.protein.ff14SB', f'm = sequence {{ ACE {name} ALA NME }}']
    if res != 'PRO':
        s.append(f'impose m {{ 1 }} {{ {{ "C" "N" "CA" "C" {phi:.3f} }} }}')                       # phi of X
    s.append(f'impose m {{ 2 }} {{ {{ "N" "CA" "C" "N" {psi:.3f} }} }}')                           # psi of X
    s.append(f'impose m {{ 2 }} {{ {{ "C" "N" "CA" "C" {ALA_NEXT[0]:.3f} }} }}')                    # phi of Ala
    s.append(f'impose m {{ 3 }} {{ {{ "N" "CA" "C" "N" {ALA_NEXT[1]:.3f} }} }}')                    # psi of Ala
    if chi1 is not None and res not in NO_CHI1:
        s.append(f'impose m {{ 2 }} {{ {{ "N" "CA" "CB" "{CHI1_ATOM.get(res, "CG")}" {chi1:.3f} }} }}')
    s += ['savepdb m built.pdb', 'quit']
    return '\n'.join(s) + '\n'


def parse_pdb(p):
    atoms = []
    for l in Path(p).read_text().splitlines():
        if l.startswith(('ATOM', 'HETATM')):
            name = l[12:16].strip(); el = re.sub(r'[^A-Za-z]', '', name)[0].upper()
            atoms.append(dict(name=name, resname=l[17:20].strip(), resseq=int(l[22:26]),
                              x=[float(l[30:38]), float(l[38:46]), float(l[46:54])], el=el))
    seqs = sorted({a['resseq'] for a in atoms})
    for a in atoms: a['res'] = seqs.index(a['resseq']) + 1   # 1 ACE, 2 X, 3 ALA, 4 NME
    return atoms


def atom_map(atoms, res):
    def ix(r, *names):
        for n in names:  # first name present wins (AmberTools versions name the NME methyl carbon CH3 or C)
            f = [k for k, a in enumerate(atoms) if a['res'] == r and a['name'] == n]
            if len(f) == 1: return f[0]
            if len(f) > 1: raise ValueError(f'expected one {r}:{n}, got {f}')
        raise ValueError(f'none of {names} in residue {r}: {[a["name"] for a in atoms if a["res"] == r]}')
    m = dict(C1=ix(1, 'C'), CH3_1=ix(1, 'CH3'), N2=ix(2, 'N'), CA2=ix(2, 'CA'), C2=ix(2, 'C'), O2=ix(2, 'O'),
             N3=ix(3, 'N'), CA3=ix(3, 'CA'), C3=ix(3, 'C'), N4=ix(4, 'N'), CH3_4=ix(4, 'CH3', 'C'))
    if res != 'GLY': m['CB2'] = ix(2, 'CB')
    if res not in NO_CHI1: m['G2'] = ix(2, CHI1_ATOM.get(res, 'CG'))
    return m


def dihedrals(m):
    return dict(phi_X=(m['C1'], m['N2'], m['CA2'], m['C2']), psi_X=(m['N2'], m['CA2'], m['C2'], m['N3']),
                phi_A=(m['C2'], m['N3'], m['CA3'], m['C3']), psi_A=(m['N3'], m['CA3'], m['C3'], m['N4']),
                omega_1=(m['CH3_1'], m['C1'], m['N2'], m['CA2']), omega_2=(m['CA2'], m['C2'], m['N3'], m['CA3']),
                omega_3=(m['CA3'], m['C3'], m['N4'], m['CH3_4']))


def terms(x, m):
    v = {'angle_CaCN': angle(x, m['CA2'], m['C2'], m['N3']), 'angle_O_C_N': angle(x, m['O2'], m['C2'], m['N3']),
         'angle_CA_C_O': angle(x, m['CA2'], m['C2'], m['O2']), 'tau_deg': angle(x, m['N2'], m['CA2'], m['C2']),
         'angle_CNCa': angle(x, m['C1'], m['N2'], m['CA2']), 'bond_N_CA': dist(x, m['N2'], m['CA2']),
         'bond_CA_C': dist(x, m['CA2'], m['C2']), 'bond_C_O': dist(x, m['C2'], m['O2']), 'bond_C_N_next': dist(x, m['C2'], m['N3'])}
    if 'CB2' in m:
        v.update(angle_N_CA_CB=angle(x, m['N2'], m['CA2'], m['CB2']), angle_C_CA_CB=angle(x, m['C2'], m['CA2'], m['CB2']),
                 bond_CA_CB=dist(x, m['CA2'], m['CB2']))
    if 'G2' in m: v['chi1'] = dihedral(x, m['N2'], m['CA2'], m['CB2'], m['G2'])
    return v


def bonds_of(x, el):
    out = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            lim = 2.15 if 'S' in (el[i], el[j]) else (1.25 if 'H' in (el[i], el[j]) else 1.80)
            if dist(x, i, j) < lim: out.append((i, j))
    return out


# ----------------------------------------------------------------------------- QM pieces
def psi4_input(sym, x, charge, basis, ddx_extra):
    geom = "\n".join(f"{s} {a:.10f} {b:.10f} {c:.10f}" for s, (a, b, c) in zip(sym, x))
    ddx = {**DDX_BASE, **ddx_extra}
    return (f"memory 4 GB\n\nmolecule peptide {{\n{charge} 1\n{geom}\nsymmetry c1\nno_com\nno_reorient\n}}\n\n"
            f"set {{\n    basis {basis}\n    scf_type df\n    reference rhf\n    e_convergence 1.0e-10\n    d_convergence 1.0e-10\n"
            f"    maxiter 300\n    dft_radial_points 99\n    dft_spherical_points 590\n}}\n"
            "set {\n" + "".join(f"    {k} {v}\n" for k, v in ddx.items()) + "}\n\ngradient('wb97x-d')\n")


def constraints_text(m, phi, psi):
    d = dihedrals(m)
    f = lambda t: ' '.join(str(i + 1) for i in d[t])
    return (f"$set\ndihedral {f('phi_X')} {phi:.8f}\ndihedral {f('psi_X')} {psi:.8f}\n"
            f"dihedral {f('phi_A')} {ALA_NEXT[0]:.8f}\ndihedral {f('psi_A')} {ALA_NEXT[1]:.8f}\n")


def geometric_opt(d, env, threads, restarts=2):
    """Constrained optimisation with at most 2 automatic restarts from the last frame (Q5 rule)."""
    for att in range(restarts + 1):
        code, sec = run_cmd(['geometric-optimize', '--engine', 'psi4', '--nt', str(threads), '--enforce', '0.01', '--conmethod', '1',
                             '--converge', 'set', 'GAU_TIGHT', '--maxiter', '500', 'psi4_opt.in', 'constraints.txt'],
                            d, d / f'geometric_attempt_{att}.log', env)
        log = (d / f'geometric_attempt_{att}.log').read_text(errors='replace'); traj = d / 'psi4_opt_optim.xyz'
        if code == 0 and 'Converged!' in log and traj.is_file(): return att, True
        if traj.is_file() and att < restarts:
            last_frame(traj, d / 'restart.xyz'); traj.rename(d / f'attempt_{att}_optim.xyz')
            sym, x, _ = read_xyz(d / 'restart.xyz')
            txt = (d / 'psi4_opt.in').read_text()
            new = "\n".join(f"{s} {a:.10f} {b:.10f} {c:.10f}" for s, (a, b, c) in zip(sym, x))
            txt = re.sub(r"(molecule peptide \{\n-?\d+ \d+\n)(.*?)(\nsymmetry c1)", lambda mm: mm.group(1) + new + mm.group(3), txt, flags=re.S)
            (d / 'psi4_opt.in').write_text(txt)
        else:
            return att, False
    return restarts, False


def dft_stage(d, sym, x, charge, basis, ddx, m, phi, psi, env, threads):
    d.mkdir(parents=True, exist_ok=True)
    (d / 'psi4_opt.in').write_text(psi4_input(sym, x, charge, basis, ddx))
    (d / 'constraints.txt').write_text(constraints_text(m, phi, psi))
    t0 = time.time(); att, ok = geometric_opt(d, env, threads)
    sec = time.time() - t0
    if not ok: return None, sec, att
    last_frame(d / 'psi4_opt_optim.xyz', d / 'final.xyz')
    return read_xyz(d / 'final.xyz')[1], sec, att


def run_case(res, order, basis, phi, psi, out, threads, pf):
    cid = case_id(res, order, basis); d = out / 'cases' / cid; d.mkdir(parents=True, exist_ok=True)
    st = dict(case_id=cid, residue=res, order=int(order), basis=basis, phi=phi, psi=psi, status='PENDING')
    env = thread_env(threads)
    try:
        b = json.loads((out / 'build' / f'{res}-{order}' / 'build.json').read_text())
        sym, x0, _ = read_xyz(out / 'build' / f'{res}-{order}' / 'built.xyz'); m = b['atom_map']; charge = b['charge']
        bonds0 = [tuple(t) for t in b['bonds']]
        # pre-optimisation (speed only)
        x_start, st['xtb'] = x0, 'not used'
        if pf.get('xtb'):
            xd = d / 'xtb'; xd.mkdir(exist_ok=True); write_xyz(xd / 'start.xyz', sym, x0)
            dd = dihedrals(m); tgt = dict(phi_X=phi, psi_X=psi, phi_A=ALA_NEXT[0], psi_A=ALA_NEXT[1])
            (xd / 'xcontrol.inp').write_text('$constrain\n force constant=1.0\n' + ''.join(
                f" dihedral: {','.join(str(i + 1) for i in dd[k])},{v:.4f}\n" for k, v in tgt.items()) + '$end\n')
            code, sec = run_cmd([pf['xtb'], 'start.xyz', '--opt', '--alpb', 'water', '--chrg', str(charge), '--input', 'xcontrol.inp'],
                                xd, xd / 'xtb.log', env, timeout=7200)
            st['xtb_seconds'] = round(sec, 1)
            if code == 0 and (xd / 'xtbopt.xyz').is_file():
                xx = read_xyz(xd / 'xtbopt.xyz')[1]
                if set(bonds_of(xx, b['elements'])) == set(bonds0): x_start, st['xtb'] = xx, 'used'
                else: st['xtb'] = 'connectivity changed; tleap start used'
            else: st['xtb'] = f'failed (code {code}); tleap start used'
        # constrained DFT, then the stationarity rule
        x1, s1, a1 = dft_stage(d / 'dft', sym, x_start, charge, basis, pf['ddx'], m, phi, psi, env, threads)
        st.update(dft_seconds=round(s1, 1), restarts_used=a1)
        if x1 is None: raise RuntimeError('OPT_FAILED')
        prev, final, stat_sec = x1, None, 0.0
        for k in (1, 2):
            xk, sk, _ = dft_stage(d / f'stationarity_{k}', sym, prev, charge, basis, pf['ddx'], m, phi, psi, env, threads)
            stat_sec += sk
            if xk is None: raise RuntimeError('STATIONARITY_OPT_FAILED')
            ta, tb = terms(prev, m), terms(xk, m)
            dang = max(abs(ta[t] - tb[t]) for t in ANGLES)
            dbond = max(abs(ta[t] - tb[t]) for t in ta if t.startswith('bond'))
            st[f'stationarity_{k}'] = dict(max_angle_change=round(dang, 5), max_bond_change=round(dbond, 6))
            if dang <= 0.02 and dbond <= 0.0005: final = xk; break
            prev = xk
        st['stationarity_seconds'] = round(stat_sec, 1)
        if final is None: raise RuntimeError('STATIONARITY_FAILED')
        # QC
        dd = dihedrals(m); got = {k: dihedral(final, *dd[k]) for k in dd}
        tgt = dict(phi_X=phi, psi_X=psi, phi_A=ALA_NEXT[0], psi_A=ALA_NEXT[1])
        cerr = max(abs(wrap(got[k] - v)) for k, v in tgt.items())
        if cerr > 0.5: raise RuntimeError(f'CONSTRAINT_FAILED {cerr:.3f}')
        if min(abs(got[k]) for k in ('omega_1', 'omega_2', 'omega_3')) < 150: raise RuntimeError('NOT_TRANS')
        if set(bonds_of(final, b['elements'])) != set(bonds0): raise RuntimeError('CONNECTIVITY_CHANGED')
        write_xyz(d / 'final.xyz', sym, final)
        tv = terms(final, m)
        st.update(status='PASS', terms=tv, dihedrals=got, constraint_error=cerr,
                  chi1_start=b.get('chi1_start'), chi1_left_start_rotamer=(None if 'chi1' not in tv or b.get('chi1_start') is None
                                                                            else bool(abs(wrap(tv['chi1'] - b['chi1_start'])) > 60)))
    except Exception as e:
        st.update(status='FAIL', failure=str(e))
    st['seconds_total'] = round(st.get('dft_seconds', 0) + st.get('stationarity_seconds', 0) + st.get('xtb_seconds', 0), 1)
    (d / 'status.json').write_text(json.dumps(st, indent=2) + '\n')
    return st


# ----------------------------------------------------------------------------- secondary comparators (v1.1)
def cdl_group(res):
    g = 'IleVal' if res in ('ILE', 'VAL') else 'Gly' if res == 'GLY' else 'Pro' if res == 'PRO' else 'NonPGIV'
    return g + '_nonxpro'   # the residue after X is Ala


_CDL = {}


def cdl_predict(path, res, phi, psi, qt):
    import runpy
    if str(path) not in _CDL: _CDL[str(path)] = runpy.run_path(str(path))['cdl_database']
    db = _CDL[str(path)]
    if getattr(db, 'version', None) != 'CDL v1.2': sys.exit('STOP: unexpected CDL version')
    grid = np.full((36, 36), np.nan)
    for (p, q), row in db[cdl_group(res)].items():
        if row[CDL_COL[qt]] > 0: grid[(p + 180) // 10, (q + 180) // 10] = row[CDL_COL[qt]]
    return cr2(grid, np.arange(-180, 180, 10).astype(float), phi, psi)


def amber_angles(amberhome):
    """Equilibrium angles from parm10.dat, overridden by frcmod.ff14SB (ANGLE sections); either atom order."""
    d = Path(amberhome) / 'dat' / 'leap' / 'parm'; found = {}
    for f in (d / 'parm10.dat', d / 'frcmod.ff14SB'):
        if not f.exists(): continue
        sec = None
        for line in f.read_text(errors='replace').splitlines():
            u = line.strip().upper()
            if u in ('BOND', 'ANGL', 'ANGLE', 'DIHE', 'IMPR', 'NONB', 'MASS'): sec = u[:4]; continue
            m = re.match(r'^(\S{1,2})\s*-\s*(\S{1,2})\s*-\s*(\S{1,2})\s+([-\d.]+)\s+([-\d.]+)', line)
            if m and (sec in (None, 'ANGL')) and f.name.startswith('frcmod') or (m and f.name == 'parm10.dat' and not re.match(r'^\S{1,2}\s*-\s*\S{1,2}\s*-\s*\S{1,2}\s*-', line)):
                a, b, c = (x.strip() for x in m.group(1, 2, 3))
                found[(a, b, c)] = found[(c, b, a)] = float(m.group(5))
    return {qt: found.get(t) for qt, t in AMBER_TYPES.items()}


def lib_value(z, lt, res, phi, psi, residues, centers):
    return cr2(z[f'{lt}__v2'][0, residues.index(res)], centers, phi, psi)


# ----------------------------------------------------------------------------- commands
def cmd_preflight(a):
    check_spec(); load_points(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pf = dict(kit_sha={p.name: sha(p) for p in sorted(KIT.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS.txt'},
              cores=os.cpu_count(), time=time.strftime('%Y-%m-%dT%H:%M:%S'))
    pf['tleap'] = shutil.which('tleap', path=os.environ.get('LEAP_PATH', '')) or a.tleap
    pf['xtb'] = shutil.which('xtb')
    for k in ('tleap',):
        if not pf[k] or not Path(pf[k]).exists(): sys.exit(f'STOP: {k} not found (set TLEAP=/path/to/tleap)')
    print('tleap', pf['tleap']); print('xtb', pf['xtb'] or 'NOT FOUND (allowed: DFT starts from the tleap geometry)')
    import psi4, geometric  # noqa
    pf['psi4'], pf['geometric'] = psi4.__version__, geometric.__version__
    # ddX settings: tightest candidate whose water-molecule gradient succeeds
    t = out / 'preflight_ddx'; t.mkdir(exist_ok=True)
    sym, x = ['O', 'H', 'H'], np.array([[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]])
    pf['ddx'] = None
    for i, extra in enumerate(DDX_TIGHT):
        dd = t / f'cand_{i}'; dd.mkdir(exist_ok=True)
        (dd / 'w.in').write_text(psi4_input(sym, x, 0, 'def2-svp', extra))
        code, _ = run_cmd(['psi4', '-n', '2', 'w.in', 'w.out'], dd, dd / 'run.log', thread_env(2), timeout=1800)
        outtxt = (dd / 'w.out').read_text(errors='replace') if (dd / 'w.out').exists() else ''
        ok = code == 0 and 'Total Gradient' in outtxt
        print(f'ddX candidate {extra or "pilot defaults"}: {"ok" if ok else "FAILED"}')
        if not ok:
            for l in (outtxt + (dd / 'run.log').read_text(errors='replace')).splitlines():
                if 'rror' in l or 'not' in l.lower() and 'option' in l.lower(): print('   ', l.strip()[:200])
        if ok: pf['ddx'] = extra; pf['ddx_tightened'] = bool(extra); break
    if pf['ddx'] is None: sys.exit('STOP: no ddX setting returned a gradient')
    # chi1 start rotamers from the registered chi1 model (training counts), residue level
    chi = find_by_sha(a.chi1_model, 'CHI1_MODEL.npz', CHI1_SHA)
    z = np.load(chi); classes = [str(c) for c in z['classes']]; residues = [str(r) for r in z['residues']]
    nkey = [k for k in z.files if k.endswith('_n')]
    cnt = sum(z[k][:, :20].reshape(3, 20, -1).sum(-1) for k in nkey)  # [class, residue], non-next-Pro
    pf['chi1_model'] = str(chi)
    pf['chi1_start'] = {r: dict(cls=classes[int(np.argmax(cnt[:, i]))], deg=WELLS[classes[int(np.argmax(cnt[:, i]))]],
                                counts={classes[c]: int(cnt[c, i]) for c in range(3)}) for i, r in enumerate(residues) if r not in NO_CHI1}
    # pilot basis rule (B1 - R0 shape SD on the four ALA cases) and slope-scale allowance
    pf['basis_rule'] = basis_rule(Path(a.pilot), find_by_sha(a.v3, 'MECHLIB_V3_TARGETS.npz', V3_SHA))
    print('basis rule:', json.dumps(pf['basis_rule']['decision']))
    (out / 'PREFLIGHT.json').write_text(json.dumps(pf, indent=2) + '\n'); print('PREFLIGHT PASS ->', out / 'PREFLIGHT.json')


def find_by_sha(given, name, prefix):
    cands = [Path(given)] if given else [p for r in (Path('/mnt/c/Wei/PBM'), Path.home()) if r.exists()
                                         for p in r.rglob(name) if p.is_file()]
    for p in cands:
        if p.is_file() and sha(p).startswith(prefix): return p
    sys.exit(f'STOP: {name} with sha {prefix}... not found (pass its path)')


def basis_rule(pilot, v3path):
    rows = {}
    for arm in ('R0', 'B1'):
        for g in PILOT_POINTS:
            p = pilot / arm / f'C-ALA-{g}' / 'status.json'
            s = json.loads(p.read_text()) if p.exists() else {}
            if s.get('status') != 'PASS': sys.exit(f'STOP: pilot {arm} C-ALA-{g} not PASS; the basis rule needs all four')
            rows[(arm, g)] = s['terms']
    z = np.load(v3path); res = [str(r) for r in z['residues']]; cen = z['node_centers'].astype(float); ia = res.index('ALA')
    out = dict(per_angle={}, decision=None)
    tz = False
    for qt, lt in ANGLES.items():
        dlt = np.array([rows[('B1', g)][qt] - rows[('R0', g)][qt] for g in PILOT_POINTS]); c = dlt - dlt.mean()
        shape_sd = float(np.sqrt((c ** 2).sum() / (len(c) - 1)))
        v = np.array([cr2(z[f'{lt}__v2'][0, ia], cen, *PILOT_POINTS[g]) for g in PILOT_POINTS]); vc = v - v.mean()
        delta_b = float(abs((c * vc).sum() / (vc * vc).sum()))
        out['per_angle'][qt] = dict(shape_sd_B1_minus_R0=round(shape_sd, 4), delta_b_slope=round(delta_b, 4))
        tz |= shape_sd > 0.10
    out['decision'] = dict(rule_basis='def2-tzvp' if tz else 'def2-svp', threshold_deg=0.10)
    return out


def cmd_build(a):
    check_spec(); out = Path(a.out); pf = json.loads((out / 'PREFLIGHT.json').read_text())
    n = 0
    for p in load_points():
        res, order, phi, psi = p['residue'], int(p['order']), float(p['phi']), float(p['psi'])
        d = out / 'build' / f'{res}-{order}'
        if (d / 'build.json').exists(): continue
        d.mkdir(parents=True, exist_ok=True)
        chi1 = pf['chi1_start'].get(res, {}).get('deg') if res not in NO_CHI1 else None
        (d / 'leap.in').write_text(leap_script(res, phi, psi, chi1))
        code, _ = run_cmd([pf['tleap'], '-f', 'leap.in'], d, d / 'leap.log', os.environ.copy(), timeout=600)
        if code != 0 or not (d / 'built.pdb').exists(): sys.exit(f'STOP: tleap failed for {res}-{order} (see {d}/leap.log)')
        atoms = parse_pdb(d / 'built.pdb'); m = atom_map(atoms, res)
        x = np.array([q['x'] for q in atoms]); sym = [q['el'] for q in atoms]
        dd = dihedrals(m); got = {k: dihedral(x, *dd[k]) for k in dd}
        off = [abs(wrap(got['psi_X'] - psi)), abs(wrap(got['phi_A'] - ALA_NEXT[0])), abs(wrap(got['psi_A'] - ALA_NEXT[1]))]
        if res != 'PRO': off.append(abs(wrap(got['phi_X'] - phi)))
        if max(off) > 1.0:
            sys.exit(f'STOP: built dihedrals off target for {res}-{order}: {got}')
        if min(abs(got[k]) for k in ('omega_1', 'omega_2', 'omega_3')) < 150: sys.exit(f'STOP: cis peptide built for {res}-{order}')
        write_xyz(d / 'built.xyz', sym, x, f'{res}-{order} tleap ff14SB')
        charge = CHARGE.get(res, 0)
        json.dump(dict(residue=res, order=order, phi=phi, psi=psi, charge=charge, atom_map=m, elements=sym,
                       bonds=bonds_of(x, sym), built_dihedrals=got, chi1_start=chi1, leap_name=LEAP_NAME.get(res, res)),
                  open(d / 'build.json', 'w'), indent=2)
        n += 1
    print(f'built {n} new geometries; total {len(list((out / "build").glob("*/build.json")))} of 160')


def case_list(a, out, pf):
    pts = {(p['residue'], int(p['order'])): p for p in load_points()}
    if a.set == 'timing':
        bases = [pf['basis_rule']['decision']['rule_basis']]
        if bases[0] == 'def2-tzvp': bases.append('def2-svp')
        return [(r, o, b) for b in bases for (r, o) in [('ALA', k) for k in range(1, 5)]], pts
    dz = json.loads((out / 'DESIGN.json').read_text()) if (out / 'DESIGN.json').exists() else sys.exit('STOP: run design first')
    res = TWELVE if dz['residues'] == 12 else sorted({r for r, _ in pts})
    return [(r, o, dz['basis']) for o in range(1, dz['k'] + 1) for r in res], pts   # point-order-major


def cmd_run(a):
    check_spec(); out = Path(a.out); pf = json.loads((out / 'PREFLIGHT.json').read_text())
    lock = open(out / 'RUN.lock', 'a+')
    try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: sys.exit('Another runner is using this output directory')
    todo, pts = case_list(a, out, pf)
    jobs = []
    for r, o, b in todo:
        sp = out / 'cases' / case_id(r, o, b) / 'status.json'
        if sp.exists() and json.loads(sp.read_text())['status'] in ('PASS', 'FAIL'): continue
        dcase = out / 'cases' / case_id(r, o, b)
        if dcase.exists(): dcase.rename(out / 'cases' / f'{case_id(r, o, b)}__INTERRUPTED_{int(time.time())}')
        jobs.append((r, o, b))
    print(f'{len(jobs)} cases to run ({a.workers} workers x {a.threads} threads); done cases kept', flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fut = {pool.submit(run_case, r, o, b, float(pts[(r, o)]['phi']), float(pts[(r, o)]['psi']), out, a.threads, pf): (r, o, b)
               for r, o, b in jobs}
        for f in as_completed(fut):
            s = f.result(); print(time.strftime('%H:%M'), s['case_id'], s['status'], s.get('failure', ''), f"{s['seconds_total']:.0f}s", flush=True)


def statuses(out):
    return [json.loads(p.read_text()) for p in sorted((out / 'cases').glob('*/status.json'))] if (out / 'cases').exists() else []


def cmd_status(a):
    out = Path(a.out); st = statuses(out)
    by = {}
    for s in st: by.setdefault(s['basis'], []).append(s)
    for b, v in by.items():
        sec = [s['seconds_total'] for s in v if s['status'] == 'PASS']
        print(f"{b}: {sum(s['status'] == 'PASS' for s in v)} pass, {sum(s['status'] == 'FAIL' for s in v)} fail"
              + (f"; mean {np.mean(sec) / 3600:.1f} h/case" if sec else ''))
    if (out / 'DESIGN.json').exists():
        dz = json.loads((out / 'DESIGN.json').read_text()); print('DESIGN:', dz['basis'], 'k', dz['k'], 'residues', dz['residues'])


def cmd_design(a):
    check_spec(); out = Path(a.out)
    if (out / 'DESIGN.json').exists(): sys.exit('DESIGN.json exists and is frozen')
    pf = json.loads((out / 'PREFLIGHT.json').read_text()); rule = pf['basis_rule']['decision']['rule_basis']
    st = [s for s in statuses(out) if s['residue'] == 'ALA' and s['order'] <= 4]
    per = {}
    for b in {s['basis'] for s in st}:
        v = [s for s in st if s['basis'] == b]
        if len(v) < 4: sys.exit(f'STOP: timing for {b} has {len(v)} of 4 cases accounted')
        ok = [s['seconds_total'] for s in v if s['status'] == 'PASS']
        if not ok: sys.exit(f'STOP: no timing case passed for {b}')
        per[b] = float(np.mean(ok))
    W = a.workers
    def days(b, k, R):
        done = sum(1 for s in st if s['basis'] == b and s['status'] == 'PASS')
        return max(R * k - done, 0) * per[b] / W / 86400
    ladder = [(rule, KMAX, 20)]
    if rule == 'def2-tzvp': ladder.append(('def2-svp', KMAX, 20))
    b2 = ladder[-1][0]
    ladder += [(b2, k, 20) for k in range(KMAX - 1, KMIN - 1, -1)] + [(b2, KMIN, 12)]
    chosen, log = None, []
    for b, k, R in ladder:
        dd = days(b, k, R); log.append(dict(basis=b, k=k, residues=R, projected_days=round(dd, 1)))
        if dd <= BUDGET_DAYS: chosen = (b, k, R); break
    if chosen is None:
        rec = dict(status='NOT_RUN_LOCALLY', ladder=log, seconds_per_case=per, workers=W)
        (out / 'DESIGN_NOT_FEASIBLE.json').write_text(json.dumps(rec, indent=2)); sys.exit('Feasibility rule step 4: request external compute')
    b, k, R = chosen
    dz = dict(basis=b, k=k, residues=R, workers=W, threads_per_worker=a.threads, seconds_per_case=per, ladder=log,
              basis_allowance=(pf['basis_rule']['per_angle'] if (rule == 'def2-tzvp' and b == 'def2-svp') else None),
              spec_sha=SPEC_SHA, points_sha=POINTS_SHA, time=time.strftime('%Y-%m-%dT%H:%M:%S'))
    (out / 'DESIGN.json').write_text(json.dumps(dz, indent=2) + '\n')
    print(json.dumps({x: dz[x] for x in ('basis', 'k', 'residues', 'workers', 'ladder')}, indent=1)); print('DESIGN sha', sha(out / 'DESIGN.json'))


def endpoint(qm, v3, v2, res, delta_b=0.0, B=BOOT, seed=SEED):
    """Primary per-angle statistics. qm, v3, v2: arrays over points; res: residue label per point."""
    from scipy import stats
    keep = [r for r in sorted(set(res)) if (res == r).sum() >= 2]
    sxy, sxx, q_all, v_all, sxy2, sxx2 = [], [], [], [], [], []
    for r in keep:
        i = res == r
        q, v, w = qm[i] - qm[i].mean(), v3[i] - v3[i].mean(), v2[i] - v2[i].mean()
        sxy.append((q * v).sum()); sxx.append((v * v).sum()); sxy2.append((q * w).sum()); sxx2.append((w * w).sum())
        q_all.append(q); v_all.append(v)
    sxy, sxx = np.array(sxy), np.array(sxx); G = len(keep)
    s = sxy.sum() / sxx.sum(); qa, va = np.concatenate(q_all), np.concatenate(v_all)
    r = float(np.corrcoef(qa, va)[0, 1])
    rng = np.random.default_rng(seed); W = rng.multinomial(G, np.ones(G) / G, size=B)
    se = float(((W @ sxy) / (W @ sxx)).std(ddof=1)); tcrit = float(stats.t.ppf(0.975, G - 1))
    def p_for(c):
        t = max(abs(s - c) - delta_b, 0.0) / se
        return float(2 * stats.t.sf(t, G - 1))
    return dict(G=G, n_points=int(len(qa)), slope=float(s), se=se, t_crit=tcrit,
                interval=[float(s - tcrit * se - delta_b), float(s + tcrit * se + delta_b)], r=r,
                slope_on_v2=float(np.sum(sxy2) / np.sum(sxx2)), p=dict(c1=p_for(1.0)), delta_b=delta_b, _p_for=p_for)


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj


def decide(eps, rho):
    """eps: {angle: endpoint dict or None (gated)}; rho: {angle: value}. Holm over the three angles for each c in {rho, 1}."""
    names = list(eps); res = {}
    p_rho = np.array([1.0 if eps[n] is None else eps[n]['_p_for'](rho[n]) for n in names])
    p_one = np.array([1.0 if eps[n] is None else eps[n]['_p_for'](1.0) for n in names])
    a_rho, a_one = holm(p_rho), holm(p_one)
    for i, n in enumerate(names):
        if eps[n] is None: res[n] = 'pattern not reproduced by QM'; continue
        inc1, incr = p_one[i] >= 0.05, p_rho[i] >= 0.05
        exc1, excr = a_one[i] < 0.05, a_rho[i] < 0.05
        res[n] = 'v3 supported' if (inc1 and excr) else ('v2 supported' if (incr and exc1) else 'inconclusive')
        eps[n]['p'] = dict(rho=float(p_rho[i]), one=float(p_one[i]), rho_holm=float(a_rho[i]), one_holm=float(a_one[i]))
    return res


def cmd_analyse(a):
    check_spec(); out = Path(a.out)
    dz = json.loads((out / 'DESIGN.json').read_text())
    pts = {(p['residue'], int(p['order'])): p for p in load_points()}
    residues = TWELVE if dz['residues'] == 12 else sorted({r for r, _ in pts})
    design = [(r, o) for r in residues for o in range(1, dz['k'] + 1)]
    st = {(s['residue'], s['order']): s for s in statuses(out) if s['basis'] == dz['basis']}
    missing = [c for c in design if c not in st]
    if missing: sys.exit(f'STOP: {len(missing)} design cases not run yet')
    passed = [c for c in design if st[c]['status'] == 'PASS']
    frac = len(passed) / len(design)
    rho = {}
    rows = [x for x in read_tsv(KIT / 'inputs' / 'QM_POINTS_RHO_v1_0.tsv') if int(x['k']) == dz['k']]
    for qt, lt in ANGLES.items(): rho[qt] = float([x for x in rows if x['term'] == lt][0]['rho'])
    resv = np.array([c[0] for c in passed]); eps, sec = {}, {}
    for qt, lt in ANGLES.items():
        qm = np.array([st[c]['terms'][qt] for c in passed]); v3 = np.array([float(pts[c][f'v3_{lt}']) for c in passed])
        v2 = np.array([float(pts[c][f'v2_{lt}']) for c in passed])
        db = dz['basis_allowance'][qt]['delta_b_slope'] if dz.get('basis_allowance') else 0.0
        e = endpoint(qm, v3, v2, resv, db)
        def cmse(pred):
            q = np.concatenate([qm[resv == r] - qm[resv == r].mean() for r in set(resv)])
            p = np.concatenate([pred[resv == r] - pred[resv == r].mean() for r in set(resv)])
            return float(np.mean((q - p) ** 2))
        sec[qt] = dict(centred_mse=dict(v3=cmse(v3), v2=cmse(v2), flat=cmse(np.zeros_like(v3))), slope_on_v2=e['slope_on_v2'], r=e['r'])
        eps[qt] = None if e['r'] < 0.3 else e
        sec[qt]['gate_r'] = e['r']; sec[qt]['endpoint'] = {k: v for k, v in e.items() if not k.startswith('_')}
    verdict = decide(eps, rho)
    # ---- secondary comparators (v1.1): all-term absolute errors for v2/v3; CDL, CCP4 2018 TRANS, AMBER for the three angles
    zv2 = np.load(find_by_sha(a.v2, 'training_targets.npz', V2_SHA)); zv3 = np.load(find_by_sha(a.v3, 'MECHLIB_V3_TARGETS.npz', V3_SHA))
    lres = [str(r) for r in zv3['residues']]; cen = zv3['node_centers'].astype(float)
    cdlp = find_by_sha(a.cdl, 'cdl_database.py', CDL_SHA)
    amb = amber_angles(os.environ.get('AMBERHOME', str(Path.home() / 'miniconda3/envs/amber')))
    allterms = {}
    for qt, lt in LIB_TERM.items():
        cs = [c for c in passed if qt in st[c]['terms']]
        if not cs: continue
        q = np.array([st[c]['terms'][qt] for c in cs])
        row = {}
        for L, z in (('v2', zv2), ('v3', zv3)):
            pr = np.array([lib_value(z, lt, c[0], float(pts[c]['phi']), float(pts[c]['psi']), lres, cen) for c in cs])
            row[L] = dict(mae=float(np.mean(np.abs(q - pr))), mean_signed=float(np.mean(pr - q)), n=len(cs))
        allterms[qt] = row
    comp = {}
    for qt in ANGLES:
        q = np.array([st[c]['terms'][qt] for c in passed])
        cd = np.array([cdl_predict(cdlp, c[0], float(pts[c]['phi']), float(pts[c]['psi']), qt) for c in passed])
        ok = np.isfinite(cd)
        qc = np.concatenate([q[(resv == r) & ok] - q[(resv == r) & ok].mean() for r in set(resv) if ((resv == r) & ok).sum() >= 2])
        cc = np.concatenate([cd[(resv == r) & ok] - cd[(resv == r) & ok].mean() for r in set(resv) if ((resv == r) & ok).sum() >= 2])
        comp[qt] = dict(CDL=dict(centred_mse=float(np.mean((qc - cc) ** 2)), slope_on_CDL=float((qc * cc).sum() / (cc * cc).sum()),
                                 r=float(np.corrcoef(qc, cc)[0, 1]), mae=float(np.mean(np.abs(q[ok] - cd[ok]))), n=int(ok.sum())),
                        CCP4_TRANS_2018=dict(level=CCP4_TRANS_2018[qt], mae=float(np.mean(np.abs(q - CCP4_TRANS_2018[qt]))),
                                             mean_signed=float(CCP4_TRANS_2018[qt] - q.mean())),
                        AMBER_ff14SB=(None if amb[qt] is None else dict(level=amb[qt], mae=float(np.mean(np.abs(q - amb[qt]))),
                                                                       mean_signed=float(amb[qt] - q.mean()))))
    res = dict(spec_sha=SPEC_SHA, design=dz, cases=len(design), passed=len(passed), qc_fraction=frac,
               panel_complete=frac >= 0.90, rho=rho, verdict=verdict, secondary=sec,
               primary={n: (None if e is None else {k: v for k, v in e.items() if not k.startswith('_')}) for n, e in eps.items()},
               secondary_all_terms=allterms, secondary_comparators=comp, amber_types=AMBER_TYPES)
    (out / 'QM_CONFIRM_RESULT.json').write_text(json.dumps(res, indent=2, default=float) + '\n')
    print(json.dumps(dict(passed=f'{len(passed)}/{len(design)}', panel_complete=frac >= 0.9, verdict=verdict), indent=1))


def selftest():
    # geometry
    x = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [-1, 1, 0.0]])
    assert abs(abs(dihedral(x, 0, 1, 2, 3)) - 180) < 1e-9 and abs(angle(x, 0, 1, 2) - 90) < 1e-9
    # Catmull-Rom: node values reproduced; midpoint formula
    cen = np.arange(-175, 176, 10).astype(float); g = np.random.default_rng(1).normal(size=(36, 36))
    assert abs(cr2(g, cen, -65, -25) - g[11, 15]) < 1e-12
    mid = cr2(g, cen, -60, -25); ref = (-g[10, 15] + 9 * g[11, 15] + 9 * g[12, 15] - g[13, 15]) / 16
    assert abs(mid - ref) < 1e-12
    # psi4 input carries grid, convergence and ddX
    t = psi4_input(['O', 'H', 'H'], np.zeros((3, 3)), 0, 'def2-svp', DDX_TIGHT[0])
    assert 'dft_spherical_points 590' in t and 'e_convergence 1.0e-10' in t and 'ddx_solvent water' in t and 'ddx_n_lebedev 590' in t
    # endpoint and decision on synthetic data
    rng = np.random.default_rng(3); res = np.repeat([f'R{i}' for i in range(20)], 8); v3 = rng.normal(0, 0.8, 160)
    for s_true, want in ((1.0, 'v3 supported'), (0.6, 'v2 supported')):
        qm = s_true * v3 + rng.normal(0, 0.15, 160)
        eps = {n: endpoint(qm, v3, 0.6 * v3, res, B=2000) for n in ANGLES}
        out = decide(eps, {n: 0.6 for n in ANGLES})
        assert all(v == want for v in out.values()), (s_true, out)
    eps = {n: None for n in ANGLES}; assert set(decide(eps, {n: 0.6 for n in ANGLES}).values()) == {'pattern not reproduced by QM'}
    assert list(holm(np.array([0.01, 0.04, 0.03]))) == [0.03, 0.06, 0.06]
    # leap script
    s = leap_script('SER', -65, -25, -60.0); assert 'impose m { 2 } { { "N" "CA" "CB" "OG" -60.000 } }' in s and 'ACE SER ALA NME' in s
    assert 'impose m { 1 }' in s and 'impose m { 1 }' not in leap_script('PRO', -65, 145, None)
    assert 'impose m { 2 } { { "C" "N" "CA" "C" -120.000 } }' in s and 'impose m { 3 } { { "N" "CA" "C" "N" 130.000 } }' in s
    assert cdl_group('ILE') == 'IleVal_nonxpro' and cdl_group('SER') == 'NonPGIV_nonxpro' and cdl_group('GLY') == 'Gly_nonxpro'
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        pd_ = Path(td) / 'dat' / 'leap' / 'parm'; pd_.mkdir(parents=True)
        (pd_ / 'parm10.dat').write_text('title\nC  12.01\n\nC -O    570.0    1.229\n\nCX-C -N    70.0      116.60    note\nO -C -N    80.0      122.90\nCX-C -O    80.0      120.40\n')
        (pd_ / 'frcmod.ff14SB').write_text('ff14SB\nANGLE\nCX-C -N    70.0      116.70\n\nDIHE\nX -C -N -X    4   10.00   180.0   2.\n')
        am = amber_angles(td)
        assert am == {'angle_CaCN': 116.7, 'angle_O_C_N': 122.9, 'angle_CA_C_O': 120.4}, am
    print('SELFTEST PASS: geometry, Catmull-Rom, psi4 input, endpoint/decision/Holm, tleap script, CDL groups, AMBER angle parsing')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['selftest', 'preflight', 'build', 'run', 'design', 'status', 'analyse'])
    ap.add_argument('--out', default=str(OUT_DEFAULT)); ap.add_argument('--pilot', default=str(PILOT_DEFAULT))
    ap.add_argument('--v3', default=None); ap.add_argument('--chi1-model', default=None); ap.add_argument('--tleap', default=None)
    ap.add_argument('--v2', default=None); ap.add_argument('--cdl', default=None)
    ap.add_argument('--set', choices=['timing', 'production'], default='timing')
    ap.add_argument('--workers', type=int, default=2); ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    if a.command == 'selftest': return selftest()
    {'preflight': cmd_preflight, 'build': cmd_build, 'run': cmd_run, 'design': cmd_design, 'status': cmd_status, 'analyse': cmd_analyse}[a.command](a)


if __name__ == '__main__':
    main()
