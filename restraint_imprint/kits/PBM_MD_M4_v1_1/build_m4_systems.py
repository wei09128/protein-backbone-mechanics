#!/usr/bin/env python3
"""build_m4_systems.py -- add a MechLib arm (merged per-term forces, next-Pro split) to an existing AMBER_FIXED system.

The AMBER_FIXED system, prepared.pdb and protein.pdb are the M3 builds, reused unchanged (hash-checked by the runner).
For each of the 11 backbone terms of the M2-B builder, one CustomCompoundBondForce is added:
  E = lambda*0.5*k*((q - mechlib_eq3(phi,psi,g))^2 - (q - amber_eq)^2)
with k and amber_eq taken from the AMBER term the correction replaces (as in M2-B/M3), and g = residue index
(20 standard residues, alphabetical) + 20 if the next residue is proline (MechLib's next-Pro split).
Surfaces come from a MechLib target file (npz, arrays term__v2 of shape (2, 20, 36, 36) = [next-Pro, residue, phi, psi]
at bin centres -175..175 deg), evaluated by OpenMM's periodic cubic spline over phi, psi (the M3 approach) and exactly
at integer g. chi1 conditioning is not represented (as in M3).

  python build_m4_systems.py --case-dir BUILD/CASE --targets FILE.npz --arm MECHLIB_V2 [--key-suffix __v2]
Writes CASE/<ARM>_system.xml and CASE/<ARM>_AUDIT.json; exits non-zero unless every check passes.
"""
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np

RES20 = sorted(['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'LEU', 'LYS', 'MET', 'PHE', 'SER', 'THR',
                'TRP', 'TYR', 'ILE', 'VAL', 'GLY', 'PRO'])
ANGLE_TERMS = {"tau_deg": ("N", "CA", "C"), "angle_N_CA_CB": ("N", "CA", "CB"), "angle_C_CA_CB": ("C", "CA", "CB"),
               "angle_CaCN": ("CA", "C", "N+"), "angle_CNCa": ("C-", "N", "CA"), "angle_CA_C_O": ("CA", "C", "O")}
BOND_TERMS = {"bond_N_CA": ("N", "CA"), "bond_CA_C": ("CA", "C"), "bond_C_O": ("C", "O"),
              "bond_C_N_next": ("C", "N+"), "bond_CA_CB": ("CA", "CB")}
COORD = {"tau_deg": "angle(p2,p3,p4)", "angle_N_CA_CB": "angle(p2,p3,p6)", "angle_C_CA_CB": "angle(p4,p3,p6)",
         "angle_CaCN": "angle(p3,p4,p5)", "angle_CNCa": "angle(p1,p2,p3)", "angle_CA_C_O": "angle(p3,p4,p6)",
         "bond_N_CA": "distance(p2,p3)", "bond_CA_C": "distance(p3,p4)", "bond_C_O": "distance(p4,p6)",
         "bond_C_N_next": "distance(p4,p5)", "bond_CA_CB": "distance(p3,p6)"}
B = 36


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def maps(system, openmm, unit):
    bonds, angles = {}, {}
    for f in system.getForces():
        if isinstance(f, openmm.HarmonicBondForce):
            for i in range(f.getNumBonds()):
                a, b, l, k = f.getBondParameters(i)
                bonds[tuple(sorted((int(a), int(b))))] = (l.value_in_unit(unit.nanometer),
                                                          k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer ** 2))
        elif isinstance(f, openmm.HarmonicAngleForce):
            for i in range(f.getNumAngles()):
                a, b, c, th, k = f.getAngleParameters(i)
                key = min((int(a), int(b), int(c)), (int(c), int(b), int(a)))
                angles[key] = (th.value_in_unit(unit.radian), k.value_in_unit(unit.kilojoule_per_mole / unit.radian ** 2))
    return bonds, angles


def atom(residues, ri, tok):
    off = 1 if tok.endswith('+') else -1 if tok.endswith('-') else 0
    name = tok.rstrip('+-'); t = ri + off
    if t < 0 or t >= len(residues):
        return None
    return {a.name: a.index for a in residues[t].atoms()}.get(name)


def grid_values(surf):
    """surf[g, i(phi), j(psi)] -> Continuous3DFunction values (x fastest, then y, then z), periodic copies appended."""
    G = surf.shape[0]
    vals = []
    for z in list(range(G)) + [0]:
        for j in range(B + 1):
            for i in range(B + 1):
                vals.append(float(surf[z, i % B, j % B]))
    return vals


def catmull(G, g, x, y, x0=-175.0):
    u, v = (x - x0) / 10.0, (y - x0) / 10.0
    i0, j0 = int(np.floor(u)), int(np.floor(v)); tx, ty = u - i0, v - j0

    def w(t):
        return [0.5 * (-t ** 3 + 2 * t * t - t), 0.5 * (3 * t ** 3 - 5 * t * t + 2), 0.5 * (-3 * t ** 3 + 4 * t * t + t), 0.5 * (t ** 3 - t * t)]
    wx, wy = w(tx), w(ty)
    return sum(wx[a] * wy[b] * G[g, (i0 + a - 1) % B, (j0 + b - 1) % B] for a in range(4) for b in range(4))


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1); v = b0 - (b0 @ b1) * b1; w = b2 - (b2 @ b1) * b1
    return math.degrees(math.atan2(np.cross(b1, v) @ w, v @ w))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case-dir', type=Path, required=True)
    ap.add_argument('--targets', type=Path, required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--key-suffix', default='__v2')
    a = ap.parse_args()
    import openmm
    from openmm import app, unit
    base_xml = (a.case_dir / 'AMBER_FIXED_system.xml').read_text()
    system = openmm.XmlSerializer.deserialize(base_xml)
    pdb = app.PDBFile(str(a.case_dir / 'prepared.pdb'))
    with np.load(a.targets, allow_pickle=False) as z:
        if [str(r) for r in z['residues']] != RES20:
            sys.exit('STOP: residue axis differs')
        surf = {t: z[t + a.key_suffix].reshape(40, B, B).astype(float) for t in COORD}
    for t in ANGLE_TERMS:
        if not (100 < np.median(surf[t]) < 130):
            sys.exit(f'STOP: {t} median {np.median(surf[t]):.2f} deg is implausible')
    for t in BOND_TERMS:
        if not (1.2 < np.median(surf[t]) < 1.6):
            sys.exit(f'STOP: {t} median {np.median(surf[t]):.3f} A is implausible')
    residues = list(pdb.topology.residues())
    bonds, angles = maps(system, openmm, unit)
    shift = math.radians(5.0)
    phi = f"atan2(sin(dihedral(p1,p2,p3,p4)-{shift:.17g}),cos(dihedral(p1,p2,p3,p4)-{shift:.17g}))"
    psi = f"atan2(sin(dihedral(p2,p3,p4,p5)-{shift:.17g}),cos(dihedral(p2,p3,p4,p5)-{shift:.17g}))"
    forces, regs, skipped = {}, [], {}
    for term in COORD:
        conv = math.radians if term in ANGLE_TERMS else (lambda v: 0.1 * v)
        vals = [conv(v) for v in grid_values(surf[term])]
        expr = f"lambda*0.5*k*((({COORD[term]})-mechlib_eq3({phi},{psi},gidx))^2-(({COORD[term]})-amber_eq)^2)"
        f = openmm.CustomCompoundBondForce(6 if COORD[term].count('p6') else 5, expr)
        f.addGlobalParameter('lambda', 1.0)
        for p in ('k', 'amber_eq', 'gidx'):
            f.addPerBondParameter(p)
        f.addTabulatedFunction('mechlib_eq3', openmm.Continuous3DFunction(B + 1, B + 1, 41, vals, -math.pi, math.pi, -math.pi, math.pi,
                                                                           0.0, 40.0, True))
        f.setName(f'MechLib {a.arm} {term}')
        forces[term] = f
    for ri in range(1, len(residues) - 1):
        res = residues[ri]
        if res.name not in RES20:
            continue
        nxt = residues[ri + 1].name
        bb = [atom(residues, ri - 1, 'C'), atom(residues, ri, 'N'), atom(residues, ri, 'CA'), atom(residues, ri, 'C'), atom(residues, ri + 1, 'N')]
        if any(i is None for i in bb) or nxt not in RES20:
            skipped['missing_backbone_context'] = skipped.get('missing_backbone_context', 0) + 1
            continue
        g = RES20.index(res.name) + 20 * (nxt == 'PRO')
        for term, toks in {**ANGLE_TERMS, **BOND_TERMS}.items():
            tgt = [atom(residues, ri, t) for t in toks]
            if any(i is None for i in tgt):
                skipped[f'atom_absent:{term}'] = skipped.get(f'atom_absent:{term}', 0) + 1
                continue
            if term in ANGLE_TERMS:
                key = min(tuple(tgt), tuple(reversed(tgt)))
                if key not in angles:
                    skipped[f'amber_absent:{term}'] = skipped.get(f'amber_absent:{term}', 0) + 1; continue
                eq, k = angles[key]
            else:
                key = tuple(sorted(tgt))
                if key not in bonds:
                    skipped[f'amber_absent:{term}'] = skipped.get(f'amber_absent:{term}', 0) + 1; continue
                eq, k = bonds[key]
            extras = [i for i in tgt if i not in bb]
            parts = bb + extras
            if len(parts) != forces[term].getNumParticlesPerBond():
                sys.exit(f'STOP: particle count mismatch for {term}')
            forces[term].addBond(parts, [k, eq, float(g)])
            regs.append(dict(residue_index=ri + 1, residue=res.name, next=nxt, group=g, term=term, particles=parts, amber_eq=eq, k=k))
    for i, (term, f) in enumerate(forces.items()):
        f.setForceGroup(1 + i)
        system.addForce(f)
    xml = openmm.XmlSerializer.serialize(system)
    restored = openmm.XmlSerializer.deserialize(xml)

    def ef(sys_, lam=None, groups=-1):
        integ = openmm.VerletIntegrator(0.001)
        ctx = openmm.Context(sys_, integ, openmm.Platform.getPlatformByName('Reference'))
        ctx.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors()); ctx.setPositions(pdb.positions)
        if lam is not None:
            ctx.setParameter('lambda', lam)
        st = ctx.getState(getEnergy=True, getForces=True, groups=groups)
        e = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        F = np.asarray(st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
        del ctx, integ
        return e, F
    base = openmm.XmlSerializer.deserialize(base_xml)
    e0, f0 = ef(base); ez, fz = ef(system, 0.0); e1, f1 = ef(system, 1.0); er, fr = ef(restored, 1.0)
    # orientation check: per-term MechLib energy from OpenMM vs a Catmull-Rom re-evaluation in python
    X = np.array(pdb.positions.value_in_unit(unit.nanometer))
    orient = {}
    for i, term in enumerate(COORD):
        e_omm, _ = ef(system, 1.0, groups={1 + i})
        e_py = 0.0; e_abs = 0.0
        for r in (r for r in regs if r['term'] == term):
            p = X[r['particles']]
            ph = dihedral(p[0], p[1], p[2], p[3]); ps = dihedral(p[1], p[2], p[3], p[4])
            s = catmull(surf[term], r['group'], ph, ps)
            if term in ANGLE_TERMS:
                idx = {'angle(p2,p3,p4)': (1, 2, 3), 'angle(p2,p3,p6)': (1, 2, 5), 'angle(p4,p3,p6)': (3, 2, 5), 'angle(p3,p4,p5)': (2, 3, 4),
                       'angle(p1,p2,p3)': (0, 1, 2), 'angle(p3,p4,p6)': (2, 3, 5)}[COORD[term]]
                u, v = p[idx[0]] - p[idx[1]], p[idx[2]] - p[idx[1]]
                q = math.acos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)); s = math.radians(s)
            else:
                idx = {'distance(p2,p3)': (1, 2), 'distance(p3,p4)': (2, 3), 'distance(p4,p6)': (3, 5), 'distance(p4,p5)': (3, 4),
                       'distance(p3,p6)': (2, 5)}[COORD[term]]
                q = float(np.linalg.norm(p[idx[0]] - p[idx[1]])); s = 0.1 * s
            ei = 0.5 * r['k'] * ((q - s) ** 2 - (q - r['amber_eq']) ** 2); e_py += ei; e_abs += abs(ei)
        orient[term] = dict(openmm=e_omm, python_catmull=e_py, n=sum(r['term'] == term for r in regs),
                            sum_abs_terms=e_abs, rel_diff=abs(e_omm - e_py) / max(1.0, e_abs))
    cond = {'terms_registered': len(regs) > 0, 'lambda0_energy_identity': abs(ez - e0) < 1e-6,
            'lambda0_force_identity': float(np.abs(f0 - fz).max()) < 1e-6, 'lambda1_finite': math.isfinite(e1),
            'serialization_identity': abs(er - e1) < 1e-6 and float(np.abs(f1 - fr).max()) < 1e-6,
            'orientation_check': all(v['rel_diff'] < 0.05 for v in orient.values())}
    audit = dict(status='PASS' if all(cond.values()) else 'FAIL', arm=a.arm, targets=str(a.targets), targets_sha256=sha256(a.targets),
                 base_system_sha256=sha256(a.case_dir / 'AMBER_FIXED_system.xml'), registered=len(regs),
                 by_term={t: sum(r['term'] == t for r in regs) for t in COORD}, next_pro_residues=sum(1 for r in regs if r['group'] >= 20 and r['term'] == 'tau_deg'),
                 skipped=skipped, conditions=cond, orientation=orient,
                 energies=dict(base=e0, lambda0=ez, lambda1=e1), openmm_version=openmm.__version__)
    (a.case_dir / f'{a.arm}_system.xml').write_text(xml)
    (a.case_dir / f'{a.arm}_AUDIT.json').write_text(json.dumps(audit, indent=1))
    print(json.dumps({k: audit[k] for k in ('status', 'arm', 'registered', 'next_pro_residues', 'conditions')}, indent=1))
    print('orientation:', {t: round(v['rel_diff'], 4) for t, v in orient.items()})
    return 0 if audit['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
