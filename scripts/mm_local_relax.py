#!/usr/bin/env python3
"""
Representative MM local-relaxation validation (Reviewer 1's suggestion).

For each candidate residue (from a candidates CSV -- either scan_pdb_cache.py's
output, or your own features_lj_v3_FINAL_CLEAN.csv filtered/renamed to match
the required columns: pdb, chain, resnum, basin, klass), this:

  1. Loads the real crystal structure once per PDB file, adds hydrogens
     (PDBFixer), builds the base AMBER system, and -- if --mechlib is given --
     applies the MechLib overlay ONCE for that file.
  2. For each candidate residue from that file, adds a positional restraint
     (every heavy atom pinned to its crystallographic position EXCEPT
     every atom belonging to the target residue -- backbone N/CA/C/O,
     Cbeta, full sidechain, and their hydrogens, all freed together)
     plus a soft phi/psi dihedral restraint, minimizes, measures tau /
     N-Ca-Cb / C-Ca-Cb, then removes those two forces so the cached base
     system is clean for the next residue. Freeing the WHOLE residue
     (not just N/CA/C/CB) is deliberate: pinning O in place while Ca/C
     move around it would artificially strain the peptide plane, since
     O is rigidly part of the same amide unit as the rest of the
     residue's backbone.

Why this matters for runtime: PDBFixer + forcefield.createSystem() are the
expensive steps (~9-10s each on a ~400-residue protein in testing) and were
previously being redone for every candidate row, even when many rows come
from the same file. Caching per file turns "9s + 4s" per residue into
"9s once, then ~4s per residue" -- for files with N candidates, that's
roughly a (9+4N)/(13N) speedup, i.e. bigger gains the more residues you
pull per file. Minimization itself converges by ~150-200 iterations in
testing; MAX_ITER=200 below reflects that (raise it back to 500 if you
see non-converged results for some residue).

What this experiment does and doesn't test: with everything outside the
target residue pinned near-rigidly (K_REST) and its own phi/psi also
softly restrained near the crystallographic value, essentially all
freedom to relax gets funneled into tau, N-Ca-Cb/C-Ca-Cb, and sidechain
(chi1+) geometry. That's the intended design -- it isolates "what does
this one residue want to do given its real crystallographic environment
held fixed" (Reviewer 1's requested local-relaxation experiment), not
"what does the whole structure relax to at equilibrium." Don't read
results from this as a full free-energy-minimized structure.

Usage:
    python mm_local_relax.py candidates.csv /path/to/pdb_cache --out results.csv

To use YOUR corrected MechLib geometry library instead of stock ff14SB,
pass --mechlib /path/to/constants_library.json. This applies the SAME
apply_corrections() overlay used elsewhere in the pipeline
(backbone_geometry_library.py / apply_library_corrections.py) -- it does
NOT go through ForceField.loadFile(), which only accepts XML and cannot
read a .frcmod at all, and which -- even with a valid XML file -- would
need to be ADDED to amber14-all.xml, not substituted for it (a system
needs the base atom types/charges/bonded terms regardless).

Results are written incrementally (one row at a time, flushed to disk) so
a crash or Ctrl-C partway through a long run doesn't lose completed work --
rerun with the same --out and it'll still contain everything done so far.
"""
import argparse, csv, math, os, sys, time
import numpy as np
from pdbfixer import PDBFixer
from openmm.app import ForceField, Simulation, CutoffNonPeriodic
from openmm import CustomExternalForce, CustomTorsionForce, LangevinMiddleIntegrator
from openmm.unit import kelvin, picosecond, picoseconds, nanometer, angstrom

FORCEFIELD_FILES = ['amber14-all.xml']   # base force field -- do NOT put a .frcmod here;
                                          # MechLib corrections are applied separately below
                                          # via apply_corrections(), not through this list.
K_REST = 2000.0   # kJ/mol/nm^2, positional restraint on frozen heavy atoms (half that for H)
K_DIH = 100.0     # kJ/mol/rad^2, soft phi/psi restraint on target residue --
                  # override with --k_dih if basin migration turns out to be
                  # the cause of a systematic tau bias (see d_phi/d_psi columns)
MAX_ITER = 200    # converged by ~150-200 in testing; raise if you see stale-looking results
NONBONDED_CUTOFF_NM = 1.2

FIELDNAMES = ["pdb", "chain", "resnum", "basin", "klass",
              "tau_pre", "tau_post", "d_tau",
              "ncacb_pre", "ncacb_post", "ccacb_pre", "ccacb_post",
              "phi_pre", "psi_pre", "d_phi", "d_psi"]


def calc_dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x, y = np.dot(v, w), np.dot(np.cross(b1n, v), w)
    return math.degrees(math.atan2(y, x))


def calc_angle(p1, p2, p3):
    v1, v2 = p1 - p2, p3 - p2
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return math.degrees(math.acos(np.clip(c, -1, 1)))


def get_pos(top, positions, chain_id, resnum, name):
    for res in top.residues():
        if res.chain.id == chain_id and res.id == str(resnum):
            for atom in res.atoms():
                if atom.name == name:
                    return np.array(positions[atom.index].value_in_unit(angstrom)), atom.index
    return None, None


def build_atom_index(top):
    """(chain_id, resnum_str, atom_name) -> atom.index, built once per file
    so relax_residue() doesn't rescan the whole topology 6x per residue."""
    idx = {}
    for res in top.residues():
        for atom in res.atoms():
            idx[(res.chain.id, res.id, atom.name)] = atom.index
    return idx


def get_pos_fast(atom_index, positions, chain_id, resnum, name):
    i = atom_index.get((chain_id, str(resnum), name))
    if i is None:
        return None, None
    return np.array(positions[i].value_in_unit(angstrom)), i


def build_base(pdb_dir, pdbid, mechlib_json=None):
    """One-time-per-file setup: fix structure, build base system, apply
    MechLib overlay if requested. Returns (topology, positions, system,
    n_base_forces, atom_index) -- n_base_forces marks the expected force
    count so relax_residue() can assert the system is clean before it
    adds anything (catches a previous residue's forces not having been
    fully removed rather than silently relaxing against a corrupted,
    accumulating restraint set). atom_index is a (chain, resnum, name)
    -> atom.index lookup built once, so relax_residue() doesn't rescan
    the whole topology for every atom it needs.
    """
    fname = os.path.join(pdb_dir, f"{pdbid}.pdb")
    fixer = PDBFixer(filename=fname)
    fixer.findMissingResidues()
    fixer.missingResidues = {}   # don't build in missing loops
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    top, pos0 = fixer.topology, fixer.positions

    forcefield = ForceField(*FORCEFIELD_FILES)
    system = forcefield.createSystem(top, nonbondedMethod=CutoffNonPeriodic,
                                      nonbondedCutoff=NONBONDED_CUTOFF_NM * nanometer,
                                      constraints=None)

    if mechlib_json:
        from apply_library_corrections import apply_corrections as _apply_mechlib
        system = _apply_mechlib(system, top, pos0, mechlib_json, force_field='amber')

    n_base_forces = system.getNumForces()
    atom_index = build_atom_index(top)
    return top, pos0, system, n_base_forces, atom_index


def relax_residue(top, pos0, system, n_base_forces, atom_index, chain_id, resnum):
    """Run the restrained local relaxation for one residue against an
    already-built (and possibly MechLib-corrected) system, then strip the
    restraint forces back off so `system` is unchanged for the next call.
    """
    actual = system.getNumForces()
    assert actual == n_base_forces, (
        f"system has {actual} forces, expected {n_base_forces} -- a previous "
        f"residue's restraint/dihedral forces were not fully removed, so "
        f"results from here on would be silently computed against a "
        f"corrupted, accumulating restraint set. Aborting rather than "
        f"continuing.")

    N0, iN = get_pos_fast(atom_index, pos0, chain_id, resnum, "N")
    CA0, iCA = get_pos_fast(atom_index, pos0, chain_id, resnum, "CA")
    C0, iC = get_pos_fast(atom_index, pos0, chain_id, resnum, "C")
    CB0, iCB = get_pos_fast(atom_index, pos0, chain_id, resnum, "CB")
    Cprev, iCprev = get_pos_fast(atom_index, pos0, chain_id, resnum - 1, "C")
    Nnext, iNnext = get_pos_fast(atom_index, pos0, chain_id, resnum + 1, "N")

    if N0 is None or CA0 is None or C0 is None:
        raise ValueError(f"residue {chain_id}{resnum} not found or missing backbone atoms")

    tau_pre = calc_angle(N0, CA0, C0)
    ncacb_pre = calc_angle(N0, CA0, CB0) if CB0 is not None else None
    ccacb_pre = calc_angle(C0, CA0, CB0) if CB0 is not None else None
    phi_pre = calc_dihedral(Cprev, N0, CA0, C0) if Cprev is not None else None
    psi_pre = calc_dihedral(N0, CA0, C0, Nnext) if Nnext is not None else None

    added_force_indices = []

    restraint = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    for p in ("k", "x0", "y0", "z0"):
        restraint.addPerParticleParameter(p)
    for atom in top.atoms():
        is_free = (atom.residue.chain.id == chain_id and atom.residue.id == str(resnum))
        if atom.element is not None and atom.element.symbol == 'H':
            k = 0.0 if is_free else K_REST * 0.5
        else:
            k = 0.0 if is_free else K_REST
        p = pos0[atom.index]
        restraint.addParticle(atom.index, [k, p.x, p.y, p.z])
    added_force_indices.append(system.addForce(restraint))

    dih = CustomTorsionForce("0.5*k*min(dtheta,2*pi-dtheta)^2; dtheta=abs(theta-theta0); pi=3.1415926535")
    dih.addPerTorsionParameter("k")
    dih.addPerTorsionParameter("theta0")
    if phi_pre is not None and None not in (iCprev, iN, iCA, iC):
        dih.addTorsion(iCprev, iN, iCA, iC, [K_DIH, math.radians(phi_pre)])
    if psi_pre is not None and None not in (iN, iCA, iC, iNnext):
        dih.addTorsion(iN, iCA, iC, iNnext, [K_DIH, math.radians(psi_pre)])
    added_force_indices.append(system.addForce(dih))

    try:
        integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, 0.001 * picoseconds)
        sim = Simulation(top, system, integrator)
        sim.context.setPositions(pos0)
        sim.minimizeEnergy(maxIterations=MAX_ITER)
        minpos = sim.context.getState(getPositions=True).getPositions()

        def gp(idx):
            return np.array(minpos[idx].value_in_unit(angstrom))

        N1, CA1, C1 = gp(iN), gp(iCA), gp(iC)
        CB1 = gp(iCB) if iCB is not None else None
        tau_post = calc_angle(N1, CA1, C1)
        ncacb_post = calc_angle(N1, CA1, CB1) if CB1 is not None else None
        ccacb_post = calc_angle(C1, CA1, CB1) if CB1 is not None else None

        # Diagnostic: did phi/psi actually stay put, or did the residue drift
        # to a different basin during "local" relaxation despite the soft
        # restraint? If d_phi/d_psi are large for some residues but not
        # others, that -- not a real basin-dependent mechanical effect --
        # would explain a systematic tau shift that doesn't track basin.
        phi_post = psi_post = None
        if iCprev is not None:
            Cprev1 = gp(iCprev)
            phi_post = calc_dihedral(Cprev1, N1, CA1, C1)
        if iNnext is not None:
            Nnext1 = gp(iNnext)
            psi_post = calc_dihedral(N1, CA1, C1, Nnext1)
    finally:
        # Remove highest index first so lower indices don't shift underneath us.
        for idx in sorted(added_force_indices, reverse=True):
            system.removeForce(idx)

    d_phi = round(phi_post - phi_pre, 2) if (phi_post is not None and phi_pre is not None) else ""
    d_psi = round(psi_post - psi_pre, 2) if (psi_post is not None and psi_pre is not None) else ""

    return dict(tau_pre=round(tau_pre, 2), tau_post=round(tau_post, 2),
                d_tau=round(tau_post - tau_pre, 2),
                ncacb_pre=round(ncacb_pre, 2) if ncacb_pre is not None else "",
                ncacb_post=round(ncacb_post, 2) if ncacb_post is not None else "",
                ccacb_pre=round(ccacb_pre, 2) if ccacb_pre is not None else "",
                ccacb_post=round(ccacb_post, 2) if ccacb_post is not None else "",
                phi_pre=round(phi_pre, 2) if phi_pre is not None else "",
                psi_pre=round(psi_pre, 2) if psi_pre is not None else "",
                d_phi=d_phi, d_psi=d_psi)


def main():
    global MAX_ITER, K_DIH
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates_csv", help="CSV with columns: pdb,chain,resnum,basin,klass (extra columns ignored)")
    ap.add_argument("pdb_dir", help="directory containing the referenced .pdb files")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--mechlib", default=None,
                     help="path to constants_library.json -- applies your real "
                          "MechLib corrections on top of base ff14SB via the "
                          "tested overlay, instead of stock AMBER angle equilibria")
    ap.add_argument("--max_iter", type=int, default=MAX_ITER,
                     help=f"minimizer iteration cap (default {MAX_ITER}; converged by "
                          "~150-200 in testing, raise only if results look stale)")
    ap.add_argument("--k_dih", type=float, default=K_DIH,
                     help=f"phi/psi restraint stiffness, kJ/mol/rad^2 (default {K_DIH}). "
                          "Raise this (e.g. 500-1000) if d_phi/d_psi in the output show "
                          "basin migration during minimization.")
    args = ap.parse_args()

    MAX_ITER = args.max_iter
    K_DIH = args.k_dih

    with open(args.candidates_csv) as f:
        candidates = list(csv.DictReader(f))

    # Group by pdb file so build_base() runs once per file, not once per row.
    by_pdb = {}
    order = []
    for c in candidates:
        pdbid = c["pdb"]
        if pdbid not in by_pdb:
            by_pdb[pdbid] = []
            order.append(pdbid)
        by_pdb[pdbid].append(c)

    total = len(candidates)
    done = 0
    t_start = time.time()

    # Resume support: if --out already exists, skip (pdb, chain, resnum) rows
    # already present in it, and append rather than overwrite.
    already_done = set()
    write_header = True
    if os.path.exists(args.out):
        with open(args.out) as f:
            for row in csv.DictReader(f):
                already_done.add((row["pdb"], row["chain"], row["resnum"]))
        if already_done:
            write_header = False
            print(f"Resuming: {len(already_done)} rows already in {args.out}, will skip those.")

    out_f = open(args.out, "a" if not write_header else "w", newline="")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()
        out_f.flush()

    for pdbid in order:
        rows = by_pdb[pdbid]
        try:
            t0 = time.time()
            top, pos0, system, n_base, atom_index = build_base(args.pdb_dir, pdbid, mechlib_json=args.mechlib)
            print(f"[{pdbid}] base system built in {time.time()-t0:.1f}s "
                  f"({len(rows)} candidate residue(s) from this file)")
        except Exception as e:
            print(f"FAILED to build base system for {pdbid}: {e!r} "
                  f"-- skipping all {len(rows)} candidates from this file")
            done += len(rows)
            continue

        for c in rows:
            chain, resnum = c["chain"], int(c["resnum"])
            key = (pdbid, chain, str(resnum))
            label = f"{c.get('klass', c.get('resname',''))}_{c.get('basin','')}_{pdbid}_{resnum}"
            done += 1
            if key in already_done:
                print(f"  [{done}/{total}] {label}: already in {args.out}, skipping")
                continue
            try:
                t1 = time.time()
                r = relax_residue(top, pos0, system, n_base, atom_index, chain, resnum)
                r.update(pdb=pdbid, chain=chain, resnum=resnum,
                          basin=c.get("basin", ""), klass=c.get("klass", c.get("resname", "")))
                writer.writerow(r)
                out_f.flush()
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta_s = (total - done) / rate if rate > 0 else float('nan')
                print(f"  [{done}/{total}] {label}: {time.time()-t1:.1f}s "
                      f"(elapsed {elapsed/60:.1f} min, ETA {eta_s/60:.1f} min) -> {r}")
            except Exception as e:
                print(f"  [{done}/{total}] FAILED {label}: {e!r}")

    out_f.close()
    print(f"\nDone. Results in {args.out}")


if __name__ == "__main__":
    main()
