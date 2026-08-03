#!/usr/bin/env python3
"""
Scan a directory of PDB files and tabulate candidate residues for the
MM local-relaxation validation: (phi, psi, basin, residue class, tau,
N-Ca-Cb, C-Ca-Cb, chi1) for every eligible residue.

Usage:
    python scan_pdb_cache.py /path/to/pdb_cache --out candidates.csv --per_cell 15

Only used if you don't already have a feature table (e.g.
features_lj_v3_FINAL_CLEAN.csv) with pdb/chain/resnum mapping preserved.
If you do, prefer that -- it's already consistent with your published
basin/class definitions and gives you real per-residue Delta values to
compare the relaxed geometry against, not just canonical ~111.2 deg.

Quality filtering (on by default):
  - Skips any residue where N, CA, C, CB, or the chi1-defining atom has
    an alternate conformer (altloc). Biopython auto-picks the
    highest-occupancy altloc PER ATOM, but that selection can pick a
    different conformer for N than for CB in the same residue --
    silently mixing two physically inconsistent partial structures into
    one "residue" and producing geometrically nonsensical angles (tau
    values of 99-106 degrees have shown up this way in unfiltered scans).
    Disable with --keep_altloc if you'd rather flag these globally
    instead of dropping them (e.g. to inspect how many there are).
  - Optional --max_bfactor: drops residues whose mean backbone B-factor
    (N, CA, C) exceeds the given ABSOLUTE value. Off by default. Use
    with caution on a heterogeneous cache -- B-factor scale varies a lot
    structure to structure (resolution, refinement protocol, isotropic
    vs anisotropic vs TLS groups), so a fixed cutoff can end up
    filtering 0% of one structure and 60%+ of another, which mostly
    measures "which structures refine hot" rather than "which residues
    are locally disordered." Prefer --bfactor_zmax below for a mixed
    cache like a pdb_cache pulled across many depositions.
  - Optional --bfactor_zmax: drops residues whose mean backbone B-factor
    is more than Z standard deviations above THAT FILE's own backbone
    B-factor mean (computed once per structure from all its residues,
    not just candidates). This adapts to each structure's own scale
    instead of applying one global number -- a Z of 2.0-2.5 is a
    reasonable starting point (flags the locally worst-behaved ~1-2% of
    residues in a roughly normal B-factor distribution, without being
    thrown off by whether the whole structure runs hot or cold).

Stratified sampling:
  --per_cell N caps the output at N residues per (residue class, basin)
  cell instead of dumping every eligible residue in the cache. Without
  this, scanning a large cache produces candidate lists in the hundreds
  of thousands to millions of rows -- each one an expensive MM
  relaxation (~1s+), which turns "a representative validation" into a
  multi-day job. N=15-20 per cell is enough for a real mean +/- SEM per
  cell while finishing in hours, not days. Sampling is seeded
  (--seed, default 0) for reproducibility.
"""
import argparse, csv, glob, math, os, random
import numpy as np
from Bio.PDB import PDBParser

BASINS = {
    "alpha": ((-80, -40), (-60, -20)),
    "beta":  ((-150, -90), (100, 160)),
    "PPII":  ((-90, -60), (120, 160)),
    "alphaL": ((40, 80), (20, 80)),
}
TARGET_RES = {"GLY", "ALA", "LEU", "VAL", "PHE", "ILE", "THR", "TYR", "TRP", "HIS"}


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


def in_basin(phi, psi):
    for name, ((p0, p1), (s0, s1)) in BASINS.items():
        if p0 <= phi <= p1 and s0 <= psi <= s1:
            return name
    return None


def any_disordered(res, names):
    """True if any of the given atom names on this residue has an
    alternate conformer (altloc). Checking multiple atoms, not just one,
    is the point -- a residue can look fine on N/CA/C but have a
    disordered CB/CG that Biopython silently resolved to a different
    occupancy-selected conformer than the backbone."""
    for name in names:
        if name in res:
            if res[name].is_disordered():
                return True
    return False


def mean_bfactor(res, names):
    vals = [res[n].get_bfactor() for n in names if n in res]
    return sum(vals) / len(vals) if vals else None


def file_backbone_bfactor_stats(model, target_res):
    """Mean and stdev of backbone (N,CA,C) B-factor across every eligible
    residue in this structure -- the reference scale for --bfactor_zmax.
    Computed once per file from the whole structure, not just candidates,
    so a handful of extreme candidates can't skew the baseline they're
    being compared against."""
    vals = []
    for chain in model:
        for res in chain:
            if res.id[0] != ' ' or res.get_resname() not in target_res:
                continue
            mb = mean_bfactor(res, ["N", "CA", "C"])
            if mb is not None:
                vals.append(mb)
    if len(vals) < 5:
        return None, None
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std())


def scan_file(fn, parser, keep_altloc=False, max_bfactor=None, bfactor_zmax=None):
    pdbid = os.path.basename(fn).replace(".pdb", "").replace(".ent", "")
    rows = []
    n_skipped_altloc = 0
    n_skipped_bfactor = 0
    try:
        struct = parser.get_structure(pdbid, fn)
    except Exception as e:
        print(f"  skip {pdbid}: parse error {e}")
        return rows, n_skipped_altloc, n_skipped_bfactor
    model = struct[0]

    file_mean, file_std = (None, None)
    if bfactor_zmax is not None:
        file_mean, file_std = file_backbone_bfactor_stats(model, TARGET_RES)

    for chain in model:
        reslist = [r for r in chain if r.id[0] == ' ']
        for i in range(1, len(reslist) - 1):
            res = reslist[i]
            resname = res.get_resname()
            if resname not in TARGET_RES:
                continue
            prev_r, next_r = reslist[i - 1], reslist[i + 1]
            try:
                if prev_r.id[1] != res.id[1] - 1 or next_r.id[1] != res.id[1] + 1:
                    continue

                backbone_names = ["N", "CA", "C", "CB", "CG", "CG1"]
                if not keep_altloc and any_disordered(res, backbone_names):
                    n_skipped_altloc += 1
                    continue
                if max_bfactor is not None:
                    mb = mean_bfactor(res, ["N", "CA", "C"])
                    if mb is not None and mb > max_bfactor:
                        n_skipped_bfactor += 1
                        continue
                if bfactor_zmax is not None and file_std:
                    mb = mean_bfactor(res, ["N", "CA", "C"])
                    if mb is not None:
                        z = (mb - file_mean) / file_std
                        if z > bfactor_zmax:
                            n_skipped_bfactor += 1
                            continue

                C_prev = prev_r["C"].get_coord()
                N, CA, C = res["N"].get_coord(), res["CA"].get_coord(), res["C"].get_coord()
                N_next = next_r["N"].get_coord()
                phi = calc_dihedral(C_prev, N, CA, C)
                psi = calc_dihedral(N, CA, C, N_next)
                basin = in_basin(phi, psi)
                if basin is None:
                    continue
                tau = calc_angle(N, CA, C)
                row = dict(pdb=pdbid, chain=chain.id, resnum=res.id[1], resname=resname,
                           phi=round(phi, 1), psi=round(psi, 1), basin=basin, tau=round(tau, 2),
                           ncacb="", ccacb="", chi1="")
                if resname != "GLY" and "CB" in res:
                    CB = res["CB"].get_coord()
                    row["ncacb"] = round(calc_angle(N, CA, CB), 2)
                    row["ccacb"] = round(calc_angle(C, CA, CB), 2)
                    cg_name = "CG" if "CG" in res else ("CG1" if "CG1" in res else None)
                    if cg_name:
                        row["chi1"] = round(calc_dihedral(N, CA, CB, res[cg_name].get_coord()), 1)
                rows.append(row)
            except KeyError:
                continue
    return rows, n_skipped_altloc, n_skipped_bfactor


def stratified_sample(rows, per_cell, seed):
    """Cap at `per_cell` rows per (resname, basin) cell. Deterministic
    given the same seed and input order."""
    rng = random.Random(seed)
    by_cell = {}
    for r in rows:
        key = (r["resname"], r["basin"])
        by_cell.setdefault(key, []).append(r)

    sampled = []
    print("\nPer-cell counts (available -> kept):")
    for key in sorted(by_cell):
        bucket = by_cell[key]
        if len(bucket) > per_cell:
            chosen = rng.sample(bucket, per_cell)
        else:
            chosen = bucket
        sampled.extend(chosen)
        print(f"  {key[0]:>4s} / {key[1]:<7s}: {len(bucket):6d} -> {len(chosen):3d}")
    return sampled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb_dir", help="directory containing .pdb files")
    ap.add_argument("--out", default="candidates.csv")
    ap.add_argument("--per_cell", type=int, default=None,
                     help="cap output at N residues per (class, basin) cell "
                          "(recommended: 15-20). Without this, output is every "
                          "eligible residue in the cache, which for a large "
                          "cache can be hundreds of thousands to millions of "
                          "rows -- each one a ~1s+ MM relaxation later.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for --per_cell sampling")
    ap.add_argument("--keep_altloc", action="store_true",
                     help="don't skip residues with alternate conformers "
                          "(default: skip them -- see module docstring)")
    ap.add_argument("--max_bfactor", type=float, default=None,
                     help="skip residues whose mean N/CA/C B-factor exceeds this "
                          "ABSOLUTE value (off by default; prefer --bfactor_zmax "
                          "for a heterogeneous cache -- see module docstring)")
    ap.add_argument("--bfactor_zmax", type=float, default=None,
                     help="skip residues whose mean N/CA/C B-factor is more than "
                          "this many standard deviations above THAT FILE's own "
                          "backbone B-factor mean (adapts per structure; "
                          "2.0-2.5 is a reasonable starting point)")
    args = ap.parse_args()

    parser = PDBParser(QUIET=True)
    all_rows = []
    total_altloc_skipped = 0
    total_bfactor_skipped = 0
    files = sorted(glob.glob(os.path.join(args.pdb_dir, "*.pdb")))
    print(f"Found {len(files)} PDB files in {args.pdb_dir}")
    for fn in files:
        rows, n_alt, n_bf = scan_file(fn, parser, keep_altloc=args.keep_altloc,
                                       max_bfactor=args.max_bfactor,
                                       bfactor_zmax=args.bfactor_zmax)
        all_rows.extend(rows)
        total_altloc_skipped += n_alt
        total_bfactor_skipped += n_bf
        if rows or n_alt or n_bf:
            msg = f"  {os.path.basename(fn)}: {len(rows)} candidates"
            if n_alt:
                msg += f", {n_alt} skipped (altloc)"
            if n_bf:
                msg += f", {n_bf} skipped (B-factor)"
            print(msg)

    print(f"\nTotal eligible residues: {len(all_rows)}")
    if total_altloc_skipped:
        print(f"Total skipped for alternate conformers: {total_altloc_skipped} "
              f"(use --keep_altloc to include them instead)")
    if total_bfactor_skipped:
        if args.bfactor_zmax is not None:
            print(f"Total skipped for B-factor Z-score > {args.bfactor_zmax}: {total_bfactor_skipped}")
        else:
            print(f"Total skipped for B-factor > {args.max_bfactor}: {total_bfactor_skipped}")

    if args.per_cell:
        all_rows = stratified_sample(all_rows, args.per_cell, args.seed)
        print(f"\nAfter --per_cell {args.per_cell} sampling: {len(all_rows)} residues")

    fieldnames = ["pdb", "chain", "resnum", "resname", "phi", "psi", "basin",
                  "tau", "ncacb", "ccacb", "chi1"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} candidate residues to {args.out}")


if __name__ == "__main__":
    main()
