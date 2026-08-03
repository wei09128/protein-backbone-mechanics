#!/usr/bin/env python3
"""
scan_all_close_contacts.py — Batch scan for genuine crystallographic clashes
================================================================================
Runs the same close-contact diagnostic logic used on 1T1U and 9UVM across
ALL PDB structures in a directory, to build a complete, documented exclusion
list of (pdb_id, res_idx-equivalent) pairs involved in genuine sub-1.5Å
non-bonded contacts between fully-resolved (occupancy>=1.0) standard-amino-
-acid atoms. These are real crystallographic modeling defects (pre-dating
modern automated clash validation), not pipeline artifacts — occupancy and
altloc-based artifacts are already excluded by add_lj_torques.py itself.

Uses the SAME Structure loader as the pipeline (paper2_pdb_loader) so the
residue indexing matches features_lj.csv exactly.

Usage:
  python scan_all_close_contacts.py --pdb_dir ./pdb_cache --out clash_exclusions.csv
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path

from paper2_pdb_loader import load_structure
from paper2_molcore import ResidueAtomMap

_STANDARD_AA = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
}


def parse_occupancy_by_coord(pdb_path):
    lookup = {}
    with open(pdb_path, errors='ignore') as fh:
        for line in fh:
            if not line.startswith('ATOM'):
                continue
            try:
                occ = float(line[54:60])
                x = round(float(line[30:38]), 3)
                y = round(float(line[38:46]), 3)
                z = round(float(line[46:54]), 3)
            except (ValueError, IndexError):
                continue
            lookup[(x, y, z)] = occ
    return lookup


def scan_structure(pdb_path, clash_cutoff=1.5):
    """
    Returns a list of dicts describing genuine (full-occupancy, standard-AA,
    non-adjacent) close contacts found in this structure.
    """
    try:
        s = load_structure(str(pdb_path))
    except Exception as e:
        return None, f"load failed: {e}"

    occ_lookup = parse_occupancy_by_coord(str(pdb_path))
    pdb_id = Path(pdb_path).stem.upper()

    # Build flat atom list: (res_idx, res_name, atom_name, xyz, occ)
    atoms = []
    for aname, coords_arr in s.coords.items():
        for i in range(s.n_res):
            res_name = s.sequence[i]
            if res_name not in _STANDARD_AA:
                continue
            xyz = coords_arr[i]
            if np.any(np.isnan(xyz)):
                continue
            atype = ResidueAtomMap.get_atom_type(res_name, aname)
            if atype is None:
                continue
            key = (round(float(xyz[0]), 3), round(float(xyz[1]), 3), round(float(xyz[2]), 3))
            occ = occ_lookup.get(key, 1.0)
            if occ < 0.999:
                continue  # only fully-resolved atoms are "genuine" contacts
            atoms.append((i, res_name, aname, xyz))

    if len(atoms) < 2:
        return [], None

    from scipy.spatial import cKDTree
    pts = np.array([a[3] for a in atoms])
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=clash_cutoff, output_type='ndarray')

    clashes = []
    for i_idx, j_idx in pairs:
        ri, rni, ani, xi = atoms[i_idx]
        rj, rnj, anj, xj = atoms[j_idx]
        if ri == rj:
            continue  # same residue, expected to be close (covalent)
        if abs(ri - rj) <= 1:
            continue  # sequence-adjacent, expected close (backbone bond)
        d = float(np.linalg.norm(xi - xj))
        clashes.append({
            'pdb_id': pdb_id, 'res_idx_a': ri, 'res_name_a': rni, 'atom_a': ani,
            'res_idx_b': rj, 'res_name_b': rnj, 'atom_b': anj, 'dist': round(d, 4),
        })
    return clashes, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdb_dir', required=True)
    ap.add_argument('--out', default='clash_exclusions.csv')
    ap.add_argument('--clash_cutoff', type=float, default=1.5)
    ap.add_argument('--max_pdbs', type=int, default=None)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    pdb_paths = sorted(Path(args.pdb_dir).glob('*.pdb'))
    if args.max_pdbs:
        pdb_paths = pdb_paths[:args.max_pdbs]
    print(f"Scanning {len(pdb_paths)} PDB files for genuine close contacts "
          f"(cutoff={args.clash_cutoff} Å, full-occupancy standard AA only)...")

    all_clashes = []
    n_fail = 0
    t0 = time.time()

    with open(args.out, 'w') as fh:
        fh.write("pdb_id,res_idx_a,res_name_a,atom_a,res_idx_b,res_name_b,atom_b,dist\n")
        for k, pdb_path in enumerate(pdb_paths, 1):
            clashes, err = scan_structure(pdb_path, args.clash_cutoff)
            if clashes is None:
                n_fail += 1
                if args.verbose:
                    print(f"  [SKIP] {pdb_path.name}: {err}")
                continue
            for c in clashes:
                fh.write(f"{c['pdb_id']},{c['res_idx_a']},{c['res_name_a']},"
                         f"{c['atom_a']},{c['res_idx_b']},{c['res_name_b']},"
                         f"{c['atom_b']},{c['dist']}\n")
                all_clashes.append(c)

            if k % 500 == 0 or k == len(pdb_paths):
                elapsed = time.time() - t0
                print(f"  [{k:5d}/{len(pdb_paths)}]  "
                      f"clashes_found={len(all_clashes)}  fail={n_fail}  "
                      f"{elapsed/60:.1f} min")

    print(f"\n{'='*60}")
    print(f"Done. {len(all_clashes)} genuine close-contact pairs found "
          f"across {len(pdb_paths) - n_fail} structures.")
    n_pdbs_affected = len(set(c['pdb_id'] for c in all_clashes))
    print(f"{n_pdbs_affected} unique PDB structures affected "
          f"({100*n_pdbs_affected/len(pdb_paths):.2f}% of dataset)")
    print(f"Written to {args.out}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
