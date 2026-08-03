#!/usr/bin/env python3
"""
add_lj_torques.py — Lennard-Jones steric torques (reconstruction)
===================================================================
Replaces the v5 simplified steric proxy (unit(CB-CA) x sc_mass/100)
with real pairwise Lennard-Jones forces, projected onto the phi/psi
rotation axes as torques.

For each residue i:
  1. Find all atoms within a cutoff radius of residue i's atoms
     (excluding 1-2 / 1-3 bonded neighbours, same convention as
     features_collector.py's _covalent_exclusion_set).
  2. For each pair (atom_i, atom_j), look up AMBER LJ parameters via
     ResidueAtomMap.get_atom_type() + AtomTypes.lj_pair() (Lorentz-
     Berthelot combining rules).
  3. Compute the LJ force magnitude via the analytic derivative of
     lennard_jones(r, sigma, epsilon):
         dE/dr = 4*epsilon * (-12*sigma^12/r^13 + 6*sigma^6/r^7)
         F     = -dE/dr        (positive = repulsive, pushes atoms apart)
     Force vector on atom_i points along unit(pos_i - pos_j) * F.
  4. Project the force at each atom onto the phi axis (N[i]-CA[i]) and
     psi axis (CA[i]-C[i]) using torque() from paper2_geom_utils.py.
  5. Sum over all atom pairs -> tau_phi_lj, tau_psi_lj for residue i.

This REPLACES tau_phi_steric / tau_psi_steric (the v5 proxy columns)
with tau_phi_lj / tau_psi_lj. Both are kept in the output so Paper 1's
original proxy can still be compared against the LJ upgrade.

Usage:
  python add_lj_torques.py \\
      --csv features.csv \\
      --pdb_dir ./pdb_cache \\
      --output features_lj.csv \\
      --cutoff 8.0

Author: Wei (Cvek Lab, LSUHSC) — reconstructed after LSUS -> LSUHSC move
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree

from pdb_loader import load_structure
from molcore import AtomTypes, ResidueAtomMap, lennard_jones
from geom_utils import torque, unit


# ══════════════════════════════════════════════════════════════════════════════
# LJ force from analytic derivative of lennard_jones()
# ══════════════════════════════════════════════════════════════════════════════

def lj_force_magnitude(r, sigma, epsilon):
    """
    F = -dE/dr for E_LJ = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6]
    Positive F = repulsive. Returns 0.0 if epsilon == 0 (ghost atoms).
    """
    if epsilon == 0.0 or r < 1e-6:
        return 0.0
    sr6 = (sigma / r) ** 6
    sr12 = sr6 * sr6
    return 4.0 * epsilon * (12.0 * sr12 - 6.0 * sr6) / r


def parse_occupancy_by_coord(pdb_path):
    """
    Independent raw-PDB parse building a coordinate -> occupancy lookup.
    Keyed by rounded (x,y,z), same rationale as parse_altloc_by_coord.
    """
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


def _occupancy_compatible(occ_a, occ_b, threshold=0.999):
    """
    Any atom with occupancy below `threshold` is structurally uncertain —
    it represents a position that is not always physically present (an
    alternate conformer, a disordered water/ion, etc). Pairing two such
    atoms across DIFFERENT residues in a non-bonded force calculation can
    produce nonsensical near-zero-distance "clashes" between states that
    never coexist simultaneously in the real crystal. We therefore only
    compute cross-residue LJ interactions between FULLY-RESOLVED atoms
    (occupancy >= threshold on both sides). This is simpler and safer than
    trying to match altLoc letters across residues, since altLoc codes are
    assigned independently per-residue and same-letter codes on different
    residues carry no guaranteed physical correlation.
    """
    return occ_a >= threshold and occ_b >= threshold


def _phi_axis(s, i):
    return s.coords['N'][i], s.coords['CA'][i]


def _psi_axis(s, i):
    return s.coords['CA'][i], s.coords['C'][i]


def _covalent_exclusion_set(s, res_idx):
    """1-2 / 1-3 bonded exclusions — same convention as features_collector.py."""
    excl = set()
    i, n = res_idx, s.n_res
    for aname in s.coords:
        excl.add((i, aname))
    if i + 1 < n and i not in s.chain_breaks:
        excl.update([(i + 1, 'N'), (i + 1, 'CA'), (i + 1, 'H')])
    if i > 0 and (i - 1) not in s.chain_breaks:
        excl.update([(i - 1, 'C'), (i - 1, 'CA'), (i - 1, 'O')])
    return excl


_DISULFIDE_CUTOFF = 2.5  # Å — real S-S covalent bond is ~2.05 Å


def _find_disulfide_bonds(s):
    """
    Detect CYS-CYS disulfide bonds by SG-SG distance across the whole
    structure (not just sequence neighbours). Returns a dict mapping
    res_idx -> set of res_idx it is disulfide-bonded to.
    """
    if 'SG' not in s.coords:
        return {}
    sg = s.coords['SG']
    cys_idxs = [i for i in range(s.n_res)
                if s.sequence[i] == 'CYS' and not np.any(np.isnan(sg[i]))]
    bonds = {}
    for a_pos, i in enumerate(cys_idxs):
        for j in cys_idxs[a_pos + 1:]:
            d = float(np.linalg.norm(sg[i] - sg[j]))
            if d <= _DISULFIDE_CUTOFF:
                bonds.setdefault(i, set()).add(j)
                bonds.setdefault(j, set()).add(i)
    return bonds


_STANDARD_AA = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
}


def _build_kdtree(s, occ_lookup):
    """Build a KD-tree over all non-hydrogen heavy atoms with known types.
    Also returns the occupancy (from the independent raw-PDB parse) for
    every included atom, matched by coordinate.

    HARD FILTER: only atoms belonging to one of the 20 standard amino
    acids are included. Waters, ligands, ions, and other heteroatoms are
    never part of intrinsic backbone LJ/steric mechanics — their inclusion
    should be a deliberate modeling choice, not an accidental side effect
    of a residue name that happens to look typeable. This also guards
    against files where waters/hetero residues get positions that are
    duplicated or overlapping (seen in some depositions), which would
    otherwise produce nonsensical near-zero-distance "clashes" against
    real protein atoms.
    """
    pts, res_list, atom_list, type_list, occ_list = [], [], [], [], []
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
                continue  # unknown atom for this residue type
            key = (round(float(xyz[0]), 3), round(float(xyz[1]), 3), round(float(xyz[2]), 3))
            occ = occ_lookup.get(key, 1.0)  # unmatched -> assume fully resolved
            pts.append(xyz)
            res_list.append(i)
            atom_list.append(aname)
            type_list.append(atype)
            occ_list.append(occ)
    if not pts:
        return None, np.empty((0, 3)), np.array([], dtype=int), [], [], []
    return (cKDTree(np.array(pts, dtype=float)),
            np.array(pts, dtype=float),
            np.array(res_list, dtype=int),
            atom_list, type_list, occ_list)


def compute_lj_torques_for_residue(s, i, tree, all_pts, res_ids,
                                    atom_names, atom_types, occ_list,
                                    occ_lookup, disulfide_bonds, cutoff=8.0):
    """
    Compute (tau_phi_lj, tau_psi_lj) for residue i by summing pairwise LJ
    forces between residue i's atoms and all atoms within `cutoff` Å,
    excluding 1-2/1-3 bonded neighbours, disulfide-bonded SG-SG pairs, and
    cross-residue pairs involving any partial-occupancy (disordered) atom.
    """
    n = s.n_res
    excl = _covalent_exclusion_set(s, i)
    for partner in disulfide_bonds.get(i, set()):
        excl.add((partner, 'SG'))
    phi_a, phi_b = _phi_axis(s, i)
    psi_a, psi_b = _psi_axis(s, i)

    have_phi = not any(np.any(np.isnan(p)) for p in (phi_a, phi_b))
    have_psi = not any(np.any(np.isnan(p)) for p in (psi_a, psi_b))
    phi_axis_hat = unit(phi_b - phi_a) if have_phi else None
    psi_axis_hat = unit(psi_b - psi_a) if have_psi else None

    tau_phi = 0.0
    tau_psi = 0.0
    res_name_i = s.sequence[i]

    # Residue i's own atoms (the ones that will feel the force)
    for aname_i, coords_arr in s.coords.items():
        pos_i = coords_arr[i]
        if np.any(np.isnan(pos_i)):
            continue
        type_i = ResidueAtomMap.get_atom_type(res_name_i, aname_i)
        if type_i is None:
            continue

        key_i = (round(float(pos_i[0]), 3), round(float(pos_i[1]), 3), round(float(pos_i[2]), 3))
        occ_i = occ_lookup.get(key_i, 1.0)

        neighbor_idxs = tree.query_ball_point(pos_i, cutoff)
        for j in neighbor_idxs:
            res_j = int(res_ids[j])
            if res_j == i:
                continue  # skip same-residue pairs (handled by internal geometry)
            if (res_j, atom_names[j]) in excl:
                continue
            if not _occupancy_compatible(occ_i, occ_list[j]):
                continue  # disordered/partial-occupancy atom — unreliable position

            pos_j = all_pts[j]
            r_vec = pos_i - pos_j
            r = float(np.linalg.norm(r_vec))
            if r < 1e-3:
                continue

            sigma_ij, epsilon_ij = AtomTypes.lj_pair(type_i, atom_types[j])
            if epsilon_ij == 0.0:
                continue

            F_mag = lj_force_magnitude(r, sigma_ij, epsilon_ij)
            if F_mag == 0.0:
                continue

            r_hat = r_vec / r
            force_vec = F_mag * r_hat  # force ON atom i, pointing away from j

            if have_phi:
                tau_phi += torque(force_vec, pos_i, phi_a, phi_axis_hat)
            if have_psi:
                tau_psi += torque(force_vec, pos_i, psi_a, psi_axis_hat)

    return float(tau_phi), float(tau_psi)


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def process_pdb(pdb_id, pdb_path, cutoff=8.0):
    """Compute LJ torques for every residue in one PDB. Returns dict[res_idx] = (phi, psi)."""
    try:
        s = load_structure(str(pdb_path))
    except Exception as e:
        return None, f"load failed: {e}"

    occ_lookup = parse_occupancy_by_coord(str(pdb_path))
    tree, all_pts, res_ids, atom_names, atom_types, occ_list = _build_kdtree(s, occ_lookup)
    if tree is None:
        return None, "no typed atoms found"

    disulfide_bonds = _find_disulfide_bonds(s)

    out = {}
    for i in range(s.n_res):
        if s.sequence[i] not in _STANDARD_AA:
            continue  # skip water/ligand/hetero residues entirely
        tau_phi, tau_psi = compute_lj_torques_for_residue(
            s, i, tree, all_pts, res_ids, atom_names, atom_types,
            occ_list, occ_lookup, disulfide_bonds, cutoff)
        out[i] = (round(tau_phi, 5), round(tau_psi, 5))
    return out, None


def main():
    ap = argparse.ArgumentParser(
        description='Compute real Lennard-Jones torques and merge into '
                    'an existing features CSV')
    ap.add_argument('--csv', required=True, help='Input features CSV')
    ap.add_argument('--pdb_dir', required=True, help='Directory of PDB files')
    ap.add_argument('--output', required=True, help='Output CSV path')
    ap.add_argument('--cutoff', type=float, default=8.0,
                    help='LJ neighbor search cutoff in Angstrom (default: 8.0)')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 70)
    print("add_lj_torques.py")
    print("=" * 70)

    print(f"[1] Loading {args.csv} ...")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"  {len(df):,} rows loaded")

    if 'chain' not in df.columns:
        df['chain'] = 'A'

    df['tau_phi_lj'] = np.nan
    df['tau_psi_lj'] = np.nan

    pdb_ids = df['pdb_id'].unique()
    print(f"[2] Processing {len(pdb_ids)} unique PDB structures "
          f"(cutoff={args.cutoff} Å) ...")

    n_ok = n_fail = 0
    for k, pdb_id in enumerate(pdb_ids, 1):
        pdb_path = Path(args.pdb_dir) / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            # PDB files on disk are lowercase (e.g. 1ubq.pdb) while pdb_id
            # in the CSV is uppercase (e.g. 1UBQ) — fall back to lowercase.
            pdb_path = Path(args.pdb_dir) / f"{str(pdb_id).lower()}.pdb"
        if not pdb_path.exists():
            n_fail += 1
            if args.verbose:
                print(f"  [SKIP] {pdb_id}: no PDB file found")
            continue

        lj_map, err = process_pdb(pdb_id, pdb_path, cutoff=args.cutoff)
        if lj_map is None:
            if args.verbose:
                print(f"  [SKIP] {pdb_id}: {err}")
            n_fail += 1
            continue

        mask = df['pdb_id'] == pdb_id
        res_idx_col = df.loc[mask, 'res_idx']
        tau_phi_vals = res_idx_col.map(lambda i: lj_map.get(i, (np.nan, np.nan))[0])
        tau_psi_vals = res_idx_col.map(lambda i: lj_map.get(i, (np.nan, np.nan))[1])
        df.loc[mask, 'tau_phi_lj'] = tau_phi_vals.values
        df.loc[mask, 'tau_psi_lj'] = tau_psi_vals.values
        n_ok += 1

        if k % 500 == 0 or k == len(pdb_ids):
            elapsed = time.time() - t0
            rate = k / max(elapsed, 1)
            print(f"  [{k:5d}/{len(pdb_ids)}]  ok={n_ok}  fail={n_fail}  "
                  f"{rate:.2f} PDB/s  {elapsed/60:.1f} min")

    print(f"\n[3] Writing {args.output} ...")
    df.to_csv(args.output, index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Done in {elapsed/60:.1f} min")
    print(f"  PDBs processed : {n_ok}")
    print(f"  PDBs failed    : {n_fail}")
    print(f"  Rows written   : {len(df):,}")
    print(f"  New columns    : tau_phi_lj, tau_psi_lj")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
