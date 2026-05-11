"""
pdb_loader.py — Real-coordinate PDB loader for feature extraction v6
======================================================================

Design goals (matches mkdssp and other production tools):
  - All coordinates come from the PDB file, no NeRF reconstruction
  - Altloc handling: keep highest-occupancy, break ties with altloc A
  - Hydrogens: always recompute via place_H_dssp from real N/C_prev/O_prev,
    ignoring any H atoms already in the file
  - Missing atoms → NaN (not zero), downstream features handle NaN gracefully
  - HETATM records ignored
  - Multi-chain: extract first protein chain (matches generate_fixtures.py)
  - NMR: stop at first ENDMDL

Returns a Structure namedtuple with everything downstream features need:
  sequence, n_res, coords (dict of atom_name → (n_res, 3) array),
  chain_id, chain_breaks, bfactors, omega_measured

Run this file directly for a self-test:
    python pdb_loader.py --pdb tests/fixtures/1ubq_chainA.pdb
"""

import argparse
import sys
import numpy as np
from collections import defaultdict, namedtuple
from pathlib import Path

from hbond_finder import place_H_dssp
from molcore import dihedral_angle as _molcore_dihedral


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

# Standard 20 amino acids + common variants we treat as protein
_PROTEIN_RESIDUES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'MSE',              # selenomethionine → treated as MET
    'SEC', 'PYL',       # rare
    'HID', 'HIE', 'HIP',
    'CYX', 'CYM',
}

# Residues to rename silently on load (so downstream code sees standard names)
_RENAME = {
    'MSE': 'MET',
    'HID': 'HIS', 'HIE': 'HIS', 'HIP': 'HIS',
    'CYX': 'CYS', 'CYM': 'CYS',
    'SEC': 'CYS',   # closest standard equivalent
}

# Backbone atoms always expected
_BACKBONE_ATOMS = ('N', 'CA', 'C', 'O')

# Threshold for detecting chain breaks (|C(i) - N(i+1)| > 2.5 Å)
_CHAIN_BREAK_CN = 2.5


Structure = namedtuple('Structure', [
    'pdb_id',           # str
    'chain_id',         # str
    'sequence',         # list of 3-letter residue names, length n_res
    'n_res',            # int
    'coords',           # dict: atom_name → (n_res, 3) numpy array, NaN for missing
    'bfactors',         # dict: atom_name → (n_res,) numpy array, NaN for missing
    'omega_measured',   # (n_res - 1,) array of omega in radians, from PDB
    'chain_breaks',     # list of residue indices where a break follows (i.e. after i)
])


# ══════════════════════════════════════════════════════════════════════════════
# Low-level parser
# ══════════════════════════════════════════════════════════════════════════════

def _parse_atom_line(line):
    """
    Parse an ATOM record using strict PDB fixed-column format.
    Returns (atom_name, altloc, resname, chain, resseq_str, x, y, z, occ, bfactor, element)
    or None if the line is malformed.
    """
    if len(line) < 54 or not line.startswith('ATOM'):
        return None
    try:
        atom_name = line[12:16].strip()
        altloc    = line[16]
        resname   = line[17:20].strip()
        chain     = line[21]
        resseq    = line[22:27]  # includes insertion code, keep as string for keys
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        # Occupancy and B-factor fields are optional
        occ  = float(line[54:60]) if len(line) >= 60 else 1.0
        bf   = float(line[60:66]) if len(line) >= 66 else 0.0
        elem = line[76:78].strip() if len(line) >= 78 else atom_name[0]
    except (ValueError, IndexError):
        return None
    return atom_name, altloc, resname, chain, resseq, x, y, z, occ, bf, elem


def _first_protein_chain(pdb_path, min_protein_residues=10):
    """
    Return the ID of the first chain in this PDB that has at least
    `min_protein_residues` standard amino acid residues. Walks only model 1.
    """
    seen = {}
    seen_residues = set()
    order = 0
    in_model = None  # None, 1, or 'done'

    with open(pdb_path) as f:
        for line in f:
            if line.startswith('MODEL'):
                in_model = 1 if in_model is None else 'done'
                continue
            if line.startswith('ENDMDL'):
                if in_model == 1:
                    in_model = 'done'
                continue
            if in_model == 'done':
                continue
            if not line.startswith('ATOM'):
                continue
            if len(line) < 22:
                continue
            chain = line[21]
            resname = line[17:20].strip()
            resseq = line[22:27]

            if chain not in seen:
                seen[chain] = {'n_prot': 0, 'order': order}
                order += 1
            key = (chain, resseq)
            if key in seen_residues:
                continue
            seen_residues.add(key)
            if resname in _PROTEIN_RESIDUES:
                seen[chain]['n_prot'] += 1

    for chain_id, info in sorted(seen.items(), key=lambda kv: kv[1]['order']):
        if info['n_prot'] >= min_protein_residues:
            return chain_id
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Main loader
# ══════════════════════════════════════════════════════════════════════════════

def load_structure(pdb_path, chain_id=None, verbose=False):
    """
    Load a protein structure from a PDB file into a Structure object.

    Rules:
      - Only model 1 is read (ATOMs after first ENDMDL are ignored)
      - HETATM records are ignored
      - Only the specified chain (or first protein chain if None) is kept
      - Altloc: keep the one with highest occupancy; tie-break with altloc 'A'
      - H atoms in the file are ignored (always recomputed via place_H_dssp)
      - Missing heavy atoms → NaN in the coord dict

    Returns a Structure namedtuple.
    """
    pdb_path = Path(pdb_path)
    pdb_id = pdb_path.stem.split('_')[0].upper()

    # Choose chain
    if chain_id is None:
        chain_id = _first_protein_chain(pdb_path)
        if chain_id is None:
            raise ValueError(f"{pdb_path}: no protein chain found")

    # ── First pass: collect all atoms, keyed by (resseq, atom_name), with altloc handling ──
    # For each (resseq, atom_name) we keep the record with the highest occupancy.
    # Ties broken by preferring altloc 'A', then ' ' (blank), then alphabetical.
    residue_order = []           # ordered list of resseq strings (first-appearance)
    seen_resseqs  = set()
    residue_names = {}           # resseq → resname (from first record)
    atoms_best    = {}           # (resseq, atom_name) → (occ, altloc_rank, record)

    def altloc_rank(a):
        # Lower rank = more preferred. 'A' is best, ' ' second, others alphabetical.
        if a == 'A': return 0
        if a == ' ': return 1
        return 2 + ord(a)

    in_model = None
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('MODEL'):
                in_model = 1 if in_model is None else 'done'
                continue
            if line.startswith('ENDMDL'):
                if in_model == 1:
                    in_model = 'done'
                continue
            if in_model == 'done':
                continue
            if not line.startswith('ATOM'):
                continue

            parsed = _parse_atom_line(line)
            if parsed is None:
                continue
            atom_name, altloc, resname, chain, resseq, x, y, z, occ, bf, elem = parsed

            if chain != chain_id:
                continue

            # Skip hydrogens — we recompute them
            if elem == 'H' or atom_name.startswith('H') or atom_name in ('D', '1H', '2H', '3H'):
                continue

            # Rename non-standard residues to standard
            resname_std = _RENAME.get(resname, resname)
            if resname_std not in _PROTEIN_RESIDUES and resname_std not in {
                'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}:
                # Not a residue we can process (weird ligand mislabeled as ATOM)
                continue

            if resseq not in seen_resseqs:
                seen_resseqs.add(resseq)
                residue_order.append(resseq)
                residue_names[resseq] = resname_std

            key = (resseq, atom_name)
            coord = np.array([x, y, z])
            record = (coord, bf)
            if key not in atoms_best:
                atoms_best[key] = (occ, altloc_rank(altloc), record)
            else:
                old_occ, old_rank, _ = atoms_best[key]
                # Prefer higher occupancy; if tied prefer lower rank (A > ' ' > others)
                if (occ, -altloc_rank(altloc)) > (old_occ, -old_rank):
                    atoms_best[key] = (occ, altloc_rank(altloc), record)

    n_res = len(residue_order)
    if n_res == 0:
        raise ValueError(f"{pdb_path}: no atoms found for chain {chain_id}")

    # ── Second pass: build the coord/bfactor arrays with NaN for missing atoms ──
    # We need to know all atom names that appear anywhere in the structure.
    atom_names_set = set(a for (_, a) in atoms_best.keys())
    atom_names_set.update(_BACKBONE_ATOMS)
    atom_names_set.add('CB')   # always present in dict even if missing for GLY

    coords   = {a: np.full((n_res, 3), np.nan) for a in atom_names_set}
    bfactors = {a: np.full((n_res,),   np.nan) for a in atom_names_set}
    sequence = [residue_names[rs] for rs in residue_order]

    resseq_to_idx = {rs: i for i, rs in enumerate(residue_order)}
    for (resseq, aname), (occ, rank, (xyz, bf)) in atoms_best.items():
        i = resseq_to_idx[resseq]
        coords[aname][i]   = xyz
        bfactors[aname][i] = bf

    # ── Place H atoms via place_H_dssp (Kabsch-Sander) ────────────────────────
    H = np.full((n_res, 3), np.nan)
    for i in range(1, n_res):
        if sequence[i] == 'PRO':
            continue  # proline has no amide H
        N_i    = coords['N'][i]
        C_prev = coords['C'][i - 1]
        O_prev = coords['O'][i - 1]
        if (np.any(np.isnan(N_i)) or np.any(np.isnan(C_prev))
                or np.any(np.isnan(O_prev))):
            continue
        H[i] = place_H_dssp(N_i, C_prev, O_prev)
    coords['H'] = H
    bfactors['H'] = np.full((n_res,), np.nan)

    # ── Detect chain breaks (|C(i)-N(i+1)| > 2.5 Å) ───────────────────────────
    chain_breaks = []
    for i in range(n_res - 1):
        c_i = coords['C'][i]
        n_next = coords['N'][i + 1]
        if np.any(np.isnan(c_i)) or np.any(np.isnan(n_next)):
            chain_breaks.append(i)
            continue
        if np.linalg.norm(n_next - c_i) > _CHAIN_BREAK_CN:
            chain_breaks.append(i)

    # ── Compute measured omega dihedrals from real coords ─────────────────────
    # Use molcore.dihedral_angle to stay consistent with nerf_builder's
    # convention (sign matters: our internal helper had an off-by-sign bug).
    omega_measured = np.full((n_res - 1,), np.nan) if n_res > 1 else np.array([])
    for i in range(n_res - 1):
        if i in chain_breaks:
            continue
        ca_i = coords['CA'][i]
        c_i  = coords['C'][i]
        n_j  = coords['N'][i + 1]
        ca_j = coords['CA'][i + 1]
        if any(np.any(np.isnan(p)) for p in (ca_i, c_i, n_j, ca_j)):
            continue
        try:
            omega_measured[i] = _molcore_dihedral(ca_i, c_i, n_j, ca_j)
        except Exception:
            pass

    if verbose:
        n_missing_bb = sum(
            1 for i in range(n_res)
            for a in _BACKBONE_ATOMS
            if np.any(np.isnan(coords[a][i]))
        )
        n_gly = sum(1 for r in sequence if r == 'GLY')
        n_missing_cb = sum(
            1 for i in range(n_res)
            if sequence[i] != 'GLY' and np.any(np.isnan(coords['CB'][i]))
        )
        print(f"  {pdb_id} chain {chain_id}: "
              f"{n_res} residues, {len(atom_names_set)} atom types, "
              f"{len(chain_breaks)} breaks, "
              f"{n_missing_bb} missing backbone atoms, "
              f"{n_missing_cb} missing non-GLY Cβ ({n_gly} GLY)")

    return Structure(
        pdb_id=pdb_id,
        chain_id=chain_id,
        sequence=sequence,
        n_res=n_res,
        coords=coords,
        bfactors=bfactors,
        omega_measured=omega_measured,
        chain_breaks=chain_breaks,
    )


_ONE = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
        'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
        'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

def _three_to_one(r):
    return _ONE.get(r, 'X')


# ══════════════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════════════

def _self_test(pdb_path):
    """Load a PDB and print a readable sanity report."""
    print("=" * 70)
    print("pdb_loader.py — self-test")
    print("=" * 70)

    s = load_structure(pdb_path, verbose=True)

    print(f"\npdb_id:     {s.pdb_id}")
    print(f"chain:      {s.chain_id}")
    print(f"n_res:      {s.n_res}")
    print(f"sequence:   {''.join(_three_to_one(r) for r in s.sequence[:50])}"
          f"{'...' if s.n_res > 50 else ''}")
    print(f"atoms:      {sorted(s.coords.keys())}")
    print(f"breaks:     {s.chain_breaks if s.chain_breaks else 'none'}")

    # Verify backbone integrity — C(i)→N(i+1) distances
    print("\nC(i) → N(i+1) bond lengths (should be ~1.33 Å):")
    bond_cn = []
    for i in range(min(5, s.n_res - 1)):
        d = np.linalg.norm(s.coords['N'][i + 1] - s.coords['C'][i])
        bond_cn.append(d)
        print(f"  res {i:3d}→{i+1:3d}: {d:.3f} Å")

    all_cn = []
    for i in range(s.n_res - 1):
        c = s.coords['C'][i]
        n = s.coords['N'][i + 1]
        if not (np.any(np.isnan(c)) or np.any(np.isnan(n))):
            all_cn.append(float(np.linalg.norm(n - c)))
    if all_cn:
        print(f"  All {len(all_cn)}: mean={np.mean(all_cn):.3f}Å  "
              f"std={np.std(all_cn):.3f}  "
              f"range=[{min(all_cn):.3f}, {max(all_cn):.3f}]")

    # Omega check
    om = s.omega_measured[~np.isnan(s.omega_measured)]
    if len(om):
        om_deg = np.degrees(om)
        # Circular mean/std: shift all values into a window centred on the
        # circular mean so wrap-around at ±180° doesn't inflate the std.
        # np.angle(mean(exp(i*x))) gives the true circular mean.
        circ_mean = np.degrees(np.angle(np.mean(np.exp(1j * om))))
        # Wrap each value to [circ_mean-180, circ_mean+180] before computing std
        delta = ((om_deg - circ_mean) + 180) % 360 - 180
        circ_std = delta.std()
        print(f"\nomega (deg): mean={circ_mean:.1f}  std={circ_std:.1f}  "
              f"median={float(np.median(om_deg)):.1f}  "
              f"range=[{om_deg.min():.0f}, {om_deg.max():.0f}]")
        n_cis = np.sum(np.abs(om_deg) < 90)
        print(f"  cis peptides (|omega| < 90°): {n_cis}")
        p1, p99 = np.percentile(delta, [1, 99])
        print(f"  1st–99th percentile: [{circ_mean + p1:.1f}°, {circ_mean + p99:.1f}°]")

    # Verify H placement on residue 5 (middle of structure)
    if s.n_res > 5:
        print("\nSample H placement check (residue 5):")
        N = s.coords['N'][5]
        H = s.coords['H'][5]
        C_prev = s.coords['C'][4]
        O_prev = s.coords['O'][4]
        if not np.any(np.isnan(H)):
            d_NH = np.linalg.norm(H - N)
            # Verify H direction matches Kabsch-Sander
            expected = N + 1.01 * (C_prev - O_prev) / np.linalg.norm(C_prev - O_prev)
            d_err = np.linalg.norm(H - expected)
            print(f"  |N-H| = {d_NH:.4f} Å (expected 1.0100)")
            print(f"  H matches place_H_dssp formula: {d_err < 1e-6}")

    # Missing atom accounting
    print("\nAtom counts per residue type:")
    # PRO residues must have H = NaN (no amide hydrogen)
    pro_indices = [i for i, r in enumerate(s.sequence) if r == 'PRO']
    if pro_indices:
        pro_H = s.coords['H'][pro_indices]
        n_pro_with_H = int(np.sum(~np.any(np.isnan(pro_H), axis=1)))
        assert n_pro_with_H == 0, (
            f"FAIL: {n_pro_with_H} PRO residues have a placed H — "
            f"prolines must not have amide H"
        )
        print(f"\n[OK] All {len(pro_indices)} PRO residues correctly have H=NaN "
              f"(indices: {pro_indices})")
    atom_names = sorted(s.coords.keys())
    missing_counts = {a: int(np.sum(np.any(np.isnan(s.coords[a]), axis=1)))
                      for a in atom_names}
    for a in ('N', 'CA', 'C', 'O', 'CB', 'H'):
        if a in missing_counts:
            m = missing_counts[a]
            pct = 100 * m / s.n_res
            print(f"  {a:<3s}: {s.n_res - m:4d} present, "
                  f"{m:4d} missing ({pct:5.1f}%)")

    # B-factor sanity (should be in [0, 100] for real structures)
    bf_ca = s.bfactors.get('CA')
    if bf_ca is not None:
        bf_ca_valid = bf_ca[~np.isnan(bf_ca)]
        if len(bf_ca_valid):
            print(f"\nCA B-factors: mean={bf_ca_valid.mean():.1f}  "
                  f"range=[{bf_ca_valid.min():.1f}, {bf_ca_valid.max():.1f}]")

    print("\n[PASS] pdb_loader.py loaded structure successfully")
    return True


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdb', required=True, help='Path to PDB file')
    ap.add_argument('--chain', default=None, help='Chain ID (default: first protein chain)')
    args = ap.parse_args()

    try:
        ok = _self_test(args.pdb)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0 if ok else 1)