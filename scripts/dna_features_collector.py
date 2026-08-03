#!/usr/bin/env python3
"""
dna_features_collector.py — DNA backbone geometry extraction for MechLib
============================================================================
Extracts the sugar-phosphate backbone torsions, glycosidic angle, and
sugar pucker for every nucleotide in a set of DNA PDB structures.

BACKBONE TORSIONS (6, per nucleotide i; P belongs to residue i, links
residue i-1's O3' to residue i's O5'):
    alpha  (a): O3'(i-1) - P(i)   - O5'(i)  - C5'(i)
    beta   (b): P(i)     - O5'(i) - C5'(i)  - C4'(i)
    gamma  (g): O5'(i)   - C5'(i) - C4'(i)  - C3'(i)
    delta  (d): C5'(i)   - C4'(i) - C3'(i)  - O3'(i)
    epsilon(e): C4'(i)   - C3'(i) - O3'(i)  - P(i+1)
    zeta   (z): C3'(i)   - O3'(i) - P(i+1)  - O5'(i+1)

GLYCOSIDIC TORSION (chi):
    Purines   (A, G): O4' - C1' - N9 - C4
    Pyrimidines (C, T): O4' - C1' - N1 - C2

SUGAR PUCKER (Altona-Sundaralingam pseudorotation):
    Ring torsions v0..v4 around C1'-C2'-C3'-C4'-O4':
      v0: C4'-O4'-C1'-C2'
      v1: O4'-C1'-C2'-C3'
      v2: C1'-C2'-C3'-C4'
      v3: C2'-C3'-C4'-O4'
      v4: C3'-C4'-O4'-C1'
    Phase angle P = atan2( sum_k v_k*sin(4pi*k/5), sum_k v_k*cos(4pi*k/5) )
    Amplitude   Amp = sqrt( sum_k v_k^2 ) * sqrt(2/5)   (v0..v4, k=0..4)

BI/BII CLASSIFICATION:
    BI:  epsilon - zeta  < 0
    BII: epsilon - zeta  > 0
    (Raw eps-zeta value is also reported so you can re-derive either
     sign convention downstream if a paper you're citing uses the flip.)

Usage:
  python dna_features_collector.py --pdb_dir ./dna_pdb_cache --out dna_features.csv
  python dna_features_collector.py --pdb PDB_ID.pdb --out test.csv --verbose
"""

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')

_PURINES = {'A', 'G', 'DA', 'DG'}
_PYRIMIDINES = {'C', 'T', 'U', 'DC', 'DT', 'DU'}


# ══════════════════════════════════════════════════════════════════════════════
# Minimal PDB parser for nucleic acid backbone + sugar + base-anchor atoms
# ══════════════════════════════════════════════════════════════════════════════

_NEEDED_ATOMS = {
    "P", "O5'", "C5'", "C4'", "C3'", "O3'", "C2'", "C1'", "O4'",
    "OP1", "OP2",  # non-bridging phosphate oxygens
    "N9", "C4", "N1", "C2",  # base-anchor atoms for chi (purine: N9,C4; pyrimidine: N1,C2)
}


class NucleicStructure:
    """Holds per-residue atom coordinates for one chain of one PDB entry."""
    def __init__(self, pdb_id, chain_id):
        self.pdb_id = pdb_id
        self.chain_id = chain_id
        self.res_names = []
        self.res_seq_ids = []
        self.coords = {a: [] for a in _NEEDED_ATOMS}
        self.bfactors_c1 = []

    @property
    def n_res(self):
        return len(self.res_names)


def parse_pdb_nucleic(pdb_path):
    """
    Parse a PDB file and return a dict: chain_id -> NucleicStructure,
    for chains that contain nucleic acid residues (DA/DC/DG/DT/A/C/G/U).
    Only ATOM records are used; first altloc encountered wins.
    """
    pdb_id = Path(pdb_path).stem.upper()
    chains = {}
    res_order = {}
    res_name_map = {}
    seen_altloc = {}

    nuc_res_names = {'DA', 'DC', 'DG', 'DT', 'DU', 'A', 'C', 'G', 'U'}

    with open(pdb_path, errors='ignore') as fh:
        for line in fh:
            if not line.startswith('ATOM'):
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21].strip() or 'A'
            res_seq = line[22:26].strip()

            if res_name not in nuc_res_names:
                continue
            if atom_name not in _NEEDED_ATOMS:
                continue

            key = (chain_id, res_seq, atom_name)
            if key in seen_altloc:
                continue
            seen_altloc[key] = True

            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                bf = float(line[60:66])
            except ValueError:
                continue

            chains.setdefault(chain_id, {})
            chains[chain_id].setdefault(res_seq, {})
            chains[chain_id][res_seq][atom_name] = (np.array([x, y, z]), bf)

            res_order.setdefault(chain_id, [])
            if res_seq not in res_order[chain_id]:
                res_order[chain_id].append(res_seq)
            res_name_map.setdefault(chain_id, {})
            res_name_map[chain_id][res_seq] = res_name

    structures = {}
    for chain_id, res_dict in chains.items():
        order = res_order[chain_id]
        s = NucleicStructure(pdb_id, chain_id)
        for res_seq in order:
            atoms = res_dict[res_seq]
            s.res_names.append(res_name_map[chain_id][res_seq])
            s.res_seq_ids.append(res_seq)
            for a in _NEEDED_ATOMS:
                if a in atoms:
                    s.coords[a].append(atoms[a][0])
                else:
                    s.coords[a].append(None)
            c1_bf = atoms.get("C1'", (None, np.nan))[1]
            s.bfactors_c1.append(c1_bf if c1_bf is not None else np.nan)
        if s.n_res >= 3:
            structures[chain_id] = s
    return structures


# ══════════════════════════════════════════════════════════════════════════════
# Dihedral geometry
# ══════════════════════════════════════════════════════════════════════════════

def _dihedral(p0, p1, p2, p3):
    """Dihedral angle in degrees, [-180, 180]. Returns NaN if any point is None."""
    if p0 is None or p1 is None or p2 is None or p3 is None:
        return float('nan')
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def _bond_length(p0, p1):
    """Distance |p0-p1| in Angstrom. Returns NaN if either point is None."""
    if p0 is None or p1 is None:
        return float('nan')
    return float(np.linalg.norm(p0 - p1))


def _bond_angle(p0, p1, p2):
    """Bond angle at vertex p1, in degrees. Returns NaN if any point is None."""
    if p0 is None or p1 is None or p2 is None:
        return float('nan')
    v1 = p0 - p1
    v2 = p2 - p1
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return float('nan')
    cos_a = float(np.dot(v1, v2) / (n1 * n2))
    cos_a = max(-1.0, min(1.0, cos_a))
    return float(np.degrees(np.arccos(cos_a)))


def _pseudorotation(ring_torsions):
    """
    Altona-Sundaralingam pseudorotation phase (deg) and amplitude from the
    5 ring torsions v0..v4 (degrees). Returns (phase_deg, amplitude_deg).
    """
    v = np.asarray(ring_torsions, dtype=float)
    if np.any(np.isnan(v)):
        return float('nan'), float('nan')
    k = np.arange(5)
    num = np.sum(v * np.sin(4 * np.pi * k / 5))
    den = np.sum(v * np.cos(4 * np.pi * k / 5))
    phase = np.degrees(np.arctan2(num, den))
    amp = np.sqrt(2.0 / 5.0) * np.sqrt(np.sum(v ** 2))
    return float(phase), float(amp)


def _sugar_pucker_class(phase_deg):
    """Classic Altona-Sundaralingam pucker classes, 36-degree sectors."""
    if np.isnan(phase_deg):
        return 'undefined'
    p = phase_deg % 360
    classes = [
        (0, 36, 'C3-endo'), (36, 72, 'C4-exo'), (72, 108, 'O4-endo'),
        (108, 144, 'C1-exo'), (144, 180, 'C2-endo'),
        (180, 216, 'C3-exo'), (216, 252, 'C4-endo'), (252, 288, 'O4-exo'),
        (288, 324, 'C1-endo'), (324, 360, 'C2-exo'),
    ]
    for lo, hi, name in classes:
        if lo <= p < hi:
            return name
    return 'undefined'


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction per structure
# ══════════════════════════════════════════════════════════════════════════════

def _detect_chain_breaks(s, max_bond_dist=2.0):
    """
    Detect genuine chain breaks between residue i and i+1 using the actual
    O3'(i)-P(i+1) distance, rather than trusting residue-index adjacency
    alone. Some depositions (e.g. multi-stranded junctions, several DNA
    copies sharing one chain ID) do not reliably signal a strand break via
    chain ID or residue numbering, which can cause the parser to compute a
    "bond" between two atoms that are not actually chemically connected --
    producing wild, unphysical bond lengths (we have observed values of
    6-8 Angstrom for what should be a ~1.61 A P-O ester bond) that also
    corrupt the derived epsilon/zeta torsions and BI/BII classification
    for the affected residues.

    Returns a set of indices i (0 <= i < n-1) for which the bond FROM
    residue i TO residue i+1 is broken.
    """
    n = s.n_res
    breaks = set()
    O3 = s.coords["O3'"]
    P = s.coords['P']
    for i in range(n - 1):
        o3_i = O3[i]
        p_next = P[i + 1]
        if o3_i is None or p_next is None:
            breaks.add(i)
            continue
        d = float(np.linalg.norm(o3_i - p_next))
        if d > max_bond_dist:
            breaks.add(i)
    return breaks


def extract_dna_features(s: NucleicStructure):
    """Extract backbone torsions, chi, and sugar pucker for every nucleotide."""
    n = s.n_res
    c = s.coords
    rows = []
    chain_breaks = _detect_chain_breaks(s)

    for i in range(n):
        res_name = s.res_names[i].strip()
        is_purine = res_name in _PURINES
        is_pyrimidine = res_name in _PYRIMIDINES
        if not (is_purine or is_pyrimidine):
            continue

        P_i    = c['P'][i]
        O5_i   = c["O5'"][i]
        C5_i   = c["C5'"][i]
        C4_i   = c["C4'"][i]
        C3_i   = c["C3'"][i]
        O3_i   = c["O3'"][i]
        C2_i   = c["C2'"][i]
        C1_i   = c["C1'"][i]
        O4_i   = c["O4'"][i]

        # Only use the neighbouring residue's atoms if the bond connecting
        # them is a real, geometrically verified bond (see
        # _detect_chain_breaks) -- not just adjacent in residue index.
        O3_prev = c["O3'"][i - 1] if (i > 0 and (i - 1) not in chain_breaks) else None
        P_next  = c['P'][i + 1] if (i + 1 < n and i not in chain_breaks) else None
        O5_next = c["O5'"][i + 1] if (i + 1 < n and i not in chain_breaks) else None
        OP1_i   = c['OP1'][i]
        OP2_i   = c['OP2'][i]

        alpha   = _dihedral(O3_prev, P_i, O5_i, C5_i)
        beta    = _dihedral(P_i, O5_i, C5_i, C4_i)
        gamma   = _dihedral(O5_i, C5_i, C4_i, C3_i)
        delta   = _dihedral(C5_i, C4_i, C3_i, O3_i)
        epsilon = _dihedral(C4_i, C3_i, O3_i, P_next)
        zeta    = _dihedral(C3_i, O3_i, P_next, O5_next)

        if is_purine:
            N9_i = c['N9'][i]; C4_base_i = c['C4'][i]
            chi = _dihedral(O4_i, C1_i, N9_i, C4_base_i)
            N_i = N9_i
        else:
            N1_i = c['N1'][i]; C2_base_i = c['C2'][i]
            chi = _dihedral(O4_i, C1_i, N1_i, C2_base_i)
            N_i = N1_i

        v0 = _dihedral(C4_i, O4_i, C1_i, C2_i)
        v1 = _dihedral(O4_i, C1_i, C2_i, C3_i)
        v2 = _dihedral(C1_i, C2_i, C3_i, C4_i)
        v3 = _dihedral(C2_i, C3_i, C4_i, O4_i)
        v4 = _dihedral(C3_i, C4_i, O4_i, C1_i)
        phase, amp = _pseudorotation([v0, v1, v2, v3, v4])
        pucker_class = _sugar_pucker_class(phase)

        # ── Bond lengths (compared against AMBER parm10.dat) ──────────────
        bond_P_O5    = _bond_length(P_i, O5_i)               # OS-P,  eq=1.610
        bond_O5_C5   = _bond_length(O5_i, C5_i)               # CT-OS, eq=1.410
        bond_C5_C4   = _bond_length(C5_i, C4_i)               # CT-CT, eq=1.526
        bond_C4_O4   = _bond_length(C4_i, O4_i)               # CT-OS, eq=1.410
        bond_C4_C3   = _bond_length(C4_i, C3_i)               # CT-CT, eq=1.526
        bond_C3_O3   = _bond_length(C3_i, O3_i)               # CT-OS, eq=1.410
        bond_C3_C2   = _bond_length(C3_i, C2_i)               # CT-CT, eq=1.526
        bond_C2_C1   = _bond_length(C2_i, C1_i)               # CT-CT, eq=1.526
        bond_C1_O4   = _bond_length(C1_i, O4_i)               # CT-OS, eq=1.410 (ring closure)
        bond_C1_N    = _bond_length(C1_i, N_i)                # CT-N*, eq=1.475 (glycosidic)
        bond_O3_Pnext = _bond_length(O3_i, P_next)            # OS-P,  eq=1.610 (phosphodiester)

        # ── Bond angles (compared against AMBER parm10.dat) ───────────────
        angle_O5_P_O3prev = _bond_angle(O5_i, P_i, O3_prev)       # OS-P-OS,  eq=102.60
        angle_OP1_P_OP2    = _bond_angle(OP1_i, P_i, OP2_i)       # OP-P-OP,  eq=119.90
        angle_C5_O5_P      = _bond_angle(C5_i, O5_i, P_i)         # CT-OS-P,  eq=120.50
        angle_C5_C4_O4     = _bond_angle(C5_i, C4_i, O4_i)        # CT-CT-OS, eq=109.50
        angle_C5_C4_C3     = _bond_angle(C5_i, C4_i, C3_i)        # CT-CT-CT, eq=109.50
        angle_O4_C4_C3     = _bond_angle(O4_i, C4_i, C3_i)        # CT-CT-OS, eq=109.50
        angle_C4_C3_O3     = _bond_angle(C4_i, C3_i, O3_i)        # CT-CT-OS, eq=109.50
        angle_C4_C3_C2     = _bond_angle(C4_i, C3_i, C2_i)        # CT-CT-CT, eq=109.50
        angle_O3_C3_C2     = _bond_angle(O3_i, C3_i, C2_i)        # CT-CT-OS, eq=109.50
        angle_C3_C2_C1     = _bond_angle(C3_i, C2_i, C1_i)        # CT-CT-CT, eq=109.50
        angle_C2_C1_O4     = _bond_angle(C2_i, C1_i, O4_i)        # CT-CT-OS, eq=109.50
        angle_C1_O4_C4     = _bond_angle(C1_i, O4_i, C4_i)        # CT-OS-CT, eq=109.50 (ring closure)
        angle_O4_C1_N      = _bond_angle(O4_i, C1_i, N_i)         # OS-CT-N*, eq=109.50 (anomeric)
        angle_C2_C1_N      = _bond_angle(C2_i, C1_i, N_i)         # CT-CT-N*, eq=109.50
        angle_C3_O3_Pnext  = _bond_angle(C3_i, O3_i, P_next)      # CT-OS-P,  eq=120.50

        eps_zeta_raw = (epsilon - zeta if not (np.isnan(epsilon) or np.isnan(zeta))
                        else float('nan'))
        # Wrap into [-180, 180] before classifying — the raw difference of
        # two angles each individually wrapped to [-180,180] can swing to
        # +-360, which flips the BI/BII sign call if left unwrapped.
        if np.isnan(eps_zeta_raw):
            eps_zeta = float('nan')
        else:
            eps_zeta = ((eps_zeta_raw + 180.0) % 360.0) - 180.0
        bi_bii = 'undefined' if np.isnan(eps_zeta) else ('BI' if eps_zeta < 0 else 'BII')

        rows.append({
            'pdb_id': s.pdb_id, 'chain': s.chain_id,
            'res_idx': i, 'res_seq': s.res_seq_ids[i], 'res_name': res_name,
            'base_type': 'purine' if is_purine else 'pyrimidine',
            'alpha_deg': round(alpha, 3) if not np.isnan(alpha) else float('nan'),
            'beta_deg':  round(beta, 3)  if not np.isnan(beta)  else float('nan'),
            'gamma_deg': round(gamma, 3) if not np.isnan(gamma) else float('nan'),
            'delta_deg': round(delta, 3) if not np.isnan(delta) else float('nan'),
            'epsilon_deg': round(epsilon, 3) if not np.isnan(epsilon) else float('nan'),
            'zeta_deg':  round(zeta, 3)  if not np.isnan(zeta)  else float('nan'),
            'chi_deg':   round(chi, 3)   if not np.isnan(chi)   else float('nan'),
            'eps_zeta_deg': round(eps_zeta, 3) if not np.isnan(eps_zeta) else float('nan'),
            'bi_bii': bi_bii,
            'pucker_phase_deg': round(phase, 3) if not np.isnan(phase) else float('nan'),
            'pucker_amplitude_deg': round(amp, 3) if not np.isnan(amp) else float('nan'),
            'pucker_class': pucker_class,
            'bfactor_c1': round(float(s.bfactors_c1[i]), 2)
                          if not np.isnan(s.bfactors_c1[i]) else float('nan'),
            # Bond lengths
            'bond_P_O5': round(bond_P_O5, 4) if not np.isnan(bond_P_O5) else float('nan'),
            'bond_O5_C5': round(bond_O5_C5, 4) if not np.isnan(bond_O5_C5) else float('nan'),
            'bond_C5_C4': round(bond_C5_C4, 4) if not np.isnan(bond_C5_C4) else float('nan'),
            'bond_C4_O4': round(bond_C4_O4, 4) if not np.isnan(bond_C4_O4) else float('nan'),
            'bond_C4_C3': round(bond_C4_C3, 4) if not np.isnan(bond_C4_C3) else float('nan'),
            'bond_C3_O3': round(bond_C3_O3, 4) if not np.isnan(bond_C3_O3) else float('nan'),
            'bond_C3_C2': round(bond_C3_C2, 4) if not np.isnan(bond_C3_C2) else float('nan'),
            'bond_C2_C1': round(bond_C2_C1, 4) if not np.isnan(bond_C2_C1) else float('nan'),
            'bond_C1_O4': round(bond_C1_O4, 4) if not np.isnan(bond_C1_O4) else float('nan'),
            'bond_C1_N': round(bond_C1_N, 4) if not np.isnan(bond_C1_N) else float('nan'),
            'bond_O3_Pnext': round(bond_O3_Pnext, 4) if not np.isnan(bond_O3_Pnext) else float('nan'),
            # Bond angles
            'angle_O5_P_O3prev': round(angle_O5_P_O3prev, 3) if not np.isnan(angle_O5_P_O3prev) else float('nan'),
            'angle_OP1_P_OP2': round(angle_OP1_P_OP2, 3) if not np.isnan(angle_OP1_P_OP2) else float('nan'),
            'angle_C5_O5_P': round(angle_C5_O5_P, 3) if not np.isnan(angle_C5_O5_P) else float('nan'),
            'angle_C5_C4_O4': round(angle_C5_C4_O4, 3) if not np.isnan(angle_C5_C4_O4) else float('nan'),
            'angle_C5_C4_C3': round(angle_C5_C4_C3, 3) if not np.isnan(angle_C5_C4_C3) else float('nan'),
            'angle_O4_C4_C3': round(angle_O4_C4_C3, 3) if not np.isnan(angle_O4_C4_C3) else float('nan'),
            'angle_C4_C3_O3': round(angle_C4_C3_O3, 3) if not np.isnan(angle_C4_C3_O3) else float('nan'),
            'angle_C4_C3_C2': round(angle_C4_C3_C2, 3) if not np.isnan(angle_C4_C3_C2) else float('nan'),
            'angle_O3_C3_C2': round(angle_O3_C3_C2, 3) if not np.isnan(angle_O3_C3_C2) else float('nan'),
            'angle_C3_C2_C1': round(angle_C3_C2_C1, 3) if not np.isnan(angle_C3_C2_C1) else float('nan'),
            'angle_C2_C1_O4': round(angle_C2_C1_O4, 3) if not np.isnan(angle_C2_C1_O4) else float('nan'),
            'angle_C1_O4_C4': round(angle_C1_O4_C4, 3) if not np.isnan(angle_C1_O4_C4) else float('nan'),
            'angle_O4_C1_N': round(angle_O4_C1_N, 3) if not np.isnan(angle_O4_C1_N) else float('nan'),
            'angle_C2_C1_N': round(angle_C2_C1_N, 3) if not np.isnan(angle_C2_C1_N) else float('nan'),
            'angle_C3_O3_Pnext': round(angle_C3_O3_Pnext, 3) if not np.isnan(angle_C3_O3_Pnext) else float('nan'),
        })

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

ALL_COLS = [
    'pdb_id', 'chain', 'res_idx', 'res_seq', 'res_name', 'base_type',
    'alpha_deg', 'beta_deg', 'gamma_deg', 'delta_deg', 'epsilon_deg', 'zeta_deg',
    'chi_deg', 'eps_zeta_deg', 'bi_bii',
    'pucker_phase_deg', 'pucker_amplitude_deg', 'pucker_class',
    'bfactor_c1',
    'bond_P_O5', 'bond_O5_C5', 'bond_C5_C4', 'bond_C4_O4', 'bond_C4_C3',
    'bond_C3_O3', 'bond_C3_C2', 'bond_C2_C1', 'bond_C1_O4', 'bond_C1_N',
    'bond_O3_Pnext',
    'angle_O5_P_O3prev', 'angle_OP1_P_OP2', 'angle_C5_O5_P',
    'angle_C5_C4_O4', 'angle_C5_C4_C3', 'angle_O4_C4_C3', 'angle_C4_C3_O3',
    'angle_C4_C3_C2', 'angle_O3_C3_C2', 'angle_C3_C2_C1', 'angle_C2_C1_O4',
    'angle_C1_O4_C4', 'angle_O4_C1_N', 'angle_C2_C1_N', 'angle_C3_O3_Pnext',
]


def run_pipeline(pdb_paths, out_csv, verbose=False):
    t0 = time.time()
    n_pdbs = n_res_total = n_skipped = 0

    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=ALL_COLS, extrasaction='ignore')
        w.writeheader()

        for pdb_path in pdb_paths:
            try:
                structures = parse_pdb_nucleic(pdb_path)
            except Exception as e:
                if verbose:
                    print(f'  [SKIP] {pdb_path.name}: {e}')
                n_skipped += 1
                continue

            if not structures:
                n_skipped += 1
                continue

            def _fmt(v):
                if isinstance(v, float) and (v != v):
                    return ''
                return v

            n_res_this_pdb = 0
            for chain_id, s in structures.items():
                try:
                    rows = extract_dna_features(s)
                except Exception as e:
                    if verbose:
                        print(f'  [SKIP] {pdb_path.name} chain {chain_id}: {e}')
                    continue
                for row in rows:
                    w.writerow({c: _fmt(row.get(c, '')) for c in ALL_COLS})
                n_res_this_pdb += len(rows)

            n_pdbs += 1
            n_res_total += n_res_this_pdb
            if verbose:
                print(f'  {pdb_path.name:20s}  {n_res_this_pdb:4d} nt  '
                      f'cumulative: {n_res_total:7d}')

    return dict(n_pdbs=n_pdbs, n_residues=n_res_total,
                n_skipped=n_skipped, elapsed_s=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser(
        description='Extract DNA backbone dihedrals, chi, and sugar pucker')
    ap.add_argument('--pdb', default=None)
    ap.add_argument('--pdb_dir', default=None)
    ap.add_argument('--out', default='dna_features.csv')
    ap.add_argument('--max_pdbs', type=int, default=None)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    if args.pdb:
        pdb_paths = [Path(args.pdb)]
    elif args.pdb_dir:
        pdb_paths = sorted(Path(args.pdb_dir).glob('*.pdb'))
        if args.max_pdbs:
            pdb_paths = pdb_paths[:args.max_pdbs]
        if not pdb_paths:
            print(f"No *.pdb files found in {args.pdb_dir}")
            sys.exit(1)
    else:
        ap.print_help()
        print("\nERROR: must specify --pdb or --pdb_dir")
        sys.exit(1)

    print(f"Processing {len(pdb_paths)} PDB file(s) -> {args.out}")
    stats = run_pipeline(pdb_paths, Path(args.out), verbose=args.verbose)
    print(f"\nDone.")
    print(f"  PDBs processed    : {stats['n_pdbs']}")
    print(f"  Nucleotides total : {stats['n_residues']:,}")
    print(f"  Skipped           : {stats['n_skipped']}")
    print(f"  Elapsed           : {stats['elapsed_s']} s")
    print(f"  Written           : {args.out}")


if __name__ == '__main__':
    main()
