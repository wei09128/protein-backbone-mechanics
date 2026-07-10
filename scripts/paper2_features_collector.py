"""
features_collector.py — Real-coordinate feature extraction v6
===============================================================

v6 differences from v5:
  - All coordinates from real PDB (no NeRF reconstruction)
  - Correct Kabsch-Sander H placement
  - Real omega, not forced to 180°
  - Strict NaN for missing atoms (Policy C)
  - Writes TWO CSVs:
      features_v5.csv — v5 column names and schema preserved for paper 1
      features_p2.csv — new clean-named columns for paper 2 (bond lengths,
                        Cβ angles, impropers, measured omega)

Current implementation status:
    [2a] Geometry + targets + Group C context features ← this file
    [2b] Group B (torques + H-bond summary)             ← next
    [2c] Group A (steric field) + orchestrator          ← after that

Run the self-test:
    python features_collector.py --self_test
"""

import argparse
import csv
import sys
import time
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

from pdb_loader import load_structure, Structure
from molcore import dihedral_angle as _dihedral_rad
from hbond_finder import find_hbonds


# ══════════════════════════════════════════════════════════════════════════════
# Constants — all match v5's collect_backbone_features.py exactly
# ══════════════════════════════════════════════════════════════════════════════

_SIDECHAIN_MASS = {
    'GLY':0.0,   'ALA':15.0,  'VAL':43.0,  'LEU':57.0,  'ILE':57.0,
    'PRO':42.0,  'PHE':91.0,  'TYR':107.0, 'TRP':130.0, 'SER':31.0,
    'THR':45.0,  'CYS':47.0,  'MET':75.0,  'ASP':58.0,  'ASN':58.0,
    'GLU':72.0,  'GLN':72.0,  'LYS':72.0,  'ARG':100.0, 'HIS':81.0,
}

_RES_ORDER = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
              'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
_RES_IDX = {r: i for i, r in enumerate(_RES_ORDER)}

_HAS_CHI2 = {'ARG','ASN','ASP','GLN','GLU','HIS','ILE','LEU','LYS','MET',
             'PHE','TRP','TYR'}

_SC_N_HEAVY = {
    'GLY':0,'ALA':1,'VAL':3,'LEU':4,'ILE':4,'PRO':3,'PHE':7,'TYR':8,
    'TRP':10,'SER':2,'THR':3,'CYS':2,'MET':4,'ASP':4,'ASN':4,'GLU':5,
    'GLN':5,'LYS':5,'ARG':7,'HIS':6,
}

_SC_N_ROTATABLE = {
    'GLY':0,'ALA':0,'VAL':1,'LEU':2,'ILE':2,'PRO':0,'PHE':2,'TYR':2,
    'TRP':2,'SER':1,'THR':1,'CYS':1,'MET':3,'ASP':2,'ASN':2,'GLU':3,
    'GLN':3,'LYS':4,'ARG':4,'HIS':2,
}

_SC_BRANCHED = {'VAL','ILE','THR'}
_SC_AROMATIC = {'PHE','TYR','TRP','HIS'}

# First-atom name for chi1 dihedral (fourth atom: N - CA - CB - X)
_CHI1_FOURTH_ATOM = {
    'SER':'OG','THR':'OG1','CYS':'SG','VAL':'CG1','LEU':'CG','ILE':'CG1',
    'MET':'CG','PHE':'CG','TYR':'CG','TRP':'CG','HIS':'CG','ASP':'CG',
    'ASN':'CG','GLU':'CG','GLN':'CG','LYS':'CG','ARG':'CG','PRO':'CG',
}

# Atom chains for chi2 dihedral (four atoms: CA - CB - X - Y)
_CHI2_CHAIN = {
    'LEU': ('CA','CB','CG','CD1'),
    'ILE': ('CA','CB','CG1','CD1'),
    'MET': ('CA','CB','CG','SD'),
    'PHE': ('CA','CB','CG','CD1'),
    'TYR': ('CA','CB','CG','CD1'),
    'TRP': ('CA','CB','CG','CD1'),
    'HIS': ('CA','CB','CG','ND1'),
    'ASP': ('CA','CB','CG','OD1'),
    'ASN': ('CA','CB','CG','OD1'),
    'GLU': ('CA','CB','CG','CD'),
    'GLN': ('CA','CB','CG','CD'),
    'LYS': ('CA','CB','CG','CD'),
    'ARG': ('CA','CB','CG','CD'),
    'PRO': ('CA','CB','CG','CD'),
}


# ══════════════════════════════════════════════════════════════════════════════
# Geometric primitives — NaN-safe wrappers around numpy
# ══════════════════════════════════════════════════════════════════════════════

def _safe_bond_length(a, b):
    """Distance |a - b| or NaN if any input has NaN."""
    a = np.asarray(a); b = np.asarray(b)
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        return float('nan')
    return float(np.linalg.norm(a - b))


def _safe_bond_angle(a, b, c):
    """Bond angle at vertex b (degrees) or NaN if any input has NaN."""
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return float('nan')
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return float('nan')
    cos_a = float(np.dot(v1, v2) / (n1 * n2))
    cos_a = max(-1.0, min(1.0, cos_a))
    return float(np.degrees(np.arccos(cos_a)))


def _safe_dihedral_deg(a, b, c, d):
    """Dihedral angle in degrees, using molcore's convention. NaN-safe."""
    a = np.asarray(a); b = np.asarray(b); c = np.asarray(c); d = np.asarray(d)
    if any(np.any(np.isnan(p)) for p in (a, b, c, d)):
        return float('nan')
    return float(np.degrees(_dihedral_rad(a, b, c, d)))


def _safe_improper(center, p1, p2, p3):
    """Improper dihedral at `center` via p1-center-p2-p3. NaN-safe."""
    return _safe_dihedral_deg(p1, center, p2, p3)


# ══════════════════════════════════════════════════════════════════════════════
# Ramachandran binning — identical to v5
# ══════════════════════════════════════════════════════════════════════════════

def _ss_bin(phi_deg, psi_deg):
    """
    Coarse Ramachandran region bin (matches v5 exactly).
      0 = αR   1 = β    2 = PPII   3 = 3₁₀   4 = loop   5 = αL
    """
    if np.isnan(phi_deg) or np.isnan(psi_deg):
        return 4  # loop (matches v5 fallback)
    p, q = phi_deg, psi_deg
    if p > 0 and -20 <= q <= 80:
        return 5
    if -100 <= p <= -40 and -60 <= q <= 20:
        return 0
    if p <= -90 and q >= 90:
        return 1
    if -90 <= p <= -50 and q >= 120:
        return 2
    if -80 <= p <= -30 and -40 <= q <= 0:
        return 3
    return 4


# ══════════════════════════════════════════════════════════════════════════════
# Dihedral extraction — phi, psi, omega, chi1, chi2 from real coordinates
# ══════════════════════════════════════════════════════════════════════════════

def _compute_phi_psi_arrays(s: Structure):
    """
    Compute phi and psi arrays (in radians) from the real PDB coordinates.
    NaN for residues at chain breaks or with missing backbone atoms.

    phi(i) = dihedral(C[i-1], N[i], CA[i], C[i])    (undefined for i=0)
    psi(i) = dihedral(N[i], CA[i], C[i], N[i+1])    (undefined for last)
    """
    n = s.n_res
    phi = np.full(n, np.nan)
    psi = np.full(n, np.nan)
    N = s.coords['N']; CA = s.coords['CA']; C = s.coords['C']

    for i in range(1, n):
        if (i - 1) in s.chain_breaks:
            continue
        if any(np.any(np.isnan(p)) for p in (C[i-1], N[i], CA[i], C[i])):
            continue
        try:
            phi[i] = _dihedral_rad(C[i-1], N[i], CA[i], C[i])
        except Exception:
            pass

    for i in range(n - 1):
        if i in s.chain_breaks:
            continue
        if any(np.any(np.isnan(p)) for p in (N[i], CA[i], C[i], N[i+1])):
            continue
        try:
            psi[i] = _dihedral_rad(N[i], CA[i], C[i], N[i+1])
        except Exception:
            pass

    return phi, psi


def _compute_chi1(s: Structure, res_idx: int):
    """
    Compute chi1 (radians) for a single residue, or None if missing atoms
    or the residue has no chi1.
    """
    res = s.sequence[res_idx]
    if res not in _CHI1_FOURTH_ATOM:
        return None
    fourth = _CHI1_FOURTH_ATOM[res]
    if fourth not in s.coords:
        return None
    N  = s.coords['N'][res_idx]
    CA = s.coords['CA'][res_idx]
    CB = s.coords['CB'][res_idx]
    X  = s.coords[fourth][res_idx]
    if any(np.any(np.isnan(p)) for p in (N, CA, CB, X)):
        return None
    try:
        return float(_dihedral_rad(N, CA, CB, X))
    except Exception:
        return None


def _compute_chi2(s: Structure, res_idx: int):
    """
    Compute chi2 (radians) for a single residue, or None if missing atoms
    or the residue has no chi2.
    """
    res = s.sequence[res_idx]
    if res not in _CHI2_CHAIN:
        return None
    chain = _CHI2_CHAIN[res]
    pts = []
    for a in chain:
        if a not in s.coords:
            return None
        p = s.coords[a][res_idx]
        if np.any(np.isnan(p)):
            return None
        pts.append(p)
    try:
        return float(_dihedral_rad(*pts))
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction — Sub-chunk 2a
# ══════════════════════════════════════════════════════════════════════════════

def extract_features_2a(s: Structure, res_idx: int,
                        phi_rad: np.ndarray, psi_rad: np.ndarray):
    """
    Extract geometry + targets + Group C context features for one residue.

    Returns a dict with:
      - Metadata: pdb_id, chain, res_idx, res_name, res_type
      - Targets: phi_deg, psi_deg, omega_deg, ss_bin
      - v5 Group C context: chi1_rad, chi2_rad, has_chi1, has_chi2,
        sc_mass, sc_n_heavy, sc_n_rotatable, sc_rigidity, sc_is_branched,
        sc_is_aromatic, sc_lever_arm, bfactor_ca, is_pro_np1,
        angle_NCaC, angle_CaCN, angle_CNCa, dist_ca_m2, dist_ca_p2,
        sc_mass_nm1, sc_mass_np1
      - Paper 2 geometry: bond_N_CA, bond_CA_C, bond_C_O, bond_C_N_next,
        bond_CA_CB, angle_N_CA_CB, angle_C_CA_CB, angle_CA_C_O,
        angle_O_C_N_next, improper_CB_out_of_plane, improper_ca, tau_deg,
        omega_measured_deg

    Group A (steric) and Group B (forces) will be added in sub-chunks 2c/2b.
    """
    n = s.n_res
    i = res_idx
    res = s.sequence[i]

    N  = s.coords['N'][i]
    CA = s.coords['CA'][i]
    C  = s.coords['C'][i]
    O  = s.coords['O'][i]
    CB = s.coords['CB'][i] if 'CB' in s.coords else np.full(3, np.nan)

    phi_i = phi_rad[i]
    psi_i = psi_rad[i]
    phi_deg = float(np.degrees(phi_i)) if not np.isnan(phi_i) else float('nan')
    psi_deg = float(np.degrees(psi_i)) if not np.isnan(psi_i) else float('nan')

    # omega for residue i is the dihedral CA(i)-C(i)-N(i+1)-CA(i+1),
    # stored at index i in s.omega_measured. If i == n-1, no omega defined.
    if i < n - 1 and i not in s.chain_breaks:
        om_rad = s.omega_measured[i]
        omega_deg = float(np.degrees(om_rad)) if not np.isnan(om_rad) else 180.0
    else:
        omega_deg = 180.0   # v5 default for terminal residue

    # ── Metadata + targets ────────────────────────────────────────────────────
    f = {
        'pdb_id':    s.pdb_id,
        'chain':     s.chain_id,
        'res_idx':   i,
        'res_name':  res,
        'res_type':  _RES_IDX.get(res, 20),
        'phi_deg':   round(phi_deg, 3) if not np.isnan(phi_deg) else float('nan'),
        'psi_deg':   round(psi_deg, 3) if not np.isnan(psi_deg) else float('nan'),
        'omega_deg': round(omega_deg, 3),
        'ss_bin':    _ss_bin(phi_deg, psi_deg),
    }

    # ── Group C context features ──────────────────────────────────────────────
    chi1 = _compute_chi1(s, i)
    chi2 = _compute_chi2(s, i)
    f['chi1_rad'] = round(chi1, 5) if chi1 is not None else 0.0
    f['has_chi1'] = int(chi1 is not None and res not in ('GLY', 'ALA'))
    f['chi2_rad'] = round(chi2, 5) if chi2 is not None else 0.0
    f['has_chi2'] = int(chi2 is not None and res in _HAS_CHI2)

    sc_mass = _SIDECHAIN_MASS.get(res, 0.0)
    sc_nh   = _SC_N_HEAVY.get(res, 0)
    sc_nr   = _SC_N_ROTATABLE.get(res, 0)
    f['sc_mass']        = sc_mass
    f['sc_n_heavy']     = sc_nh
    f['sc_n_rotatable'] = sc_nr
    f['sc_rigidity']    = round(sc_nh / max(sc_nr, 1), 2)
    f['sc_is_branched'] = int(res in _SC_BRANCHED)
    f['sc_is_aromatic'] = int(res in _SC_AROMATIC)

    # sc_lever_arm: |CA - CB|, or max(|CA-CB|, |CA-CG|) if a CG-like atom exists
    sc_lever = 0.0
    if not np.any(np.isnan(CB)):
        sc_lever = float(np.linalg.norm(CB - CA))
        # If we have the first sidechain atom beyond Cβ, use it for the
        # farther reach (matches v5's use of CG position)
        fourth = _CHI1_FOURTH_ATOM.get(res)
        if fourth and fourth in s.coords:
            p = s.coords[fourth][i]
            if not np.any(np.isnan(p)):
                sc_lever = max(sc_lever, float(np.linalg.norm(p - CA)))
    f['sc_lever_arm'] = round(sc_lever, 3)

    # B-factor for CA
    bf_ca_arr = s.bfactors.get('CA')
    if bf_ca_arr is not None and not np.isnan(bf_ca_arr[i]):
        f['bfactor_ca'] = round(float(bf_ca_arr[i]), 2)
    else:
        f['bfactor_ca'] = 0.0

    # is_pro_np1: 1 if the next residue is proline
    f['is_pro_np1'] = int(i + 1 < n and s.sequence[i + 1] == 'PRO')

    # Backbone bond angles (classical τ and neighbours), in degrees
    # angle_NCaC is the internal angle of residue i (N-Cα-C)
    # angle_CaCN is the angle from residue i to i+1 at the peptide bond
    # angle_CNCa is the angle from residue i-1 to i at the peptide bond
    f['angle_NCaC'] = round(_safe_bond_angle(N, CA, C), 3)

    if i + 1 < n and i not in s.chain_breaks:
        f['angle_CaCN'] = round(
            _safe_bond_angle(CA, C, s.coords['N'][i + 1]), 3)
    else:
        f['angle_CaCN'] = 117.0  # v5 default

    if i > 0 and (i - 1) not in s.chain_breaks:
        f['angle_CNCa'] = round(
            _safe_bond_angle(s.coords['C'][i - 1], N, CA), 3)
    else:
        f['angle_CNCa'] = 121.0  # v5 default

    # dist_ca_m2, dist_ca_p2: distances to CA two residues away
    if i >= 2:
        f['dist_ca_m2'] = round(_safe_bond_length(s.coords['CA'][i - 2], CA), 3)
    else:
        f['dist_ca_m2'] = 0.0
    if i < n - 2:
        f['dist_ca_p2'] = round(_safe_bond_length(s.coords['CA'][i + 2], CA), 3)
    else:
        f['dist_ca_p2'] = 0.0

    # Neighbouring sidechain masses
    f['sc_mass_nm1'] = _SIDECHAIN_MASS.get(s.sequence[i - 1], 0.0) if i > 0 else 0.0
    f['sc_mass_np1'] = _SIDECHAIN_MASS.get(s.sequence[i + 1], 0.0) if i < n - 1 else 0.0

    # ── Paper 2 geometry features ─────────────────────────────────────────────
    # Bond lengths
    f['bond_N_CA']    = _round_or_nan(_safe_bond_length(N, CA),  3)
    f['bond_CA_C']    = _round_or_nan(_safe_bond_length(CA, C),  3)
    f['bond_C_O']     = _round_or_nan(_safe_bond_length(C, O),   3)
    f['bond_CA_CB']   = _round_or_nan(_safe_bond_length(CA, CB), 3)
    if i + 1 < n and i not in s.chain_breaks:
        f['bond_C_N_next'] = _round_or_nan(
            _safe_bond_length(C, s.coords['N'][i + 1]), 3)
    else:
        f['bond_C_N_next'] = float('nan')

    # tau_deg = ∠N-Cα-C (same as angle_NCaC but named for paper 2 clarity)
    f['tau_deg'] = f['angle_NCaC']

    # Cβ angles — the paper 2 centrepiece
    f['angle_N_CA_CB'] = _round_or_nan(_safe_bond_angle(N, CA, CB), 3)
    f['angle_C_CA_CB'] = _round_or_nan(_safe_bond_angle(C, CA, CB), 3)

    # Carbonyl angles
    f['angle_CA_C_O'] = _round_or_nan(_safe_bond_angle(CA, C, O), 3)
    if i + 1 < n and i not in s.chain_breaks:
        f['angle_O_C_N_next'] = _round_or_nan(
            _safe_bond_angle(O, C, s.coords['N'][i + 1]), 3)
    else:
        f['angle_O_C_N_next'] = float('nan')

    # Cα improper dihedral: N - Cα - C - Cβ. Measures L/D chirality and
    # tetrahedral distortion. For L-amino acids the ideal is ~35-37°.
    f['improper_ca'] = _round_or_nan(_safe_improper(CA, N, C, CB), 3)

    # Cβ out-of-plane from N-Cα-C. Another chirality probe.
    f['improper_CB_out_of_plane'] = _round_or_nan(
        _safe_improper(CA, N, CB, C), 3)

    # Measured omega in degrees (with NaN at breaks, not 180° fallback)
    # — this is the paper 2 version, distinct from f['omega_deg'] which
    # uses the v5 fallback convention.
    if i < n - 1 and i not in s.chain_breaks:
        om_rad = s.omega_measured[i]
        f['omega_measured_deg'] = (
            round(float(np.degrees(om_rad)), 3)
            if not np.isnan(om_rad) else float('nan'))
    else:
        f['omega_measured_deg'] = float('nan')

    return f


def _round_or_nan(value, ndigits):
    """Round a float, or return NaN if value is NaN."""
    if np.isnan(value):
        return float('nan')
    return round(value, ndigits)


# ══════════════════════════════════════════════════════════════════════════════
# Column lists — separating v5 schema from paper 2 schema
# ══════════════════════════════════════════════════════════════════════════════

# Identifier columns shared between both CSVs
V5_ID_COLS = ['pdb_id', 'res_idx', 'res_name', 'res_type']
P2_ID_COLS = ['pdb_id', 'chain', 'res_idx', 'res_name']

# v5 columns produced by sub-chunk 2a. Groups B (forces) and A (steric) are
# added in later sub-chunks.
V5_2A_COLS = [
    'phi_deg', 'psi_deg', 'omega_deg', 'ss_bin',
    'chi1_rad', 'has_chi1', 'chi2_rad', 'has_chi2',
    'sc_mass', 'sc_n_heavy', 'sc_n_rotatable', 'sc_rigidity',
    'sc_is_branched', 'sc_is_aromatic', 'sc_lever_arm',
    'bfactor_ca', 'is_pro_np1',
    'angle_NCaC', 'angle_CaCN', 'angle_CNCa',
    'dist_ca_m2', 'dist_ca_p2',
    'sc_mass_nm1', 'sc_mass_np1',
]

# New paper 2 columns
P2_COLS = [
    'bond_N_CA', 'bond_CA_C', 'bond_C_O', 'bond_C_N_next', 'bond_CA_CB',
    'tau_deg', 'angle_N_CA_CB', 'angle_C_CA_CB',
    'angle_CA_C_O', 'angle_O_C_N_next',
    'improper_ca', 'improper_CB_out_of_plane',
    'omega_measured_deg',
]

# ══════════════════════════════════════════════════════════════════════════════
# Sub-chunk 2b — Group B: Forces / torques / H-bond summary
# ══════════════════════════════════════════════════════════════════════════════

V5_2B_COLS = [
    # torques (12)
    'tau_phi_correct',   'tau_psi_correct',
    'tau_phi_bb_donor',  'tau_psi_bb_donor',
    'tau_phi_bb_acc',    'tau_psi_bb_acc',
    'tau_phi_sc_hb',     'tau_psi_sc_hb',
    'tau_phi_steric',    'tau_psi_steric',
    'tau_phi_elec_corr', 'tau_psi_elec_corr',
    # H-bond summary (13)
    'hb_n_bonds',
    'hb_tau_phi_sum',    'hb_tau_phi_rms',    'hb_tau_phi_cancel',
    'hb_tau_psi_rms',    'hb_tau_psi_cancel',
    'hb_mean_d_HO',     'hb_best_d_HO',
    'hb_mean_energy',
    'hb_n_donor',        'hb_n_acceptor',
    'hb_n_strong',       'hb_best_e',
]


# ── Geometry helpers for torques ──────────────────────────────────────────────

def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-10 else np.zeros(3)


def _phi_axes(s, i):
    """Return (N[i], CA[i]) — the rotation bond for phi."""
    return s.coords['N'][i], s.coords['CA'][i]


def _psi_axes(s, i):
    """Return (CA[i], C[i]) — the rotation bond for psi."""
    return s.coords['CA'][i], s.coords['C'][i]


def _torque_cos(axis_a, axis_b, vec_HO_unit, energy):
    """
    τ = energy × cos(angle between rotation axis and H→O vector).

    axis_a, axis_b : atoms defining the rotation bond (e.g. N, CA for φ)
    vec_HO_unit    : pre-normalised H→O unit vector  (list or array)
    energy         : K-S energy in kcal/mol (negative = stable)

    Returns 0.0 if any coordinate is NaN.
    """
    if any(np.any(np.isnan(p)) for p in (axis_a, axis_b)):
        return 0.0
    axis = _unit(axis_b - axis_a)
    return float(energy * np.dot(axis, np.asarray(vec_HO_unit, dtype=float)))


def _r(x, n=5):
    """Round finite float; pass through NaN / inf unchanged."""
    return round(float(x), n) if np.isfinite(x) else float(x)


def _rms(xs):
    return float(np.sqrt(np.mean(np.square(xs)))) if xs else 0.0


def _cancel(xs):
    """Cancellation ratio: |Σxᵢ| / Σ|xᵢ|. 1=all same sign, 0=full cancel."""
    abs_sum = sum(abs(x) for x in xs)
    return abs(sum(xs)) / abs_sum if abs_sum > 1e-10 else 0.0


# ── Steric torque proxy (v5 simplified model) ────────────────────────────────

_STERIC_SCALE = 0.3


def _steric_torque(s, i):
    """
    Repulsion from Cβ atoms of residues i-1 and i+1 projected onto φ/ψ axes.
    Returns (tau_phi_steric, tau_psi_steric).
    """
    n = s.n_res
    CA = s.coords['CA']
    CB = s.coords.get('CB', np.full((n, 3), np.nan))
    phi_a, phi_b = _phi_axes(s, i)
    psi_a, psi_b = _psi_axes(s, i)
    tau_phi = 0.0
    tau_psi = 0.0

    for j in (i - 1, i + 1):
        if j < 0 or j >= n:
            continue
        cb_j = CB[j]
        ca_i = CA[i]
        if np.any(np.isnan(cb_j)) or np.any(np.isnan(ca_i)):
            continue
        d = float(np.linalg.norm(cb_j - ca_i))
        if d < 1e-3:
            continue
        repulsion = _STERIC_SCALE / max(d ** 2, 0.1)
        direction = _unit(ca_i - cb_j)
        if not any(np.any(np.isnan(p)) for p in (phi_a, phi_b)):
            tau_phi += repulsion * float(
                np.dot(_unit(phi_b - phi_a), direction))
        if not any(np.any(np.isnan(p)) for p in (psi_a, psi_b)):
            tau_psi += repulsion * float(
                np.dot(_unit(psi_b - psi_a), direction))

    return float(tau_phi), float(tau_psi)


# ── Electrostatic correction proxy (v5 partial-charge model) ──────────────────

_PARTIAL_Q = {
    'N': -0.415, 'H': +0.271, 'CA': +0.034,
    'C': +0.597, 'O': -0.568,
}


def _elec_torque(s, i):
    """
    Sum of q_j / r² × cos(θ) for backbone partial charges within 5 Å,
    projected onto φ/ψ axes.  Returns (tau_phi_elec, tau_psi_elec).
    """
    n = s.n_res
    phi_a, phi_b = _phi_axes(s, i)
    psi_a, psi_b = _psi_axes(s, i)
    ca_i = s.coords['CA'][i]
    if np.any(np.isnan(ca_i)):
        return 0.0, 0.0

    tau_phi = 0.0
    tau_psi = 0.0
    for j in range(max(0, i - 3), min(n, i + 4)):
        if j == i:
            continue
        for atom_name, q_j in _PARTIAL_Q.items():
            if atom_name not in s.coords:
                continue
            p_j = s.coords[atom_name][j]
            if np.any(np.isnan(p_j)):
                continue
            d = float(np.linalg.norm(p_j - ca_i))
            if d < 1e-3 or d > 5.0:
                continue
            e_term = q_j / max(d ** 2, 0.01)
            direction = _unit(p_j - ca_i)
            if not any(np.any(np.isnan(p)) for p in (phi_a, phi_b)):
                tau_phi += e_term * float(
                    np.dot(_unit(phi_b - phi_a), direction))
            if not any(np.any(np.isnan(p)) for p in (psi_a, psi_b)):
                tau_psi += e_term * float(
                    np.dot(_unit(psi_b - psi_a), direction))

    return float(tau_phi), float(tau_psi)


# ── H-bond index builder ─────────────────────────────────────────────────────

def build_hbond_index(s, e_threshold=-0.3):
    """
    Run find_hbonds on a Structure's real coordinates and build a
    per-residue index.

    Returns (raw_bonds, index) where:
      raw_bonds = list of bond dicts from find_hbonds
      index     = dict[int → list[bond_dict]] keyed by residue index
                  (each bond appears under both its donor and acceptor)
    """
    coords = {
        'N': s.coords['N'],
        'H': s.coords['H'],
        'C': s.coords['C'],
        'O': s.coords['O'],
    }
    raw = find_hbonds(coords, s.sequence, e_threshold=e_threshold)
    index = {i: [] for i in range(s.n_res)}
    for b in raw:
        index[b['donor']].append(b)
        index[b['acceptor']].append(b)
    return raw, index


# ── Main 2b feature extractor ────────────────────────────────────────────────

def extract_features_2b(s: Structure, res_idx: int,
                        hbond_index: dict) -> dict:
    """
    Extract Group B (forces / torques / H-bond summary) features for one
    residue.  Returns a dict with exactly the 25 keys in V5_2B_COLS.
    """
    i = res_idx
    bonds_all   = hbond_index.get(i, [])
    bonds_donor = [b for b in bonds_all if b['donor']    == i]
    bonds_acc   = [b for b in bonds_all if b['acceptor'] == i]

    phi_a, phi_b = _phi_axes(s, i)
    psi_a, psi_b = _psi_axes(s, i)

    # ── Per-bond torque accumulation ─────────────────────────────────────────
    tau_phi_bb_donor = 0.0
    tau_psi_bb_donor = 0.0
    tau_phi_bb_acc   = 0.0
    tau_psi_bb_acc   = 0.0
    per_bond_phi = []
    per_bond_psi = []

    for b in bonds_donor:
        vec_HO = b['vec_HO']
        t_phi = _torque_cos(phi_a, phi_b, vec_HO, b['energy'])
        t_psi = _torque_cos(psi_a, psi_b, vec_HO, b['energy'])
        tau_phi_bb_donor += t_phi
        tau_psi_bb_donor += t_psi
        per_bond_phi.append(t_phi)
        per_bond_psi.append(t_psi)

    for b in bonds_acc:
        vec_OH = [-v for v in b['vec_HO']]
        t_phi = _torque_cos(phi_a, phi_b, vec_OH, b['energy'])
        t_psi = _torque_cos(psi_a, psi_b, vec_OH, b['energy'])
        tau_phi_bb_acc += t_phi
        tau_psi_bb_acc += t_psi
        per_bond_phi.append(t_phi)
        per_bond_psi.append(t_psi)

    tau_phi_correct = tau_phi_bb_donor + tau_phi_bb_acc
    tau_psi_correct = tau_psi_bb_donor + tau_psi_bb_acc

    # ── Steric & electrostatic proxies ───────────────────────────────────────
    tau_phi_steric, tau_psi_steric = _steric_torque(s, i)
    tau_phi_elec,   tau_psi_elec   = _elec_torque(s, i)

    # ── H-bond summary statistics ─────────────────────────────────────────────
    n_bonds    = len(bonds_all)
    n_donor    = len(bonds_donor)
    n_acceptor = len(bonds_acc)
    n_strong   = sum(1 for b in bonds_all if b['energy'] < -0.5)

    if n_bonds > 0:
        d_HO_vals   = [b['r_oh']   for b in bonds_all]
        energies    = [b['energy']  for b in bonds_all]
        mean_d_HO   = float(np.mean(d_HO_vals))
        best_d_HO   = float(np.min(d_HO_vals))
        mean_energy = float(np.mean(energies))
        best_e      = float(np.min(energies))
    else:
        mean_d_HO = best_d_HO = mean_energy = best_e = 0.0

    return {
        'tau_phi_correct':   _r(tau_phi_correct),
        'tau_psi_correct':   _r(tau_psi_correct),
        'tau_phi_bb_donor':  _r(tau_phi_bb_donor),
        'tau_psi_bb_donor':  _r(tau_psi_bb_donor),
        'tau_phi_bb_acc':    _r(tau_phi_bb_acc),
        'tau_psi_bb_acc':    _r(tau_psi_bb_acc),
        'tau_phi_sc_hb':     0.0,
        'tau_psi_sc_hb':     0.0,
        'tau_phi_steric':    _r(tau_phi_steric),
        'tau_psi_steric':    _r(tau_psi_steric),
        'tau_phi_elec_corr': _r(tau_phi_elec),
        'tau_psi_elec_corr': _r(tau_psi_elec),
        'hb_n_bonds':        n_bonds,
        'hb_tau_phi_sum':    _r(tau_phi_correct),
        'hb_tau_phi_rms':    _r(_rms(per_bond_phi)),
        'hb_tau_phi_cancel': _r(_cancel(per_bond_phi)),
        'hb_tau_psi_rms':    _r(_rms(per_bond_psi)),
        'hb_tau_psi_cancel': _r(_cancel(per_bond_psi)),
        'hb_mean_d_HO':     _r(mean_d_HO),
        'hb_best_d_HO':     _r(best_d_HO),
        'hb_mean_energy':    _r(mean_energy),
        'hb_n_donor':        n_donor,
        'hb_n_acceptor':     n_acceptor,
        'hb_n_strong':       n_strong,
        'hb_best_e':         _r(best_e),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Sub-chunk 2c — Group A: Steric field
# ══════════════════════════════════════════════════════════════════════════════

V5_2C_COLS = [
    'steric_N_3A',  'steric_N_4A',  'steric_N_5A',
    'steric_CA_3A', 'steric_CA_4A', 'steric_CA_5A',
    'steric_C_3A',  'steric_C_4A',  'steric_C_5A',
    'steric_O_3A',  'steric_O_4A',  'steric_O_5A',
    'steric_asym_x', 'steric_asym_y', 'steric_asym_z',
    'steric_clash_phi_plus',  'steric_clash_phi_minus',
    'steric_clash_psi_plus',  'steric_clash_psi_minus',
    'sc_contact_nm1_to_bb', 'sc_contact_np1_to_bb',
]

_STERIC_RADII = (3.0, 4.0, 5.0)
_ASYM_RADIUS  = 8.0
_CLASH_STEP   = 0.5
_CLASH_CUTOFF = 2.5
_SC_BB_CUTOFF = 4.5
_BB_ATOMS     = ('N', 'CA', 'C', 'O')


def _build_kdtree(s):
    """Build a KD-tree from all non-H heavy atom coordinates."""
    pts_list, res_list, atom_list = [], [], []
    for aname, coords_arr in s.coords.items():
        if aname == 'H':
            continue
        for i in range(s.n_res):
            xyz = coords_arr[i]
            if np.any(np.isnan(xyz)):
                continue
            pts_list.append(xyz)
            res_list.append(i)
            atom_list.append(aname)
    if not pts_list:
        return None, np.empty((0, 3)), np.array([], dtype=int), np.array([])
    all_pts  = np.array(pts_list, dtype=float)
    res_ids  = np.array(res_list,  dtype=int)
    atom_ids = np.array(atom_list)
    return cKDTree(all_pts), all_pts, res_ids, atom_ids


def _covalent_exclusion_set(s, res_idx):
    """Return set of (res_idx, atom_name) pairs that are 1-2 or 1-3 bonded."""
    excl = set()
    i, n = res_idx, s.n_res
    for aname in s.coords:
        excl.add((i, aname))
    if i + 1 < n and i not in s.chain_breaks:
        excl.update([(i+1, 'N'), (i+1, 'CA'), (i+1, 'H')])
    if i > 0 and (i-1) not in s.chain_breaks:
        excl.update([(i-1, 'C'), (i-1, 'CA'), (i-1, 'O')])
    return excl


def extract_features_2c(s, res_idx, tree, all_pts, res_ids, atom_ids):
    """Compute Group A (steric) features for one residue."""
    i, n = res_idx, s.n_res
    f = {k: 0 for k in V5_2C_COLS}
    if tree is None:
        return f
    excl = _covalent_exclusion_set(s, i)

    # Backbone atom-centred steric counts
    for aname, prefix in (('N','steric_N'),('CA','steric_CA'),
                          ('C','steric_C'),('O','steric_O')):
        center = s.coords[aname][i]
        if np.any(np.isnan(center)):
            continue
        idxs = tree.query_ball_point(center, _STERIC_RADII[-1])
        valid = [j for j in idxs
                 if (int(res_ids[j]), str(atom_ids[j])) not in excl]
        dists = (np.linalg.norm(all_pts[valid] - center, axis=1)
                 if valid else np.array([]))
        for r, sfx in zip(_STERIC_RADII, ('_3A','_4A','_5A')):
            f[prefix + sfx] = int(np.sum(dists <= r))

    # Asymmetry vector
    ca_i = s.coords['CA'][i]
    if not np.any(np.isnan(ca_i)):
        ca_idxs = tree.query_ball_point(ca_i, _ASYM_RADIUS)
        vecs = []
        for j in ca_idxs:
            if str(atom_ids[j]) != 'CA' or int(res_ids[j]) == i:
                continue
            diff = ca_i - all_pts[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-6:
                vecs.append(diff / norm)
        if vecs:
            asym = np.mean(vecs, axis=0)
            f['steric_asym_x'] = round(float(asym[0]), 4)
            f['steric_asym_y'] = round(float(asym[1]), 4)
            f['steric_asym_z'] = round(float(asym[2]), 4)

    # Clash probes
    N_i, CA_i, C_i = s.coords['N'][i], s.coords['CA'][i], s.coords['C'][i]

    def _probe_count(probe_pt):
        if np.any(np.isnan(probe_pt)):
            return 0
        idxs = tree.query_ball_point(probe_pt, _CLASH_CUTOFF)
        return sum(1 for j in idxs
                   if (int(res_ids[j]), str(atom_ids[j])) not in excl)

    if not any(np.any(np.isnan(p)) for p in (N_i, CA_i)):
        phi_u = CA_i - N_i
        pn = np.linalg.norm(phi_u)
        if pn > 1e-6:
            phi_u = phi_u / pn
            f['steric_clash_phi_plus']  = _probe_count(CA_i + _CLASH_STEP*phi_u)
            f['steric_clash_phi_minus'] = _probe_count(CA_i - _CLASH_STEP*phi_u)
    if not any(np.any(np.isnan(p)) for p in (CA_i, C_i)):
        psi_u = C_i - CA_i
        pn = np.linalg.norm(psi_u)
        if pn > 1e-6:
            psi_u = psi_u / pn
            f['steric_clash_psi_plus']  = _probe_count(CA_i + _CLASH_STEP*psi_u)
            f['steric_clash_psi_minus'] = _probe_count(CA_i - _CLASH_STEP*psi_u)

    # Sidechain-to-backbone contacts for neighbouring residues
    def _sc_to_bb_count(sc_res, bb_res):
        if sc_res < 0 or sc_res >= n:
            return 0
        bb_pts = [s.coords[a][bb_res] for a in _BB_ATOMS
                  if not np.any(np.isnan(s.coords[a][bb_res]))]
        if not bb_pts:
            return 0
        count = 0
        for aname, arr in s.coords.items():
            if aname in _BB_ATOMS or aname == 'H':
                continue
            p = arr[sc_res]
            if np.any(np.isnan(p)):
                continue
            for bp in bb_pts:
                if np.linalg.norm(p - bp) <= _SC_BB_CUTOFF:
                    count += 1; break
        return count

    f['sc_contact_nm1_to_bb'] = _sc_to_bb_count(i-1, i)
    f['sc_contact_np1_to_bb'] = _sc_to_bb_count(i+1, i)
    return f


# ══════════════════════════════════════════════════════════════════════════════
# Full feature extraction — combines 2a + 2b + 2c
# ══════════════════════════════════════════════════════════════════════════════

def extract_all_features(s):
    """Extract the full feature set for every residue in Structure s."""
    phi, psi = _compute_phi_psi_arrays(s)
    tree, all_pts, res_ids, atom_ids = _build_kdtree(s)
    raw_bonds, hbond_index = build_hbond_index(s)

    rows = []
    for i in range(s.n_res):
        row = extract_features_2a(s, i, phi, psi)
        row.update(extract_features_2b(s, i, hbond_index))
        row.update(extract_features_2c(s, i, tree, all_pts, res_ids, atom_ids))
        rows.append(row)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# CSV column order — single unified schema
# ══════════════════════════════════════════════════════════════════════════════

# Unified column order. `chain` is included once. P2_COLS geometry features
# append after the v5 context/force/steric columns. Deduplication preserves
# first-occurrence order so paper 1 scripts reading by name still work.
def _dedup_keep_order(*lists):
    seen = set()
    out = []
    for lst in lists:
        for c in lst:
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out

ALL_COLS = _dedup_keep_order(
    ['pdb_id', 'chain', 'res_idx', 'res_name', 'res_type'],  # identifiers
    V5_2A_COLS,   # geometry + targets + context (24)
    V5_2B_COLS,   # torques + H-bond summary (25)
    V5_2C_COLS,   # steric field (21)
    P2_COLS,      # paper 2 geometry (13)
)

# Kept for backwards compatibility with any code importing these names
V5_ALL_COLS = ALL_COLS
P2_ALL_COLS = ALL_COLS


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(pdb_paths, out_csv, verbose=False):
    """Process PDB files and write a single unified features.csv."""
    t0 = time.time()
    n_pdbs = n_res_total = n_skipped = 0

    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=ALL_COLS, extrasaction='ignore')
        w.writeheader()

        for pdb_path in pdb_paths:
            try:
                s = load_structure(str(pdb_path))
            except Exception as e:
                if verbose: print(f'  [SKIP] {pdb_path.name}: {e}')
                n_skipped += 1; continue
            try:
                rows = extract_all_features(s)
            except Exception as e:
                if verbose: print(f'  [SKIP] {pdb_path.name}: {e}')
                n_skipped += 1; continue

            def _fmt(v):
                if isinstance(v, float) and (v != v): return ''
                return v

            for row in rows:
                w.writerow({c: _fmt(row.get(c, '')) for c in ALL_COLS})
            n_pdbs += 1; n_res_total += s.n_res
            if verbose:
                print(f'  {pdb_path.name:30s}  {s.n_res:4d} res  '
                      f'cumulative: {n_res_total:7d}')

    return dict(n_pdbs=n_pdbs, n_residues=n_res_total,
                n_skipped=n_skipped, elapsed_s=round(time.time()-t0, 1))


# ══════════════════════════════════════════════════════════════════════════════
# Self-test — combined 2a + 2b + 2c
# ══════════════════════════════════════════════════════════════════════════════

def _check(label, condition, detail=''):
    tag = 'PASS' if condition else 'FAIL'
    suffix = f'  ({detail})' if detail else ''
    print(f"  [{tag}] {label}{suffix}")
    return bool(condition)


def _self_test(pdb_path='tests/fixtures/1ubq_chainA.pdb'):
    print("=" * 70)
    print("features_collector.py — self-test (2a + 2b + 2c)")
    print("=" * 70)

    if not Path(pdb_path).exists():
        print(f"  FAIL: {pdb_path} not found"); return False

    s = load_structure(pdb_path)
    print(f"  Loaded {s.n_res} residues from chain {s.chain_id}")

    rows = extract_all_features(s)
    print(f"  Extracted {len(rows)} rows × {len(rows[0])} columns")

    all_pass = True

    # ── 2a: Geometry ──────────────────────────────────────────────────────────
    print("\n[2a — Geometry & Context]")
    all_pass &= _check("row count = n_res", len(rows) == s.n_res)

    expected_cols = set(V5_ALL_COLS) | set(P2_ALL_COLS)
    missing = expected_cols - set(rows[0].keys())
    all_pass &= _check("all declared columns present",
                        len(missing) == 0,
                        f"missing: {missing}" if missing else "")
    all_pass &= _check("res 0 phi is NaN", np.isnan(rows[0]['phi_deg']))
    all_pass &= _check("res N-1 psi is NaN", np.isnan(rows[-1]['psi_deg']))

    helix_phi = [rows[i]['phi_deg'] for i in range(22, 34)
                 if not np.isnan(rows[i]['phi_deg'])]
    mp = np.mean(helix_phi)
    helix_psi = [rows[i]['psi_deg'] for i in range(22, 34)
                 if not np.isnan(rows[i]['psi_deg'])]
    ms = np.mean(helix_psi)
    all_pass &= _check("helix phi/psi", -80 < mp < -40 and -70 < ms < -20,
                        f"phi={mp:.1f} psi={ms:.1f}")

    omegas = [r['omega_deg'] for r in rows[:-1]]
    n_trans = sum(1 for o in omegas if 150 < abs(o) <= 180)
    all_pass &= _check("≥90% omega trans", n_trans >= len(omegas)*0.9,
                        f"{n_trans}/{len(omegas)}")

    # Widened bounds for real crystal strain
    taus = [r['angle_NCaC'] for r in rows if not np.isnan(r['angle_NCaC'])]
    all_pass &= _check("τ in [100, 125]",
                        all(100 < t < 125 for t in taus),
                        f"range=[{min(taus):.1f}, {max(taus):.1f}]")

    non_gly_cb = [r['angle_N_CA_CB'] for r in rows
                  if r['res_name'] != 'GLY' and not np.isnan(r['angle_N_CA_CB'])]
    all_pass &= _check("non-GLY ∠N-Cα-Cβ in [98, 120]",
                        all(98 < v < 120 for v in non_gly_cb),
                        f"range=[{min(non_gly_cb):.1f}, {max(non_gly_cb):.1f}]")

    bonds_nca = [r['bond_N_CA'] for r in rows if not np.isnan(r['bond_N_CA'])]
    all_pass &= _check("bond N-Cα in [1.40, 1.52]",
                        all(1.40 < b < 1.52 for b in bonds_nca),
                        f"mean={np.mean(bonds_nca):.4f}")

    # ── 2b: H-bonds & torques ─────────────────────────────────────────────────
    print("\n[2b — H-bonds & Torques]")
    raw_bonds, _ = build_hbond_index(s)
    n_strong = sum(1 for b in raw_bonds if b['energy'] < -0.5)
    total_counted = sum(r['hb_n_bonds'] for r in rows)
    all_pass &= _check("Σ hb_n_bonds == 2 × distinct",
                        total_counted == 2 * len(raw_bonds),
                        f"{total_counted} vs {2*len(raw_bonds)}")
    all_pass &= _check("≥20 strong bonds", n_strong >= 20, f"{n_strong}")

    pro_ok = all(rows[i]['hb_n_donor'] == 0
                 for i, r in enumerate(s.sequence) if r == 'PRO')
    all_pass &= _check("PRO hb_n_donor = 0", pro_ok)

    all_pass &= _check("hb_n_donor + hb_n_acceptor == hb_n_bonds",
                        all(r['hb_n_donor']+r['hb_n_acceptor']==r['hb_n_bonds']
                            for r in rows))
    all_pass &= _check("hb_tau_phi_sum == tau_phi_correct",
                        all(abs(r['hb_tau_phi_sum']-r['tau_phi_correct'])<1e-9
                            for r in rows))

    if raw_bonds:
        energies = [b['energy'] for b in raw_bonds]
        all_pass &= _check("all bond energies negative",
                            all(e < 0 for e in energies))

    tau_ok = True
    for b in raw_bonds[:50]:  # check first 50
        pa, pb = _phi_axes(s, b['donor'])
        t = abs(_torque_cos(pa, pb, b['vec_HO'], b['energy']))
        if t > abs(b['energy']) + 1e-6:
            tau_ok = False; break
    all_pass &= _check("|τ per bond| ≤ |E|", tau_ok)

    # ── 2c: Steric field ──────────────────────────────────────────────────────
    print("\n[2c — Steric field]")

    # Monotone: 3A ≤ 4A ≤ 5A
    for atom in ('N','CA','C','O'):
        bad = [i for i, r in enumerate(rows)
               if not (r[f'steric_{atom}_3A'] <= r[f'steric_{atom}_4A']
                       <= r[f'steric_{atom}_5A'])]
        all_pass &= _check(f"steric_{atom}: 3A≤4A≤5A", len(bad)==0,
                            f"{len(bad)} violations" if bad else "")

    # Asymmetry in [-1, 1]
    for ax in ('x','y','z'):
        vals = [r[f'steric_asym_{ax}'] for r in rows]
        all_pass &= _check(f"asym_{ax} in [-1,1]",
                            all(-1.0 <= v <= 1.0 for v in vals))

    # Interior steric_CA_5A > 0
    interior_ca5 = [rows[i]['steric_CA_5A'] for i in range(5, s.n_res-5)]
    all_pass &= _check("interior steric_CA_5A > 0",
                        all(v > 0 for v in interior_ca5),
                        f"min={min(interior_ca5)}")

    # sc_contact non-negative ints
    for col in ('sc_contact_nm1_to_bb', 'sc_contact_np1_to_bb'):
        vals = [r[col] for r in rows]
        all_pass &= _check(f"{col} non-neg int",
                            all(isinstance(v, int) and v >= 0 for v in vals))

    # ── CSV orchestrator quick test ───────────────────────────────────────────
    print("\n[Orchestrator]")
    import tempfile
    fixtures = list(Path('tests/fixtures').glob('*_chain*.pdb'))
    if len(fixtures) >= 2:
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / 'features.csv'
            stats = run_pipeline(fixtures[:3], out_path)
            all_pass &= _check(f"pipeline ran {stats['n_pdbs']} PDBs",
                                stats['n_pdbs'] >= 2)
            all_pass &= _check(f"CSV written with {stats['n_residues']} residues",
                                stats['n_residues'] > 0)
            with open(out_path) as fh:
                hdr = fh.readline().strip().split(',')
            all_pass &= _check("CSV header matches ALL_COLS",
                                hdr == ALL_COLS)
    else:
        print("  [SKIP] not enough fixtures for orchestrator test")

    # ── Column count summary ──────────────────────────────────────────────────
    print(f"\n[Schema summary]")
    print(f"  unified: {len(ALL_COLS)} columns  "
          f"(2a:{len(V5_2A_COLS)} + 2b:{len(V5_2B_COLS)} "
          f"+ 2c:{len(V5_2C_COLS)} + p2:{len(P2_COLS)})")

    print("\n" + "=" * 70)
    if all_pass:
        print("  features_collector.py self-test: ALL PASS")
    else:
        print("  *** SOME FAILURES ***")
    print("=" * 70)
    return all_pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='features_collector.py — extract paper 1 + paper 2 '
                    'features from real PDB coordinates')
    ap.add_argument('--self_test', action='store_true',
                    help='Run combined 2a+2b+2c self-test on 1UBQ fixture')
    ap.add_argument('--pdb', default=None,
                    help='Single PDB file to process')
    ap.add_argument('--pdb_dir', default=None,
                    help='Directory of PDB files to process')
    ap.add_argument('--out', default='features.csv',
                    help='Output CSV path (default: features.csv)')
    ap.add_argument('--max_pdbs', type=int, default=None,
                    help='Limit number of PDBs (for quick tests)')
    ap.add_argument('--verbose', action='store_true',
                    help='Print per-PDB progress')
    args = ap.parse_args()

    if args.self_test:
        ok = _self_test(args.pdb or 'tests/fixtures/1ubq_chainA.pdb')
        sys.exit(0 if ok else 1)

    # Collect PDB paths
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
        print("\nERROR: must specify --self_test, --pdb, or --pdb_dir")
        sys.exit(1)

    print(f"Processing {len(pdb_paths)} PDB file(s) → {args.out}")
    print(f"Schema: {len(ALL_COLS)} columns")
    stats = run_pipeline(pdb_paths, Path(args.out), verbose=args.verbose)
    print(f"\nDone.")
    print(f"  PDBs processed : {stats['n_pdbs']}")
    print(f"  Residues total : {stats['n_residues']:,}")
    print(f"  Skipped        : {stats['n_skipped']}")
    print(f"  Elapsed        : {stats['elapsed_s']} s")
    print(f"  Written        : {args.out}")