#!/usr/bin/env python3
"""
Paper 4 — NeRF Geometry Library Integration
=============================================

A drop-in replacement for hardcoded AMBER constants in NeRF builders.
Reads the constant library JSON and provides lookup functions for
(φ,ψ)-dependent backbone geometry.

Two modes of operation:

  1. LOOKUP MODE (for NeRF reconstruction):
     Given (φ, ψ, residue_name), returns the optimal bond lengths,
     bond angles, and ω for placing the next atom.

  2. BENCHMARK MODE (standalone script):
     Rebuilds backbone coordinates for test proteins using
     Library-NeRF vs AMBER-NeRF vs PDB ground truth.
     Reports per-residue RMSD and bond-angle deviations.

Usage as module:
    from paper4_02_nerf_integration import GeometryLibrary
    
    lib = GeometryLibrary('paper4_library/constants_library.json')
    
    # Get all geometry for one residue
    geom = lib.lookup(phi=-63.0, psi=-43.0, res_name='ALA')
    # geom = {'tau': 110.5, 'bond_NCA': 1.458, 'bond_CAC': 1.524, ...}
    
    # Or individual values
    tau = lib.get_tau(phi=-63.0, psi=-43.0, res_name='ALA')

Usage as benchmark:
    python paper4_02_nerf_integration.py \
        --library ./paper4_library/constants_library.json \
        --pdb_dir /mnt/f/Protein_Folding/pdb_cache \
        --test_pdbs 1ubq 1l2y 2evq 2rlj \
        --out ./paper4_benchmark/

Author: Wei (Cvek Lab, LSUS)
"""

import argparse
import gzip
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# AMBER ff14SB fallback constants
# ══════════════════════════════════════════════════════════════════════════════

AMBER_DEFAULTS = {
    'tau':        111.1,    # ∠N-Cα-C [deg]
    'angle_NCaCB': 110.1,  # ∠N-Cα-Cβ [deg]
    'angle_CCaCB': 110.1,  # ∠C-Cα-Cβ [deg]
    'angle_CaCN':  116.6,  # ∠Cα-C-N [deg]
    'angle_CNCa':  121.9,  # ∠C-N-Cα [deg]
    'angle_CaCO':  120.4,  # ∠Cα-C=O [deg]
    'bond_NCA':    1.458,  # N-Cα [Å]
    'bond_CAC':    1.522,  # Cα-C [Å]
    'bond_CO':     1.229,  # C=O [Å]
    'bond_CN':     1.335,  # C-N peptide [Å]
    'bond_CACB':   1.526,  # Cα-Cβ [Å]
    'omega':       180.0,  # ω [deg]
}

# Mapping from library JSON keys to our short names
LIB_KEY_MAP = {
    'tau':         'tau_deg_eq',
    'angle_NCaCB': 'angle_N_CA_CB_eq',
    'angle_CCaCB': 'angle_C_CA_CB_eq',
    'angle_CaCN':  'angle_CaCN_eq',
    'angle_CNCa':  'angle_CNCa_eq',
    'angle_CaCO':  'angle_CA_C_O_eq',
    'bond_NCA':    'bond_N_CA_eq',
    'bond_CAC':    'bond_CA_C_eq',
    'bond_CO':     'bond_C_O_eq',
    'bond_CN':     'bond_C_N_next_eq',
    'bond_CACB':   'bond_CA_CB_eq',
    'omega':       'omega_deg_eq',
}

# Which keys also have coupling corrections
COUPLING_KEYS = {
    'tau':         'tau_deg_coupling',
    'angle_NCaCB': 'angle_N_CA_CB_coupling',
    'angle_CCaCB': 'angle_C_CA_CB_coupling',
    'angle_CaCN':  'angle_CaCN_coupling',
    'angle_CNCa':  'angle_CNCa_coupling',
    'angle_CaCO':  'angle_CA_C_O_coupling',
    'bond_NCA':    'bond_N_CA_coupling',
    'bond_CAC':    'bond_CA_C_coupling',
    'bond_CO':     'bond_C_O_coupling',
    'bond_CN':     'bond_C_N_next_coupling',
    'bond_CACB':   'bond_CA_CB_coupling',
    'omega':       'omega_deg_coupling',
}


# ══════════════════════════════════════════════════════════════════════════════
# GeometryLibrary class
# ══════════════════════════════════════════════════════════════════════════════

class GeometryLibrary:
    """PDB-derived backbone geometry lookup table.
    
    Replaces hardcoded AMBER constants with (φ,ψ,residue)-dependent
    values from the Paper 4 constant library.
    """
    
    def __init__(self, json_path, bin_size=10, use_coupling=True):
        """Load the library from JSON.
        
        Args:
            json_path: path to constants_library.json
            bin_size: Ramachandran grid spacing (must match extraction)
            use_coupling: if True, apply coupling corrections to eq values
        """
        with open(json_path) as f:
            self._lib = json.load(f)
        
        self._bin_size = bin_size
        self._use_coupling = use_coupling
        self._half_bin = bin_size / 2.0
        
        # Available residue classes
        self._classes = set(self._lib.keys())
        
        # Cache bin centers
        self._bin_centers = np.arange(-180 + self._half_bin,
                                      180 + self._half_bin,
                                      bin_size)
        
        # Build fallback chain: specific AA → ALL → AMBER
        self._stats = {
            'hits': 0, 'fallback_all': 0, 'fallback_amber': 0,
            'interpolated': 0
        }
    
    def _angle_to_bin_key(self, angle):
        """Convert angle to bin center string key."""
        # Wrap to [-180, 180)
        angle = ((angle + 180) % 360) - 180
        # Find nearest bin center
        idx = int(np.round((angle - (-180 + self._half_bin)) / self._bin_size))
        idx = max(0, min(idx, len(self._bin_centers) - 1))
        center = self._bin_centers[idx]
        return str(int(center))
    
    def _get_cell(self, phi, psi, res_name):
        """Get library cell with fallback chain.
        
        Fallback: res_name → 'ALL' → None (use AMBER)
        """
        phi_key = self._angle_to_bin_key(phi)
        psi_key = self._angle_to_bin_key(psi)
        
        # Try specific residue
        if res_name in self._lib:
            phi_dict = self._lib[res_name].get(phi_key)
            if phi_dict:
                cell = phi_dict.get(psi_key)
                if cell:
                    self._stats['hits'] += 1
                    return cell
        
        # Fallback to ALL
        if 'ALL' in self._lib:
            phi_dict = self._lib['ALL'].get(phi_key)
            if phi_dict:
                cell = phi_dict.get(psi_key)
                if cell:
                    self._stats['fallback_all'] += 1
                    return cell
        
        # No data for this cell
        self._stats['fallback_amber'] += 1
        return None
    
    def _get_value(self, cell, param_name):
        """Extract a single value from a library cell.
        
        Applies coupling correction if enabled.
        """
        lib_key = LIB_KEY_MAP.get(param_name)
        if lib_key is None or cell is None:
            return AMBER_DEFAULTS.get(param_name, 0.0)
        
        value = cell.get(lib_key)
        if value is None:
            return AMBER_DEFAULTS.get(param_name, 0.0)
        
        # Apply coupling correction
        if self._use_coupling:
            coup_key = COUPLING_KEYS.get(param_name)
            if coup_key:
                coup = cell.get(coup_key, 0.0)
                if coup is not None:
                    value += coup
        
        return value
    
    def lookup(self, phi, psi, res_name='ALA'):
        """Look up all geometry parameters for one residue.
        
        Args:
            phi: backbone φ angle [degrees]
            psi: backbone ψ angle [degrees]
            res_name: 3-letter amino acid code
        
        Returns:
            dict with keys: tau, angle_NCaCB, angle_CCaCB, angle_CaCN,
                           angle_CNCa, angle_CaCO, bond_NCA, bond_CAC,
                           bond_CO, bond_CN, bond_CACB, omega
        """
        cell = self._get_cell(phi, psi, res_name)
        
        result = {}
        for param in AMBER_DEFAULTS:
            result[param] = self._get_value(cell, param)
        
        return result
    
    def get_tau(self, phi, psi, res_name='ALA'):
        """Get τ (∠N-Cα-C) for given conformation."""
        cell = self._get_cell(phi, psi, res_name)
        return self._get_value(cell, 'tau')
    
    def get_bond_lengths(self, phi, psi, res_name='ALA'):
        """Get (bond_NCA, bond_CAC, bond_CO, bond_CN, bond_CACB)."""
        cell = self._get_cell(phi, psi, res_name)
        return {
            'N_CA': self._get_value(cell, 'bond_NCA'),
            'CA_C': self._get_value(cell, 'bond_CAC'),
            'C_O':  self._get_value(cell, 'bond_CO'),
            'C_N':  self._get_value(cell, 'bond_CN'),
            'CA_CB': self._get_value(cell, 'bond_CACB'),
        }
    
    def get_angles(self, phi, psi, res_name='ALA'):
        """Get all bond angles in degrees."""
        cell = self._get_cell(phi, psi, res_name)
        return {
            'N_CA_C':   self._get_value(cell, 'tau'),
            'N_CA_CB':  self._get_value(cell, 'angle_NCaCB'),
            'C_CA_CB':  self._get_value(cell, 'angle_CCaCB'),
            'CA_C_N':   self._get_value(cell, 'angle_CaCN'),
            'C_N_CA':   self._get_value(cell, 'angle_CNCa'),
            'CA_C_O':   self._get_value(cell, 'angle_CaCO'),
        }
    
    def get_omega(self, phi, psi, res_name='ALA'):
        """Get peptide ω angle."""
        cell = self._get_cell(phi, psi, res_name)
        val = self._get_value(cell, 'omega')
        # Fix the ω averaging issue: if close to 0, force to 180
        if abs(val) < 90:
            val = 180.0
        return val
    
    @property
    def stats(self):
        """Return lookup statistics."""
        total = sum(self._stats.values())
        return {**self._stats, 'total': total}
    
    def reset_stats(self):
        """Reset lookup counters."""
        self._stats = {k: 0 for k in self._stats}


# ══════════════════════════════════════════════════════════════════════════════
# NeRF reconstruction functions
# ══════════════════════════════════════════════════════════════════════════════

def place_atom(prev3, bond_length, bond_angle_deg, dihedral_deg):
    """Place atom using Natural Extension Reference Frame (NeRF).
    
    Given 3 previous atoms [A, B, C], place atom D such that:
      |C-D| = bond_length
      ∠B-C-D = bond_angle
      dihedral A-B-C-D = dihedral
    
    Returns: coordinates of D as numpy array.
    """
    A, B, C = [np.asarray(p, dtype=float) for p in prev3]
    
    bond_angle = np.radians(bond_angle_deg)
    dihedral = np.radians(dihedral_deg)
    
    # Build local frame at C
    bc = C - B
    bc_norm = bc / np.linalg.norm(bc)
    
    ab = B - A
    n = np.cross(ab, bc)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-10:
        # Degenerate: pick arbitrary perpendicular
        if abs(bc_norm[0]) < 0.9:
            n = np.cross(bc_norm, np.array([1, 0, 0]))
        else:
            n = np.cross(bc_norm, np.array([0, 1, 0]))
        n_norm = np.linalg.norm(n)
    n = n / n_norm
    
    m = np.cross(n, bc_norm)
    
    # Position in local frame
    d_x = -bond_length * np.cos(bond_angle)
    d_y = bond_length * np.sin(bond_angle) * np.cos(dihedral)
    d_z = bond_length * np.sin(bond_angle) * np.sin(dihedral)
    
    D = C + d_x * bc_norm + d_y * m + d_z * n
    
    return D


def rebuild_backbone(sequence, phi_list, psi_list, omega_list=None,
                     geometry_source='amber', library=None):
    """Rebuild backbone N, CA, C coordinates from dihedrals.
    
    Args:
        sequence: list of 3-letter residue names
        phi_list: φ angles in degrees (NaN for first residue)
        psi_list: ψ angles in degrees (NaN for last residue)
        omega_list: ω angles (default: all 180°)
        geometry_source: 'amber' or 'library'
        library: GeometryLibrary instance (required if source='library')
    
    Returns:
        dict with 'N', 'CA', 'C' keys -> (n_res, 3) arrays
    """
    n = len(sequence)
    if omega_list is None:
        omega_list = [180.0] * n
    
    coords = {
        'N':  np.full((n, 3), np.nan),
        'CA': np.full((n, 3), np.nan),
        'C':  np.full((n, 3), np.nan),
    }
    
    # Seed first residue at origin with standard geometry
    coords['N'][0]  = np.array([0.0, 0.0, 0.0])
    coords['CA'][0] = np.array([1.458, 0.0, 0.0])
    coords['C'][0]  = np.array([2.009, 1.420, 0.0])  # approximate
    
    def get_geom(i):
        """Get geometry constants for residue i."""
        phi_i = phi_list[i] if not np.isnan(phi_list[i]) else -63.0
        psi_i = psi_list[i] if not np.isnan(psi_list[i]) else -43.0
        res_i = sequence[i]
        
        if geometry_source == 'library' and library is not None:
            return library.lookup(phi_i, psi_i, res_i)
        else:
            return dict(AMBER_DEFAULTS)
    
    for i in range(n):
        geom = get_geom(i)
        
        if i == 0:
            # First residue: just set CA-C bond using tau
            # Already seeded above; refine C position
            tau = geom['tau']
            bond_cac = geom['bond_CAC']
            # Place C from N, CA using tau angle
            coords['C'][0] = place_atom(
                [np.array([-1.335, -0.5, 0.0]),  # virtual prev C
                 coords['N'][0], coords['CA'][0]],
                bond_cac, tau, 
                psi_list[0] if not np.isnan(psi_list[0]) else -43.0
            )
            continue
        
        # Get geometry for the peptide bond from i-1 to i
        geom_prev = get_geom(i - 1)
        omega_i = omega_list[i - 1] if not np.isnan(omega_list[i - 1]) else 180.0
        # Fix ω
        if abs(omega_i) < 90:
            omega_i = 180.0
        
        # Place N[i]: angle at C[i-1] = ∠CA-C-N, dihedral = ψ[i-1]
        psi_prev = psi_list[i - 1] if not np.isnan(psi_list[i - 1]) else -43.0
        angle_CaCN = geom_prev['angle_CaCN']
        bond_CN = geom_prev['bond_CN']
        
        coords['N'][i] = place_atom(
            [coords['N'][i-1], coords['CA'][i-1], coords['C'][i-1]],
            bond_CN, angle_CaCN, psi_prev
        )
        
        # Place CA[i]: angle at N[i] = ∠C-N-CA, dihedral = ω
        angle_CNCa = geom['angle_CNCa']
        bond_NCA = geom['bond_NCA']
        
        coords['CA'][i] = place_atom(
            [coords['CA'][i-1], coords['C'][i-1], coords['N'][i]],
            bond_NCA, angle_CNCa, omega_i
        )
        
        # Place C[i]: angle at CA[i] = τ, dihedral = φ[i]
        phi_i = phi_list[i] if not np.isnan(phi_list[i]) else -63.0
        tau = geom['tau']
        bond_CAC = geom['bond_CAC']
        
        coords['C'][i] = place_atom(
            [coords['C'][i-1], coords['N'][i], coords['CA'][i]],
            bond_CAC, tau, phi_i
        )
    
    return coords


# ══════════════════════════════════════════════════════════════════════════════
# PDB parsing (minimal, for benchmark)
# ══════════════════════════════════════════════════════════════════════════════

def parse_pdb_backbone(path):
    """Parse PDB, return sequence and backbone coords."""
    opener = gzip.open if str(path).endswith('.gz') else open
    residues = []  # list of (resseq, resname, {atom: xyz})
    current = None
    
    try:
        with opener(path, 'rt') as f:
            for line in f:
                if not line.startswith('ATOM'):
                    continue
                aname = line[12:16].strip()
                if aname not in ('N', 'CA', 'C', 'O'):
                    continue
                alt = line[16:17].strip()
                if alt and alt != 'A':
                    continue
                rname = line[17:20].strip()
                chain = line[21:22].strip()
                rseq = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                key = (chain, rseq)
                if current is None or current[0] != key:
                    if current is not None:
                        residues.append(current[1])
                    current = (key, {'resname': rname, 'resseq': rseq, 'chain': chain, 'atoms': {}})
                current[1]['atoms'][aname] = np.array([x, y, z])
            
            if current is not None:
                residues.append(current[1])
    except Exception:
        pass
    
    return residues


def extract_dihedrals(residues):
    """Extract φ, ψ, ω from parsed residues."""
    n = len(residues)
    phi = np.full(n, np.nan)
    psi = np.full(n, np.nan)
    omega = np.full(n, np.nan)
    
    def dihedral(p1, p2, p3, p4):
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        n1_len = np.linalg.norm(n1)
        n2_len = np.linalg.norm(n2)
        if n1_len < 1e-6 or n2_len < 1e-6:
            return np.nan
        n1 /= n1_len
        n2 /= n2_len
        b2_u = b2 / np.linalg.norm(b2)
        m1 = np.cross(n1, b2_u)
        return np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))
    
    for i in range(n):
        atoms_i = residues[i]['atoms']
        if not all(a in atoms_i for a in ['N', 'CA', 'C']):
            continue
        
        # φ[i] = dihedral(C[i-1], N[i], CA[i], C[i])
        if i > 0 and 'C' in residues[i-1]['atoms']:
            phi[i] = dihedral(residues[i-1]['atoms']['C'],
                              atoms_i['N'], atoms_i['CA'], atoms_i['C'])
        
        # ψ[i] = dihedral(N[i], CA[i], C[i], N[i+1])
        if i < n - 1 and 'N' in residues[i+1]['atoms']:
            psi[i] = dihedral(atoms_i['N'], atoms_i['CA'], atoms_i['C'],
                              residues[i+1]['atoms']['N'])
        
        # ω[i] = dihedral(CA[i], C[i], N[i+1], CA[i+1])
        if i < n - 1:
            ri1 = residues[i+1]['atoms']
            if all(a in ri1 for a in ['N', 'CA']):
                omega[i] = dihedral(atoms_i['CA'], atoms_i['C'],
                                    ri1['N'], ri1['CA'])
    
    return phi, psi, omega


def find_pdb(pdb_id, pdb_dir):
    d = Path(pdb_dir)
    for pat in [f"{pdb_id}.pdb", f"{pdb_id}.pdb.gz",
                f"pdb{pdb_id}.ent", f"pdb{pdb_id}.ent.gz",
                f"{pdb_id.lower()}.pdb", f"{pdb_id.lower()}.pdb.gz"]:
        p = d / pat
        if p.exists():
            return p
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark
# ══════════════════════════════════════════════════════════════════════════════

def compute_rmsd(coords1, coords2):
    """Compute RMSD between two coordinate sets after Kabsch alignment."""
    # Remove NaN rows
    mask = ~(np.any(np.isnan(coords1), axis=1) | np.any(np.isnan(coords2), axis=1))
    c1 = coords1[mask]
    c2 = coords2[mask]
    
    if len(c1) < 3:
        return np.nan
    
    # Center
    c1_center = c1 - c1.mean(axis=0)
    c2_center = c2 - c2.mean(axis=0)
    
    # Kabsch
    H = c1_center.T @ c2_center
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, d])
    R = Vt.T @ sign_matrix @ U.T
    
    c1_aligned = c1_center @ R.T
    
    rmsd = np.sqrt(np.mean(np.sum((c1_aligned - c2_center)**2, axis=1)))
    return rmsd


def benchmark_protein(pdb_id, pdb_dir, library):
    """Benchmark Library-NeRF vs AMBER-NeRF on one protein."""
    pdb_path = find_pdb(pdb_id, pdb_dir)
    if pdb_path is None:
        return None
    
    residues = parse_pdb_backbone(pdb_path)
    if len(residues) < 10:
        return None
    
    # Extract ground truth
    sequence = [r['resname'] for r in residues]
    phi, psi, omega = extract_dihedrals(residues)
    
    # Ground truth coords
    n = len(residues)
    pdb_N  = np.array([r['atoms'].get('N', [np.nan]*3) for r in residues])
    pdb_CA = np.array([r['atoms'].get('CA', [np.nan]*3) for r in residues])
    pdb_C  = np.array([r['atoms'].get('C', [np.nan]*3) for r in residues])
    
    # Rebuild with AMBER constants
    library.reset_stats()
    amber_coords = rebuild_backbone(sequence, phi, psi, omega,
                                     geometry_source='amber')
    
    # Rebuild with library
    library.reset_stats()
    lib_coords = rebuild_backbone(sequence, phi, psi, omega,
                                   geometry_source='library', library=library)
    lib_stats = library.stats
    
    # Compute RMSDs
    pdb_all = np.vstack([pdb_N, pdb_CA, pdb_C])
    amber_all = np.vstack([amber_coords['N'], amber_coords['CA'], amber_coords['C']])
    lib_all = np.vstack([lib_coords['N'], lib_coords['CA'], lib_coords['C']])
    
    rmsd_amber = compute_rmsd(amber_all, pdb_all)
    rmsd_lib = compute_rmsd(lib_all, pdb_all)
    
    # Per-atom-type RMSD
    rmsd_amber_ca = compute_rmsd(amber_coords['CA'], pdb_CA)
    rmsd_lib_ca = compute_rmsd(lib_coords['CA'], pdb_CA)
    
    # Bond length deviations
    def bond_lengths(coords, atom1, atom2):
        """Compute consecutive bond lengths."""
        c1 = coords[atom1]
        c2 = coords[atom2]
        dists = np.linalg.norm(c2 - c1, axis=1)
        return dists
    
    # N-CA bond lengths
    pdb_nca = np.array([np.linalg.norm(r['atoms']['CA'] - r['atoms']['N'])
                         for r in residues
                         if 'N' in r['atoms'] and 'CA' in r['atoms']])
    
    amber_nca = np.linalg.norm(amber_coords['CA'] - amber_coords['N'], axis=1)
    lib_nca = np.linalg.norm(lib_coords['CA'] - lib_coords['N'], axis=1)
    
    # Remove NaN
    mask = ~(np.isnan(pdb_nca) | np.isnan(amber_nca[:len(pdb_nca)]) | np.isnan(lib_nca[:len(pdb_nca)]))
    if mask.sum() > 3:
        bond_mae_amber = np.mean(np.abs(amber_nca[:len(pdb_nca)][mask] - pdb_nca[mask]))
        bond_mae_lib = np.mean(np.abs(lib_nca[:len(pdb_nca)][mask] - pdb_nca[mask]))
    else:
        bond_mae_amber = bond_mae_lib = np.nan
    
    return {
        'pdb_id': pdb_id,
        'n_residues': n,
        'rmsd_amber_all': rmsd_amber,
        'rmsd_library_all': rmsd_lib,
        'rmsd_amber_ca': rmsd_amber_ca,
        'rmsd_library_ca': rmsd_lib_ca,
        'bond_mae_amber': bond_mae_amber,
        'bond_mae_library': bond_mae_lib,
        'lib_hits': lib_stats['hits'],
        'lib_fallback_all': lib_stats['fallback_all'],
        'lib_fallback_amber': lib_stats['fallback_amber'],
        'improvement_rmsd': rmsd_amber - rmsd_lib,
        'improvement_pct': (rmsd_amber - rmsd_lib) / rmsd_amber * 100 if rmsd_amber > 0 else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main (benchmark mode)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Paper 4 — NeRF geometry library benchmark')
    ap.add_argument('--library', required=True,
                    help='Path to constants_library.json')
    ap.add_argument('--pdb_dir', required=True,
                    help='Directory with PDB files')
    ap.add_argument('--test_pdbs', nargs='+', default=['1ubq'],
                    help='PDB IDs to benchmark')
    ap.add_argument('--out', default='./paper4_benchmark')
    ap.add_argument('--no_coupling', action='store_true',
                    help='Disable coupling corrections')
    args = ap.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    
    print(f"Loading library from {args.library}...")
    lib = GeometryLibrary(args.library, use_coupling=not args.no_coupling)
    print(f"  Classes: {len(lib._classes)}")
    print(f"  Coupling: {'ON' if not args.no_coupling else 'OFF'}")
    
    print(f"\nBenchmarking {len(args.test_pdbs)} proteins...")
    
    results = []
    for pdb_id in args.test_pdbs:
        print(f"\n  {pdb_id}...", end=" ", flush=True)
        res = benchmark_protein(pdb_id, args.pdb_dir, lib)
        if res is None:
            print("SKIP (not found)")
            continue
        results.append(res)
        print(f"RMSD amber={res['rmsd_amber_all']:.3f}  "
              f"lib={res['rmsd_library_all']:.3f}  "
              f"Δ={res['improvement_rmsd']:+.3f} Å  "
              f"({res['improvement_pct']:+.1f}%)")
    
    if not results:
        print("No proteins processed!")
        return
    
    # ── Summary report ───────────────────────────────────────────────────
    R = []
    R.append("=" * 70)
    R.append("Paper 4 — NeRF Benchmark: Library vs AMBER Constants")
    R.append("=" * 70)
    
    header = (f"  {'PDB':>6s}  {'n_res':>5s}  {'RMSD_amb':>8s}  {'RMSD_lib':>8s}  "
              f"{'Δ':>8s}  {'%':>6s}  {'bond_amb':>8s}  {'bond_lib':>8s}")
    R.append(header)
    R.append("  " + "─" * (len(header.strip())))
    
    for res in results:
        R.append(f"  {res['pdb_id']:>6s}  {res['n_residues']:>5d}  "
                 f"{res['rmsd_amber_all']:>8.3f}  {res['rmsd_library_all']:>8.3f}  "
                 f"{res['improvement_rmsd']:>+8.3f}  {res['improvement_pct']:>+6.1f}  "
                 f"{res['bond_mae_amber']:>8.4f}  {res['bond_mae_library']:>8.4f}")
    
    # Averages
    if len(results) > 1:
        R.append("  " + "─" * (len(header.strip())))
        avg_amber = np.mean([r['rmsd_amber_all'] for r in results])
        avg_lib = np.mean([r['rmsd_library_all'] for r in results])
        avg_pct = np.mean([r['improvement_pct'] for r in results])
        R.append(f"  {'AVG':>6s}  {'':>5s}  {avg_amber:>8.3f}  {avg_lib:>8.3f}  "
                 f"{avg_amber-avg_lib:>+8.3f}  {avg_pct:>+6.1f}")
    
    R.append(f"\n  Library lookup stats:")
    total_hits = sum(r['lib_hits'] for r in results)
    total_fb_all = sum(r['lib_fallback_all'] for r in results)
    total_fb_amber = sum(r['lib_fallback_amber'] for r in results)
    total = total_hits + total_fb_all + total_fb_amber
    R.append(f"    Direct hits:      {total_hits:>6d} ({total_hits/total*100:.1f}%)")
    R.append(f"    Fallback to ALL:  {total_fb_all:>6d} ({total_fb_all/total*100:.1f}%)")
    R.append(f"    Fallback to AMBER:{total_fb_amber:>6d} ({total_fb_amber/total*100:.1f}%)")
    
    report = '\n'.join(R)
    
    report_path = os.path.join(args.out, 'benchmark_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n{report}")
    print(f"\n  Saved {report_path}")
    
    # Save CSV
    import pandas as pd
    pd.DataFrame(results).to_csv(
        os.path.join(args.out, 'benchmark_results.csv'), index=False)
    
    print(f"  Done in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()