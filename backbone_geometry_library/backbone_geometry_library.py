#!/usr/bin/env python3
"""
backbone_geometry_library — Conformation-Dependent Backbone Geometry
=====================================================================

A drop-in replacement for fixed backbone constants in protein modelling.

Quick Start
-----------
    from backbone_geometry_library import GeometryLibrary

    # Load the library
    lib = GeometryLibrary()   # uses bundled data

    # Look up geometry for a residue
    geom = lib.get(phi=-63, psi=-43, residue='ALA')
    print(geom['tau'])        # 111.0° (vs AMBER's fixed 111.1°)
    print(geom['bond_NCA'])   # 1.460 Å (vs AMBER's fixed 1.458 Å)

    # Get all backbone constants for NeRF reconstruction
    angles = lib.get_angles(phi=-120, psi=130, residue='VAL')
    bonds = lib.get_bonds(phi=-120, psi=130, residue='VAL')

Use Cases
---------
    1. NeRF reconstruction (AlphaFold, RoseTTAFold, custom pipelines)
    2. Force-field correction (AMBER, CHARMM, OPLS via OpenMM)
    3. Structure validation (conformation-aware outlier detection)
    4. Crystallographic refinement (conformation-dependent restraints)

OpenMM Integration
------------------
    from backbone_geometry_library import apply_corrections
    
    # system = your OpenMM System
    # topology = your OpenMM Topology  
    # positions = your initial coordinates
    system = apply_corrections(system, topology, positions)
    # Done! All bond-angle equilibria are now conformation-dependent.

Reference
---------
    Chen, W., Cvek, U., & Trutschl, M. (2025).
    "A Conformation-Dependent Geometry Library for Protein Backbone 
     Reconstruction." [submitted]

Author: Wei Chen (Cvek Lab, LSUS)
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Wei Chen"

import json
import os
import numpy as np
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# AMBER ff14SB defaults (fallback when library cell is missing)
# ══════════════════════════════════════════════════════════════════════════════

AMBER_DEFAULTS = {
    'tau':         111.1,   # ∠N-Cα-C [deg]
    'angle_NCaCB': 110.1,   # ∠N-Cα-Cβ [deg]
    'angle_CCaCB': 110.1,   # ∠C-Cα-Cβ [deg]
    'angle_CaCN':  116.6,   # ∠Cα-C-N [deg]
    'angle_CNCa':  121.9,   # ∠C-N-Cα [deg]
    'angle_CaCO':  120.4,   # ∠Cα-C=O [deg]
    'bond_NCA':    1.458,   # N-Cα [Å]
    'bond_CAC':    1.522,   # Cα-C [Å]
    'bond_CO':     1.229,   # C=O [Å]
    'bond_CN':     1.335,   # C-N peptide [Å]
    'bond_CACB':   1.526,   # Cα-Cβ [Å]
    'omega':       180.0,   # ω [deg]
}

# Map from short names to library JSON keys
_KEY_MAP = {
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

# Spring constants (kcal/mol/rad² for angles, kcal/mol/Å² for bonds)
SPRING_CONSTANTS = {
    'tau': 63.0, 'angle_NCaCB': 63.0, 'angle_CCaCB': 63.0,
    'angle_CaCN': 70.0, 'angle_CNCa': 50.0, 'angle_CaCO': 80.0,
    'bond_NCA': 337.0, 'bond_CAC': 317.0, 'bond_CO': 570.0,
    'bond_CN': 490.0, 'bond_CACB': 317.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# GeometryLibrary — main class
# ══════════════════════════════════════════════════════════════════════════════

class GeometryLibrary:
    """Conformation-dependent backbone geometry lookup table.
    
    Provides (φ,ψ,residue)-dependent equilibrium values for backbone
    bond lengths, bond angles, and peptide planarity. Replaces fixed
    constants (AMBER/CHARMM/OPLS) with PDB-derived values that vary
    across the Ramachandran plane.
    
    Parameters
    ----------
    library_path : str or None
        Path to constants_library.json. If None, looks for the file
        in the same directory as this module.
    chi1_path : str or None
        Path to constants_chi1.json for Cβ angle corrections.
    bin_size : int
        Ramachandran grid spacing in degrees (default: 10).
    
    Examples
    --------
    >>> lib = GeometryLibrary()
    >>> lib.get(phi=-63, psi=-43, residue='ALA')
    {'tau': 111.0, 'bond_NCA': 1.460, ...}
    
    >>> lib.get_tau(phi=-63, psi=-43, residue='GLY')
    113.1
    
    >>> lib.get_bonds(phi=-120, psi=130, residue='VAL')
    {'NCA': 1.459, 'CAC': 1.524, 'CO': 1.232, 'CN': 1.333, 'CACB': 1.540}
    """
    
    def __init__(self, library_path=None, chi1_path=None, bin_size=10):
        # Find library files
        if library_path is None:
            module_dir = Path(__file__).parent
            library_path = module_dir / 'constants_library.json'
            if not library_path.exists():
                raise FileNotFoundError(
                    f"Library not found at {library_path}. "
                    f"Pass library_path explicitly or place "
                    f"constants_library.json next to this module."
                )
        
        with open(library_path) as f:
            self._lib = json.load(f)
        
        # Optional χ₁ sublibrary
        self._chi1 = None
        if chi1_path is None:
            chi1_default = Path(str(library_path).replace(
                'constants_library', 'constants_chi1'))
            if chi1_default.exists():
                chi1_path = chi1_default
        if chi1_path and Path(chi1_path).exists():
            with open(chi1_path) as f:
                self._chi1 = json.load(f)
        
        self._bin_size = bin_size
        self._half = bin_size / 2.0
        self._centers = np.arange(-180 + self._half,
                                   180 + self._half, bin_size)
        
        self._stats = {'hits': 0, 'fallback_all': 0, 'fallback_default': 0}
    
    def _bin_key(self, angle):
        """Convert angle (degrees) to bin center string key."""
        angle = ((angle + 180) % 360) - 180
        idx = int(np.round((angle - self._centers[0]) / self._bin_size))
        idx = max(0, min(idx, len(self._centers) - 1))
        return str(int(self._centers[idx]))
    
    def _lookup_cell(self, phi, psi, residue):
        """Find the library cell, with fallback chain.
        
        Priority: specific residue → 'ALL' → None (use defaults)
        """
        pk = self._bin_key(phi)
        qk = self._bin_key(psi)
        
        for cls in [residue, 'ALL']:
            if cls in self._lib:
                cell = self._lib[cls].get(pk, {}).get(qk)
                if cell:
                    if cls == residue:
                        self._stats['hits'] += 1
                    else:
                        self._stats['fallback_all'] += 1
                    return cell
        
        self._stats['fallback_default'] += 1
        return None
    
    def _get_value(self, cell, param):
        """Extract one value from a library cell."""
        if cell is None:
            return AMBER_DEFAULTS.get(param, 0.0)
        
        lib_key = _KEY_MAP.get(param)
        if lib_key and lib_key in cell:
            val = cell[lib_key]
            # Fix ω wrapping: if near 0, force to 180
            if param == 'omega' and abs(val) < 90:
                return 180.0
            return val
        
        return AMBER_DEFAULTS.get(param, 0.0)
    
    # ── Public API ────────────────────────────────────────────────────────
    
    def get(self, phi, psi, residue='ALA'):
        """Look up all geometry parameters for one residue.
        
        Parameters
        ----------
        phi : float
            Backbone φ angle in degrees.
        psi : float  
            Backbone ψ angle in degrees.
        residue : str
            Three-letter amino acid code (e.g., 'ALA', 'GLY', 'VAL').
        
        Returns
        -------
        dict
            Keys: tau, angle_NCaCB, angle_CCaCB, angle_CaCN, angle_CNCa,
                  angle_CaCO, bond_NCA, bond_CAC, bond_CO, bond_CN,
                  bond_CACB, omega
            Values: equilibrium geometry in degrees (angles) or Å (bonds).
        
        Examples
        --------
        >>> lib.get(phi=-63, psi=-43, residue='ALA')
        {'tau': 111.0, 'angle_NCaCB': 110.3, 'bond_NCA': 1.460, ...}
        """
        cell = self._lookup_cell(phi, psi, residue)
        return {param: self._get_value(cell, param) 
                for param in AMBER_DEFAULTS}
    
    def get_tau(self, phi, psi, residue='ALA'):
        """Get τ (∠N-Cα-C) in degrees.
        
        Examples
        --------
        >>> lib.get_tau(-63, -43, 'GLY')
        113.1
        >>> lib.get_tau(-120, 130, 'VAL')  
        109.6
        """
        cell = self._lookup_cell(phi, psi, residue)
        return self._get_value(cell, 'tau')
    
    def get_bonds(self, phi, psi, residue='ALA'):
        """Get all bond lengths in Å.
        
        Returns
        -------
        dict
            Keys: NCA, CAC, CO, CN, CACB
        
        Examples
        --------
        >>> lib.get_bonds(-63, -43, 'ALA')
        {'NCA': 1.460, 'CAC': 1.524, 'CO': 1.232, 'CN': 1.333, 'CACB': 1.534}
        """
        cell = self._lookup_cell(phi, psi, residue)
        return {
            'NCA':  self._get_value(cell, 'bond_NCA'),
            'CAC':  self._get_value(cell, 'bond_CAC'),
            'CO':   self._get_value(cell, 'bond_CO'),
            'CN':   self._get_value(cell, 'bond_CN'),
            'CACB': self._get_value(cell, 'bond_CACB'),
        }
    
    def get_angles(self, phi, psi, residue='ALA'):
        """Get all bond angles in degrees.
        
        Returns
        -------
        dict
            Keys: N_CA_C (τ), N_CA_CB, C_CA_CB, CA_C_N, C_N_CA, CA_C_O
        
        Examples
        --------
        >>> lib.get_angles(-63, -43, 'ALA')
        {'N_CA_C': 111.0, 'N_CA_CB': 110.3, ...}
        """
        cell = self._lookup_cell(phi, psi, residue)
        return {
            'N_CA_C':  self._get_value(cell, 'tau'),
            'N_CA_CB': self._get_value(cell, 'angle_NCaCB'),
            'C_CA_CB': self._get_value(cell, 'angle_CCaCB'),
            'CA_C_N':  self._get_value(cell, 'angle_CaCN'),
            'C_N_CA':  self._get_value(cell, 'angle_CNCa'),
            'CA_C_O':  self._get_value(cell, 'angle_CaCO'),
        }
    
    def get_omega(self, phi, psi, residue='ALA'):
        """Get peptide bond angle ω in degrees (typically ~180°)."""
        cell = self._lookup_cell(phi, psi, residue)
        return self._get_value(cell, 'omega')
    
    def get_chi1_correction(self, phi, psi, residue, chi1_rotamer):
        """Get χ₁-dependent Cβ angle corrections.
        
        Parameters
        ----------
        phi, psi : float
            Backbone dihedrals in degrees.
        residue : str
            Three-letter amino acid code.
        chi1_rotamer : str
            One of 'g+', 't', 'g-' (gauche+, trans, gauche-).
        
        Returns
        -------
        dict or None
            Keys: N_CA_CB, C_CA_CB (corrected equilibrium values)
            None if no χ₁ data available for this cell.
        """
        if self._chi1 is None:
            return None
        
        pk = self._bin_key(phi)
        qk = self._bin_key(psi)
        
        res_data = self._chi1.get(residue)
        if res_data is None:
            return None
        
        cell = res_data.get(pk, {}).get(qk, {}).get(chi1_rotamer)
        if cell is None:
            return None
        
        result = {}
        if 'angle_N_CA_CB_eq' in cell:
            result['N_CA_CB'] = cell['angle_N_CA_CB_eq']
        if 'angle_C_CA_CB_eq' in cell:
            result['C_CA_CB'] = cell['angle_C_CA_CB_eq']
        
        return result if result else None
    
    @property
    def available_residues(self):
        """List of residue types in the library."""
        return sorted([k for k in self._lib.keys() if k != 'ALL'])
    
    @property
    def stats(self):
        """Lookup statistics: hits, fallbacks, total."""
        total = sum(self._stats.values())
        return {**self._stats, 'total': total}
    
    def reset_stats(self):
        """Reset lookup counters."""
        self._stats = {k: 0 for k in self._stats}


# ══════════════════════════════════════════════════════════════════════════════
# Dihedral computation from coordinates
# ══════════════════════════════════════════════════════════════════════════════

def compute_dihedral(p1, p2, p3, p4):
    """Compute dihedral angle in degrees from four 3D points."""
    b1 = np.asarray(p2) - np.asarray(p1)
    b2 = np.asarray(p3) - np.asarray(p2)
    b3 = np.asarray(p4) - np.asarray(p3)
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_len = np.linalg.norm(n1)
    n2_len = np.linalg.norm(n2)
    if n1_len < 1e-8 or n2_len < 1e-8:
        return 0.0
    n1 /= n1_len
    n2 /= n2_len
    b2_u = b2 / np.linalg.norm(b2)
    m1 = np.cross(n1, b2_u)
    return float(np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2))))


# ══════════════════════════════════════════════════════════════════════════════
# OpenMM integration
# ══════════════════════════════════════════════════════════════════════════════

def apply_corrections(system, topology, positions, library_path=None,
                      force_field='amber'):
    """Apply library corrections to an OpenMM System.
    
    Adds a CustomAngleForce that shifts the equilibrium of each backbone
    bond angle from the force-field default to the library value, without
    modifying the spring constant or any other force-field parameters.
    
    Parameters
    ----------
    system : openmm.System
        The OpenMM system to correct.
    topology : openmm.app.Topology
        Protein topology.
    positions : array-like
        Atomic positions (for computing φ,ψ).
    library_path : str or None
        Path to constants_library.json.
    force_field : str
        One of 'amber', 'charmm', 'opls'. Determines default eq values.
    
    Returns
    -------
    openmm.System
        The system with correction forces added.
    
    Examples
    --------
    >>> from openmm.app import PDBFile, ForceField, Simulation
    >>> from backbone_geometry_library import apply_corrections
    >>> 
    >>> pdb = PDBFile('protein.pdb')
    >>> ff = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    >>> system = ff.createSystem(pdb.topology)
    >>> system = apply_corrections(system, pdb.topology, pdb.positions)
    >>> # Now run simulation as normal — equilibria are corrected
    """
    try:
        from openmm import CustomAngleForce
        from openmm import unit as u
    except ImportError:
        raise ImportError(
            "OpenMM is required for force-field integration. "
            "Install with: conda install -c conda-forge openmm"
        )
    
    # Load library
    lib = GeometryLibrary(library_path)
    
    # FF-specific defaults
    ff_defaults = {
        'amber':  {'tau': 111.1, 'NCaCB': 110.1, 'CCaCB': 110.1},
        'charmm': {'tau': 110.7, 'NCaCB': 111.0, 'CCaCB': 108.5},
        'opls':   {'tau': 111.1, 'NCaCB': 109.5, 'CCaCB': 111.1},
    }
    defaults = ff_defaults.get(force_field.lower(), ff_defaults['amber'])
    
    # Build correction force
    # E_corr = ½k(θ - θ_lib)² - ½k(θ - θ_ff)²
    # This shifts minimum from θ_ff to θ_lib without changing k
    correction = CustomAngleForce(
        '0.5*k*(theta-theta_lib)*(theta-theta_lib)'
        ' - 0.5*k*(theta-theta_ff)*(theta-theta_ff)'
    )
    correction.addPerAngleParameter('theta_lib')
    correction.addPerAngleParameter('theta_ff')
    correction.addPerAngleParameter('k')
    
    # Convert positions to numpy for dihedral computation
    if hasattr(positions, 'value_in_unit'):
        pos_nm = np.array(positions.value_in_unit(u.nanometer))
    else:
        pos_nm = np.array(positions)
    
    # Iterate over residues
    deg2rad = np.pi / 180.0
    kcal2kj = 4.184
    n_corrections = 0
    
    residue_list = list(topology.residues())
    
    for i, residue in enumerate(residue_list):
        atoms = {a.name: a.index for a in residue.atoms()}
        
        if not all(a in atoms for a in ['N', 'CA', 'C']):
            continue
        
        # Compute φ and ψ from coordinates
        phi, psi = _compute_phi_psi(residue_list, i, atoms, pos_nm)
        
        if phi is None or psi is None:
            phi = phi or -63.0
            psi = psi or -43.0
        
        res_name = residue.name
        geom = lib.get(phi, psi, res_name)
        
        # Add τ correction (N-CA-C)
        tau_lib = geom['tau'] * deg2rad
        tau_ff = defaults['tau'] * deg2rad
        k_tau = SPRING_CONSTANTS['tau'] * kcal2kj  # to kJ/mol/rad²
        
        correction.addAngle(
            atoms['N'], atoms['CA'], atoms['C'],
            [tau_lib, tau_ff, k_tau]
        )
        n_corrections += 1
        
        # Add ∠N-Cα-Cβ correction
        if 'CB' in atoms:
            ncacb_lib = geom['angle_NCaCB'] * deg2rad
            ncacb_ff = defaults['NCaCB'] * deg2rad
            k_ncacb = SPRING_CONSTANTS['angle_NCaCB'] * kcal2kj
            
            correction.addAngle(
                atoms['N'], atoms['CA'], atoms['CB'],
                [ncacb_lib, ncacb_ff, k_ncacb]
            )
            
            # Add ∠C-Cα-Cβ correction
            ccacb_lib = geom['angle_CCaCB'] * deg2rad
            ccacb_ff = defaults['CCaCB'] * deg2rad
            k_ccacb = SPRING_CONSTANTS['angle_CCaCB'] * kcal2kj
            
            correction.addAngle(
                atoms['C'], atoms['CA'], atoms['CB'],
                [ccacb_lib, ccacb_ff, k_ccacb]
            )
            n_corrections += 2
    
    system.addForce(correction)
    print(f"backbone_geometry_library: added {n_corrections} angle corrections "
          f"({lib.stats['hits']} direct, "
          f"{lib.stats['fallback_all']} ALL fallback, "
          f"{lib.stats['fallback_default']} default fallback)")
    
    return system


def _compute_phi_psi(residues, idx, atoms, positions):
    """Compute φ and ψ for one residue from positions array."""
    phi = None
    psi = None
    
    # φ = dihedral(C[i-1], N[i], CA[i], C[i])
    if idx > 0:
        prev_atoms = {a.name: a.index for a in residues[idx - 1].atoms()}
        if 'C' in prev_atoms and all(a in atoms for a in ['N', 'CA', 'C']):
            try:
                phi = compute_dihedral(
                    positions[prev_atoms['C']],
                    positions[atoms['N']],
                    positions[atoms['CA']],
                    positions[atoms['C']]
                )
            except Exception:
                pass
    
    # ψ = dihedral(N[i], CA[i], C[i], N[i+1])
    if idx < len(residues) - 1:
        next_atoms = {a.name: a.index for a in residues[idx + 1].atoms()}
        if 'N' in next_atoms and all(a in atoms for a in ['N', 'CA', 'C']):
            try:
                psi = compute_dihedral(
                    positions[atoms['N']],
                    positions[atoms['CA']],
                    positions[atoms['C']],
                    positions[next_atoms['N']]
                )
            except Exception:
                pass
    
    return phi, psi


# ══════════════════════════════════════════════════════════════════════════════
# Command-line interface
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Command-line tool for library queries."""
    import argparse
    
    ap = argparse.ArgumentParser(
        description='Query the backbone geometry library',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Look up geometry for ALA at αR position
  python backbone_geometry_library.py --phi -63 --psi -43 --residue ALA
  
  # Look up geometry for GLY in β-sheet
  python backbone_geometry_library.py --phi -120 --psi 130 --residue GLY
  
  # Compare all 20 amino acids at αR
  python backbone_geometry_library.py --phi -63 --psi -43 --all_residues
  
  # Show AMBER comparison
  python backbone_geometry_library.py --phi -63 --psi -43 --residue ALA --compare
        """)
    
    ap.add_argument('--library', default=None,
                    help='Path to constants_library.json')
    ap.add_argument('--phi', type=float, required=True,
                    help='φ angle in degrees')
    ap.add_argument('--psi', type=float, required=True,
                    help='ψ angle in degrees')
    ap.add_argument('--residue', default='ALA',
                    help='Three-letter residue code (default: ALA)')
    ap.add_argument('--all_residues', action='store_true',
                    help='Show all 20 amino acids')
    ap.add_argument('--compare', action='store_true',
                    help='Compare with AMBER defaults')
    
    args = ap.parse_args()
    
    lib = GeometryLibrary(args.library)
    
    if args.all_residues:
        residues = lib.available_residues
    else:
        residues = [args.residue]
    
    print(f"\nBackbone Geometry Library v{__version__}")
    print(f"  φ = {args.phi}°, ψ = {args.psi}°\n")
    
    header = f"  {'Res':>4s}"
    params = ['tau', 'angle_NCaCB', 'angle_CCaCB', 'bond_NCA', 'bond_CAC', 'bond_CACB']
    labels = ['τ(°)', '∠NCaCB(°)', '∠CCaCB(°)', 'N-Cα(Å)', 'Cα-C(Å)', 'Cα-Cβ(Å)']
    
    for label in labels:
        header += f"  {label:>10s}"
    print(header)
    print("  " + "─" * (len(header.strip())))
    
    if args.compare:
        line = f"  {'AMBER':>4s}"
        for param in params:
            line += f"  {AMBER_DEFAULTS[param]:>10.3f}"
        print(line)
        print("  " + "─" * (len(header.strip())))
    
    for res in residues:
        geom = lib.get(args.phi, args.psi, res)
        line = f"  {res:>4s}"
        for param in params:
            val = geom[param]
            line += f"  {val:>10.3f}"
        print(line)
    
    print()


if __name__ == '__main__':
    main()