#!/usr/bin/env python3
"""
backbone_geometry_library — Conformation-Dependent Backbone Geometry
=====================================================================

A drop-in replacement for fixed backbone constants in protein modelling.
"""

__version__ = "1.0.0"
__author__ = "Wei Chen"

import json
import os
import numpy as np
from pathlib import Path


AMBER_DEFAULTS = {
    'tau':         111.1,
    'angle_NCaCB': 110.1,
    'angle_CCaCB': 110.1,
    'angle_CaCN':  116.6,
    'angle_CNCa':  121.9,
    'angle_CaCO':  120.4,
    'bond_NCA':    1.458,
    'bond_CAC':    1.522,
    'bond_CO':     1.229,
    'bond_CN':     1.335,
    'bond_CACB':   1.526,
    'omega':       180.0,
}

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

SPRING_CONSTANTS = {
    'tau': 63.0, 'angle_NCaCB': 63.0, 'angle_CCaCB': 63.0,
    'angle_CaCN': 70.0, 'angle_CNCa': 50.0, 'angle_CaCO': 80.0,
    'bond_NCA': 337.0, 'bond_CAC': 317.0, 'bond_CO': 570.0,
    'bond_CN': 490.0, 'bond_CACB': 317.0,
}


class GeometryLibrary:
    def __init__(self, library_path=None, chi1_path=None, bin_size=10):
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
        self._centers = np.arange(-180 + self._half, 180 + self._half, bin_size)
        self._stats = {'hits': 0, 'fallback_all': 0, 'fallback_default': 0}

    def _bin_key(self, angle):
        angle = ((angle + 180) % 360) - 180
        idx = int(np.round((angle - self._centers[0]) / self._bin_size))
        idx = max(0, min(idx, len(self._centers) - 1))
        return str(int(self._centers[idx]))

    def _lookup_cell(self, phi, psi, residue):
        pk, qk = self._bin_key(phi), self._bin_key(psi)
        for cls in [residue, 'ALL']:
            if cls in self._lib:
                cell = self._lib[cls].get(pk, {}).get(qk)
                if cell:
                    self._stats['hits' if cls == residue else 'fallback_all'] += 1
                    return cell
        self._stats['fallback_default'] += 1
        return None

    def _get_value(self, cell, param):
        if cell is None:
            return AMBER_DEFAULTS.get(param, 0.0)
        lib_key = _KEY_MAP.get(param)
        if lib_key and lib_key in cell:
            val = cell[lib_key]
            if param == 'omega' and abs(val) < 90:
                return 180.0
            return val
        return AMBER_DEFAULTS.get(param, 0.0)

    def get(self, phi, psi, residue='ALA'):
        cell = self._lookup_cell(phi, psi, residue)
        return {param: self._get_value(cell, param) for param in AMBER_DEFAULTS}

    def get_tau(self, phi, psi, residue='ALA'):
        cell = self._lookup_cell(phi, psi, residue)
        return self._get_value(cell, 'tau')

    def get_bonds(self, phi, psi, residue='ALA'):
        cell = self._lookup_cell(phi, psi, residue)
        return {
            'NCA':  self._get_value(cell, 'bond_NCA'),
            'CAC':  self._get_value(cell, 'bond_CAC'),
            'CO':   self._get_value(cell, 'bond_CO'),
            'CN':   self._get_value(cell, 'bond_CN'),
            'CACB': self._get_value(cell, 'bond_CACB'),
        }

    def get_angles(self, phi, psi, residue='ALA'):
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
        cell = self._lookup_cell(phi, psi, residue)
        return self._get_value(cell, 'omega')

    def get_chi1_correction(self, phi, psi, residue, chi1_rotamer):
        if self._chi1 is None:
            return None
        pk, qk = self._bin_key(phi), self._bin_key(psi)
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
        return sorted([k for k in self._lib.keys() if k != 'ALL'])

    @property
    def stats(self):
        total = sum(self._stats.values())
        return {**self._stats, 'total': total}

    def reset_stats(self):
        self._stats = {k: 0 for k in self._stats}


def compute_dihedral(p1, p2, p3, p4):
    b1 = np.asarray(p2) - np.asarray(p1)
    b2 = np.asarray(p3) - np.asarray(p2)
    b3 = np.asarray(p4) - np.asarray(p3)
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_len, n2_len = np.linalg.norm(n1), np.linalg.norm(n2)
    if n1_len < 1e-8 or n2_len < 1e-8:
        return 0.0
    n1, n2 = n1 / n1_len, n2 / n2_len
    b2_u = b2 / np.linalg.norm(b2)
    m1 = np.cross(n1, b2_u)
    return float(np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2))))


def _compute_phi_psi(residues, idx, atoms, positions):
    phi = None
    psi = None
    if idx > 0:
        prev_atoms = {a.name: a.index for a in residues[idx - 1].atoms()}
        if 'C' in prev_atoms and all(a in atoms for a in ['N', 'CA', 'C']):
            try:
                phi = compute_dihedral(
                    positions[prev_atoms['C']], positions[atoms['N']],
                    positions[atoms['CA']], positions[atoms['C']])
            except Exception:
                pass
    if idx < len(residues) - 1:
        next_atoms = {a.name: a.index for a in residues[idx + 1].atoms()}
        if 'N' in next_atoms and all(a in atoms for a in ['N', 'CA', 'C']):
            try:
                psi = compute_dihedral(
                    positions[atoms['N']], positions[atoms['CA']],
                    positions[atoms['C']], positions[next_atoms['N']])
            except Exception:
                pass
    return phi, psi


def apply_corrections(system, topology, positions, library_path=None,
                       force_field='amber'):
    try:
        from openmm import CustomAngleForce
        from openmm import unit as u
    except ImportError:
        raise ImportError(
            "OpenMM is required for force-field integration. "
            "Install with: conda install -c conda-forge openmm"
        )

    lib = GeometryLibrary(library_path)

    ff_defaults = {
        'amber':  {'tau': 111.1, 'NCaCB': 110.1, 'CCaCB': 110.1},
        'charmm': {'tau': 110.7, 'NCaCB': 111.0, 'CCaCB': 108.5},
        'opls':   {'tau': 111.1, 'NCaCB': 109.5, 'CCaCB': 111.1},
    }
    defaults = ff_defaults.get(force_field.lower(), ff_defaults['amber'])

    correction = CustomAngleForce(
        '0.5*k*(theta-theta_lib)*(theta-theta_lib)'
        ' - 0.5*k*(theta-theta_ff)*(theta-theta_ff)'
    )
    correction.addPerAngleParameter('theta_lib')
    correction.addPerAngleParameter('theta_ff')
    correction.addPerAngleParameter('k')

    if hasattr(positions, 'value_in_unit'):
        pos_nm = np.array(positions.value_in_unit(u.nanometer))
    else:
        pos_nm = np.array(positions)

    deg2rad = np.pi / 180.0
    kcal2kj = 4.184
    n_corrections = 0
    residue_list = list(topology.residues())

    for i, residue in enumerate(residue_list):
        atoms = {a.name: a.index for a in residue.atoms()}
        if not all(a in atoms for a in ['N', 'CA', 'C']):
            continue

        phi, psi = _compute_phi_psi(residue_list, i, atoms, pos_nm)
        if phi is None or psi is None:
            phi = phi or -63.0
            psi = psi or -43.0

        res_name = residue.name
        geom = lib.get(phi, psi, res_name)

        tau_lib = geom['tau'] * deg2rad
        tau_ff = defaults['tau'] * deg2rad
        k_tau = SPRING_CONSTANTS['tau'] * kcal2kj

        correction.addAngle(atoms['N'], atoms['CA'], atoms['C'],
                             [tau_lib, tau_ff, k_tau])
        n_corrections += 1

        if 'CB' in atoms:
            ncacb_lib = geom['angle_NCaCB'] * deg2rad
            ncacb_ff = defaults['NCaCB'] * deg2rad
            k_ncacb = SPRING_CONSTANTS['angle_NCaCB'] * kcal2kj
            correction.addAngle(atoms['N'], atoms['CA'], atoms['CB'],
                                 [ncacb_lib, ncacb_ff, k_ncacb])

            ccacb_lib = geom['angle_CCaCB'] * deg2rad
            ccacb_ff = defaults['CCaCB'] * deg2rad
            k_ccacb = SPRING_CONSTANTS['angle_CCaCB'] * kcal2kj
            correction.addAngle(atoms['C'], atoms['CA'], atoms['CB'],
                                 [ccacb_lib, ccacb_ff, k_ccacb])
            n_corrections += 2

    system.addForce(correction)
    print(f"backbone_geometry_library: added {n_corrections} angle corrections "
          f"({lib.stats['hits']} direct, "
          f"{lib.stats['fallback_all']} ALL fallback, "
          f"{lib.stats['fallback_default']} default fallback)")
    return system
