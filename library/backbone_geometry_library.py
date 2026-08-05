#!/usr/bin/env python3
"""
backbone_geometry_library — Conformation-Dependent Backbone Geometry
=====================================================================

A drop-in replacement for fixed backbone constants in protein and DNA
modelling.

Changelog (this revision fixes 6 validation failures found by
validate_mechlib_package.py in the prior release):
  1. Chi1 sub-library now resolves lookups using ITS OWN bin size
     (auto-detected from constants_chi1.json, typically 20 deg),
     instead of silently reusing the main library's bin size (10 deg).
     Previously this made get_chi1_correction() a dead code path that
     always returned None.
  2. apply_corrections() now applies ALL 11 library observables (all
     6 backbone angles + all 5 bond lengths) via CustomAngleForce and
     a new CustomBondForce, not just tau/N-Ca-Cb/C-Ca-Cb. See the
     NOTE in apply_corrections() about which of these have verified
     per-force-field (AMBER/CHARMM/OPLS) baselines vs. AMBER-only.
  3. The phi/psi fallback now uses an explicit `is None` check instead
     of Python truthiness, so a legitimately computed dihedral of
     exactly 0.0 deg is no longer silently replaced by the default.
  4. Added DNAGeometryLibrary, a new class that loads and queries
     MechLib_DNA_library.csv. Previously this file was shipped with
     no supporting code anywhere.
  5. Added a CLI (`if __name__ == '__main__'`) matching the usage
     documented in README.md, which previously did not exist.
  6. library_correction.frcmod is no longer part of this release (see
     README -- it was never valid tleap syntax and duplicated a
     removal the README already documented).
"""

__version__ = "1.1.0"
__author__ = "Wei Chen"

import csv
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

# NOTE on fix #2 scope: only tau/angle_NCaCB/angle_CCaCB have
# per-force-field-verified baselines (see ff_defaults in
# apply_corrections, sourced from Paper 4 Methods 2.2). The 3
# additional angles and 5 bond lengths wired in this revision use the
# AMBER ff14SB values in AMBER_DEFAULTS as their baseline regardless
# of the requested force_field, because CHARMM36/OPLS-AA/M-specific
# values for those 8 terms were not independently verified in this
# release. apply_corrections() prints a one-time warning when this
# applies. Treat those 8 terms' corrections as AMBER-referenced even
# when force_field='charmm' or 'opls'.


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

        self._bin_size = bin_size
        self._half = bin_size / 2.0
        self._centers = np.arange(-180 + self._half, 180 + self._half, bin_size)

        # --- chi1 sub-library: FIX #1 -----------------------------------
        # The chi1 table is commonly exported on a coarser (phi,psi) grid
        # than the main library (e.g. 20 deg vs 10 deg) because splitting
        # by rotamer thins out cell populations fast. It must be resolved
        # with ITS OWN bin size, auto-detected from its own keys, not the
        # main library's self._bin_size/self._centers.
        self._chi1 = None
        self._chi1_bin_size = None
        self._chi1_half = None
        self._chi1_centers = None
        if chi1_path is None:
            chi1_default = Path(str(library_path).replace(
                'constants_library', 'constants_chi1'))
            if chi1_default.exists():
                chi1_path = chi1_default
        if chi1_path and Path(chi1_path).exists():
            with open(chi1_path) as f:
                self._chi1 = json.load(f)
            detected = self._infer_bin_size(self._chi1)
            self._chi1_bin_size = detected if detected is not None else bin_size
            self._chi1_half = self._chi1_bin_size / 2.0
            self._chi1_centers = np.arange(
                -180 + self._chi1_half, 180 + self._chi1_half, self._chi1_bin_size)

        self._stats = {'hits': 0, 'fallback_all': 0, 'fallback_default': 0}

    @staticmethod
    def _infer_bin_size(nested_dict):
        """Detect the (phi,psi) bin spacing actually used in a
        res -> phi_bin -> psi_bin -> ... nested dict, from its own
        keys, rather than assuming it matches some other table's bin
        size. Returns None if it can't be determined (e.g. empty dict
        or a residue with only one phi bin), in which case the caller
        should fall back to an explicit default."""
        for res, phis in nested_dict.items():
            keys = sorted(int(k) for k in phis.keys())
            if len(keys) > 1:
                return keys[1] - keys[0]
        return None

    def _bin_key(self, angle, bin_size=None, centers=None):
        bin_size = self._bin_size if bin_size is None else bin_size
        centers = self._centers if centers is None else centers
        angle = ((angle + 180) % 360) - 180
        idx = int(np.round((angle - centers[0]) / bin_size))
        idx = max(0, min(idx, len(centers) - 1))
        return str(int(centers[idx]))

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
        # FIX #1: use the chi1 sub-library's own bin size/centers, not
        # the main library's.
        pk = self._bin_key(phi, self._chi1_bin_size, self._chi1_centers)
        qk = self._bin_key(psi, self._chi1_bin_size, self._chi1_centers)
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


class DNAGeometryLibrary:
    """Conformation-dependent DNA sugar-phosphate backbone geometry.

    FIX #4: this class did not exist in the prior release --
    MechLib_DNA_library.csv shipped with no supporting code anywhere.

    Unlike the protein library, DNA geometry here is keyed by
    (base, BI/BII backbone state, sugar pucker class) rather than a
    continuous (phi,psi) grid -- those are the coordinates the DNA
    analysis (Paper 4, Section 3.4-3.5) actually bins by. There is no
    residue-agnostic 'ALL' pooled fallback in this release (the CSV
    has 74 rows total; a pooled fallback was not exported), so an
    unpopulated (base, state, pucker) combination returns None rather
    than a fallback value -- callers should handle that explicitly,
    e.g. by falling back to their own AMBER parm10 defaults.
    """

    def __init__(self, library_csv=None):
        if library_csv is None:
            module_dir = Path(__file__).parent
            library_csv = module_dir / 'MechLib_DNA_library.csv'
            if not Path(library_csv).exists():
                raise FileNotFoundError(
                    f"DNA library not found at {library_csv}. "
                    f"Pass library_csv explicitly or place "
                    f"MechLib_DNA_library.csv next to this module."
                )
        self._lib = {}
        meta_cols = {'res_name', 'bi_bii', 'pucker_class', 'n_nucleotides'}
        with open(library_csv, newline='') as f:
            reader = csv.DictReader(f)
            self._fields = [c for c in reader.fieldnames if c not in meta_cols]
            for row in reader:
                key = (row['res_name'].upper(), row['bi_bii'], row['pucker_class'])
                cell = {}
                for k in self._fields:
                    v = row.get(k, '')
                    if v not in ('', None):
                        try:
                            cell[k] = float(v)
                        except ValueError:
                            pass
                self._lib[key] = cell

    def get(self, base, bi_bii, pucker_class):
        """Return the dict of corrected geometry terms for this exact
        (base, BI/BII state, pucker class) combination, or None if
        this release's library doesn't have that combination
        populated (no fallback tier is defined for DNA -- see class
        docstring)."""
        return self._lib.get((base.upper(), bi_bii, pucker_class))

    @property
    def available_keys(self):
        return sorted(self._lib.keys())


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
    """Add conformation-dependent bonded-geometry corrections to an
    OpenMM system.

    FIX #2: this now corrects ALL 11 library observables (6 angles +
    5 bond lengths) via CustomAngleForce + CustomBondForce, not just
    tau/N-Ca-Cb/C-Ca-Cb. Only tau, N-Ca-Cb, and C-Ca-Cb have baselines
    independently verified per force field (Paper 4 Methods 2.2); the
    other 3 angles and all 5 bonds use AMBER ff14SB baselines
    regardless of the `force_field` argument -- see the module-level
    NOTE above SPRING_CONSTANTS. A one-time warning is printed when
    force_field != 'amber' to make this explicit rather than silently
    mixing baseline sources.
    """
    try:
        from openmm import CustomAngleForce, CustomBondForce
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
    ff_key = force_field.lower()
    defaults = ff_defaults.get(ff_key, ff_defaults['amber'])
    if ff_key != 'amber':
        print(f"backbone_geometry_library: NOTE -- tau/N-Ca-Cb/C-Ca-Cb use "
              f"{ff_key}-specific baselines; the additional 3 angles and all "
              f"5 bond-length corrections use AMBER ff14SB baselines "
              f"regardless (not independently verified for {ff_key} in this "
              f"release). See apply_corrections() docstring.")

    angle_correction = CustomAngleForce(
        '0.5*k*(theta-theta_lib)*(theta-theta_lib)'
        ' - 0.5*k*(theta-theta_ff)*(theta-theta_ff)'
    )
    angle_correction.addPerAngleParameter('theta_lib')
    angle_correction.addPerAngleParameter('theta_ff')
    angle_correction.addPerAngleParameter('k')

    bond_correction = CustomBondForce(
        '0.5*k*(r-r_lib)*(r-r_lib)'
        ' - 0.5*k*(r-r_ff)*(r-r_ff)'
    )
    bond_correction.addPerBondParameter('r_lib')
    bond_correction.addPerBondParameter('r_ff')
    bond_correction.addPerBondParameter('k')

    if hasattr(positions, 'value_in_unit'):
        pos_nm = np.array(positions.value_in_unit(u.nanometer))
    else:
        pos_nm = np.array(positions)

    deg2rad = np.pi / 180.0
    kcal2kj = 4.184
    n_angle_corrections = 0
    n_bond_corrections = 0
    residue_list = list(topology.residues())

    for i, residue in enumerate(residue_list):
        atoms = {a.name: a.index for a in residue.atoms()}
        if not all(a in atoms for a in ['N', 'CA', 'C']):
            continue

        phi, psi = _compute_phi_psi(residue_list, i, atoms, pos_nm)
        if phi is None or psi is None:
            # FIX #3: explicit None-check, not truthiness -- a
            # legitimately computed 0.0 deg dihedral must not be
            # treated as missing.
            phi = phi if phi is not None else -63.0
            psi = psi if psi is not None else -43.0

        res_name = residue.name
        geom = lib.get(phi, psi, res_name)

        def add_angle(a1, a2, a3, lib_val, ff_val, spring_key):
            nonlocal n_angle_corrections
            k = SPRING_CONSTANTS[spring_key] * kcal2kj
            angle_correction.addAngle(
                a1, a2, a3, [lib_val * deg2rad, ff_val * deg2rad, k])
            n_angle_corrections += 1

        def add_bond(a1, a2, lib_val, ff_val, spring_key):
            nonlocal n_bond_corrections
            k = SPRING_CONSTANTS[spring_key] * kcal2kj
            bond_correction.addBond(
                a1, a2, [lib_val * 0.1, ff_val * 0.1, k])  # Angstrom -> nm
            n_bond_corrections += 1

        # tau (N-CA-C): per-force-field baseline
        add_angle(atoms['N'], atoms['CA'], atoms['C'],
                   geom['tau'], defaults['tau'], 'tau')

        if 'CB' in atoms:
            add_angle(atoms['N'], atoms['CA'], atoms['CB'],
                       geom['angle_NCaCB'], defaults['NCaCB'], 'angle_NCaCB')
            add_angle(atoms['C'], atoms['CA'], atoms['CB'],
                       geom['angle_CCaCB'], defaults['CCaCB'], 'angle_CCaCB')
            add_bond(atoms['CA'], atoms['CB'],
                     geom['bond_CACB'], AMBER_DEFAULTS['bond_CACB'], 'bond_CACB')

        # intra-residue additions (AMBER-referenced baseline -- see docstring)
        if 'O' in atoms:
            add_angle(atoms['CA'], atoms['C'], atoms['O'],
                       geom['angle_CaCO'], AMBER_DEFAULTS['angle_CaCO'], 'angle_CaCO')
            add_bond(atoms['C'], atoms['O'],
                     geom['bond_CO'], AMBER_DEFAULTS['bond_CO'], 'bond_CO')

        add_bond(atoms['N'], atoms['CA'],
                 geom['bond_NCA'], AMBER_DEFAULTS['bond_NCA'], 'bond_NCA')
        add_bond(atoms['CA'], atoms['C'],
                 geom['bond_CAC'], AMBER_DEFAULTS['bond_CAC'], 'bond_CAC')

        # inter-residue additions: need the next residue's N/CA
        if i < len(residue_list) - 1:
            next_atoms = {a.name: a.index for a in residue_list[i + 1].atoms()}
            if 'N' in next_atoms:
                add_bond(atoms['C'], next_atoms['N'],
                         geom['bond_CN'], AMBER_DEFAULTS['bond_CN'], 'bond_CN')
                add_angle(atoms['CA'], atoms['C'], next_atoms['N'],
                           geom['angle_CaCN'], AMBER_DEFAULTS['angle_CaCN'], 'angle_CaCN')
                if 'CA' in next_atoms:
                    add_angle(atoms['C'], next_atoms['N'], next_atoms['CA'],
                               geom['angle_CNCa'], AMBER_DEFAULTS['angle_CNCa'], 'angle_CNCa')

    system.addForce(angle_correction)
    system.addForce(bond_correction)
    print(f"backbone_geometry_library: added {n_angle_corrections} angle "
          f"corrections and {n_bond_corrections} bond corrections "
          f"({lib.stats['hits']} direct, "
          f"{lib.stats['fallback_all']} ALL fallback, "
          f"{lib.stats['fallback_default']} default fallback)")
    return system


def _cli():
    """FIX #5: CLI matching the usage documented in README.md, which
    previously did not exist -- `python backbone_geometry_library.py
    --phi ... --psi ... --residue ...` could not run at all."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Query conformation-dependent backbone geometry "
                     "for a given (phi, psi) and residue.")
    parser.add_argument('--phi', type=float, required=True)
    parser.add_argument('--psi', type=float, required=True)
    parser.add_argument('--residue', type=str, default='ALA',
                         help="Three-letter residue code (ignored if --all_residues)")
    parser.add_argument('--compare', action='store_true',
                         help="Also print the fixed AMBER default for comparison")
    parser.add_argument('--all_residues', action='store_true',
                         help="Print geometry for all 20 amino acids at this (phi,psi)")
    parser.add_argument('--library_path', type=str, default=None)
    parser.add_argument('--chi1_path', type=str, default=None)
    args = parser.parse_args()

    lib = GeometryLibrary(args.library_path, args.chi1_path)
    residues = lib.available_residues if args.all_residues else [args.residue.upper()]

    for res in residues:
        geom = lib.get(args.phi, args.psi, res)
        if args.compare:
            print(f"{res}: tau={geom['tau']:.2f} deg "
                  f"(AMBER default {AMBER_DEFAULTS['tau']:.2f}), "
                  f"N-Ca-Cb={geom['angle_NCaCB']:.2f} deg "
                  f"(AMBER default {AMBER_DEFAULTS['angle_NCaCB']:.2f}), "
                  f"bond N-CA={geom['bond_NCA']:.4f} A "
                  f"(AMBER default {AMBER_DEFAULTS['bond_NCA']:.4f})")
        else:
            print(f"{res}: {geom}")


if __name__ == '__main__':
    _cli()
