"""
mechlib.py — Conformation-dependent backbone geometry correction library
============================================================================
Usage:
    from mechlib import MechLibProtein, MechLibDNA

    protein_lib = MechLibProtein("MechLib_protein_library.csv")
    corrected_tau = protein_lib.get("ALA", phi=-60.0, psi=-45.0, "tau_deg")

    dna_lib = MechLibDNA("MechLib_DNA_library.csv")
    corrected_bond = dna_lib.get("DA", bi_bii="BI", pucker_class="C2-endo",
                                 observable="bond_P_O5")

If a requested combination isn't in the library (insufficient data in the
original PDB-derived dataset), both classes fall back to the nearest
available pooled average, and finally to the AMBER/OPLS-AA/M fixed
equilibrium value, so a lookup always returns a usable number.

Citation: [add DOI / paper reference once published]
"""

import csv
from bisect import bisect_left

# Fixed force-field equilibria used as the final fallback layer.
# Protein values verified against AMBER ff14SB (protein.ff14SB.xml).
_AMBER_FALLBACK = {
    'tau_deg': 110.10, 'angle_N_CA_CB': 109.70, 'angle_C_CA_CB': 111.10,
    'angle_CaCN': 116.6, 'angle_CNCa': 121.9, 'angle_CA_C_O': 120.4,
    'bond_N_CA': 1.449, 'bond_CA_C': 1.522, 'bond_C_O': 1.229,
    'bond_C_N_next': 1.335, 'bond_CA_CB': 1.526,
}

# DNA values verified against AMBER parm10.dat (shared across bsc0/bsc1/OL15).
_AMBER_DNA_FALLBACK = {
    'bond_P_O5': 1.610, 'bond_O5_C5': 1.410, 'bond_C5_C4': 1.526,
    'bond_C4_O4': 1.410, 'bond_C4_C3': 1.526, 'bond_C3_O3': 1.410,
    'bond_C3_C2': 1.526, 'bond_C2_C1': 1.526, 'bond_C1_O4': 1.410,
    'bond_C1_N': 1.475, 'bond_O3_Pnext': 1.610,
    'angle_O5_P_O3prev': 102.60, 'angle_OP1_P_OP2': 119.90,
    'angle_C5_O5_P': 120.50, 'angle_C5_C4_O4': 109.50, 'angle_C5_C4_C3': 109.50,
    'angle_O4_C4_C3': 109.50, 'angle_C4_C3_O3': 109.50, 'angle_C4_C3_C2': 109.50,
    'angle_O3_C3_C2': 109.50, 'angle_C3_C2_C1': 109.50, 'angle_C2_C1_O4': 109.50,
    'angle_C1_O4_C4': 109.50, 'angle_O4_C1_N': 109.50, 'angle_C2_C1_N': 109.50,
    'angle_C3_O3_Pnext': 120.50,
}


class MechLibProtein:
    """Loads MechLib_protein_library.csv and provides conformation-dependent
    lookup of corrected backbone bond/angle equilibrium values."""

    BIN_SIZE = 10.0  # degrees; must match the bin size used to build the CSV

    def __init__(self, csv_path):
        self._by_res = {}     # {(res, phi_bin, psi_bin): {col: value}}
        self._pooled = {}      # {(phi_bin, psi_bin): {col: [values, count]}}
        self._load(csv_path)

    def _bin_center(self, angle_deg):
        wrapped = ((angle_deg + 180.0) % 360.0) - 180.0
        n_bins = int(round(wrapped / self.BIN_SIZE))
        return n_bins * self.BIN_SIZE + self.BIN_SIZE / 2.0 * (1 if wrapped >= 0 else -1)

    def _load(self, csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            cols = [c for c in reader.fieldnames
                    if c not in ('res_name', 'phi_bin_center', 'psi_bin_center', 'n_residues')]
            for row in reader:
                res = row['res_name']
                phi_c = float(row['phi_bin_center'])
                psi_c = float(row['psi_bin_center'])
                n = int(row['n_residues'])
                values = {c: (float(row[c]) if row[c] != '' else None) for c in cols}

                self._by_res[(res, phi_c, psi_c)] = values

                pkey = (phi_c, psi_c)
                if pkey not in self._pooled:
                    self._pooled[pkey] = {c: [0.0, 0] for c in cols}
                for c in cols:
                    if values[c] is not None:
                        self._pooled[pkey][c][0] += values[c] * n
                        self._pooled[pkey][c][1] += n

    def get(self, res_name, phi, psi, observable):
        """Return the MechLib-corrected equilibrium value for `observable`
        at the given residue type and backbone conformation. Falls back to
        pooled (phi,psi) average, then to the fixed AMBER equilibrium, if
        the specific (residue, phi, psi) bin has insufficient data."""
        phi_c = self._nearest_bin_center(phi)
        psi_c = self._nearest_bin_center(psi)

        key = (res_name, phi_c, psi_c)
        if key in self._by_res and self._by_res[key].get(observable) is not None:
            return self._by_res[key][observable]

        pkey = (phi_c, psi_c)
        if pkey in self._pooled:
            total, count = self._pooled[pkey].get(observable, [0.0, 0])
            if count > 0:
                return total / count

        return _AMBER_FALLBACK.get(observable)

    def _nearest_bin_center(self, angle_deg):
        wrapped = ((angle_deg + 180.0) % 360.0) - 180.0
        bin_idx = int((wrapped + 180.0) // self.BIN_SIZE)
        return -180.0 + bin_idx * self.BIN_SIZE + self.BIN_SIZE / 2.0


class MechLibDNA:
    """Loads MechLib_DNA_library.csv and provides conformation-dependent
    lookup of corrected DNA backbone bond/angle equilibrium values."""

    def __init__(self, csv_path):
        self._by_base = {}   # {(base, bi_bii, pucker): {col: value}}
        self._pooled = {}     # {(bi_bii, pucker): {col: [sum, count]}}
        self._load(csv_path)

    def _load(self, csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            cols = [c for c in reader.fieldnames
                    if c not in ('res_name', 'bi_bii', 'pucker_class', 'n_nucleotides')]
            for row in reader:
                base = row['res_name']
                bibii = row['bi_bii']
                pucker = row['pucker_class']
                n = int(row['n_nucleotides'])
                values = {c: (float(row[c]) if row[c] != '' else None) for c in cols}

                self._by_base[(base, bibii, pucker)] = values

                pkey = (bibii, pucker)
                if pkey not in self._pooled:
                    self._pooled[pkey] = {c: [0.0, 0] for c in cols}
                for c in cols:
                    if values[c] is not None:
                        self._pooled[pkey][c][0] += values[c] * n
                        self._pooled[pkey][c][1] += n

    def get(self, base, bi_bii, pucker_class, observable):
        """Return the MechLib-DNA-corrected equilibrium value for
        `observable`, given base identity, BI/BII state, and sugar pucker
        class. Falls back to pooled (BI/BII, pucker) average, then to the
        fixed AMBER parm10 equilibrium."""
        key = (base, bi_bii, pucker_class)
        if key in self._by_base and self._by_base[key].get(observable) is not None:
            return self._by_base[key][observable]

        pkey = (bi_bii, pucker_class)
        if pkey in self._pooled:
            total, count = self._pooled[pkey].get(observable, [0.0, 0])
            if count > 0:
                return total / count

        return _AMBER_DNA_FALLBACK.get(observable)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--selftest':
        print("Run with real MechLib_protein_library.csv / MechLib_DNA_library.csv "
              "to test; see module docstring for usage.")
