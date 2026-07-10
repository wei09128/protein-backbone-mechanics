"""
nerf_builder.py — NeRF Protein Structure Builder v2.4
======================================================

v2.4 changes:
  OPT 1:  Rotamer-library clash avoidance replaces degree-step grid scan.
          _find_clash_free_chis now tries 3–9 canonical rotamers per residue
          (from Lovell 2000 backbone-independent modes) instead of scanning
          all angles at 10°/15° steps.

          Trials comparison:
            Single-chi residues (SER/CYS/THR):   36 →  3   (~12×  faster)
            Two-chi residues    (ILE/LEU/TRP):   576 →  5   (~115× faster)
            Aromatic            (PHE/TYR/HIS):   576 →  5   (~115× faster)
            Long chains         (LYS/ARG/GLN):   576 →  5   (~115× faster)

          Strategy: try each (χ1, χ2) library pair; keep χ3+ from chi_list
          unchanged. Accept first clash-free hit; if none, keep best
          (largest min_d) to minimise energy distortion rather than silently
          leaving the clash.

  Everything else unchanged from v2.3.
"""

import numpy as np
from molcore import (
    BondGeometry, AtomTypes, AminoAcids,
    ResidueAtomMap, PDBParser,
    dihedral_angle, bond_angle, rmsd
)

_DEFAULT_ANGLES_DEG = {
    'GLY': (116.4, 120.6, 112.5), 'PRO': (116.9, 122.6, 111.8),
    'ALA': (116.2, 121.7, 111.2), 'VAL': (116.0, 121.4, 111.4),
    'LEU': (116.3, 121.8, 111.0), 'ILE': (115.9, 121.5, 111.5),
    'PHE': (116.2, 121.6, 111.2), 'TYR': (116.2, 121.6, 111.2),
    'TRP': (116.2, 121.6, 111.2), 'SER': (116.3, 121.7, 111.2),
    'THR': (116.1, 121.5, 111.5), 'CYS': (116.2, 121.7, 111.2),
    'MET': (116.3, 121.8, 111.0), 'ASP': (116.4, 121.9, 111.0),
    'ASN': (116.4, 121.8, 111.0), 'GLU': (116.3, 121.8, 111.0),
    'GLN': (116.3, 121.8, 111.0), 'LYS': (116.3, 121.8, 111.0),
    'ARG': (116.3, 121.8, 111.0), 'HIS': (116.2, 121.7, 111.2),
}

DEFAULT_ANGLES = {
    res: tuple(np.radians(a) for a in angles)
    for res, angles in _DEFAULT_ANGLES_DEG.items()
}

DEFAULT_CHI = {
    'GLY': [], 'ALA': [],
    'VAL': [np.radians(177)],
    'LEU': [np.radians(-65), np.radians(175)],
    'ILE': [np.radians(-60), np.radians(170)],
    'PRO': [np.radians(30), np.radians(-40)],
    'PHE': [np.radians(-65), np.radians(90)],
    'TYR': [np.radians(-65), np.radians(90)],
    'TRP': [np.radians(-65), np.radians(-90)],
    'SER': [np.radians(-60)], 'THR': [np.radians(-60)],
    'CYS': [np.radians(-60)],
    'MET': [np.radians(-65), np.radians(180), np.radians(75)],
    'ASP': [np.radians(-70), np.radians(-15)],
    'ASN': [np.radians(-65), np.radians(-20)],
    'GLU': [np.radians(-65), np.radians(180), np.radians(-20)],
    'GLN': [np.radians(-65), np.radians(180), np.radians(-20)],
    'LYS': [np.radians(-65), np.radians(180), np.radians(180), np.radians(180)],
    'ARG': [np.radians(-65), np.radians(180), np.radians(180), np.radians(180)],
    'HIS': [np.radians(-65), np.radians(-70)],
}

SIDECHAIN_BUILD = {
    'SER': [('OG',('N','CA','CB'),1.417,np.radians(111.0),'chi1')],
    'THR': [('OG1',('N','CA','CB'),1.433,np.radians(109.5),'chi1'),
            ('CG2',('N','CA','CB'),1.521,np.radians(111.5),np.radians(120))],
    'CYS': [('SG',('N','CA','CB'),1.808,np.radians(113.8),'chi1')],
    'VAL': [('CG1',('N','CA','CB'),1.521,np.radians(110.7),'chi1'),
            ('CG2',('N','CA','CB'),1.521,np.radians(110.4),'chi1_plus120')],
    'LEU': [('CG',('N','CA','CB'),1.530,np.radians(116.3),'chi1'),
            ('CD1',('CA','CB','CG'),1.524,np.radians(110.7),'chi2'),
            ('CD2',('CA','CB','CG'),1.525,np.radians(110.7),'chi2_plus120')],
    'ILE': [('CG1',('N','CA','CB'),1.530,np.radians(110.7),'chi1'),
            ('CG2',('N','CA','CB'),1.521,np.radians(110.4),'chi1_plus120'),
            ('CD1',('CA','CB','CG1'),1.513,np.radians(113.8),'chi2')],
    'ASP': [('CG',('N','CA','CB'),1.516,np.radians(112.6),'chi1'),
            ('OD1',('CA','CB','CG'),1.250,np.radians(118.4),'chi2'),
            ('OD2',('CA','CB','CG'),1.250,np.radians(118.4),'chi2_plus180')],
    'ASN': [('CG',('N','CA','CB'),1.516,np.radians(112.6),'chi1'),
            ('OD1',('CA','CB','CG'),1.231,np.radians(120.8),'chi2'),
            ('ND2',('CA','CB','CG'),1.335,np.radians(116.5),'chi2_plus180')],
    'GLU': [('CG',('N','CA','CB'),1.520,np.radians(113.4),'chi1'),
            ('CD',('CA','CB','CG'),1.516,np.radians(113.0),'chi2'),
            ('OE1',('CB','CG','CD'),1.250,np.radians(118.4),'chi3'),
            ('OE2',('CB','CG','CD'),1.250,np.radians(118.4),'chi3_plus180')],
    'GLN': [('CG',('N','CA','CB'),1.520,np.radians(113.4),'chi1'),
            ('CD',('CA','CB','CG'),1.516,np.radians(113.0),'chi2'),
            ('OE1',('CB','CG','CD'),1.231,np.radians(120.8),'chi3'),
            ('NE2',('CB','CG','CD'),1.335,np.radians(116.5),'chi3_plus180')],
    'LYS': [('CG',('N','CA','CB'),1.520,np.radians(113.8),'chi1'),
            ('CD',('CA','CB','CG'),1.520,np.radians(111.3),'chi2'),
            ('CE',('CB','CG','CD'),1.520,np.radians(111.9),'chi3'),
            ('NZ',('CG','CD','CE'),1.489,np.radians(111.9),'chi4')],
    'ARG': [('CG',('N','CA','CB'),1.520,np.radians(113.8),'chi1'),
            ('CD',('CA','CB','CG'),1.520,np.radians(111.3),'chi2'),
            ('NE',('CB','CG','CD'),1.460,np.radians(111.9),'chi3'),
            ('CZ',('CG','CD','NE'),1.326,np.radians(124.0),'chi4'),
            ('NH1',('CD','NE','CZ'),1.326,np.radians(120.0),np.radians(0)),
            ('NH2',('CD','NE','CZ'),1.326,np.radians(120.0),np.radians(180))],
    'MET': [('CG',('N','CA','CB'),1.520,np.radians(113.7),'chi1'),
            ('SD',('CA','CB','CG'),1.808,np.radians(112.7),'chi2'),
            ('CE',('CB','CG','SD'),1.792,np.radians(100.9),'chi3')],
    'PHE': [('CG',('N','CA','CB'),1.512,np.radians(113.8),'chi1'),
            ('CD1',('CA','CB','CG'),1.390,np.radians(120.8),'chi2'),
            ('CD2',('CA','CB','CG'),1.390,np.radians(120.8),'chi2_plus180'),
            ('CE1',('CB','CG','CD1'),1.390,np.radians(120.0),np.radians(180)),
            ('CE2',('CB','CG','CD2'),1.390,np.radians(120.0),np.radians(180)),
            ('CZ',('CG','CD1','CE1'),1.390,np.radians(120.0),np.radians(0))],
    'TYR': [('CG',('N','CA','CB'),1.512,np.radians(113.8),'chi1'),
            ('CD1',('CA','CB','CG'),1.390,np.radians(120.8),'chi2'),
            ('CD2',('CA','CB','CG'),1.390,np.radians(120.8),'chi2_plus180'),
            ('CE1',('CB','CG','CD1'),1.390,np.radians(120.0),np.radians(180)),
            ('CE2',('CB','CG','CD2'),1.390,np.radians(120.0),np.radians(180)),
            ('CZ',('CG','CD1','CE1'),1.390,np.radians(120.0),np.radians(0)),
            ('OH',('CD1','CE1','CZ'),1.376,np.radians(119.9),np.radians(180))],
    'TRP': [('CG',('N','CA','CB'),1.498,np.radians(113.7),'chi1'),
            ('CD1',('CA','CB','CG'),1.365,np.radians(126.9),'chi2'),
            ('CD2',('CA','CB','CG'),1.433,np.radians(126.9),'chi2_plus180'),
            ('NE1',('CB','CG','CD1'),1.374,np.radians(110.0),np.radians(180)),
            ('CE2',('CB','CG','CD2'),1.412,np.radians(107.0),np.radians(180)),
            ('CE3',('CG','CD2','CE2'),1.400,np.radians(120.0),np.radians(180)),
            ('CZ2',('CD2','CE2','NE1'),1.394,np.radians(120.0),np.radians(180)),
            ('CZ3',('CD2','CE3','CZ2'),1.380,np.radians(120.0),np.radians(0)),
            ('CH2',('CE2','CZ2','CZ3'),1.368,np.radians(120.0),np.radians(0))],
    'HIS': [('CG',('N','CA','CB'),1.497,np.radians(113.8),'chi1'),
            ('ND1',('CA','CB','CG'),1.378,np.radians(122.6),'chi2'),
            ('CD2',('CA','CB','CG'),1.354,np.radians(130.6),'chi2_plus180'),
            ('CE1',('CB','CG','ND1'),1.321,np.radians(108.5),np.radians(180)),
            ('NE2',('CG','ND1','CE1'),1.321,np.radians(108.5),np.radians(0))],
    'PRO': [('CG',('N','CA','CB'),1.492,np.radians(104.5),'chi1'),
            ('CD',('CA','CB','CG'),1.503,np.radians(105.5),'chi2')],
}


def _place_h_from_n(N_pos, CA_pos, bond_length=1.01):
    """Place H on N using N-CA bond direction (for residue 0 / no preceding C)."""
    n_to_ca = CA_pos - N_pos
    n_to_ca /= np.linalg.norm(n_to_ca)
    # H sits opposite CA, in the N-CA-H plane, ~120° from CA
    # Use a fixed perpendicular to avoid dependence on missing prev-C
    perp = np.array([0., 0., 1.])
    if abs(np.dot(n_to_ca, perp)) > 0.9:
        perp = np.array([1., 0., 0.])
    perp = np.cross(n_to_ca, perp)
    perp /= np.linalg.norm(perp)
    # 120° tetrahedral: -cos(120°)=0.5 along -n_to_ca, sin(120°)≈0.866 along perp
    h_dir = -0.5 * n_to_ca + 0.866 * perp
    h_dir /= np.linalg.norm(h_dir)
    return N_pos + bond_length * h_dir

def _place_h_from_nerf(prev_C, N_pos, CA_pos, bond_length=1.01):
    """Place amide H using prev_C → N → CA chain (omega ≈ 180°)."""
    b1 = N_pos - prev_C
    b2 = CA_pos - N_pos
    b1 /= np.linalg.norm(b1)
    b2 /= np.linalg.norm(b2)
    # H is trans to CA across the N (omega=180 means H is on same side as prev_C)
    n = np.cross(b1, b2)
    if np.linalg.norm(n) < 1e-6:
        return _place_h_from_n(N_pos, CA_pos, bond_length)
    n /= np.linalg.norm(n)
    # Bond angle N-H ≈ 120°
    h_dir = -0.5 * b2 + 0.866 * np.cross(n, b2)  # not normalized yet
    h_dir /= np.linalg.norm(h_dir)
    return N_pos + bond_length * h_dir

class NeRFBuilder:
    L_CN   = BondGeometry.C_N
    L_NCA  = BondGeometry.N_CA
    L_CAC  = BondGeometry.CA_C
    L_CO   = BondGeometry.C_O
    L_NH   = BondGeometry.N_H
    L_CACB = BondGeometry.CA_CB
    OMEGA  = np.pi
    CB_DIHEDRAL   = np.radians(-122.66)
    N_CA_CB_ANGLE = np.radians(BondGeometry.N_CA_CB)

    def __init__(self, pdb_file=None, sequence=None):
        self.pdb_file = pdb_file
        if pdb_file is not None:
            self.data = PDBParser.parse_backbone(pdb_file)
            self.n_res = len(self.data['CA'])
            self.sequence = sequence or PDBParser.extract_sequence(pdb_file)
        elif sequence is not None:
            self.sequence = sequence
            self.n_res = len(sequence)
            self.data = None
        else:
            raise ValueError("Must provide pdb_file or sequence")

    @staticmethod
    def nerf(A, B, C, length, angle_BCD, dihedral_ABCD):
        bc = C - B; bc_len = np.linalg.norm(bc)
        if bc_len < 1e-10: return C + np.array([length, 0.0, 0.0])
        bc = bc / bc_len
        ab = B - A; ab_len = np.linalg.norm(ab)
        if ab_len < 1e-10:
            perp = np.array([1,0,0]) if abs(bc[0]) < 0.9 else np.array([0,1,0])
            n = np.cross(bc, perp); n = n / np.linalg.norm(n)
        else:
            ab = ab / ab_len; n = np.cross(bc, ab); n_len = np.linalg.norm(n)
            if n_len < 1e-10:
                perp = np.array([1,0,0]) if abs(bc[0]) < 0.9 else np.array([0,1,0])
                n = np.cross(bc, perp); n = n / np.linalg.norm(n)
            else: n = n / n_len
        m = np.cross(bc, n)
        d = np.array([-np.cos(angle_BCD),
                      np.sin(angle_BCD)*np.cos(-dihedral_ABCD),
                      np.sin(angle_BCD)*np.sin(-dihedral_ABCD)])
        return C + length * (d[0]*bc + d[1]*m + d[2]*n)

    def _build_raw(self, phi, psi, chi_list=None, angles=None, origin=None):
        """Internal raw build — no clash check. Called by build() and _find_clash_free_chis()."""
        N = self.n_res
        coords = {'N':np.zeros((N,3)),'CA':np.zeros((N,3)),'C':np.zeros((N,3)),
                  'O':np.zeros((N,3)),'H':np.zeros((N,3)),'CB':np.zeros((N,3))}
        if angles is None:
            angles = [DEFAULT_ANGLES.get(self.sequence[i], DEFAULT_ANGLES['ALA']) for i in range(N)]
        if chi_list is None:
            chi_list = [DEFAULT_CHI.get(self.sequence[i], []) for i in range(N)]
        if origin is None: origin = np.array([0.0,0.0,0.0])
        coords['N'][0] = origin
        coords['CA'][0] = origin + np.array([self.L_NCA,0.0,0.0])
        coords['C'][0] = self.nerf(origin-np.array([1.0,0.0,0.0]),
            coords['N'][0], coords['CA'][0], self.L_CAC, angles[0][2], phi[0])
        coords['O'][0] = self._place_carbonyl_O(coords['N'][0],coords['CA'][0],coords['C'][0])
        # Residue 0 has no preceding peptide to define H direction.
        # Set H[0] = NaN (Option 1) — hbond_finder and v5 features skip i=0.
        coords['H'][0] = np.full(3, np.nan)
        for i in range(N-1):
            t_cacn, t_cnca, t_ncac = angles[i]
            coords['N'][i+1]  = self.nerf(coords['N'][i], coords['CA'][i], coords['C'][i],
                                           self.L_CN, t_cacn, psi[i])
            coords['CA'][i+1] = self.nerf(coords['CA'][i], coords['C'][i], coords['N'][i+1],
                                           self.L_NCA, t_cnca, self.OMEGA)
            t_ncac_next = angles[i+1][2] if i+1 < len(angles) else angles[i][2]
            coords['C'][i+1]  = self.nerf(coords['C'][i], coords['N'][i+1], coords['CA'][i+1],
                                           self.L_CAC, t_ncac_next, phi[i+1])
            coords['O'][i+1]  = self._place_carbonyl_O(coords['N'][i+1], coords['CA'][i+1],
                                                        coords['C'][i+1])
            # FIXED: uses backbone geometry (prev_C → N → CA), matches DSSP convention
            # coords['H'][i+1]  = _place_h_from_nerf(
            #     coords['C'][i], coords['N'][i+1], coords['CA'][i+1]
            # )
            coords['H'][i+1] = self._place_amide_H(
                coords['C'][i], coords['O'][i], coords['N'][i+1]
            )
            if self.sequence[i+1] != 'GLY':
                coords['CB'][i+1] = self._place_CB(coords['C'][i+1], coords['N'][i+1],
                                                    coords['CA'][i+1], coords['C'][i])
        self._build_all_sidechains(coords, chi_list)
        return coords

    def build(self, phi, psi, chi_list=None, angles=None, origin=None):
        _chi_was_provided = chi_list is not None
        coords = self._build_raw(phi, psi, chi_list, angles, origin)
        if _chi_was_provided:
            _, coords = self._find_clash_free_chis(chi_list, phi, psi, coords)
        return coords

    def extract_angles(self):
        if self.data is None: raise ValueError("No PDB data loaded")
        d = self.data; N = self.n_res
        phi = np.zeros(N); psi = np.zeros(N); omega = np.zeros(N-1); bond_list = []
        theta_ncac_0 = bond_angle(d['N'][0],d['CA'][0],d['C'][0])
        res0 = self.sequence[0] if self.sequence else 'ALA'
        default_0 = DEFAULT_ANGLES.get(res0, DEFAULT_ANGLES['ALA'])
        angle_list = [(default_0[0],default_0[1],theta_ncac_0)]
        for i in range(N-1):
            psi[i] = dihedral_angle(d['N'][i],d['CA'][i],d['C'][i],d['N'][i+1])
            omega[i] = dihedral_angle(d['CA'][i],d['C'][i],d['N'][i+1],d['CA'][i+1])
            phi[i+1] = dihedral_angle(d['C'][i],d['N'][i+1],d['CA'][i+1],d['C'][i+1])
            angle_list.append((bond_angle(d['CA'][i],d['C'][i],d['N'][i+1]),
                               bond_angle(d['C'][i],d['N'][i+1],d['CA'][i+1]),
                               bond_angle(d['N'][i+1],d['CA'][i+1],d['C'][i+1])))
            bond_list.append((np.linalg.norm(d['N'][i+1]-d['C'][i]),
                              np.linalg.norm(d['CA'][i+1]-d['N'][i+1]),
                              np.linalg.norm(d['C'][i+1]-d['CA'][i+1])))
        chi_list = self._extract_all_chi()
        return phi, psi, omega, chi_list, angle_list, bond_list

    def extract_and_verify(self):
        phi,psi,omega,chi_list,angles,bonds = self.extract_angles()
        N = self.n_res
        print(f"Structure: {N} residues")
        l_cn=[b[0] for b in bonds]; l_nca=[b[1] for b in bonds]; l_cac=[b[2] for b in bonds]
        print(f"  C-N:  fixed={self.L_CN:.3f}  PDB={np.mean(l_cn):.3f}±{np.std(l_cn):.3f}")
        print(f"  N-CA: fixed={self.L_NCA:.3f}  PDB={np.mean(l_nca):.3f}±{np.std(l_nca):.3f}")
        print(f"  CA-C: fixed={self.L_CAC:.3f}  PDB={np.mean(l_cac):.3f}±{np.std(l_cac):.3f}")
        recon = self.build(phi,psi,chi_list,angles)
        print(f"  CA RMSD (PDB angles): {rmsd(recon['CA'],self.data['CA']):.4f} Å")
        recon2 = self.build(phi,psi,chi_list)
        print(f"  CA RMSD (default angles): {rmsd(recon2['CA'],self.data['CA']):.4f} Å")
        return phi,psi,chi_list

    # ── Atom typing ──────────────────────────────────────────────────────────

    def get_typed_structure(self, coords):
        """All atoms including hydrogens. NaN-safe."""
        typed = []
        for i in range(self.n_res):
            res_name = self.sequence[i]
            for atom_name, atom_type in ResidueAtomMap.get_all_types(res_name):
                if atom_name not in coords: continue
                c = coords[atom_name]
                if c.ndim != 2 or i >= len(c): continue
                pos = c[i]
                if np.any(np.isnan(pos)): continue
                if np.linalg.norm(pos) < 0.01: continue
                p = AtomTypes.get(atom_type)
                typed.append({
                    'name': atom_name, 'type': atom_type,
                    'res_idx': i, 'res_name': res_name,
                    'pos': pos.copy(),
                    'sigma': p.sigma, 'epsilon': p.epsilon,
                    'charge': p.charge, 'vdw_radius': p.vdw_radius,
                    'element': p.element,
                    'is_donor': p.is_donor, 'is_acceptor': p.is_acceptor,
                })
        return typed

    def get_typed_structure_heavy(self, coords):
        """Heavy atoms only — ~4× faster for pairwise energy."""
        typed = []
        for i in range(self.n_res):
            res_name = self.sequence[i]
            for atom_name, atom_type in ResidueAtomMap.get_all_types(res_name):
                p = AtomTypes.get(atom_type)
                if p.element == 'H': continue
                if atom_name not in coords: continue
                c = coords[atom_name]
                if c.ndim != 2 or i >= len(c): continue
                pos = c[i]
                if np.any(np.isnan(pos)): continue
                if np.linalg.norm(pos) < 0.01: continue
                typed.append({
                    'name': atom_name, 'type': atom_type,
                    'res_idx': i, 'res_name': res_name,
                    'pos': pos.copy(),
                    'sigma': p.sigma, 'epsilon': p.epsilon,
                    'charge': p.charge, 'vdw_radius': p.vdw_radius,
                    'element': p.element,
                    'is_donor': p.is_donor, 'is_acceptor': p.is_acceptor,
                })
        return typed

    def parameter_count(self):
        N = self.n_res
        n_phi_psi = 2*N; n_chi = sum(len(DEFAULT_CHI.get(r,[])) for r in self.sequence)
        n_search = n_phi_psi + n_chi; n_fixed = 7*N
        n_atoms = sum(ResidueAtomMap.count_heavy_atoms(r) for r in self.sequence)
        print(f"Parameters for {N}-residue protein:")
        print(f"  SEARCH: φ,ψ={n_phi_psi} + χ={n_chi} = {n_search}")
        print(f"  FIXED:  bonds+angles+ω = {n_fixed}")
        print(f"  ATOMS:  {n_atoms} heavy")
        return {'n_search':n_search,'n_chi':n_chi,'n_fixed':n_fixed,'n_atoms':n_atoms}

    # ── Private helpers ──────────────────────────────────────────────────────

    def _place_carbonyl_O(self, N, CA, C):
        return self.nerf(N,CA,C,self.L_CO,np.radians(BondGeometry.CA_C_O),np.pi)

    def _place_amide_H(self, C_prev, O_prev, N):
        """Place amide H using Kabsch & Sander (1983) convention:
            H = N + L_NH * (C_prev - O_prev) / |C_prev - O_prev|
        H lies in the peptide plane, antiparallel to the preceding C=O
        bond. Matches mkdssp and gives C_prev-N-H angle of ~120°.

        v2.5 fix: previous versions used (N - C_prev)/|...| which placed H
        collinear with C_prev-N (angle 180°) and was off by ~1.8 Å. This
        was the cause of large errors in hbond_finder.py vs mkdssp; see
        diagnose_hbond.py for the diagnostic that caught it.
        """
        v = C_prev - O_prev
        vn = np.linalg.norm(v)
        if vn < 1e-10:
            return np.full(3, np.nan)
        return N + self.L_NH * v / vn

    def _place_CB(self, C, N, CA, C_prev=None):
        if C_prev is not None:
            return self.nerf(C_prev,N,CA,self.L_CACB,self.N_CA_CB_ANGLE,self.CB_DIHEDRAL)
        return self.nerf(C,N,CA,self.L_CACB,self.N_CA_CB_ANGLE,-self.CB_DIHEDRAL)

    def _build_all_sidechains(self, coords, chi_list):
        for i in range(self.n_res):
            res = self.sequence[i]
            if res not in SIDECHAIN_BUILD or res in ('GLY','ALA'): continue
            chis = chi_list[i] if i<len(chi_list) else DEFAULT_CHI.get(res,[])
            for atom_name,parents,blen,bang,dih_src in SIDECHAIN_BUILD[res]:
                pc = []
                for pn in parents:
                    if pn not in coords: break
                    pc.append(coords[pn][i])
                if len(pc)!=3 or any(np.linalg.norm(p)<0.01 for p in pc): continue
                dih = self._resolve_dihedral(dih_src,chis)
                if dih is None: continue
                pos = self.nerf(pc[0],pc[1],pc[2],blen,bang,dih)
                if atom_name not in coords:
                    coords[atom_name] = np.zeros((self.n_res,3))
                coords[atom_name][i] = pos

    def _rebuild_single_sidechain(self, coords, res_idx, chis):
        """Rebuild only one residue's sidechain atoms in-place."""
        res = self.sequence[res_idx]
        if res not in SIDECHAIN_BUILD or res in ('GLY', 'ALA'):
            return
        for atom_name, parents, blen, bang, dih_src in SIDECHAIN_BUILD[res]:
            pc = []
            for pn in parents:
                if pn not in coords:
                    return
                pc.append(coords[pn][res_idx])
            if len(pc) != 3:
                continue
            dih = self._resolve_dihedral(dih_src, chis)
            if dih is None:
                continue
            pos = self.nerf(pc[0], pc[1], pc[2], blen, bang, dih)
            if atom_name not in coords:
                coords[atom_name] = np.zeros((self.n_res, 3))
            coords[atom_name][res_idx] = pos

    def _resolve_dihedral(self, src, chis):
        if isinstance(src,(int,float)): return float(src)
        chi_map = {'chi1':0,'chi2':1,'chi3':2,'chi4':3}
        if src in chi_map:
            idx=chi_map[src]; return chis[idx] if idx<len(chis) else None
        if '_plus' in src:
            base,off=src.rsplit('_plus',1); idx=chi_map.get(base)
            if idx is None or idx>=len(chis): return None
            return chis[idx]+np.radians(float(off))
        return None

    def _extract_all_chi(self):
        if self.data is None or self.pdb_file is None:
            return [[] for _ in range(self.n_res)]
        try: all_atoms = PDBParser.parse_all_atoms(self.pdb_file)
        except: return [[] for _ in range(self.n_res)]
        def get_atom(name,ri):
            if name in all_atoms['atoms']:
                for idx,c in all_atoms['atoms'][name]:
                    if idx==ri: return c
            return None
        d = self.data; chi_list = []
        chi1_map = {'SER':'OG','THR':'OG1','CYS':'SG','VAL':'CG1','LEU':'CG',
            'ILE':'CG1','MET':'CG','PHE':'CG','TYR':'CG','TRP':'CG',
            'HIS':'CG','ASP':'CG','ASN':'CG','GLU':'CG','GLN':'CG',
            'LYS':'CG','ARG':'CG','PRO':'CG'}
        chi2_chains = {
            'LEU':[('CA','CB','CG','CD1')],'ILE':[('CA','CB','CG1','CD1')],
            'MET':[('CA','CB','CG','SD')],'PHE':[('CA','CB','CG','CD1')],
            'TYR':[('CA','CB','CG','CD1')],'TRP':[('CA','CB','CG','CD1')],
            'HIS':[('CA','CB','CG','ND1')],'ASP':[('CA','CB','CG','OD1')],
            'ASN':[('CA','CB','CG','OD1')],
            'GLU':[('CA','CB','CG','CD'),('CB','CG','CD','OE1')],
            'GLN':[('CA','CB','CG','CD'),('CB','CG','CD','OE1')],
            'LYS':[('CA','CB','CG','CD'),('CB','CG','CD','CE'),('CG','CD','CE','NZ')],
            'ARG':[('CA','CB','CG','CD'),('CB','CG','CD','NE'),('CG','CD','NE','CZ')],
            'PRO':[('CA','CB','CG','CD')]}
        for i in range(self.n_res):
            res=self.sequence[i]; chis=[]
            cb = d['CB'][i]
            if np.any(np.isnan(cb)) or np.linalg.norm(cb)<0.1:
                chi_list.append([]); continue
            if res in chi1_map:
                atom_d=get_atom(chi1_map[res],i)
                if atom_d is not None:
                    chis.append(dihedral_angle(d['N'][i],d['CA'][i],cb,atom_d))
            if res in chi2_chains:
                fixed={'CA':d['CA'][i],'CB':cb}
                for chain in chi2_chains[res]:
                    atoms=[]
                    for nm in chain:
                        if nm in fixed: atoms.append(fixed[nm])
                        else:
                            a=get_atom(nm,i)
                            if a is None: break
                            atoms.append(a)
                    if len(atoms)==4: chis.append(dihedral_angle(*atoms))
            chi_list.append(chis)
        return chi_list

    # ── Clash-avoidance ───────────────────────────────────────────────────────
    _CLASH_MIN_DIST = 2.2    # Å — minimum sidechain→backbone distance
    #_CLASH_SEP_MIN  = 2      # skip pairs closer than this in sequence
    _CLASH_SEP_MIN  = 1      # skip same-residue pairs only

    _SC_BEYOND_CB = {
        'ILE': ['CG1','CG2','CD1'],
        'LEU': ['CG','CD1','CD2'],
        'VAL': ['CG1','CG2'],
        'PHE': ['CG','CD1','CD2','CE1','CE2','CZ'],
        'TYR': ['CG','CD1','CD2','CE1','CE2','CZ','OH'],
        'TRP': ['CG','CD1','CD2','NE1','CE2','CE3','CZ2','CZ3','CH2'],
        'MET': ['CG','SD','CE'],
        'CYS': ['SG'],
        'SER': ['OG'],
        'THR': ['OG1','CG2'],
        'ASN': ['CG','OD1','ND2'],
        'ASP': ['CG','OD1','OD2'],
        'GLN': ['CG','CD','OE1','NE2'],
        'GLU': ['CG','CD','OE1','OE2'],
        'LYS': ['CG','CD','CE','NZ'],
        'ARG': ['CG','CD','NE','CZ','NH1','NH2'],
        'HIS': ['CG','ND1','CD2','CE1','NE2'],
        'PRO': ['CG','CD'],
    }

    # ── Rotamer library (Lovell 2000 backbone-independent modes) ─────────────
    # Format: list of (chi1_rad, chi2_rad_or_None).
    # chi2=None → leave chi2+ from chi_list unchanged (single-chi residues).
    # chi3+ are always kept from chi_list unchanged.
    #
    # v2.4: replaces 10°/15° degree-step scan (~12–115× faster per residue).
    _r = np.radians
    _ROTAMERS = {
        # ── Single-chi (3 trials) ─────────────────────────────────────────
        'SER': [(_r(-60), None), (_r(60),  None), (_r(180), None)],
        'CYS': [(_r(-60), None), (_r(60),  None), (_r(180), None)],
        'THR': [(_r(-60), None), (_r(60),  None), (_r(180), None)],
        'VAL': [(_r(-60), None), (_r(180), None), (_r(60),  None)],
        # ── Two-chi, branched (5 trials) ──────────────────────────────────
        'ILE': [(_r(-60), _r(170)), (_r(-60), _r(-60)),
                (_r(60),  _r(170)), (_r(180), _r(170)), (_r(180), _r(-60))],
        'LEU': [(_r(-60), _r(180)), (_r(-60), _r(60)),
                (_r(180), _r(60)),  (_r(60),  _r(180)), (_r(180), _r(180))],
        # ── Two-chi, polar/aromatic (5 trials) ────────────────────────────
        'ASN': [(_r(-60), _r(0)),   (_r(-60), _r(180)),
                (_r(180), _r(0)),   (_r(180), _r(180)), (_r(60),  _r(0))],
        'ASP': [(_r(-60), _r(0)),   (_r(-60), _r(180)),
                (_r(180), _r(0)),   (_r(180), _r(180)), (_r(60),  _r(0))],
        'HIS': [(_r(-60), _r(-75)), (_r(-60), _r(105)),
                (_r(180), _r(75)),  (_r(60),  _r(75)),  (_r(180), _r(-75))],
        'PHE': [(_r(-60), _r(90)),  (_r(-60), _r(-90)),
                (_r(180), _r(90)),  (_r(60),  _r(90)),  (_r(180), _r(-90))],
        'TYR': [(_r(-60), _r(90)),  (_r(-60), _r(-90)),
                (_r(180), _r(90)),  (_r(60),  _r(90)),  (_r(180), _r(-90))],
        'TRP': [(_r(-60), _r(-90)), (_r(-60), _r(90)),
                (_r(180), _r(-90)), (_r(60),  _r(90)),  (_r(180), _r(90))],
        'MET': [(_r(-60), _r(180)), (_r(-60), _r(60)),
                (_r(180), _r(180)), (_r(60),  _r(180)), (_r(180), _r(60))],
        # ── Long chains — chi3+ kept from chi_list (5 trials) ─────────────
        'GLN': [(_r(-60), _r(180)), (_r(-60), _r(60)),
                (_r(180), _r(180)), (_r(60),  _r(180)), (_r(180), _r(60))],
        'GLU': [(_r(-60), _r(180)), (_r(-60), _r(60)),
                (_r(180), _r(180)), (_r(60),  _r(180)), (_r(180), _r(60))],
        'LYS': [(_r(-60), _r(180)), (_r(180), _r(180)),
                (_r(60),  _r(180)), (_r(-60), _r(60)),  (_r(180), _r(60))],
        'ARG': [(_r(-60), _r(180)), (_r(180), _r(180)),
                (_r(60),  _r(180)), (_r(-60), _r(60)),  (_r(180), _r(60))],
        # ── Proline ring puckering (2 trials) ─────────────────────────────
        'PRO': [(_r(30), _r(-40)), (_r(-40), _r(30))],
    }
    del _r

    def _min_sc_bb_dist(self, coords, res_idx):
        BACKBONE = {'N', 'CA', 'C', 'O'}
        res_name = self.sequence[res_idx]
        sc_atoms = self._SC_BEYOND_CB.get(res_name, [])
        min_d = 999.0
    
        ca_i = coords['CA'][res_idx]   # ← grab once outside both loops
    
        for other in range(self.n_res):
            if abs(other - res_idx) < self._CLASH_SEP_MIN:
                continue
            # CA–CA pre-filter: skip residues that can't possibly clash
            ca_j = coords['CA'][other]
            if np.linalg.norm(ca_i - ca_j) > 14.0:
                continue
            for sc in sc_atoms:
                if sc not in coords:
                    continue
                p = coords[sc][res_idx]
                if not np.all(np.isfinite(p)) or np.linalg.norm(p) < 0.01:
                    continue
                for bb in BACKBONE:
                    if bb not in coords:
                        continue
                    q = coords[bb][other]
                    if not np.all(np.isfinite(q)) or np.linalg.norm(q) < 0.01:
                        continue
                    d = np.linalg.norm(p - q)
                    if d < min_d:
                        min_d = d
        return min_d

    def _find_clash_free_chis(self, chi_list, phi, psi, coords_in):
        """
        Clash-free rotamer search using a precomputed rotamer library.

        v2.4: Replaces the 10°/15° degree-step grid scan with Lovell 2000
        backbone-independent rotamer modes. For each clashing residue:

          1. Look up _ROTAMERS[res_name] → 3–9 (chi1, chi2) pairs.
          2. Try each pair; chi3+ are kept unchanged from chi_list.
          3. Accept the FIRST clash-free hit (min_d >= _CLASH_MIN_DIST).
          4. If no hit is clash-free, keep BEST (largest min_d) to
             minimise energy distortion rather than leaving the clash.

        Trial counts vs v2.3 degree-step scan:
          SER/CYS/THR :   36 →  3   (~12×  faster)
          ILE/LEU/TRP :  576 →  5   (~115× faster)
          PHE/TYR/HIS :  576 →  5   (~115× faster)
          LYS/ARG/GLN :  576 →  5   (~115× faster)
        """
        chi_out = [list(c) for c in chi_list]

        for res_idx in range(self.n_res):
            if not chi_out[res_idx]:
                continue

            # Fast-exit: already clash-free
            min_d = self._min_sc_bb_dist(coords_in, res_idx)
            if min_d >= self._CLASH_MIN_DIST:
                continue

            res_name = self.sequence[res_idx]
            rotamers = self._ROTAMERS.get(res_name)
            if rotamers is None:
                continue  # GLY/ALA have no chi; unknown residues ignored

            best_d   = min_d
            best_chi = None

            for c1, c2 in rotamers:
                trial_chi = list(chi_out[res_idx])
                trial_chi[0] = c1
                if c2 is not None and len(trial_chi) >= 2:
                    trial_chi[1] = c2
                # chi3+ left unchanged from chi_out[res_idx]

                coords_t = {k: v.copy() for k, v in coords_in.items()}
                self._rebuild_single_sidechain(coords_t, res_idx, trial_chi)
                d = self._min_sc_bb_dist(coords_t, res_idx)

                if d > best_d:
                    best_d   = d
                    best_chi = trial_chi

                if d >= self._CLASH_MIN_DIST:
                    break  # first clash-free rotamer wins

            if best_chi is not None:
                chi_out[res_idx] = best_chi
                self._rebuild_single_sidechain(coords_in, res_idx, best_chi)

        return chi_out, coords_in

if __name__ == '__main__':
    print("nerf_builder.py v2.4 — Self Test")
    seq = ['ALA','GLY','TYR','ASP','LEU','LYS','SER','ARG']
    builder = NeRFBuilder(sequence=seq); N = len(seq)
    phi = np.full(N,np.radians(-63)); psi = np.full(N,np.radians(-43))
    coords = builder.build(phi,psi)
    typed_all = builder.get_typed_structure(coords)
    typed_heavy = builder.get_typed_structure_heavy(coords)
    print(f"  All atoms:   {len(typed_all)}")
    print(f"  Heavy atoms: {len(typed_heavy)}")
    for i in range(N-1):
        assert abs(np.linalg.norm(coords['N'][i+1]-coords['C'][i])-1.329)<0.01
    print(f"✓ nerf_builder.py v2.4 OK")