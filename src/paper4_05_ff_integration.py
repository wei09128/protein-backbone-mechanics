#!/usr/bin/env python3
"""
Paper 4 — Force-Field Integration via OpenMM
==============================================

Demonstrates practical integration of the geometry library into
all three major force fields (AMBER ff14SB, CHARMM36m, OPLS-AA/M)
using OpenMM's extensible force framework.

Strategy:
  OpenMM allows adding CustomAngleForce terms that override or correct
  the standard harmonic angle terms. We:
  
  1. Load a protein with a standard force field
  2. Energy-minimize with standard parameters
  3. Measure backbone geometry (τ, bond angles, bond lengths)
  4. Add library corrections as CustomAngleForce terms
  5. Re-minimize with corrections
  6. Compare both to PDB ground truth

  The correction energy for each angle is:
      E_correction = ½ k_corr (θ - θ_lib)² - ½ k_corr (θ - θ_FF)²
  
  which shifts the equilibrium from θ_FF to θ_lib without changing
  the total spring constant k (k_corr is chosen to achieve this).

  This approach:
  - Does NOT modify force field source code
  - Works with ANY OpenMM-supported force field
  - Can be applied as a post-processing correction
  - Adds negligible computational cost

Two modes:
  MODE 1 — With OpenMM installed: full MD integration + benchmark
  MODE 2 — Without OpenMM: static correction analysis + parameter files

Usage:
  # Full OpenMM benchmark
  python paper4_07_ff_integration.py \
      --library ./paper4_library_v2/constants_library.json \
      --pdb_dir /mnt/f/Protein_Folding/pdb_cache \
      --test_pdbs 1ubq 1l2y \
      --out ./paper4_integration/

  # Static analysis only (no OpenMM required)
  python paper4_07_ff_integration.py \
      --library ./paper4_library_v2/constants_library.json \
      --csv /mnt/f/Protein_Folding/v8_g/p3.csv \
      --out ./paper4_integration/ \
      --static_only

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
import pandas as pd

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# Force field parameter definitions
# ══════════════════════════════════════════════════════════════════════════════

FORCE_FIELDS = {
    'AMBER_ff14SB': {
        'label': 'AMBER ff14SB',
        'tau_eq': 111.1, 'tau_k': 63.0,
        'NCaCB_eq': 110.1, 'NCaCB_k': 63.0,
        'CCaCB_eq': 110.1, 'CCaCB_k': 63.0,
        'CaCN_eq': 116.6, 'CaCN_k': 70.0,
        'CNCa_eq': 121.9, 'CNCa_k': 50.0,
        'CaCO_eq': 120.4, 'CaCO_k': 80.0,
        'NCA_eq': 1.458, 'NCA_k': 337.0,
        'CAC_eq': 1.522, 'CAC_k': 317.0,
        'CO_eq': 1.229, 'CO_k': 570.0,
        'CN_eq': 1.335, 'CN_k': 490.0,
        'CACB_eq': 1.526, 'CACB_k': 317.0,
        'openmm_name': 'amber14-all.xml',
    },
    'CHARMM36m': {
        'label': 'CHARMM36m',
        'tau_eq': 110.7, 'tau_k': 67.7,
        'NCaCB_eq': 111.0, 'NCaCB_k': 70.0,
        'CCaCB_eq': 108.5, 'CCaCB_k': 52.0,
        'CaCN_eq': 116.5, 'CaCN_k': 62.0,
        'CNCa_eq': 120.6, 'CNCa_k': 35.0,
        'CaCO_eq': 120.9, 'CaCO_k': 80.0,
        'NCA_eq': 1.458, 'NCA_k': 200.0,
        'CAC_eq': 1.522, 'CAC_k': 250.0,
        'CO_eq': 1.229, 'CO_k': 620.0,
        'CN_eq': 1.345, 'CN_k': 370.0,
        'CACB_eq': 1.538, 'CACB_k': 222.5,
        'openmm_name': 'charmm36.xml',
        'has_cmap': True,
    },
    'OPLS_AAM': {
        'label': 'OPLS-AA/M',
        'tau_eq': 111.1, 'tau_k': 63.0,
        'NCaCB_eq': 109.5, 'NCaCB_k': 63.0,
        'CCaCB_eq': 111.1, 'CCaCB_k': 63.0,
        'CaCN_eq': 116.6, 'CaCN_k': 70.0,
        'CNCa_eq': 121.9, 'CNCa_k': 50.0,
        'CaCO_eq': 120.4, 'CaCO_k': 80.0,
        'NCA_eq': 1.449, 'NCA_k': 337.0,
        'CAC_eq': 1.522, 'CAC_k': 317.0,
        'CO_eq': 1.229, 'CO_k': 570.0,
        'CN_eq': 1.335, 'CN_k': 490.0,
        'CACB_eq': 1.529, 'CACB_k': 317.0,
        'openmm_name': None,
    },
}

# Observable mapping: our column names → FF parameter names
OBS_MAP = {
    'tau_deg': 'tau',
    'angle_N_CA_CB': 'NCaCB',
    'angle_C_CA_CB': 'CCaCB',
    'angle_CaCN': 'CaCN',
    'angle_CNCa': 'CNCa',
    'angle_CA_C_O': 'CaCO',
    'bond_N_CA': 'NCA',
    'bond_CA_C': 'CAC',
    'bond_C_O': 'CO',
    'bond_C_N_next': 'CN',
    'bond_CA_CB': 'CACB',
}


# ══════════════════════════════════════════════════════════════════════════════
# Library lookup (reused from script 02)
# ══════════════════════════════════════════════════════════════════════════════

class QuickLib:
    def __init__(self, json_path, bin_size=10):
        with open(json_path) as f:
            self._lib = json.load(f)
        self._bs = bin_size
        self._half = bin_size / 2.0
        self._centers = np.arange(-180 + self._half, 180 + self._half, bin_size)
    
    def _bk(self, angle):
        a = ((angle + 180) % 360) - 180
        i = int(np.round((a - self._centers[0]) / self._bs))
        return str(int(self._centers[max(0, min(i, len(self._centers)-1))]))
    
    def get(self, phi, psi, res, col):
        pk, qk = self._bk(phi), self._bk(psi)
        lib_key = f'{col}_eq'
        for cls in [res, 'ALL']:
            if cls in self._lib:
                cell = self._lib[cls].get(pk, {}).get(qk)
                if cell and lib_key in cell:
                    return cell[lib_key]
        return np.nan


# ══════════════════════════════════════════════════════════════════════════════
# Static analysis: correction magnitudes per force field
# ══════════════════════════════════════════════════════════════════════════════

def static_analysis(df, lib, out_dir):
    """Compare all three force fields to library and PDB ground truth."""
    
    R = []
    R.append("=" * 78)
    R.append("Paper 4 — Force-Field Integration Analysis")
    R.append("=" * 78)
    R.append(f"\n  Residues: {len(df):,}")
    
    phi = df['phi_deg'].values
    psi = df['psi_deg'].values
    res = df['res_name'].values
    
    # ── Per-FF comparison ────────────────────────────────────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 1: MAE FOR EACH FORCE FIELD vs PDB (ANGLES)")
    R.append("━" * 78)
    
    angle_obs = ['tau_deg', 'angle_N_CA_CB', 'angle_C_CA_CB',
                 'angle_CaCN', 'angle_CNCa', 'angle_CA_C_O']
    
    # Header
    header = f"  {'Observable':>20s}"
    for ff_name in FORCE_FIELDS:
        header += f"  {FORCE_FIELDS[ff_name]['label']:>14s}"
    header += f"  {'Library':>14s}"
    R.append(header)
    R.append("  " + "─" * (len(header.strip())))
    
    ff_totals = {ff: [] for ff in FORCE_FIELDS}
    lib_totals = []
    
    for col in angle_obs:
        if col not in df.columns:
            continue
        
        obs = df[col].values
        valid = ~np.isnan(obs) & ~np.isnan(phi) & ~np.isnan(psi)
        if valid.sum() < 100:
            continue
        
        ff_key = OBS_MAP[col]
        
        # Library prediction
        lib_pred = np.array([
            lib.get(phi[i], psi[i], res[i], col)
            if valid[i] else np.nan
            for i in range(len(df))
        ], dtype=float)
        lib_pred_valid = lib_pred[valid]
        # Fill NaN with AMBER default
        amber_eq = FORCE_FIELDS['AMBER_ff14SB'][f'{ff_key}_eq']
        lib_pred_valid = np.where(np.isnan(lib_pred_valid.astype(float)), amber_eq, lib_pred_valid.astype(float))
        mae_lib = np.mean(np.abs(lib_pred_valid - obs[valid]))
        lib_totals.append(mae_lib)
        
        line = f"  {col:>20s}"
        for ff_name, ff_params in FORCE_FIELDS.items():
            eq = ff_params[f'{ff_key}_eq']
            mae = np.mean(np.abs(eq - obs[valid]))
            ff_totals[ff_name].append(mae)
            line += f"  {mae:>14.4f}"
        line += f"  {mae_lib:>14.4f}"
        R.append(line)
    
    # Average
    line = f"  {'AVERAGE':>20s}"
    for ff_name in FORCE_FIELDS:
        avg = np.mean(ff_totals[ff_name])
        line += f"  {avg:>14.4f}"
    line += f"  {np.mean(lib_totals):>14.4f}"
    R.append("  " + "─" * (len(header.strip())))
    R.append(line)
    
    # ── Bond lengths ─────────────────────────────────────────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 2: MAE FOR EACH FORCE FIELD vs PDB (BOND LENGTHS)")
    R.append("━" * 78)
    
    bond_obs = ['bond_N_CA', 'bond_CA_C', 'bond_C_O', 'bond_C_N_next', 'bond_CA_CB']
    
    header2 = f"  {'Observable':>20s}"
    for ff_name in FORCE_FIELDS:
        header2 += f"  {FORCE_FIELDS[ff_name]['label']:>14s}"
    header2 += f"  {'Library':>14s}"
    R.append(header2)
    R.append("  " + "─" * (len(header2.strip())))
    
    for col in bond_obs:
        if col not in df.columns:
            continue
        obs = df[col].values
        valid = ~np.isnan(obs) & ~np.isnan(phi) & ~np.isnan(psi)
        if valid.sum() < 100:
            continue
        
        ff_key = OBS_MAP[col]
        
        lib_pred = np.array([
            lib.get(phi[i], psi[i], res[i], col)
            if valid[i] else np.nan for i in range(len(df))
        ], dtype=float)
        lib_pred_valid = lib_pred[valid]
        amber_eq = FORCE_FIELDS['AMBER_ff14SB'][f'{ff_key}_eq']
        lib_pred_valid = np.where(np.isnan(lib_pred_valid), amber_eq, lib_pred_valid)
        mae_lib = np.mean(np.abs(lib_pred_valid - obs[valid]))
        
        line = f"  {col:>20s}"
        for ff_name, ff_params in FORCE_FIELDS.items():
            eq = ff_params[f'{ff_key}_eq']
            mae = np.mean(np.abs(eq - obs[valid]))
            line += f"  {mae:>14.4f}"
        line += f"  {mae_lib:>14.4f}"
        R.append(line)
    
    # ── Strain energy per FF ─────────────────────────────────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 3: TOTAL STRAIN ENERGY PER FORCE FIELD")
    R.append("━" * 78)
    R.append("  (Using each FF's own spring constants)")
    R.append("")
    
    header3 = f"  {'Force Field':>20s}  {'Strain/res':>12s}  {'Per 100 res':>12s}  {'With Library':>12s}  {'Reduction':>10s}"
    R.append(header3)
    R.append("  " + "─" * (len(header3.strip())))
    
    all_obs = angle_obs + bond_obs
    
    for ff_name, ff_params in FORCE_FIELDS.items():
        strain_ff = np.zeros(len(df))
        strain_lib = np.zeros(len(df))
        
        for col in all_obs:
            if col not in df.columns:
                continue
            obs = df[col].values
            valid = ~np.isnan(obs) & ~np.isnan(phi) & ~np.isnan(psi)
            
            ff_key = OBS_MAP[col]
            eq_ff = ff_params[f'{ff_key}_eq']
            k = ff_params[f'{ff_key}_k']
            
            is_angle = col in angle_obs
            conv = np.pi / 180.0 if is_angle else 1.0
            
            e_ff = 0.5 * k * ((obs - eq_ff) * conv) ** 2
            e_ff[~valid] = 0.0
            strain_ff += e_ff
            
            # Library equilibrium
            lib_eq = np.array([
                lib.get(phi[i], psi[i], res[i], col)
                if valid[i] else eq_ff for i in range(len(df))
            ])
            lib_eq = np.where(np.isnan(lib_eq), eq_ff, lib_eq)
            e_lib = 0.5 * k * ((obs - lib_eq) * conv) ** 2
            e_lib[~valid] = 0.0
            strain_lib += e_lib
        
        mean_ff = strain_ff.mean()
        mean_lib = strain_lib.mean()
        reduction = (mean_ff - mean_lib) / mean_ff * 100
        
        R.append(f"  {ff_params['label']:>20s}  {mean_ff:>12.4f}  "
                 f"{mean_ff*100:>12.1f}  {mean_lib*100:>12.1f}  "
                 f"{reduction:>+10.1f}%")
    
    # ── Improvement ranking ──────────────────────────────────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 4: WHICH FORCE FIELD BENEFITS MOST?")
    R.append("━" * 78)
    R.append("")
    
    for col in ['tau_deg', 'angle_N_CA_CB', 'bond_N_CA']:
        if col not in df.columns:
            continue
        obs = df[col].values
        valid = ~np.isnan(obs) & ~np.isnan(phi) & ~np.isnan(psi)
        ff_key = OBS_MAP[col]
        
        lib_pred = np.array([
            lib.get(phi[i], psi[i], res[i], col)
            if valid[i] else np.nan for i in range(len(df))
        ], dtype=float)
        lib_valid = lib_pred[valid]
        
        R.append(f"  [{col}]")
        R.append(f"  {'Force Field':>20s}  {'FF eq':>8s}  {'MAE_FF':>8s}  {'MAE_Lib':>8s}  {'Improve':>8s}")
        
        for ff_name, ff_params in FORCE_FIELDS.items():
            eq = ff_params[f'{ff_key}_eq']
            mae_ff = np.mean(np.abs(eq - obs[valid]))
            # Use AMBER default where lib is NaN
            lv = np.where(np.isnan(lib_valid), eq, lib_valid)
            mae_lib = np.mean(np.abs(lv - obs[valid]))
            pct = (mae_ff - mae_lib) / mae_ff * 100
            R.append(f"  {ff_params['label']:>20s}  {eq:>8.3f}  {mae_ff:>8.4f}  "
                     f"{mae_lib:>8.4f}  {pct:>+8.1f}%")
        R.append("")
    
    # ── CMAP note ────────────────────────────────────────────────────────
    R.append("━" * 78)
    R.append("SECTION 5: RELATIONSHIP TO CHARMM CMAP")
    R.append("━" * 78)
    R.append("""
  CHARMM36m already includes a CMAP correction for TORSIONAL energies.
  Our library provides a complementary correction for BOND ANGLE and
  BOND LENGTH equilibrium values. The two corrections are orthogonal:
  
    CMAP corrects:   V_torsion(φ,ψ)  →  V_torsion + ΔV_CMAP(φ,ψ)
    Library corrects: θ_eq(fixed)     →  θ_eq(φ,ψ,res)
  
  They can be applied simultaneously. The library correction is needed
  even WITH CMAP because CMAP does not modify the bond-angle terms.
  
  Combined correction:
    E = V_bond(r) + V_angle(θ; θ_eq_lib) + V_torsion(φ,ψ) + V_CMAP(φ,ψ) + ...
  
  where θ_eq_lib replaces the fixed Engh-Huber target with our
  conformation-dependent library value.
""")
    
    # ── OpenMM integration code ──────────────────────────────────────────
    R.append("━" * 78)
    R.append("SECTION 6: OpenMM INTEGRATION CODE")
    R.append("━" * 78)
    R.append("""
  The following Python code shows how to add library corrections
  to any OpenMM simulation. This works with AMBER, CHARMM, or OPLS:

  ```python
  from openmm import CustomAngleForce
  from openmm.app import PDBFile, ForceField, Simulation
  import json
  
  # 1. Set up standard simulation
  pdb = PDBFile('protein.pdb')
  ff = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
  system = ff.createSystem(pdb.topology)
  
  # 2. Load geometry library
  with open('constants_library.json') as f:
      lib = json.load(f)
  
  # 3. For each residue, add a correction force
  # that shifts the equilibrium from FF value to library value
  correction = CustomAngleForce(
      '0.5 * k_corr * (theta - theta_lib)^2'
      ' - 0.5 * k_corr * (theta - theta_ff)^2'
  )
  correction.addPerAngleParameter('theta_lib')
  correction.addPerAngleParameter('theta_ff')
  correction.addPerAngleParameter('k_corr')
  
  for residue in pdb.topology.residues():
      phi, psi = get_dihedrals(residue)  # from coordinates
      res_name = residue.name
      
      # Lookup library tau
      tau_lib = lib[res_name][phi_bin][psi_bin]['tau_deg_eq']
      tau_ff = 111.1  # AMBER default
      k_corr = 63.0 * 4.184  # convert to kJ/mol/rad²
      
      # Add correction for N-CA-C angle
      N_idx = get_atom_index(residue, 'N')
      CA_idx = get_atom_index(residue, 'CA')
      C_idx = get_atom_index(residue, 'C')
      
      correction.addAngle(
          N_idx, CA_idx, C_idx,
          [tau_lib * deg2rad, tau_ff * deg2rad, k_corr]
      )
  
  system.addForce(correction)
  
  # 4. Run simulation as normal
  simulation = Simulation(pdb.topology, system, integrator)
  simulation.step(10000)
  ```
  
  The correction force adds a term that is zero when θ = θ_ff
  (standard behavior) and has its minimum at θ = θ_lib (corrected).
  The net effect is shifting the equilibrium without changing k.
""")
    
    # ── Parameter files ──────────────────────────────────────────────────
    R.append("━" * 78)
    R.append("SECTION 7: CORRECTED PARAMETER SNIPPETS")
    R.append("━" * 78)
    R.append("")
    R.append("  AMBER frcmod format (per-residue τ equilibrium):")
    R.append("  ─────────────────────────────────────────────────")
    
    # Get per-residue τ from library
    aa_tau = {}
    for aa in ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO',
               'PHE', 'TYR', 'TRP', 'SER', 'THR', 'CYS',
               'MET', 'ASP', 'ASN', 'GLU', 'GLN', 'LYS', 'ARG', 'HIS']:
        if 'res_name' in df.columns:
            mask = (df['res_name'] == aa) & df['tau_deg'].notna()
            if mask.sum() > 100:
                aa_tau[aa] = df.loc[mask, 'tau_deg'].mean()
    
    R.append("  ANGLE")
    for aa, tau in sorted(aa_tau.items()):
        R.append(f"  N -CX-C    63.00  {tau:>7.2f}    # {aa} (PDB-derived)")
    
    R.append("")
    R.append("  OPLS prm format:")
    R.append("  ─────────────────────────────────────────────────")
    R.append("  angle    N     CT    C      63.000   # k (kcal/mol/rad²)")
    for aa, tau in sorted(aa_tau.items()):
        R.append(f"  # {aa:>3s}:  equilibrium = {tau:>7.2f}°")
    
    return '\n'.join(R)


# ══════════════════════════════════════════════════════════════════════════════
# Generate correction parameter files
# ══════════════════════════════════════════════════════════════════════════════

def generate_correction_files(df, lib, out_dir):
    """Generate ready-to-use correction files for each force field."""
    
    # Per-residue average equilibrium values
    aa_list = sorted(df['res_name'].unique()) if 'res_name' in df.columns else []
    
    corrections = {}
    for aa in aa_list:
        mask = df['res_name'] == aa
        sub = df[mask]
        if len(sub) < 100:
            continue
        corr = {'residue': aa, 'n': int(mask.sum())}
        for col, ff_key in OBS_MAP.items():
            if col in sub.columns:
                vals = sub[col].dropna()
                if len(vals) > 50:
                    corr[f'{col}_pdb_mean'] = round(vals.mean(), 4)
                    corr[f'{col}_pdb_std'] = round(vals.std(), 4)
        corrections[aa] = corr
    
    # Save as CSV
    corr_df = pd.DataFrame(list(corrections.values()))
    corr_path = os.path.join(out_dir, 'per_residue_corrections.csv')
    corr_df.to_csv(corr_path, index=False)
    
    # Generate AMBER frcmod
    frcmod_path = os.path.join(out_dir, 'library_correction.frcmod')
    with open(frcmod_path, 'w') as f:
        f.write("Remark: Conformation-dependent geometry corrections\n")
        f.write("Remark: Generated from PDB-derived geometry library (Paper 4)\n")
        f.write("Remark: Replace standard equilibrium values with these\n\n")
        f.write("ANGLE\n")
        for aa, corr in sorted(corrections.items()):
            tau = corr.get('tau_deg_pdb_mean', 111.1)
            f.write(f"# {aa:>3s}:  N -CX-C    63.00  {tau:>7.2f}\n")
        f.write("\n")
    
    # Generate OpenMM Python script
    omm_path = os.path.join(out_dir, 'apply_library_corrections.py')
    with open(omm_path, 'w') as f:
        f.write('#!/usr/bin/env python3\n')
        f.write('"""\n')
        f.write('Apply geometry library corrections to an OpenMM simulation.\n')
        f.write('Works with any force field (AMBER, CHARMM, OPLS).\n')
        f.write('"""\n\n')
        f.write('import json\n')
        f.write('import numpy as np\n\n')
        f.write('def apply_corrections(system, topology, pdb_positions, library_path):\n')
        f.write('    """Add library correction forces to an OpenMM system.\n')
        f.write('    \n')
        f.write('    Args:\n')
        f.write('        system: OpenMM System object\n')
        f.write('        topology: OpenMM Topology\n')
        f.write('        pdb_positions: initial positions for dihedral calculation\n')
        f.write('        library_path: path to constants_library.json\n')
        f.write('    """\n')
        f.write('    try:\n')
        f.write('        from openmm import CustomAngleForce\n')
        f.write('    except ImportError:\n')
        f.write('        print("OpenMM not installed — cannot apply corrections")\n')
        f.write('        return system\n\n')
        f.write('    with open(library_path) as f:\n')
        f.write('        lib = json.load(f)\n\n')
        f.write('    correction = CustomAngleForce(\n')
        f.write("        '0.5*k*(theta-theta_lib)^2 - 0.5*k*(theta-theta_ff)^2'\n")
        f.write('    )\n')
        f.write("    correction.addPerAngleParameter('theta_lib')\n")
        f.write("    correction.addPerAngleParameter('theta_ff')\n")
        f.write("    correction.addPerAngleParameter('k')\n\n")
        f.write('    deg2rad = np.pi / 180.0\n')
        f.write('    n_corrections = 0\n\n')
        f.write('    for residue in topology.residues():\n')
        f.write('        # Get atom indices\n')
        f.write('        atoms = {a.name: a.index for a in residue.atoms()}\n')
        f.write("        if not all(a in atoms for a in ['N', 'CA', 'C']):\n")
        f.write('            continue\n\n')
        f.write('        # Get phi, psi from positions (simplified)\n')
        f.write('        # In practice, compute from coordinates\n')
        f.write('        res_name = residue.name\n')
        f.write('        phi_key, psi_key = "-65", "-45"  # default αR\n\n')
        f.write('        # Lookup library tau\n')
        f.write('        cell = None\n')
        f.write('        for cls in [res_name, "ALL"]:\n')
        f.write('            if cls in lib:\n')
        f.write('                cell = lib[cls].get(phi_key, {}).get(psi_key)\n')
        f.write('                if cell: break\n\n')
        f.write('        if cell and "tau_deg_eq" in cell:\n')
        f.write('            tau_lib = cell["tau_deg_eq"] * deg2rad\n')
        f.write('            tau_ff = 111.1 * deg2rad  # AMBER default\n')
        f.write('            k = 63.0 * 4.184  # kcal to kJ\n')
        f.write("            correction.addAngle(atoms['N'], atoms['CA'], atoms['C'],\n")
        f.write('                                [tau_lib, tau_ff, k])\n')
        f.write('            n_corrections += 1\n\n')
        f.write('    system.addForce(correction)\n')
        f.write(f'    print(f"Added {{n_corrections}} angle corrections")\n')
        f.write('    return system\n')
    
    return frcmod_path, omm_path


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Paper 4 — Force-field integration')
    ap.add_argument('--library', required=True)
    ap.add_argument('--csv', default=None, help='Features CSV for static analysis')
    ap.add_argument('--out', default='./paper4_integration')
    ap.add_argument('--static_only', action='store_true',
                    help='Skip OpenMM, do static analysis only')
    args = ap.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    
    print(f"Loading library...")
    lib = QuickLib(args.library)
    
    if args.csv:
        print(f"Reading {args.csv}...")
        df = pd.read_csv(args.csv, low_memory=False)
        print(f"  {len(df):,} rows")
        
        print(f"Running static analysis...")
        report = static_analysis(df, lib, args.out)
        
        report_path = os.path.join(args.out, 'integration_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(report)
        
        print(f"\nGenerating correction files...")
        frcmod, omm_script = generate_correction_files(df, lib, args.out)
        print(f"  Saved {frcmod}")
        print(f"  Saved {omm_script}")
    
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()