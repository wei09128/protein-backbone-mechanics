#!/usr/bin/env python3
"""
Paper 3 — Hyperconjugation Geometry Extraction & Analysis
==========================================================

Paper 2 finding: β-branched residues (VAL, ILE, THR) drive bond-angle
deformations at Cα BEYOND what steric bulk alone predicts. The classical
explanation (bigger sidechain = more steric push) fails because LEU has
similar mass/volume to VAL but does NOT show the same anomalous behavior.

The quantum-mechanical explanation: hyperconjugation.

  σ(Cβ–Cγ) → σ*(Cα–N)   and   σ(Cβ–Cγ) → σ*(Cα–C)

These are orbital overlap interactions where electron density from the
Cβ–Cγ bond delocalises into the antibonding orbitals of the backbone
bonds at Cα. The key geometric requirement is ANTIPERIPLANAR alignment:
the Cγ–Cβ–Cα–N (or Cγ–Cβ–Cα–C) dihedral must be near ±180° for
maximal overlap.

β-branched residues have TWO Cγ atoms, so BOTH Cβ–Cγ bonds can
interact with backbone σ* simultaneously — doubling the effect and
creating a conformational coupling that non-branched residues lack.

This script:
  1. Extracts Cγ–Cβ–Cα–N and Cγ–Cβ–Cα–C dihedrals for all residues
  2. Computes cos²(dihedral) as the hyperconjugation overlap proxy
  3. For β-branched: measures BOTH branches and their combined effect
  4. Tests whether these geometric descriptors explain Paper 2 residuals
     (the excess bond-angle variance in VAL/ILE/THR vs LEU/MET/PHE)

Usage:
  python paper3_02_hyperconjugation.py \\
      --csv /mnt/f/Protein_Folding/v6_GeometryDeformation/features.csv \\
      --pdb_dir /mnt/f/Protein_Folding/pdb_cache \\
      --out ./paper3_hyperconj/ --max_pdbs 5

Author: Wei (Cvek Lab, LSUS)
"""

import argparse
import gzip
import os
import sys
import time
import warnings
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from tqdm import tqdm

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# Sidechain atom definitions for hyperconjugation
# ══════════════════════════════════════════════════════════════════════════════
# Which Cγ atoms exist per residue, and which are "branch" atoms

SIDECHAIN_CG = {
    # β-branched: TWO Cγ atoms (the key players)
    'VAL': ['CG1', 'CG2'],
    'ILE': ['CG1', 'CG2'],    # CG1 continues to CD1; CG2 is methyl
    'THR': ['OG1', 'CG2'],    # OG1 is hydroxyl (different σ orbital character)
    
    # Non-branched with Cγ (controls)
    'LEU': ['CG'],             # single Cγ, branches at Cγ not Cβ
    'MET': ['CG'],
    'PHE': ['CG'],
    'TYR': ['CG'],
    'TRP': ['CG'],
    'PRO': ['CG'],
    'LYS': ['CG'],
    'ARG': ['CG'],
    'GLU': ['CG'],
    'GLN': ['CG'],
    'HIS': ['CG'],
    'ASP': ['CG'],
    'ASN': ['CG'],
    
    # No Cγ
    'ALA': [],      # only CB, no γ
    'GLY': [],      # no CB at all
    'SER': ['OG'],  # hydroxyl at γ position
    'CYS': ['SG'],  # thiol at γ position
}


# ══════════════════════════════════════════════════════════════════════════════
# PDB parsing — backbone + sidechain up to Cγ
# ══════════════════════════════════════════════════════════════════════════════

SKIP_RES = {'HOH', 'WAT', 'DOD', 'SO4', 'GOL', 'EDO', 'ACE', 'NME'}
ATOMS_NEEDED = {'N', 'CA', 'C', 'O', 'CB',
                'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG',
                'CD', 'CD1', 'CD2'}  # include δ for ILE/LEU context

def parse_pdb_gamma(path):
    """Parse PDB, return backbone + Cγ-level atoms per residue."""
    opener = gzip.open if str(path).endswith('.gz') else open
    residues = defaultdict(dict)
    try:
        with opener(path, 'rt') as f:
            for line in f:
                if not line.startswith('ATOM'):
                    continue
                aname = line[12:16].strip()
                if aname not in ATOMS_NEEDED:
                    continue
                alt = line[16:17].strip()
                if alt and alt != 'A':
                    continue
                rname = line[17:20].strip()
                if rname in SKIP_RES:
                    continue
                chain = line[21:22].strip()
                rseq = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                key = (chain, rseq)
                residues[key][aname] = np.array([x, y, z])
                residues[key]['_resname'] = rname
    except Exception:
        pass
    return residues


def find_pdb(pdb_id, pdb_dir):
    d = Path(pdb_dir)
    for pat in [f"{pdb_id}.pdb", f"{pdb_id}.pdb.gz",
                f"pdb{pdb_id}.ent", f"pdb{pdb_id}.ent.gz",
                f"{pdb_id.upper()}.pdb", f"{pdb_id.lower()}.pdb",
                f"{pdb_id.lower()}.pdb.gz"]:
        p = d / pat
        if p.exists():
            return p
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Dihedral calculation
# ══════════════════════════════════════════════════════════════════════════════

def dihedral(p1, p2, p3, p4):
    """Compute dihedral angle (degrees) for atoms p1-p2-p3-p4."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    n1_len = np.linalg.norm(n1)
    n2_len = np.linalg.norm(n2)
    
    if n1_len < 1e-6 or n2_len < 1e-6:
        return np.nan
    
    n1 = n1 / n1_len
    n2 = n2 / n2_len
    
    b2_unit = b2 / np.linalg.norm(b2)
    m1 = np.cross(n1, b2_unit)
    
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    
    return np.degrees(np.arctan2(y, x))


def measure_hyperconj(residues, chain, resseq):
    """Measure hyperconjugation geometry for one residue.
    
    For each Cγ atom:
      - dihedral Cγ–Cβ–Cα–N  (overlap with σ*(Cα–N))
      - dihedral Cγ–Cβ–Cα–C  (overlap with σ*(Cα–C))
      - cos²(dihedral) as overlap proxy
        → cos² = 1.0 at ±180° (antiperiplanar, maximum overlap)
        → cos² = 0.0 at ±90° (gauche, minimum overlap)
    
    Returns dict of features or None.
    """
    key = (chain, resseq)
    res = residues.get(key, {})
    
    resname = res.get('_resname', '')
    if resname not in SIDECHAIN_CG:
        return None
    
    cg_atoms = SIDECHAIN_CG[resname]
    if not cg_atoms:
        return None
    
    # Need backbone N, CA, C and CB
    for needed in ('N', 'CA', 'C', 'CB'):
        if needed not in res:
            return None
    
    N = res['N']
    CA = res['CA']
    C = res['C']
    CB = res['CB']
    
    result = {
        'hc_resname': resname,
        'hc_n_branches': len(cg_atoms),
        'hc_is_beta_branched': int(len(cg_atoms) >= 2),
    }
    
    all_cos2_N = []
    all_cos2_C = []
    all_dihedrals_N = []
    all_dihedrals_C = []
    
    for i, cg_name in enumerate(cg_atoms):
        if cg_name not in res:
            continue
        
        CG = res[cg_name]
        
        # Dihedral: Cγ–Cβ–Cα–N
        dih_N = dihedral(CG, CB, CA, N)
        # Dihedral: Cγ–Cβ–Cα–C
        dih_C = dihedral(CG, CB, CA, C)
        
        if np.isnan(dih_N) or np.isnan(dih_C):
            continue
        
        cos2_N = np.cos(np.radians(dih_N)) ** 2
        cos2_C = np.cos(np.radians(dih_C)) ** 2
        
        suffix = f'_{i+1}' if len(cg_atoms) > 1 else ''
        result[f'hc_dih_CgCbCaN{suffix}'] = dih_N
        result[f'hc_dih_CgCbCaC{suffix}'] = dih_C
        result[f'hc_cos2_N{suffix}'] = cos2_N
        result[f'hc_cos2_C{suffix}'] = cos2_C
        result[f'hc_cg_name{suffix}'] = cg_name
        
        # Distance Cβ–Cγ (bond length affects σ orbital energy)
        d_cb_cg = np.linalg.norm(CG - CB)
        result[f'hc_d_CbCg{suffix}'] = d_cb_cg
        
        all_cos2_N.append(cos2_N)
        all_cos2_C.append(cos2_C)
        all_dihedrals_N.append(dih_N)
        all_dihedrals_C.append(dih_C)
    
    if not all_cos2_N:
        return None
    
    # Aggregate measures
    # Sum of cos² across all Cγ atoms — total hyperconjugation overlap
    result['hc_sum_cos2_N'] = sum(all_cos2_N)
    result['hc_sum_cos2_C'] = sum(all_cos2_C)
    result['hc_sum_cos2_total'] = sum(all_cos2_N) + sum(all_cos2_C)
    
    # Max cos² (strongest single overlap)
    result['hc_max_cos2_N'] = max(all_cos2_N)
    result['hc_max_cos2_C'] = max(all_cos2_C)
    
    # For β-branched: product of cos² (both branches antiperiplanar?)
    if len(all_cos2_N) >= 2:
        result['hc_prod_cos2_N'] = all_cos2_N[0] * all_cos2_N[1]
        result['hc_prod_cos2_C'] = all_cos2_C[0] * all_cos2_C[1]
        # Angle between the two Cγ–Cβ vectors (tetrahedral ideally ~109.5°)
        CG1 = res.get(cg_atoms[0])
        CG2 = res.get(cg_atoms[1])
        if CG1 is not None and CG2 is not None:
            v1 = CG1 - CB
            v2 = CG2 - CB
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            result['hc_cg_cg_angle'] = np.degrees(np.arccos(cos_angle))
    
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Worker
# ══════════════════════════════════════════════════════════════════════════════

def worker_extract(pdb_id, idx_resseq_list, pdb_dir):
    """Extract hyperconjugation geometry for all residues of one PDB."""
    pdb_path = find_pdb(pdb_id, pdb_dir)
    if pdb_path is None:
        return []
    
    residues = parse_pdb_gamma(pdb_path)
    if len(residues) < 3:
        return []
    
    resseq_to_chain = {}
    for (chain, rseq) in residues:
        if rseq not in resseq_to_chain:
            resseq_to_chain[rseq] = chain
    
    results = []
    for df_idx, resseq in idx_resseq_list:
        chain = resseq_to_chain.get(resseq)
        if chain is None:
            continue
        
        hc = measure_hyperconj(residues, chain, resseq)
        if hc is not None:
            results.append((df_idx, hc))
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis(df, out_dir):
    """Analyse hyperconjugation geometry vs backbone angle deformations."""
    
    R = []  # report lines
    R.append("=" * 72)
    R.append("Paper 3 — Hyperconjugation Geometry Analysis")
    R.append("=" * 72)
    R.append(f"Total residues with HC geometry: {len(df):,}")
    
    # ── 1. Basic statistics ──────────────────────────────────────────────
    R.append("\n[1] BASIC STATISTICS")
    R.append("-" * 60)
    
    for col in ['hc_sum_cos2_N', 'hc_sum_cos2_C', 'hc_sum_cos2_total',
                'hc_max_cos2_N', 'hc_max_cos2_C']:
        if col in df.columns:
            s = df[col].dropna()
            R.append(f"  {col:25s}  n={len(s):>8,}  "
                     f"mean={s.mean():.4f}  std={s.std():.4f}  "
                     f"[{s.quantile(0.05):.3f}, {s.quantile(0.95):.3f}]")
    
    # ── 2. β-branched vs non-branched comparison ─────────────────────────
    R.append("\n[2] β-BRANCHED vs NON-BRANCHED")
    R.append("-" * 60)
    
    if 'hc_is_beta_branched' in df.columns:
        for col in ['hc_sum_cos2_N', 'hc_sum_cos2_C', 'hc_sum_cos2_total']:
            if col not in df.columns:
                continue
            branched = df.loc[df['hc_is_beta_branched'] == 1, col].dropna()
            unbranched = df.loc[df['hc_is_beta_branched'] == 0, col].dropna()
            if len(branched) < 10 or len(unbranched) < 10:
                continue
            t, p = sp_stats.ttest_ind(branched, unbranched)
            R.append(f"  {col:25s}")
            R.append(f"    β-branched:  mean={branched.mean():.4f}±{branched.std():.4f}  n={len(branched):,}")
            R.append(f"    non-branch:  mean={unbranched.mean():.4f}±{unbranched.std():.4f}  n={len(unbranched):,}")
            R.append(f"    t={t:+.2f}  p={p:.2e}")
    
    # ── 3. Per-residue HC strength ───────────────────────────────────────
    R.append("\n[3] HYPERCONJUGATION BY RESIDUE TYPE")
    R.append("-" * 60)
    
    if 'hc_resname' in df.columns and 'hc_sum_cos2_total' in df.columns:
        res_order = ['VAL', 'ILE', 'THR',  # β-branched
                     'LEU', 'MET', 'PHE', 'TYR', 'TRP',  # non-branched bulky
                     'PRO', 'LYS', 'ARG', 'GLU', 'GLN',  # non-branched other
                     'HIS', 'ASP', 'ASN',
                     'SER', 'CYS', 'ALA']
        
        R.append(f"  {'Res':>4s} {'branch':>6s} {'n':>7s} "
                 f"{'⟨Σcos²_N⟩':>10s} {'⟨Σcos²_C⟩':>10s} {'⟨Σcos²_tot⟩':>12s}")
        R.append(f"  {'─'*4} {'─'*6} {'─'*7} {'─'*10} {'─'*10} {'─'*12}")
        
        for res in res_order:
            mask = df['hc_resname'] == res
            if mask.sum() < 5:
                continue
            sub = df.loc[mask]
            branched = 'YES' if res in ('VAL', 'ILE', 'THR') else 'no'
            sN = sub['hc_sum_cos2_N'].dropna()
            sC = sub['hc_sum_cos2_C'].dropna()
            sT = sub['hc_sum_cos2_total'].dropna()
            R.append(f"  {res:>4s} {branched:>6s} {mask.sum():>7,} "
                     f"{sN.mean():>10.4f} {sC.mean():>10.4f} {sT.mean():>12.4f}")
    
    # ── 4. Correlation with backbone angles ──────────────────────────────
    R.append("\n[4] CORRELATION WITH BACKBONE ANGLES")
    R.append("-" * 60)
    
    geom_cols = [c for c in ['tau_deg', 'angle_NCaC', 'angle_CaCN',
                             'angle_C_CA_CB', 'angle_N_CA_CB',
                             'phi_deg', 'psi_deg']
                 if c in df.columns]
    
    hc_num = [c for c in ['hc_sum_cos2_N', 'hc_sum_cos2_C', 'hc_sum_cos2_total',
                          'hc_max_cos2_N', 'hc_max_cos2_C',
                          'hc_prod_cos2_N', 'hc_prod_cos2_C']
              if c in df.columns]
    
    if geom_cols and hc_num:
        header = f"  {'':30s}" + "".join(f" {g:>14s}" for g in geom_cols)
        R.append(header)
        
        for h in hc_num:
            line = f"  {h:30s}"
            for g in geom_cols:
                shared = df[[h, g]].dropna()
                if len(shared) < 30:
                    line += f" {'n/a':>14s}"
                else:
                    r, p = sp_stats.pearsonr(shared[h], shared[g])
                    star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                    line += f" {r:+8.4f}{star:>6s}"
            R.append(line)
    
    # ── 4b. Same but ONLY for β-branched residues ────────────────────────
    R.append("\n[4b] CORRELATION — β-BRANCHED ONLY (VAL/ILE/THR)")
    R.append("-" * 60)
    
    df_bb = df[df['hc_is_beta_branched'] == 1].copy() if 'hc_is_beta_branched' in df.columns else pd.DataFrame()
    
    if len(df_bb) > 30 and geom_cols and hc_num:
        header = f"  {'':30s}" + "".join(f" {g:>14s}" for g in geom_cols)
        R.append(header)
        
        for h in hc_num:
            line = f"  {h:30s}"
            for g in geom_cols:
                shared = df_bb[[h, g]].dropna()
                if len(shared) < 30:
                    line += f" {'n/a':>14s}"
                else:
                    r, p = sp_stats.pearsonr(shared[h], shared[g])
                    star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                    line += f" {r:+8.4f}{star:>6s}"
            R.append(line)
    
    # ── 5. The key test: does HC explain β-branching anomaly? ────────────
    R.append("\n[5] KEY TEST — Does hyperconjugation explain the β-branching anomaly?")
    R.append("-" * 60)
    R.append("  Paper 2 found: VAL/ILE/THR shift ∠C-Cα-Cβ by 5.3× more than")
    R.append("  non-branched residues. Sterics alone don't explain this.")
    R.append("  Hypothesis: cos²(Cγ-Cβ-Cα-N) overlap drives the excess shift.")
    R.append("")
    
    # Compute residuals: angle - median(angle | basin, res_name)
    target_angles = [c for c in ['tau_deg', 'angle_NCaC', 'angle_C_CA_CB', 
                                  'angle_N_CA_CB']
                     if c in df.columns]
    
    # Assign basins
    if 'phi_deg' in df.columns and 'psi_deg' in df.columns:
        def assign_basin(phi, psi):
            if -180 <= phi < 0 and -120 < psi < 50:
                return 'alphaR'
            elif -180 <= phi < 0 and (psi >= 50 or psi <= -120):
                return 'beta'
            elif phi >= 0:
                return 'alphaL'
            return 'other'
        df['basin'] = df.apply(lambda r: assign_basin(r['phi_deg'], r['psi_deg']), axis=1)
    
    if 'basin' in df.columns and 'res_name' in df.columns and target_angles:
        for angle in target_angles:
            resid_col = f'{angle}_resid'
            df[resid_col] = df[angle] - df.groupby(['basin', 'res_name'])[angle].transform('median')
            
            # Test 1: r(residual, sum_cos2) for β-branched only
            if len(df_bb) > 30:
                for hc_col in ['hc_sum_cos2_N', 'hc_sum_cos2_C', 'hc_sum_cos2_total']:
                    if hc_col not in df_bb.columns:
                        continue
                    # Need to recompute residuals for the bb subset  
                    df_bb_r = df.loc[df['hc_is_beta_branched'] == 1].copy()
                    df_bb_r[resid_col] = df_bb_r[angle] - df_bb_r.groupby(['basin', 'res_name'])[angle].transform('median')
                    shared = df_bb_r[[resid_col, hc_col]].dropna()
                    if len(shared) < 30:
                        continue
                    r, p = sp_stats.pearsonr(shared[resid_col], shared[hc_col])
                    star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                    R.append(f"  β-branched: r({angle}_resid, {hc_col}) = {r:+.4f}  p={p:.2e} {star}")
            
            # Test 2: r(residual, sum_cos2) for NON-branched (should be weaker)
            df_nb = df[df['hc_is_beta_branched'] == 0].copy() if 'hc_is_beta_branched' in df.columns else pd.DataFrame()
            if len(df_nb) > 30:
                for hc_col in ['hc_sum_cos2_N', 'hc_sum_cos2_C']:
                    if hc_col not in df_nb.columns:
                        continue
                    df_nb[resid_col] = df_nb[angle] - df_nb.groupby(['basin', 'res_name'])[angle].transform('median')
                    shared = df_nb[[resid_col, hc_col]].dropna()
                    if len(shared) < 30:
                        continue
                    r, p = sp_stats.pearsonr(shared[resid_col], shared[hc_col])
                    star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                    R.append(f"  non-branch: r({angle}_resid, {hc_col}) = {r:+.4f}  p={p:.2e} {star}")
            
            R.append("")
    
    # ── 6. GBR ΔR² test ──────────────────────────────────────────────────
    R.append("\n[6] GBR ΔR² — Does HC improve angle prediction?")
    R.append("-" * 60)
    
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
        
        for angle in target_angles:
            hc_features = [c for c in ['hc_sum_cos2_N', 'hc_sum_cos2_C',
                                        'hc_sum_cos2_total', 'hc_max_cos2_N',
                                        'hc_max_cos2_C']
                           if c in df.columns]
            
            cols_needed = [angle, 'basin', 'res_name'] + hc_features
            sub = df[cols_needed].dropna()
            
            if len(sub) < 200:
                R.append(f"  {angle}: too few samples ({len(sub)})")
                continue
            
            if len(sub) > 50000:
                sub = sub.sample(50000, random_state=42)
            
            y = sub[angle].values
            basin_d = pd.get_dummies(sub['basin'], prefix='basin')
            res_d = pd.get_dummies(sub['res_name'], prefix='res')
            
            X_base = pd.concat([basin_d, res_d], axis=1).values
            X_hc = sub[hc_features].values
            X_full = np.hstack([X_base, X_hc])
            
            gbr = GradientBoostingRegressor(
                n_estimators=100, max_depth=4, random_state=42)
            
            cv_base = cross_val_score(gbr, X_base, y, cv=5, scoring='r2')
            cv_full = cross_val_score(gbr, X_full, y, cv=5, scoring='r2')
            
            delta = cv_full.mean() - cv_base.mean()
            R.append(f"  {angle}:")
            R.append(f"    Baseline (basin+res):  R²={cv_base.mean():.4f}±{cv_base.std():.4f}")
            R.append(f"    + HC features:         R²={cv_full.mean():.4f}±{cv_full.std():.4f}")
            R.append(f"    ΔR²: {delta:+.4f}")
            
            if delta > 0.01:
                R.append(f"    → HC adds measurable signal to {angle}")
            elif delta > 0.001:
                R.append(f"    → HC adds marginal signal to {angle}")
            else:
                R.append(f"    → HC does NOT improve {angle} prediction")
        
        # ── 6b. Same but β-branched only ─────────────────────────────────
        R.append("\n  [6b] β-BRANCHED ONLY:")
        for angle in target_angles:
            hc_features = [c for c in ['hc_sum_cos2_N', 'hc_sum_cos2_C',
                                        'hc_sum_cos2_total', 'hc_prod_cos2_N',
                                        'hc_prod_cos2_C']
                           if c in df.columns]
            
            cols_needed = [angle, 'basin', 'res_name'] + hc_features
            sub = df.loc[df['hc_is_beta_branched'] == 1, cols_needed].dropna()
            
            if len(sub) < 100:
                R.append(f"  {angle}: too few β-branched ({len(sub)})")
                continue
            
            y = sub[angle].values
            basin_d = pd.get_dummies(sub['basin'], prefix='basin')
            res_d = pd.get_dummies(sub['res_name'], prefix='res')
            
            X_base = pd.concat([basin_d, res_d], axis=1).values
            X_hc = sub[hc_features].values
            X_full = np.hstack([X_base, X_hc])
            
            gbr = GradientBoostingRegressor(
                n_estimators=100, max_depth=4, random_state=42)
            
            cv_base = cross_val_score(gbr, X_base, y, cv=5, scoring='r2')
            cv_full = cross_val_score(gbr, X_full, y, cv=5, scoring='r2')
            
            delta = cv_full.mean() - cv_base.mean()
            R.append(f"  {angle} (β-branched):")
            R.append(f"    Baseline:  R²={cv_base.mean():.4f}±{cv_base.std():.4f}")
            R.append(f"    + HC:      R²={cv_full.mean():.4f}±{cv_full.std():.4f}")
            R.append(f"    ΔR²: {delta:+.4f}")
            
    except ImportError:
        R.append("  (sklearn not available)")
    
    # ── 7. Decision ──────────────────────────────────────────────────────
    R.append("\n" + "=" * 72)
    R.append("SCOPING DECISION")
    R.append("=" * 72)
    R.append("""
  Interpretation guide:

  SCENARIO A — HC explains β-branching anomaly:
    β-branched r(resid, cos²) > 0.15 AND ΔR² > 1% for Cβ angles
    → Paper 3 has a QM mechanism: "hyperconjugation mediates β-branching
       effect on backbone geometry beyond steric prediction"
    → Next: alanine dipeptide QM scan to validate energy decomposition

  SCENARIO B — HC geometry is (φ,ψ)-redundant:
    r(resid, cos²) ≈ 0 even for β-branched
    → Like n→π*, HC is real but already captured by dihedral binning
    → Pivot: the "QM overlay" paper may not have enough new signal

  SCENARIO C — HC adds signal but not specifically for β-branched:
    r(resid, cos²) similar for branched and non-branched
    → HC is a general rotamer effect, not specific to branching
    → Paper 3 reframes as "sidechain rotamer modulates backbone 
       geometry via orbital overlap" (broader but less mechanistic)
""")
    
    return '\n'.join(R)


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def make_plots(df, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    # ── Plot 1: cos²(dihedral) distributions by branching ────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, col, title in zip(axes,
        ['hc_sum_cos2_N', 'hc_sum_cos2_C', 'hc_sum_cos2_total'],
        ['Σcos²(Cγ-Cβ-Cα-N)', 'Σcos²(Cγ-Cβ-Cα-C)', 'Σcos²(total)']):
        
        if col not in df.columns:
            continue
        
        bb = df.loc[df['hc_is_beta_branched'] == 1, col].dropna()
        nb = df.loc[df['hc_is_beta_branched'] == 0, col].dropna()
        
        ax.hist(nb, bins=50, alpha=0.5, density=True, label=f'non-branched (n={len(nb):,})',
                color='#4292C6')
        ax.hist(bb, bins=50, alpha=0.5, density=True, label=f'β-branched (n={len(bb):,})',
                color='#EF6548')
        ax.set_xlabel(title)
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'hc_cos2_distributions.png'), dpi=150)
    plt.close()
    print(f"  Saved hc_cos2_distributions.png")
    
    # ── Plot 2: per-residue cos² bar chart ───────────────────────────────
    if 'hc_resname' in df.columns and 'hc_sum_cos2_total' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        
        order = ['VAL', 'ILE', 'THR',
                 'LEU', 'MET', 'PHE', 'TYR', 'TRP',
                 'PRO', 'LYS', 'ARG', 'GLU', 'GLN',
                 'HIS', 'ASP', 'ASN', 'SER', 'CYS']
        
        means, stds, labels, colors = [], [], [], []
        for res in order:
            mask = df['hc_resname'] == res
            if mask.sum() < 5:
                continue
            s = df.loc[mask, 'hc_sum_cos2_total'].dropna()
            means.append(s.mean())
            stds.append(s.std() / np.sqrt(len(s)))
            labels.append(res)
            colors.append('#EF6548' if res in ('VAL', 'ILE', 'THR') else '#4292C6')
        
        x = range(len(means))
        ax.bar(x, means, yerr=stds, color=colors, edgecolor='#333333',
               capsize=2, linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('⟨Σcos²⟩ total overlap')
        ax.set_title('Hyperconjugation overlap by residue (red = β-branched)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'hc_per_residue.png'), dpi=150)
        plt.close()
        print(f"  Saved hc_per_residue.png")
    
    # ── Plot 3: scatter cos² vs angle residual (β-branched only) ─────────
    if 'hc_is_beta_branched' in df.columns:
        target = None
        for c in ['angle_C_CA_CB', 'angle_N_CA_CB', 'tau_deg']:
            if c in df.columns:
                target = c
                break
        
        if target and 'hc_sum_cos2_total' in df.columns:
            fig, ax = plt.subplots(figsize=(7, 6))
            sub = df[df['hc_is_beta_branched'] == 1].copy()
            if 'basin' in sub.columns and 'res_name' in sub.columns:
                sub['resid'] = sub[target] - sub.groupby(['basin', 'res_name'])[target].transform('median')
                sub = sub[['resid', 'hc_sum_cos2_total']].dropna()
                if len(sub) > 10000:
                    sub = sub.sample(10000, random_state=42)
                ax.scatter(sub['hc_sum_cos2_total'], sub['resid'],
                          s=3, alpha=0.2, c='#EF6548')
                r, p = sp_stats.pearsonr(sub['hc_sum_cos2_total'], sub['resid'])
                ax.set_xlabel('Σcos²(total) — hyperconjugation overlap')
                ax.set_ylabel(f'{target} residual [°]')
                ax.set_title(f'β-branched: HC overlap vs {target} residual\n'
                            f'r = {r:+.4f}, p = {p:.2e}')
            
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'hc_scatter_beta_branched.png'), dpi=150)
            plt.close()
            print(f"  Saved hc_scatter_beta_branched.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Paper 3 — Hyperconjugation geometry extraction')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--pdb_dir', required=True)
    ap.add_argument('--out', default='./paper3_hyperconj')
    ap.add_argument('--max_pdbs', type=int, default=None)
    ap.add_argument('--workers', type=int, default=None)
    args = ap.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    
    # ── Read CSV ─────────────────────────────────────────────────────────
    print(f"[1/5] Reading {args.csv}...")
    df_full = pd.read_csv(args.csv, low_memory=False)
    print(f"  {len(df_full):,} rows, {len(df_full.columns)} columns  "
          f"({time.time()-t0:.1f}s)")
    
    df_full['_pdb_lower'] = df_full['pdb_id'].str.strip().str.lower()
    pdb_ids = sorted(df_full['_pdb_lower'].unique())
    if args.max_pdbs:
        pdb_ids = pdb_ids[:args.max_pdbs]
    
    df_work = df_full[df_full['_pdb_lower'].isin(pdb_ids)].copy()
    df_work = df_work.reset_index(drop=True)
    print(f"  Processing {len(pdb_ids)} structures, {len(df_work):,} residues")
    
    pdb_groups = defaultdict(list)
    for idx, row in df_work[['_pdb_lower', 'res_idx']].iterrows():
        pdb_groups[row['_pdb_lower']].append((idx, int(row['res_idx'])))
    
    # ── Extract HC geometry ──────────────────────────────────────────────
    print(f"[2/5] Extracting hyperconjugation geometry...")
    
    all_hc = {}
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(worker_extract, pid, pdb_groups[pid], args.pdb_dir): pid
            for pid in pdb_ids
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="HC"):
            for idx, hc_data in future.result():
                all_hc[idx] = hc_data
    
    print(f"  Measured HC for {len(all_hc):,} residues  ({time.time()-t0:.1f}s)")
    
    # ── Merge ────────────────────────────────────────────────────────────
    print(f"[3/5] Merging...")
    hc_df = pd.DataFrame.from_dict(all_hc, orient='index')
    df_merged = df_work.join(hc_df, how='inner')
    
    hc_csv = os.path.join(args.out, 'features_hc.csv')
    df_merged.to_csv(hc_csv, index=False)
    print(f"  Saved {hc_csv} ({len(df_merged):,} rows)")
    
    # ── Analysis ─────────────────────────────────────────────────────────
    print(f"[4/5] Analysis...")
    report = run_analysis(df_merged, args.out)
    
    report_path = os.path.join(args.out, 'hc_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved {report_path}")
    print(report)
    
    # ── Plots ────────────────────────────────────────────────────────────
    print(f"[5/5] Plots...")
    make_plots(df_merged, args.out)
    
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()