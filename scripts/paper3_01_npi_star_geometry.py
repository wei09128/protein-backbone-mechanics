#!/usr/bin/env python3
"""
Paper 3 — n→π* Interaction Geometry Extraction & Analysis
==========================================================

The n→π* interaction is a quantum-mechanical orbital overlap between the
lone pair (n) on the carbonyl oxygen of residue i and the π* antibonding
orbital of the carbonyl carbon of residue i+1. It stabilises αR and PPII
conformations by ~0.5–1.0 kcal/mol per residue.

No classical force field captures this interaction — it is purely QM.
But its GEOMETRY is measurable from PDB coordinates:

  Geometric signature (Bartlett et al., J Mol Biol 2010; Newberry & Raines):
    d(O_i ··· C_i+1)     < 3.22 Å   (van der Waals sum = 3.22)
    θ(Bürgi–Dunitz angle) ≈ 107±10°  (O_i ··· C_i+1 = O_i+1 angle)

This script:
  1. Extracts O_i···C_i+1 distance and Bürgi–Dunitz angle for every residue
  2. Merges with existing features.csv
  3. Analyses how n→π* geometry correlates with:
     - (φ, ψ) position on the Ramachandran plane
     - Bond angle deformations (τ, ∠N-Cα-Cβ, ∠C-Cα-Cβ) from Paper 2
     - Basin classification
     - Residue type (Pro, Gly, β-branched, aromatic, etc.)
  4. Produces diagnostic plots and a summary report

Key question for Paper 3: do the residual patterns from Papers 1–2
(the ~15–20% unexplained variance) localise to regions of strong n→π*?

Usage:
  python paper3_01_npi_star_geometry.py \
      --csv /mnt/f/Protein_Folding/v6_GeometryDeformation/features.csv \
      --pdb_dir /mnt/f/Protein_Folding/pdb_cache \
      --out ./paper3_npi/ \
      --max_pdbs 500

  # Quick test (5 structures):
  python paper3_01_npi_star_geometry.py \
      --csv features.csv --pdb_dir ./pdb_cache --out ./paper3_npi/ --max_pdbs 5

Author: Wei (Cvek Lab, LSUS)
"""

import argparse
import csv
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

csv.field_size_limit(sys.maxsize)
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# PDB parsing (reused from add_lj_torques.py, trimmed to what we need)
# ══════════════════════════════════════════════════════════════════════════════

SKIP_RES = {'HOH', 'WAT', 'DOD', 'SO4', 'GOL', 'EDO', 'ACE', 'NME'}

def parse_pdb_backbone(path):
    """Parse PDB, return only backbone heavy atoms (N, CA, C, O) per residue.
    
    Returns: dict of (chain, resseq) -> {'N': xyz, 'CA': xyz, 'C': xyz, 'O': xyz}
    """
    opener = gzip.open if str(path).endswith('.gz') else open
    residues = defaultdict(dict)
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
                if rname in SKIP_RES:
                    continue
                chain = line[21:22].strip()
                rseq = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                key = (chain, rseq)
                residues[key][aname] = np.array([x, y, z])
                if 'resname' not in residues[key]:
                    residues[key]['resname'] = rname
    except Exception as e:
        pass
    return residues


def find_pdb(pdb_id, pdb_dir):
    """Find PDB file on disk."""
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
# n→π* geometry measurement
# ══════════════════════════════════════════════════════════════════════════════

def measure_npi_star(residues, chain, resseq_i, resseq_ip1):
    """Measure n→π* geometry between residue i and i+1.
    
    The n→π* interaction is:
      Donor:   O of residue i     (the lone pair n)
      Acceptor: C of residue i+1  (the π* orbital on C=O)
    
    Geometric descriptors:
      d_OC:    distance O_i ··· C_{i+1}        (contact distance)
      theta_BD: Bürgi-Dunitz angle              (O_i ··· C_{i+1} = O_{i+1})
      d_OO:    distance O_i ··· O_{i+1}        (secondary contact)
      delta_e: estimated interaction energy      (from Bartlett empirical fit)
    
    Returns: dict with geometric parameters, or None if atoms missing.
    """
    key_i = (chain, resseq_i)
    key_ip1 = (chain, resseq_ip1)
    
    ri = residues.get(key_i, {})
    rip1 = residues.get(key_ip1, {})
    
    # Need O_i, C_{i+1}, O_{i+1} at minimum
    if 'O' not in ri or 'C' not in rip1 or 'O' not in rip1:
        return None
    # Also need C_i for the donor angle
    if 'C' not in ri:
        return None
    
    O_i = ri['O']
    C_i = ri['C']
    C_ip1 = rip1['C']
    O_ip1 = rip1['O']
    
    # 1. Contact distance: O_i ··· C_{i+1}
    vec_OC = C_ip1 - O_i
    d_OC = np.linalg.norm(vec_OC)
    
    # Sanity: if > 5 Å, there's a chain break
    if d_OC > 5.0 or d_OC < 0.5:
        return None
    
    # 2. Bürgi-Dunitz angle: O_i ··· C_{i+1} = O_{i+1}
    #    This is the angle at C_{i+1} between O_i and O_{i+1}
    vec_CO_i = O_i - C_ip1        # from C_{i+1} toward O_i (nucleophile)
    vec_CO_ip1 = O_ip1 - C_ip1    # from C_{i+1} toward O_{i+1} (the C=O bond)
    
    cos_theta = np.dot(vec_CO_i, vec_CO_ip1) / (
        np.linalg.norm(vec_CO_i) * np.linalg.norm(vec_CO_ip1))
    cos_theta = np.clip(cos_theta, -1, 1)
    theta_BD = np.degrees(np.arccos(cos_theta))
    
    # 3. O_i ··· O_{i+1} distance (secondary descriptor)
    d_OO = np.linalg.norm(O_ip1 - O_i)
    
    # 4. Donor angle: C_i = O_i ··· C_{i+1}
    #    How well aligned the lone pair is with the acceptor
    vec_OC_donor = C_i - O_i           # from O_i back toward its own C
    vec_OC_acceptor = C_ip1 - O_i      # from O_i toward acceptor C_{i+1}
    cos_donor = np.dot(vec_OC_donor, vec_OC_acceptor) / (
        np.linalg.norm(vec_OC_donor) * np.linalg.norm(vec_OC_acceptor))
    cos_donor = np.clip(cos_donor, -1, 1)
    theta_donor = np.degrees(np.arccos(cos_donor))
    
    # 5. Pyramidalisation: out-of-plane displacement of O_i above/below
    #    the plane defined by C_{i+1}, O_{i+1}, CA_{i+1}
    #    (measures how far O_i penetrates toward the π* orbital)
    if 'CA' in rip1:
        CA_ip1 = rip1['CA']
        # Plane normal from C_{i+1}, CA_{i+1}, O_{i+1}
        v1 = CA_ip1 - C_ip1
        v2 = O_ip1 - C_ip1
        normal = np.cross(v1, v2)
        n_len = np.linalg.norm(normal)
        if n_len > 0.01:
            normal = normal / n_len
            # Signed distance of O_i from this plane
            disp = np.dot(O_i - C_ip1, normal)
        else:
            disp = np.nan
    else:
        disp = np.nan
    
    # 6. Interaction strength estimate (empirical)
    #    Bartlett et al. (2010): E ∝ exp(-d/0.58) for d < 3.22 Å
    #    Newberry & Raines (2017): more refined, but this captures the trend
    if d_OC < 3.22:
        # Simple exponential decay model
        # Calibrated so E ≈ -0.5 kcal/mol at d = 2.8 Å (typical αR)
        e_npi = -0.88 * np.exp(-(d_OC - 2.8) / 0.58)
    else:
        e_npi = 0.0
    
    # 7. Binary classification
    # "Strong" n→π*: d < 3.22 Å AND θ_BD in [95°, 125°] (Bürgi-Dunitz window)
    is_strong = (d_OC < 3.22) and (95.0 < theta_BD < 125.0)
    # "Weak/marginal": d < 3.5 Å (beyond vdW sum but still measurable)
    is_present = (d_OC < 3.5) and (80.0 < theta_BD < 140.0)
    
    return {
        'npi_d_OC': d_OC,
        'npi_theta_BD': theta_BD,
        'npi_d_OO': d_OO,
        'npi_theta_donor': theta_donor,
        'npi_disp': disp,
        'npi_energy': e_npi,
        'npi_is_strong': int(is_strong),
        'npi_is_present': int(is_present),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Analysis functions
# ══════════════════════════════════════════════════════════════════════════════

NPI_DONOR_COLS = [
    'npi_d_OC_donor', 'npi_theta_BD_donor', 'npi_d_OO_donor',
    'npi_theta_donor_donor', 'npi_disp_donor', 'npi_energy_donor',
    'npi_is_strong_donor', 'npi_is_present_donor',
]

NPI_ACC_COLS = [
    'npi_d_OC_acc', 'npi_theta_BD_acc', 'npi_d_OO_acc',
    'npi_theta_donor_acc', 'npi_disp_acc', 'npi_energy_acc',
    'npi_is_strong_acc', 'npi_is_present_acc',
]


def assign_basin(phi, psi):
    """Simple basin assignment matching your features_collector logic."""
    if -180 <= phi < 0 and -120 < psi < 50:
        return 'alphaR'
    elif -180 <= phi < 0 and (psi >= 50 or psi <= -120):
        return 'beta'
    elif phi >= 0:
        return 'alphaL'
    else:
        return 'other'


def run_analysis(df, out_dir):
    """Core analysis: how does n→π* geometry relate to backbone mechanics?"""
    
    report = []
    report.append("=" * 72)
    report.append("Paper 3 — n→π* Interaction Geometry Analysis")
    report.append("=" * 72)
    report.append(f"Total residues: {len(df):,}")
    
    # ── 1. Basic statistics ──────────────────────────────────────────────
    report.append("\n[1] BASIC STATISTICS")
    report.append("-" * 50)
    
    for label, col in [("Donor d(O···C)", 'npi_d_OC_donor'),
                       ("Acceptor d(O···C)", 'npi_d_OC_acc'),
                       ("Donor θ_BD", 'npi_theta_BD_donor'),
                       ("Acceptor θ_BD", 'npi_theta_BD_acc'),
                       ("Donor E_nπ*", 'npi_energy_donor'),
                       ("Acceptor E_nπ*", 'npi_energy_acc')]:
        if col in df.columns:
            s = df[col].dropna()
            report.append(f"  {label:25s}  n={len(s):>8,}  "
                         f"mean={s.mean():7.3f}  std={s.std():7.3f}  "
                         f"median={s.median():7.3f}  "
                         f"[{s.quantile(0.05):6.3f}, {s.quantile(0.95):6.3f}]")
    
    # Strong/present fractions
    for label, col in [("Donor strong", 'npi_is_strong_donor'),
                       ("Donor present", 'npi_is_present_donor'),
                       ("Acceptor strong", 'npi_is_strong_acc'),
                       ("Acceptor present", 'npi_is_present_acc')]:
        if col in df.columns:
            s = df[col].dropna()
            frac = s.mean()
            report.append(f"  {label:25s}  {frac*100:5.1f}% ({int(s.sum()):,} / {len(s):,})")
    
    # ── 2. Per-basin breakdown ───────────────────────────────────────────
    report.append("\n[2] n→π* BY BASIN")
    report.append("-" * 50)
    
    if 'phi_deg' in df.columns and 'psi_deg' in df.columns:
        df['basin'] = df.apply(lambda r: assign_basin(r['phi_deg'], r['psi_deg']), axis=1)
    elif 'ss_bin' in df.columns:
        df['basin'] = df['ss_bin']
    
    if 'basin' in df.columns:
        for basin in ['alphaR', 'beta', 'alphaL', 'other']:
            mask = df['basin'] == basin
            if mask.sum() < 10:
                continue
            sub = df.loc[mask]
            d_col = 'npi_d_OC_donor'
            e_col = 'npi_energy_donor'
            s_col = 'npi_is_strong_donor'
            if d_col in sub.columns:
                d = sub[d_col].dropna()
                e = sub[e_col].dropna() if e_col in sub.columns else pd.Series()
                s = sub[s_col].dropna() if s_col in sub.columns else pd.Series()
                report.append(
                    f"  {basin:8s}  n={mask.sum():>8,}  "
                    f"⟨d⟩={d.mean():.3f}±{d.std():.3f} Å  "
                    f"⟨E⟩={e.mean():.4f} kcal/mol  "
                    f"strong={s.mean()*100:.1f}%")
    
    # ── 3. Correlation with backbone geometry ────────────────────────────
    report.append("\n[3] CORRELATION WITH BACKBONE ANGLES")
    report.append("-" * 50)
    
    geom_cols = []
    for c in ['tau_deg', 'angle_NCaC', 'angle_CaCN', 'angle_CNCa',
              'omega_deg', 'phi_deg', 'psi_deg']:
        if c in df.columns:
            geom_cols.append(c)
    
    npi_num_cols = []
    for c in NPI_DONOR_COLS + NPI_ACC_COLS:
        if c in df.columns and df[c].dtype in ('float64', 'float32', 'int64'):
            npi_num_cols.append(c)
    
    if geom_cols and npi_num_cols:
        header = f"  {'':30s} " + "".join(f" {g:>12s}" for g in geom_cols)
        report.append(header)
        
        for n in npi_num_cols:
            vals_n = df[n].dropna()
            line = f"  {n:30s} "
            for g in geom_cols:
                shared = df[[n, g]].dropna()
                if len(shared) < 30:
                    line += f" {'n/a':>12s}"
                else:
                    r, p = sp_stats.pearsonr(shared[n], shared[g])
                    star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                    line += f" {r:+7.4f}{star:>5s}"
            report.append(line)
    
    # ── 4. Per-residue breakdown ─────────────────────────────────────────
    report.append("\n[4] n→π* BY RESIDUE TYPE")
    report.append("-" * 50)
    
    if 'res_name' in df.columns and 'npi_d_OC_donor' in df.columns:
        res_stats = []
        for res in sorted(df['res_name'].unique()):
            mask = df['res_name'] == res
            sub = df.loc[mask, 'npi_d_OC_donor'].dropna()
            e_sub = df.loc[mask, 'npi_energy_donor'].dropna()
            s_sub = df.loc[mask, 'npi_is_strong_donor'].dropna()
            if len(sub) < 10:
                continue
            res_stats.append({
                'res': res, 'n': len(sub),
                'd_mean': sub.mean(), 'd_std': sub.std(),
                'e_mean': e_sub.mean() if len(e_sub) > 0 else np.nan,
                'frac_strong': s_sub.mean() if len(s_sub) > 0 else np.nan,
            })
        
        res_stats.sort(key=lambda x: x['d_mean'])
        report.append(f"  {'Res':>4s}  {'n':>8s}  {'⟨d(O···C)⟩':>10s}  "
                     f"{'σ':>6s}  {'⟨E_nπ*⟩':>10s}  {'%strong':>8s}")
        for r in res_stats:
            report.append(f"  {r['res']:>4s}  {r['n']:>8,}  {r['d_mean']:>10.3f}  "
                         f"{r['d_std']:>6.3f}  {r['e_mean']:>10.4f}  "
                         f"{r['frac_strong']*100:>7.1f}%")
    
    # ── 5. Key diagnostic: does n→π* explain residuals? ──────────────────
    report.append("\n[5] RESIDUAL ANALYSIS — Does n→π* explain Paper 2 residuals?")
    report.append("-" * 50)
    
    # If tau_deg exists, compute residual after median subtraction
    if 'tau_deg' in df.columns and 'res_name' in df.columns and 'basin' in df.columns:
        # Simple residual: tau - median(tau | basin, res_name)
        df['tau_resid'] = df['tau_deg'] - df.groupby(['basin', 'res_name'])['tau_deg'].transform('median')
        
        for npi_col in ['npi_d_OC_donor', 'npi_energy_donor', 'npi_is_strong_donor']:
            if npi_col not in df.columns:
                continue
            shared = df[['tau_resid', npi_col]].dropna()
            if len(shared) < 30:
                continue
            r, p = sp_stats.pearsonr(shared['tau_resid'], shared[npi_col])
            report.append(f"  Pearson r(τ_resid, {npi_col}): {r:+.4f}  p={p:.2e}")
        
        # Partial R² from adding n→π* features to basin+residue model
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.model_selection import cross_val_score
            
            # Baseline: basin + res_name predict tau
            sub = df[['tau_deg', 'basin', 'res_name'] + 
                     [c for c in NPI_DONOR_COLS if c in df.columns]].dropna()
            if len(sub) > 1000:
                # Sample for speed
                if len(sub) > 50000:
                    sub = sub.sample(50000, random_state=42)
                
                y = sub['tau_deg'].values
                
                # Encode categoricals
                basin_dummies = pd.get_dummies(sub['basin'], prefix='basin')
                res_dummies = pd.get_dummies(sub['res_name'], prefix='res')
                
                X_base = pd.concat([basin_dummies, res_dummies], axis=1).values
                
                npi_features = [c for c in NPI_DONOR_COLS if c in sub.columns]
                X_npi = sub[npi_features].values
                X_full = np.hstack([X_base, X_npi])
                
                gbr = GradientBoostingRegressor(
                    n_estimators=100, max_depth=4, random_state=42)
                
                cv_base = cross_val_score(gbr, X_base, y, cv=5, scoring='r2')
                cv_full = cross_val_score(gbr, X_full, y, cv=5, scoring='r2')
                
                report.append(f"\n  GBR cross-validated R² for τ prediction:")
                report.append(f"    Baseline (basin + res_name):      {cv_base.mean():.4f} ± {cv_base.std():.4f}")
                report.append(f"    + n→π* features:                  {cv_full.mean():.4f} ± {cv_full.std():.4f}")
                report.append(f"    ΔR²:                              {cv_full.mean()-cv_base.mean():+.4f}")
                
                if cv_full.mean() - cv_base.mean() > 0.01:
                    report.append("    → n→π* adds measurable predictive power to τ")
                elif cv_full.mean() - cv_base.mean() > 0.001:
                    report.append("    → n→π* adds marginal predictive power to τ")
                else:
                    report.append("    → n→π* does NOT add predictive power to τ")
        except ImportError:
            report.append("  (sklearn not available — skipping GBR analysis)")
    
    # ── 6. Ramachandran heatmap of n→π* strength ────────────────────────
    report.append("\n[6] RAMACHANDRAN DISTRIBUTION OF n→π* STRENGTH")
    report.append("-" * 50)
    
    if 'phi_deg' in df.columns and 'psi_deg' in df.columns and 'npi_d_OC_donor' in df.columns:
        # Bin into 10° grid
        phi_bins = np.arange(-180, 181, 10)
        psi_bins = np.arange(-180, 181, 10)
        
        sub = df[['phi_deg', 'psi_deg', 'npi_d_OC_donor', 'npi_energy_donor']].dropna()
        sub['phi_bin'] = pd.cut(sub['phi_deg'], phi_bins, labels=False)
        sub['psi_bin'] = pd.cut(sub['psi_deg'], psi_bins, labels=False)
        
        grid = sub.groupby(['phi_bin', 'psi_bin']).agg(
            d_mean=('npi_d_OC_donor', 'mean'),
            e_mean=('npi_energy_donor', 'mean'),
            count=('npi_d_OC_donor', 'count'),
        ).reset_index()
        
        # Save grid for plotting
        grid_path = os.path.join(out_dir, 'npi_ramachandran_grid.csv')
        grid.to_csv(grid_path, index=False)
        report.append(f"  Saved Ramachandran grid to {grid_path}")
        report.append(f"  Grid cells with data: {len(grid)}")
        
        # Summary: which basins have shortest d(O···C)?
        for basin_name, phi_range, psi_range in [
            ('αR', (-100, -30), (-60, -20)),
            ('β', (-160, -80), (100, 170)),
            ('PPII', (-90, -55), (120, 170)),
            ('αL', (30, 90), (10, 70)),
        ]:
            mask = ((sub['phi_deg'] >= phi_range[0]) & (sub['phi_deg'] <= phi_range[1]) &
                    (sub['psi_deg'] >= psi_range[0]) & (sub['psi_deg'] <= psi_range[1]))
            if mask.sum() < 10:
                continue
            d = sub.loc[mask, 'npi_d_OC_donor']
            e = sub.loc[mask, 'npi_energy_donor']
            report.append(f"  {basin_name:5s}  n={mask.sum():>7,}  "
                         f"⟨d⟩={d.mean():.3f} Å  ⟨E⟩={e.mean():.4f} kcal/mol")
    
    # ── 7. Decision: does Paper 3 have legs? ─────────────────────────────
    report.append("\n" + "=" * 72)
    report.append("PAPER 3 SCOPING DECISION")
    report.append("=" * 72)
    report.append("""
  Key questions answered by this analysis:
  
  Q1: Is n→π* geometry (φ,ψ)-dependent?
      → Expected YES (literature). If NO, something is wrong with extraction.
  
  Q2: Does n→π* strength differ between basins?
      → Expected: αR strongest, β weakest, PPII intermediate.
  
  Q3: Does n→π* explain residual variance in τ beyond basin+residue?
      → If ΔR² > 1%: Paper 3 has a QM channel to report.
      → If ΔR² < 0.1%: n→π* is real but already captured by (φ,ψ) binning.
  
  Q4: Are Pro/Gly outliers in n→π* geometry?
      → Pro: expected strongest (restricted φ favours short O···C).
      → Gly: expected weakest (flexible, no preferred geometry).
  
  Next steps depend on Q3:
    ΔR² > 1%  → run alanine dipeptide QM benchmark (paper3_02)
    ΔR² < 0.1% → n→π* is implicitly captured; look at hyperconjugation instead
""")
    
    return '\n'.join(report)


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def make_plots(df, out_dir):
    """Generate diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return
    
    # ── Plot 1: d(O···C) distribution by basin ───────────────────────────
    if 'basin' in df.columns and 'npi_d_OC_donor' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1a: histogram by basin
        ax = axes[0]
        for basin, color in [('alphaR', '#2166AC'), ('beta', '#B2182B'),
                             ('alphaL', '#4DAF4A'), ('other', '#999999')]:
            mask = (df['basin'] == basin) & df['npi_d_OC_donor'].notna()
            if mask.sum() < 10:
                continue
            ax.hist(df.loc[mask, 'npi_d_OC_donor'], bins=80, range=(2.0, 5.0),
                    alpha=0.5, label=f'{basin} (n={mask.sum():,})', color=color,
                    density=True)
        ax.axvline(3.22, color='red', ls='--', lw=1.5, label='vdW sum (3.22 Å)')
        ax.set_xlabel('d(O_i ··· C_{i+1}) [Å]')
        ax.set_ylabel('Density')
        ax.set_title('n→π* contact distance by basin')
        ax.legend(fontsize=8)
        
        # 1b: Bürgi-Dunitz angle vs distance (scatter)
        ax = axes[1]
        sub = df[['npi_d_OC_donor', 'npi_theta_BD_donor']].dropna()
        if len(sub) > 20000:
            sub = sub.sample(20000, random_state=42)
        ax.scatter(sub['npi_d_OC_donor'], sub['npi_theta_BD_donor'],
                  s=1, alpha=0.1, c='#333333')
        # Mark the "ideal" region
        from matplotlib.patches import Rectangle
        rect = Rectangle((2.4, 95), 0.82, 30, linewidth=2,
                         edgecolor='red', facecolor='none', ls='--',
                         label='Strong n→π* zone')
        ax.add_patch(rect)
        ax.set_xlabel('d(O_i ··· C_{i+1}) [Å]')
        ax.set_ylabel('Bürgi-Dunitz angle [°]')
        ax.set_title('n→π* geometry: distance vs angle')
        ax.set_xlim(2.0, 5.0)
        ax.set_ylim(60, 180)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'npi_distributions.png'), dpi=150)
        plt.close()
        print(f"  Saved npi_distributions.png")
    
    # ── Plot 2: n→π* energy on Ramachandran plane ────────────────────────
    if all(c in df.columns for c in ['phi_deg', 'psi_deg', 'npi_energy_donor']):
        fig, ax = plt.subplots(figsize=(7, 6))
        sub = df[['phi_deg', 'psi_deg', 'npi_energy_donor']].dropna()
        
        phi_bins = np.arange(-180, 181, 5)
        psi_bins = np.arange(-180, 181, 5)
        
        sub['phi_bin'] = pd.cut(sub['phi_deg'], phi_bins, labels=False)
        sub['psi_bin'] = pd.cut(sub['psi_deg'], psi_bins, labels=False)
        
        grid = sub.groupby(['phi_bin', 'psi_bin'])['npi_energy_donor'].mean()
        Z = np.full((len(psi_bins)-1, len(phi_bins)-1), np.nan)
        for (pb, qb), val in grid.items():
            if not np.isnan(pb) and not np.isnan(qb):
                Z[int(qb), int(pb)] = val
        
        phi_c = (phi_bins[:-1] + phi_bins[1:]) / 2
        psi_c = (psi_bins[:-1] + psi_bins[1:]) / 2
        
        im = ax.pcolormesh(phi_c, psi_c, Z, cmap='RdBu_r', vmin=-0.8, vmax=0.0,
                          shading='auto')
        plt.colorbar(im, ax=ax, label='⟨E_nπ*⟩ [kcal/mol]')
        ax.set_xlabel('φ [°]')
        ax.set_ylabel('ψ [°]')
        ax.set_title('n→π* interaction energy on Ramachandran plane')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'npi_ramachandran.png'), dpi=150)
        plt.close()
        print(f"  Saved npi_ramachandran.png")
    
    # ── Plot 3: Per-residue n→π* strength ────────────────────────────────
    if 'res_name' in df.columns and 'npi_d_OC_donor' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        res_means = df.groupby('res_name')['npi_d_OC_donor'].agg(['mean', 'std', 'count'])
        res_means = res_means[res_means['count'] >= 100].sort_values('mean')
        
        x = range(len(res_means))
        ax.bar(x, res_means['mean'], yerr=res_means['std']/np.sqrt(res_means['count']),
               color='#4292C6', edgecolor='#2166AC', capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(res_means.index, fontsize=9)
        ax.axhline(3.22, color='red', ls='--', lw=1.5, label='vdW sum')
        ax.set_ylabel('⟨d(O···C)⟩ [Å]')
        ax.set_title('Mean n→π* contact distance by residue type')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'npi_per_residue.png'), dpi=150)
        plt.close()
        print(f"  Saved npi_per_residue.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Paper 3 — n→π* interaction geometry extraction')
    ap.add_argument('--csv', required=True,
                    help='Path to features.csv (unified)')
    ap.add_argument('--pdb_dir', required=True,
                    help='Directory containing PDB files')
    ap.add_argument('--out', default='./paper3_npi',
                    help='Output directory')
    ap.add_argument('--max_pdbs', type=int, default=None,
                    help='Limit number of PDBs for testing')
    ap.add_argument('--workers', type=int, default=None,
                    help='Number of CPU cores')
    args = ap.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    
    # ── Read CSV (pandas — fast) ─────────────────────────────────────────
    print(f"[1/5] Reading {args.csv}...")
    df_full = pd.read_csv(args.csv, low_memory=False)
    print(f"  {len(df_full):,} rows, {len(df_full.columns)} columns  "
          f"({time.time()-t0:.1f}s)")
    
    # Group by PDB
    df_full['_pdb_lower'] = df_full['pdb_id'].str.strip().str.lower()
    pdb_ids = sorted(df_full['_pdb_lower'].unique())
    if args.max_pdbs:
        pdb_ids = pdb_ids[:args.max_pdbs]
    
    # Filter to only the PDBs we'll process
    df_work = df_full[df_full['_pdb_lower'].isin(pdb_ids)].copy()
    df_work = df_work.reset_index(drop=True)
    print(f"  Processing {len(pdb_ids)} structures, {len(df_work):,} residues")
    
    # Build lightweight lookup: pdb_id -> list of (df_index, resseq)
    pdb_groups = defaultdict(list)
    for idx, row in df_work[['_pdb_lower', 'res_idx']].iterrows():
        pdb_groups[row['_pdb_lower']].append((idx, int(row['res_idx'])))
    
    # ── Extract n→π* geometry ────────────────────────────────────────────
    print(f"[2/5] Extracting n→π* geometry from {len(pdb_ids)} structures...")
    
    # Worker now only needs pdb_id, list of (idx, resseq), and pdb_dir
    # No more sending the entire CSV to each process
    all_npi = {}  # df_index -> dict of npi values
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_worker_extract, pid, pdb_groups[pid], args.pdb_dir): pid
            for pid in pdb_ids
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="n→π*"):
            for idx, npi_data in future.result():
                all_npi[idx] = npi_data
    
    print(f"  Measured n→π* for {len(all_npi):,} residues  "
          f"({time.time()-t0:.1f}s)")
    
    # ── Merge ────────────────────────────────────────────────────────────
    print(f"[3/5] Merging with features...")
    
    npi_df = pd.DataFrame.from_dict(all_npi, orient='index')
    df_merged = df_work.join(npi_df, how='inner')
    
    # Save merged CSV
    npi_csv = os.path.join(args.out, 'features_npi.csv')
    df_merged.to_csv(npi_csv, index=False)
    print(f"  Saved {npi_csv} ({len(df_merged):,} rows, {len(df_merged.columns)} cols)")
    
    # ── Analysis ─────────────────────────────────────────────────────────
    print(f"[4/5] Running analysis...")
    report = run_analysis(df_merged, args.out)
    
    report_path = os.path.join(args.out, 'npi_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved {report_path}")
    print(report)
    
    # ── Plots ────────────────────────────────────────────────────────────
    print(f"[5/5] Generating plots...")
    make_plots(df_merged, args.out)
    
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")


def _worker_extract(pdb_id, idx_resseq_list, pdb_dir):
    """Lightweight worker: parse PDB once, measure n→π* for each residue.
    
    Args:
        pdb_id: PDB identifier
        idx_resseq_list: list of (dataframe_index, resseq) tuples
        pdb_dir: path to PDB directory
    
    Returns:
        list of (df_index, npi_dict) tuples
    """
    pdb_path = find_pdb(pdb_id, pdb_dir)
    if pdb_path is None:
        return []
    
    residues = parse_pdb_backbone(pdb_path)
    if len(residues) < 3:
        return []
    
    # Build resseq -> chain lookup
    resseq_to_chain = {}
    for (chain, rseq) in residues:
        if rseq not in resseq_to_chain:
            resseq_to_chain[rseq] = chain
    
    results = []
    for df_idx, resseq in idx_resseq_list:
        chain = resseq_to_chain.get(resseq)
        if chain is None:
            continue
        
        npi_data = {}
        
        # Donor: this residue's O → next residue's C
        npi = measure_npi_star(residues, chain, resseq, resseq + 1)
        if npi is not None:
            for k, v in npi.items():
                npi_data[k + '_donor'] = v
        
        # Acceptor: previous residue's O → this residue's C
        npi_acc = measure_npi_star(residues, chain, resseq - 1, resseq)
        if npi_acc is not None:
            for k, v in npi_acc.items():
                npi_data[k + '_acc'] = v
        
        if npi_data:
            results.append((df_idx, npi_data))
    
    return results


if __name__ == '__main__':
    main()