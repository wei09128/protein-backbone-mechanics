#!/usr/bin/env python3
"""
Paper 3 — Bond Length Scoping Analysis
=======================================

Purpose: determine whether bond-length variance for three backbone bonds
(N–Cα, Cα–C, C=O) is dominated by local stereochemical features (Block A)
or environmental/electronic features (Block B).

Inputs:
  --csv   : unified features.csv from features_collector.py
  --pdb_dir : directory of PDB files (same set used to generate features.csv)
  --out   : output directory for results (default: ./p3_scoping/)

Outputs:
  1. bond_lengths.csv           — per-residue bond lengths merged with features
  2. variance_decomposition.txt — ANOVA results (Block A vs B vs A+B)
  3. convergence_curves.png     — residual variance vs subgrouping depth
  4. leaf_histograms.png        — sample size distribution at stopping depth
  5. regime_summary.png         — R² bar chart (local vs env vs combined)

Design:
  Block A (local):  res_name, ss_bin, chi1_class, flanking_class_nm1, flanking_class_np1
  Block B (env):    hb_co_status, hb_nh_status, packing_density, burial_proxy
  
  Three models per bond: A-only, B-only, A+B
  Variance partition via Type II ANOVA (statsmodels)
  Subgrouping convergence via recursive partitioning with BH-FDR at q=0.01
  Stopping rule: n_leaf >= 100 or depth >= 5

Author: Wei (automated scoping for Paper 3)
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ══════════════════════════════════════════════════════════════════════════════
# Bond length extraction from PDB files
# ══════════════════════════════════════════════════════════════════════════════

def extract_bond_lengths_from_pdb(pdb_path):
    """
    Extract N–Cα, Cα–C, C=O bond lengths for each residue in a PDB file.
    
    Returns list of dicts with keys:
        pdb_id, chain, res_idx, bond_NCa, bond_CaC, bond_CO
    """
    from collections import OrderedDict
    
    # Parse ATOM records
    atoms_by_chain = defaultdict(lambda: defaultdict(dict))
    pdb_id = Path(pdb_path).stem.lower()
    
    with open(pdb_path) as fh:
        for line in fh:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            atom_name = line[12:16].strip()
            if atom_name not in ('N', 'CA', 'C', 'O'):
                continue
            alt = line[16]
            if alt not in (' ', 'A', ''):
                continue
            res_name = line[17:20].strip()
            chain = line[21]
            res_seq = int(line[22:26])
            icode = line[26]
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            
            key = (res_seq, icode)
            atoms_by_chain[chain][key][atom_name] = np.array([x, y, z])
    
    rows = []
    for chain_id in sorted(atoms_by_chain):
        residues = atoms_by_chain[chain_id]
        sorted_keys = sorted(residues.keys())
        
        for idx, key in enumerate(sorted_keys):
            atoms = residues[key]
            
            # Need N, CA, C, O for all three bonds
            if not all(a in atoms for a in ('N', 'CA', 'C', 'O')):
                continue
            
            n  = atoms['N']
            ca = atoms['CA']
            c  = atoms['C']
            o  = atoms['O']
            
            bond_NCa = np.linalg.norm(ca - n)
            bond_CaC = np.linalg.norm(c - ca)
            bond_CO  = np.linalg.norm(o - c)
            
            rows.append({
                'pdb_id':   pdb_id,
                'chain':    chain_id,
                'res_seq':  key[0],
                'icode':    key[1],
                'res_idx':  idx,
                'bond_NCa': round(bond_NCa, 4),
                'bond_CaC': round(bond_CaC, 4),
                'bond_CO':  round(bond_CO, 4),
            })
    
    return rows


def extract_all_bond_lengths(pdb_dir, max_pdbs=None, verbose=False):
    """Extract bond lengths from all PDB files in a directory."""
    pdb_files = sorted(Path(pdb_dir).glob('*.pdb'))
    if max_pdbs:
        pdb_files = pdb_files[:max_pdbs]
    
    all_rows = []
    for i, pdb_path in enumerate(pdb_files):
        try:
            rows = extract_bond_lengths_from_pdb(pdb_path)
            all_rows.extend(rows)
            if verbose and (i+1) % 100 == 0:
                print(f'  extracted {i+1}/{len(pdb_files)} PDBs, '
                      f'{len(all_rows)} residues so far')
        except Exception as e:
            if verbose:
                print(f'  [SKIP] {pdb_path.name}: {e}')
    
    print(f'  Bond lengths extracted: {len(all_rows)} residues '
          f'from {len(pdb_files)} PDBs')
    return pd.DataFrame(all_rows)


# ══════════════════════════════════════════════════════════════════════════════
# Feature engineering — Block A (local) and Block B (environmental)
# ══════════════════════════════════════════════════════════════════════════════

def _flanking_class(res_name):
    """Coarse-grain residue identity for flanking positions."""
    if res_name == 'GLY': return 'Gly'
    if res_name == 'PRO': return 'Pro'
    if res_name in ('PHE', 'TYR', 'TRP', 'HIS'): return 'aromatic'
    if res_name in ('VAL', 'ILE', 'THR'): return 'branched_beta'
    return 'other'


def _chi1_class(chi1_rad, has_chi1):
    """Classify chi1 into g+/g-/trans/none."""
    if not has_chi1 or np.isnan(chi1_rad):
        return 'none'
    deg = np.degrees(chi1_rad)
    # Wrap to [-180, 180]
    deg = ((deg + 180) % 360) - 180
    if -120 < deg <= 0:
        return 'g-'
    elif 0 < deg <= 120:
        return 'g+'
    else:
        return 'trans'


def _basin_label(phi, psi):
    """Assign Ramachandran basin from phi/psi in degrees."""
    if np.isnan(phi) or np.isnan(psi):
        return 'other'
    # Alpha-helix
    if -100 < phi < -30 and -67 < psi < -7:
        return 'alpha'
    # Beta-sheet
    if -180 < phi < -60 and 90 < psi < 180:
        return 'beta'
    if -180 < phi < -60 and -180 < psi < -120:
        return 'beta'
    # PPII
    if -90 < phi < -40 and 120 < psi < 180:
        return 'PPII'
    # Alpha-L
    if 30 < phi < 90 and 10 < psi < 70:
        return 'alphaL'
    return 'other'


def _hb_status_co(hb_n_strong, hb_best_e):
    """Classify C=O H-bond acceptor status."""
    if hb_n_strong >= 2:
        return 'bifurcated'
    elif hb_n_strong == 1:
        if hb_best_e < -1.5:
            return 'strong'
        else:
            return 'weak'
    else:
        return 'free'


def _packing_bin(contact_count):
    """Bin packing density into categories."""
    if np.isnan(contact_count):
        return 'unknown'
    if contact_count < 10:
        return 'exposed'
    elif contact_count < 20:
        return 'intermediate'
    else:
        return 'buried'


def engineer_features(df):
    """
    Add Block A and Block B categorical features to the merged dataframe.
    
    Block A (local/stereochemical):
      - res_name (already present, 20 categories)
      - basin (from phi/psi — alpha, beta, PPII, alphaL, other)
      - chi1_class (from chi1_rad — g+, g-, trans, none)
      - ss_class (from ss_bin integer — helix, sheet, PPII, turn, loop, alphaL)
      - flanking_nm1, flanking_np1 (coarse-grained flanking residues)
    
    Block B (environmental/electronic):
      - hb_co_status (from hb_n_strong, hb_best_e — free, weak, strong, bifurcated)
      - packing_bin (quartile-based from contact counts)
      - burial_class (quartile-based from burial proxy or steric counts)
    """
    print('    Engineering Block A features...')
    
    # ── Integer → residue name mapping (matches _RES_ORDER in features_collector) ──
    _IDX_TO_RES = {
        0: 'ALA', 1: 'ARG', 2: 'ASN', 3: 'ASP', 4: 'CYS',
        5: 'GLN', 6: 'GLU', 7: 'GLY', 8: 'HIS', 9: 'ILE',
        10: 'LEU', 11: 'LYS', 12: 'MET', 13: 'PHE', 14: 'PRO',
        15: 'SER', 16: 'THR', 17: 'TRP', 18: 'TYR', 19: 'VAL', 20: 'UNK'
    }
    
    # ss_bin integer → class name (matches _ss_bin in features_collector)
    _SS_LABELS = {0: 'alphaR', 1: 'beta', 2: 'PPII', 3: '310',
                  4: 'loop', 5: 'alphaL'}
    
    # ── Block A: basin ──
    phi = df['phi_deg'].values
    psi = df['psi_deg'].values
    basins = np.full(len(df), 'other', dtype=object)
    for i in range(len(df)):
        basins[i] = _basin_label(phi[i], psi[i])
    df['basin'] = basins
    
    # ── Block A: chi1_class (vectorized) ──
    chi1 = df['chi1_rad'].values if 'chi1_rad' in df.columns else np.full(len(df), np.nan)
    has_c = df['has_chi1'].values if 'has_chi1' in df.columns else np.zeros(len(df))
    chi1_cls = np.full(len(df), 'none', dtype=object)
    for i in range(len(df)):
        chi1_cls[i] = _chi1_class(chi1[i], has_c[i])
    df['chi1_class'] = chi1_cls
    
    # ── Block A: ss_class (from integer ss_bin) ──
    if 'ss_bin' in df.columns:
        df['ss_class'] = df['ss_bin'].map(_SS_LABELS).fillna('loop')
    else:
        df['ss_class'] = 'loop'
    
    # ── Block A: flanking residue classes ──
    # res_nm1 / res_np1 are integer-coded in the CSV (0=ALA, ..., 19=VAL)
    def _decode_flanking_column(col_name):
        if col_name not in df.columns:
            print(f'      {col_name}: NOT FOUND in CSV → all "other"')
            return pd.Series('other', index=df.index)
        col = df[col_name]
        # Debug: show what we're working with
        sample = col.dropna().head(5).tolist()
        print(f'      {col_name}: dtype={col.dtype}, sample={sample}')
        
        # Try to determine if values are residue names (strings) or indices (numbers)
        first_valid = col.dropna().iloc[0] if len(col.dropna()) > 0 else None
        if first_valid is None:
            return pd.Series('other', index=df.index)
        
        # If first valid value looks like a residue name string
        if isinstance(first_valid, str) and first_valid.isalpha() and len(first_valid) == 3:
            return col.apply(lambda v: _flanking_class(str(v)) if pd.notna(v) else 'other')
        
        # Otherwise treat as numeric index
        def _num_to_flanking(v):
            if pd.isna(v):
                return 'other'
            try:
                idx = int(float(v))  # handles '10', 10, 10.0
                return _flanking_class(_IDX_TO_RES.get(idx, 'UNK'))
            except (ValueError, TypeError):
                return 'other'
        
        return col.apply(_num_to_flanking)
    
    df['flanking_nm1'] = _decode_flanking_column('res_nm1')
    df['flanking_np1'] = _decode_flanking_column('res_np1')
    
    # Report flanking distributions
    for c in ['flanking_nm1', 'flanking_np1']:
        vc = df[c].value_counts()
        print(f'      {c}: {len(vc)} categories → {vc.to_dict()}')
    
    print('    Engineering Block B features...')
    
    # ── Block B: hb_co_status (vectorized) ──
    hb_ns = df['hb_n_strong'].values if 'hb_n_strong' in df.columns else np.zeros(len(df))
    hb_be = df['hb_best_e'].values if 'hb_best_e' in df.columns else np.zeros(len(df))
    hb_status = np.full(len(df), 'free', dtype=object)
    for i in range(len(df)):
        hb_status[i] = _hb_status_co(hb_ns[i], hb_be[i])
    df['hb_co_status'] = hb_status
    
    # ── Block B: packing_bin (data-driven quartiles) ──
    # Determine best packing proxy available
    packing_col = None
    if 'contact_count_8A' in df.columns and df['contact_count_8A'].notna().sum() > 100:
        packing_col = 'contact_count_8A'
    else:
        steric_5a = [c for c in df.columns if c.endswith('_5A')]
        if steric_5a:
            df['_packing_sum'] = df[steric_5a].sum(axis=1)
            packing_col = '_packing_sum'
    
    if packing_col and df[packing_col].notna().sum() > 100:
        # Use data-driven quartile bins
        try:
            df['packing_bin'] = pd.qcut(
                df[packing_col], q=4,
                labels=['exposed', 'partial', 'moderate', 'buried'],
                duplicates='drop'
            ).astype(str)
        except ValueError:
            # Fall back to equal-width bins if qcut fails
            df['packing_bin'] = pd.cut(
                df[packing_col], bins=4,
                labels=['exposed', 'partial', 'moderate', 'buried']
            ).astype(str)
    else:
        df['packing_bin'] = 'unknown'
    
    # ── Block B: burial_class (data-driven tertiles) ──
    burial_col = None
    if 'burial_proxy' in df.columns and df['burial_proxy'].notna().sum() > 100:
        burial_col = 'burial_proxy'
    elif 'bfactor_ca' in df.columns and df['bfactor_ca'].notna().sum() > 100:
        # B-factor as burial surrogate (inverse: high B → exposed)
        burial_col = 'bfactor_ca'
    
    if burial_col and df[burial_col].notna().sum() > 100:
        try:
            df['burial_class'] = pd.qcut(
                df[burial_col], q=3,
                labels=['exposed', 'intermediate', 'buried'],
                duplicates='drop'
            ).astype(str)
        except ValueError:
            df['burial_class'] = pd.cut(
                df[burial_col], bins=3,
                labels=['exposed', 'intermediate', 'buried']
            ).astype(str)
    else:
        df['burial_class'] = 'unknown'
    
    # Replace any NaN-category strings
    for col in ['packing_bin', 'burial_class', 'flanking_nm1', 'flanking_np1',
                'hb_co_status', 'basin', 'chi1_class', 'ss_class']:
        df[col] = df[col].fillna('unknown').replace('nan', 'unknown')
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ANOVA variance decomposition — Block A vs Block B
# ══════════════════════════════════════════════════════════════════════════════

BLOCK_A_COLS = ['res_name', 'basin', 'chi1_class', 'ss_class',
                'flanking_nm1', 'flanking_np1']
BLOCK_B_COLS = ['hb_co_status', 'packing_bin', 'burial_class']

BOND_NAMES   = ['bond_NCa', 'bond_CaC', 'bond_CO']
BOND_LABELS  = {'bond_NCa': 'N–Cα', 'bond_CaC': 'Cα–C', 'bond_CO': 'C=O'}


def run_anova_decomposition(df, bond_col, block_a_cols, block_b_cols):
    """
    Fit three OLS models (A-only, B-only, A+B) and return R² values
    plus Type II ANOVA SS for the combined model.
    
    Uses statsmodels OLS with categorical encoding.
    Returns dict with R²_A, R²_B, R²_AB, SS_A, SS_B, SS_interaction, SS_resid
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    
    # Drop rows with NaN in target
    mask = df[bond_col].notna()
    for col in block_a_cols + block_b_cols:
        mask &= df[col].notna()
    dfc = df.loc[mask].copy()
    
    # Encode categoricals
    for col in block_a_cols + block_b_cols:
        dfc[col] = dfc[col].astype(str)
    
    n = len(dfc)
    total_var = dfc[bond_col].var()
    
    # Drop categories with very few observations to avoid rank deficiency
    for col in block_a_cols + block_b_cols:
        vc = dfc[col].value_counts()
        rare = vc[vc < 20].index
        if len(rare) > 0:
            dfc = dfc[~dfc[col].isin(rare)]
    
    # Skip factors with only 1 level (no variance to explain)
    def _usable(cols):
        return [c for c in cols if dfc[c].nunique() >= 2]
    
    usable_a = _usable(block_a_cols)
    usable_b = _usable(block_b_cols)
    
    n = len(dfc)
    total_var = dfc[bond_col].var()
    
    # Formula strings
    a_terms = ' + '.join([f'C({c})' for c in usable_a]) if usable_a else None
    b_terms = ' + '.join([f'C({c})' for c in usable_b]) if usable_b else None
    
    results = {'n': n, 'total_var': total_var, 'bond': bond_col}
    
    # Model A: local features only
    if a_terms:
        try:
            model_a = ols(f'{bond_col} ~ {a_terms}', data=dfc).fit()
            results['R2_A'] = model_a.rsquared
            results['R2_adj_A'] = model_a.rsquared_adj
        except Exception as e:
            print(f'  WARNING: Model A failed for {bond_col}: {e}')
            results['R2_A'] = results['R2_adj_A'] = np.nan
    else:
        results['R2_A'] = results['R2_adj_A'] = 0.0
    
    # Model B: environmental features only
    if b_terms:
        try:
            model_b = ols(f'{bond_col} ~ {b_terms}', data=dfc).fit()
            results['R2_B'] = model_b.rsquared
            results['R2_adj_B'] = model_b.rsquared_adj
        except Exception as e:
            print(f'  WARNING: Model B failed for {bond_col}: {e}')
            results['R2_B'] = results['R2_adj_B'] = np.nan
    else:
        results['R2_B'] = results['R2_adj_B'] = 0.0
    
    # Model AB: both blocks (additive — no interactions to avoid rank explosion)
    if a_terms and b_terms:
        ab_formula = f'{bond_col} ~ {a_terms} + {b_terms}'
    elif a_terms:
        ab_formula = f'{bond_col} ~ {a_terms}'
    elif b_terms:
        ab_formula = f'{bond_col} ~ {b_terms}'
    else:
        ab_formula = None
    
    if ab_formula:
        try:
            model_ab = ols(ab_formula, data=dfc).fit()
            results['R2_AB'] = model_ab.rsquared
            results['R2_adj_AB'] = model_ab.rsquared_adj
            
            # Type II ANOVA on the combined model
            anova_table = anova_lm(model_ab, typ=2)
            
            # Extract SS for each block
            ss_a = sum(anova_table.loc[f'C({c})', 'sum_sq']
                       for c in usable_a
                       if f'C({c})' in anova_table.index)
            ss_b = sum(anova_table.loc[f'C({c})', 'sum_sq']
                       for c in usable_b
                       if f'C({c})' in anova_table.index)
            ss_resid = anova_table.loc['Residual', 'sum_sq']
            ss_total = ss_a + ss_b + ss_resid
            
            results['SS_A_frac'] = ss_a / ss_total
            results['SS_B_frac'] = ss_b / ss_total
            results['SS_resid_frac'] = ss_resid / ss_total
            
            # Per-feature breakdown
            results['per_feature'] = {}
            for c in usable_a + usable_b:
                key = f'C({c})'
                if key in anova_table.index:
                    row = anova_table.loc[key]
                    results['per_feature'][c] = {
                        'SS': row['sum_sq'],
                        'SS_frac': row['sum_sq'] / ss_total,
                        'F': row['F'],
                        'p': row['PR(>F)'],
                        'df': row['df'],
                    }
        except Exception as e:
            print(f'  WARNING: Model AB failed for {bond_col}: {e}')
            results['R2_AB'] = np.nan
            results['SS_A_frac'] = results['SS_B_frac'] = results['SS_resid_frac'] = np.nan
    else:
        results['R2_AB'] = 0.0
        results['SS_A_frac'] = results['SS_B_frac'] = results['SS_resid_frac'] = np.nan
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Recursive subgrouping — convergence analysis
# ══════════════════════════════════════════════════════════════════════════════

def recursive_subgroup(df, bond_col, split_cols, 
                       min_leaf=100, max_depth=5, fdr_q=0.01):
    """
    Iteratively partition data by split_cols until:
      - no split achieves BH-corrected p < fdr_q, OR
      - leaf size < min_leaf, OR
      - depth >= max_depth
    
    Returns:
      convergence: list of (depth, n_leaves, mean_leaf_var, total_residual_var)
      leaves: list of (path, n, mean, std, k) for each terminal leaf
    """
    from statsmodels.stats.multitest import multipletests
    
    convergence = []
    
    # Initial state: one leaf = entire dataset
    leaves = [{'path': (), 'mask': np.ones(len(df), dtype=bool)}]
    
    for depth in range(max_depth):
        new_leaves = []
        all_pvals = []
        candidate_splits = []
        
        for leaf in leaves:
            subset = df.loc[leaf['mask']]
            if len(subset) < min_leaf * 2:
                new_leaves.append(leaf)  # too small to split
                continue
            
            # Try each remaining split variable
            used = set(leaf['path'])
            best_col = None
            best_pval = 1.0
            
            for col in split_cols:
                if col in used:
                    continue
                groups = subset.groupby(col)[bond_col]
                group_list = [g.dropna().values for _, g in groups 
                              if len(g.dropna()) >= 10]
                if len(group_list) < 2:
                    continue
                
                # One-way ANOVA
                try:
                    F, p = sp_stats.f_oneway(*group_list)
                    if np.isnan(p):
                        continue
                except:
                    continue
                
                all_pvals.append(p)
                candidate_splits.append((leaf, col, p))
                
                if p < best_pval:
                    best_pval = p
                    best_col = col
        
        # Apply BH-FDR correction across all tests at this depth
        if not all_pvals:
            # Record convergence and stop
            leaf_vars = []
            for leaf in leaves:
                vals = df.loc[leaf['mask'], bond_col].dropna()
                if len(vals) > 1:
                    leaf_vars.append(vals.var() * len(vals))
            total_n = sum(len(df.loc[l['mask'], bond_col].dropna()) 
                          for l in leaves)
            total_resid = sum(leaf_vars) / total_n if total_n > 0 else np.nan
            convergence.append((depth, len(leaves), total_resid))
            break
        
        reject, pvals_corrected, _, _ = multipletests(
            all_pvals, alpha=fdr_q, method='fdr_bh')
        
        # Build a lookup: (leaf_id, col) -> corrected_p
        split_decisions = {}
        for (leaf, col, raw_p), corr_p, rej in zip(
                candidate_splits, pvals_corrected, reject):
            lid = id(leaf)
            if lid not in split_decisions or corr_p < split_decisions[lid][1]:
                split_decisions[lid] = (col, corr_p, rej)
        
        any_split = False
        for leaf in leaves:
            lid = id(leaf)
            if lid in split_decisions:
                col, corr_p, rej = split_decisions[lid]
                if rej and col is not None:
                    # Perform the split
                    subset = df.loc[leaf['mask']]
                    for val, grp in subset.groupby(col):
                        child_mask = leaf['mask'].copy()
                        child_mask &= (df[col] == val)
                        if child_mask.sum() >= min_leaf:
                            new_leaves.append({
                                'path': leaf['path'] + (col,),
                                'mask': child_mask
                            })
                        else:
                            new_leaves.append(leaf)
                    any_split = True
                    continue
            new_leaves.append(leaf)
        
        leaves = new_leaves
        
        # Record convergence
        leaf_vars = []
        for leaf in leaves:
            vals = df.loc[leaf['mask'], bond_col].dropna()
            if len(vals) > 1:
                leaf_vars.append(vals.var() * len(vals))
        total_n = sum(len(df.loc[l['mask'], bond_col].dropna()) 
                      for l in leaves)
        total_resid = sum(leaf_vars) / total_n if total_n > 0 else np.nan
        convergence.append((depth, len(leaves), total_resid))
        
        if not any_split:
            break
    
    # Compute leaf statistics
    RT = 0.593  # kcal/mol at 298K
    leaf_stats = []
    for leaf in leaves:
        vals = df.loc[leaf['mask'], bond_col].dropna()
        if len(vals) > 1:
            sigma = vals.std()
            k = RT / (sigma**2) if sigma > 0 else np.nan
            leaf_stats.append({
                'path': ' > '.join(leaf['path']) if leaf['path'] else 'root',
                'n': len(vals),
                'mean': vals.mean(),
                'std': sigma,
                'k': k,
            })
    
    return convergence, leaf_stats


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_regime_summary(anova_results, out_dir):
    """Bar chart: R² for local vs env vs combined, per bond."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bonds = [r['bond'] for r in anova_results]
    labels = [BOND_LABELS.get(b, b) for b in bonds]
    r2_a  = [r.get('R2_A', 0) for r in anova_results]
    r2_b  = [r.get('R2_B', 0) for r in anova_results]
    r2_ab = [r.get('R2_AB', 0) for r in anova_results]
    
    x = np.arange(len(bonds))
    w = 0.25
    
    ax.bar(x - w, r2_a,  w, label='Block A (local)', color='#4C72B0')
    ax.bar(x,     r2_b,  w, label='Block B (env)',    color='#DD8452')
    ax.bar(x + w, r2_ab, w, label='A + B (combined)', color='#55A868')
    
    ax.set_xlabel('Bond')
    ax.set_ylabel('R²')
    ax.set_title('Variance Explained: Local vs Environmental Features')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, max(r2_ab) * 1.3 if max(r2_ab) > 0 else 0.1)
    
    for i, (a, b, ab) in enumerate(zip(r2_a, r2_b, r2_ab)):
        ax.text(i - w, a + 0.005, f'{a:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(i,     b + 0.005, f'{b:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + w, ab + 0.005, f'{ab:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'regime_summary.png'), dpi=150)
    plt.close()
    print(f'  Saved regime_summary.png')


def plot_convergence(convergence_data, out_dir):
    """Residual variance vs subgrouping depth for each bond."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    
    for ax, (bond_col, conv) in zip(axes, convergence_data.items()):
        if not conv:
            ax.set_title(BOND_LABELS.get(bond_col, bond_col))
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            continue
        
        depths = [c[0] for c in conv]
        resid  = [c[2] for c in conv]
        n_leaves = [c[1] for c in conv]
        
        ax.plot(depths, resid, 'o-', color='#4C72B0', linewidth=2)
        for d, r, nl in zip(depths, resid, n_leaves):
            ax.annotate(f'{nl}L', (d, r), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=8, color='gray')
        
        ax.set_xlabel('Subgrouping depth')
        ax.set_title(BOND_LABELS.get(bond_col, bond_col))
    
    axes[0].set_ylabel('Residual variance (Å²)')
    fig.suptitle('Subgrouping Convergence', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'convergence_curves.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved convergence_curves.png')


def plot_leaf_histograms(leaf_data, out_dir):
    """Histogram of leaf sample sizes at stopping depth."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    for ax, (bond_col, stats) in zip(axes, leaf_data.items()):
        if not stats:
            ax.set_title(BOND_LABELS.get(bond_col, bond_col))
            continue
        
        ns = [s['n'] for s in stats]
        ax.hist(ns, bins=30, color='#4C72B0', edgecolor='white', alpha=0.8)
        ax.axvline(100, color='red', linestyle='--', label='min_leaf=100')
        ax.axvline(np.median(ns), color='orange', linestyle='--',
                   label=f'median={np.median(ns):.0f}')
        ax.set_xlabel('Leaf sample size')
        ax.set_title(f'{BOND_LABELS.get(bond_col, bond_col)}  '
                     f'({len(stats)} leaves)')
        ax.legend(fontsize=8)
    
    axes[0].set_ylabel('Count')
    fig.suptitle('Leaf Sample Size Distribution', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'leaf_histograms.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved leaf_histograms.png')


def plot_ss_decomposition(anova_results, out_dir):
    """Stacked bar: fraction of SS from Block A, Block B, Residual."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bonds = [r['bond'] for r in anova_results]
    labels = [BOND_LABELS.get(b, b) for b in bonds]
    
    ss_a = [r.get('SS_A_frac', 0) for r in anova_results]
    ss_b = [r.get('SS_B_frac', 0) for r in anova_results]
    ss_r = [r.get('SS_resid_frac', 0) for r in anova_results]
    
    x = np.arange(len(bonds))
    
    ax.bar(x, ss_a, label='Block A (local)', color='#4C72B0')
    ax.bar(x, ss_b, bottom=ss_a, label='Block B (env)', color='#DD8452')
    ax.bar(x, ss_r, bottom=[a+b for a,b in zip(ss_a, ss_b)],
           label='Residual', color='#CCCCCC')
    
    ax.set_xlabel('Bond')
    ax.set_ylabel('Fraction of total SS')
    ax.set_title('Type II ANOVA — Sum of Squares Partition')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ss_decomposition.png'), dpi=150)
    plt.close()
    print(f'  Saved ss_decomposition.png')


# ══════════════════════════════════════════════════════════════════════════════
# Text report
# ══════════════════════════════════════════════════════════════════════════════

def write_report(anova_results, convergence_data, leaf_data, out_dir):
    """Write human-readable variance decomposition report."""
    path = os.path.join(out_dir, 'variance_decomposition.txt')
    
    with open(path, 'w') as fh:
        fh.write('=' * 72 + '\n')
        fh.write('PAPER 3 SCOPING ANALYSIS — BOND LENGTH VARIANCE DECOMPOSITION\n')
        fh.write('=' * 72 + '\n\n')
        
        for res in anova_results:
            bond = res['bond']
            label = BOND_LABELS.get(bond, bond)
            fh.write(f'─── {label} ({bond}) ───\n')
            fh.write(f'  N = {res["n"]:,}\n')
            fh.write(f'  Total variance = {res["total_var"]:.6f} Å²\n')
            fh.write(f'  Total SD       = {np.sqrt(res["total_var"]):.4f} Å\n\n')
            
            fh.write(f'  R² (Block A — local only)       = {res.get("R2_A", np.nan):.4f}\n')
            fh.write(f'  R² (Block B — env only)         = {res.get("R2_B", np.nan):.4f}\n')
            fh.write(f'  R² (A + B — combined)           = {res.get("R2_AB", np.nan):.4f}\n\n')
            
            fh.write(f'  Type II SS fractions (combined model):\n')
            fh.write(f'    Block A:   {res.get("SS_A_frac", np.nan):.4f}\n')
            fh.write(f'    Block B:   {res.get("SS_B_frac", np.nan):.4f}\n')
            fh.write(f'    Residual:  {res.get("SS_resid_frac", np.nan):.4f}\n\n')
            
            # Per-feature breakdown
            if 'per_feature' in res:
                fh.write(f'  Per-feature breakdown:\n')
                fh.write(f'  {"Feature":<22s} {"SS_frac":>8s} {"F":>10s} {"p":>12s}\n')
                fh.write(f'  {"─"*22} {"─"*8} {"─"*10} {"─"*12}\n')
                for feat, vals in sorted(res['per_feature'].items(),
                                         key=lambda x: -x[1]['SS_frac']):
                    fh.write(f'  {feat:<22s} {vals["SS_frac"]:>8.4f} '
                             f'{vals["F"]:>10.1f} {vals["p"]:>12.2e}\n')
            fh.write('\n')
            
            # Convergence
            conv = convergence_data.get(bond, [])
            if conv:
                fh.write(f'  Subgrouping convergence:\n')
                fh.write(f'  {"Depth":>5s} {"Leaves":>7s} {"Resid var":>12s}\n')
                for depth, n_leaves, resid in conv:
                    fh.write(f'  {depth:>5d} {n_leaves:>7d} {resid:>12.6f}\n')
            fh.write('\n')
            
            # Leaf summary
            stats = leaf_data.get(bond, [])
            if stats:
                ns = [s['n'] for s in stats]
                ks = [s['k'] for s in stats if not np.isnan(s['k'])]
                fh.write(f'  Leaf summary ({len(stats)} leaves):\n')
                fh.write(f'    n:  min={min(ns)}, median={np.median(ns):.0f}, '
                         f'max={max(ns)}\n')
                if ks:
                    fh.write(f'    k:  min={min(ks):.1f}, median={np.median(ks):.1f}, '
                             f'max={max(ks):.1f} kcal/mol/Å²\n')
            
            fh.write('\n' + '=' * 72 + '\n\n')
        
        # Decision summary
        fh.write('DECISION SUMMARY\n')
        fh.write('─' * 40 + '\n')
        for res in anova_results:
            label = BOND_LABELS.get(res['bond'], res['bond'])
            r2a = res.get('R2_A', 0)
            r2b = res.get('R2_B', 0)
            r2ab = res.get('R2_AB', 0)
            
            if r2a > 2 * r2b:
                regime = 'LOCAL-DOMINATED'
            elif r2b > 2 * r2a:
                regime = 'ENVIRONMENT-DOMINATED'
            else:
                regime = 'MIXED'
            
            fh.write(f'  {label:8s}  R²_A={r2a:.3f}  R²_B={r2b:.3f}  '
                     f'R²_AB={r2ab:.3f}  →  {regime}\n')
        
        fh.write('\nOutcome interpretation:\n')
        fh.write('  1 = Clean partition → Paper 3 has a tight story\n')
        fh.write('  2 = Everything mixed → Paper 3 becomes a methods paper\n')
        fh.write('  3 = Subgrouping diverges → missing feature class\n')
    
    print(f'  Saved variance_decomposition.txt')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Paper 3 bond-length scoping analysis')
    ap.add_argument('--csv', required=True,
                    help='Path to unified features.csv')
    ap.add_argument('--pdb_dir', required=True,
                    help='Directory containing PDB files')
    ap.add_argument('--out', default='./p3_scoping',
                    help='Output directory (default: ./p3_scoping/)')
    ap.add_argument('--max_pdbs', type=int, default=None,
                    help='Limit number of PDBs to process (for testing)')
    ap.add_argument('--min_leaf', type=int, default=100,
                    help='Minimum leaf size for subgrouping (default: 100)')
    ap.add_argument('--max_depth', type=int, default=5,
                    help='Maximum subgrouping depth (default: 5)')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    
    # ── Step 1: Extract bond lengths from PDBs ────────────────────────────
    print('\n[1/5] Extracting bond lengths from PDB files...')
    bl_df = extract_all_bond_lengths(args.pdb_dir, args.max_pdbs, args.verbose)
    
    if len(bl_df) == 0:
        print('ERROR: no bond lengths extracted. Check --pdb_dir.')
        sys.exit(1)
    
    # Quick sanity check on bond length ranges
    for bond in BOND_NAMES:
        vals = bl_df[bond].dropna()
        print(f'  {BOND_LABELS[bond]:6s}: '
              f'mean={vals.mean():.3f} Å, std={vals.std():.4f} Å, '
              f'range=[{vals.min():.3f}, {vals.max():.3f}]')
    
    # ── Step 2: Load features.csv and merge ───────────────────────────────
    print('\n[2/5] Loading features.csv and merging...')
    feat_df = pd.read_csv(args.csv)
    print(f'  features.csv: {len(feat_df):,} rows, {len(feat_df.columns)} cols')
    
    # Merge on pdb_id + chain + res_idx
    # Bond lengths use sequential res_idx per chain; features.csv should too
    # Try merging on pdb_id and res_idx first
    if 'chain' in feat_df.columns and 'chain' in bl_df.columns:
        merge_cols = ['pdb_id', 'chain', 'res_idx']
    else:
        merge_cols = ['pdb_id', 'res_idx']
    
    # Normalize pdb_id casing
    feat_df['pdb_id'] = feat_df['pdb_id'].astype(str).str.lower()
    bl_df['pdb_id'] = bl_df['pdb_id'].astype(str).str.lower()
    
    merged = pd.merge(feat_df, bl_df[['pdb_id', 'chain', 'res_idx'] + BOND_NAMES],
                      on=merge_cols, how='inner')
    print(f'  Merged: {len(merged):,} rows '
          f'({len(merged)/len(feat_df)*100:.1f}% of features)')
    
    if len(merged) < 1000:
        print(f'  WARNING: very few merged rows ({len(merged)}). '
              f'Check pdb_id/res_idx alignment between CSVs.')
        print(f'  feat_df pdb_ids sample: {feat_df["pdb_id"].unique()[:5]}')
        print(f'  bl_df pdb_ids sample: {bl_df["pdb_id"].unique()[:5]}')
    
    # ── Outlier filtering ──
    # Bond lengths outside physically reasonable ranges are chain breaks,
    # alternate conformations, or parsing artifacts. Filter to ±4σ from
    # expected values (Engh & Huber reference ranges).
    n_before = len(merged)
    expected = {'bond_NCa': (1.458, 0.020),   # mean, generous_sd
                'bond_CaC': (1.525, 0.025),
                'bond_CO':  (1.231, 0.020)}
    for bond, (mu, sd) in expected.items():
        lo, hi = mu - 4*sd, mu + 4*sd
        mask = merged[bond].between(lo, hi)
        n_drop = (~mask).sum()
        if n_drop > 0:
            print(f'  Filtering {bond}: dropped {n_drop:,} rows '
                  f'outside [{lo:.3f}, {hi:.3f}]')
        merged = merged[mask]
    print(f'  After outlier filter: {len(merged):,} rows '
          f'(removed {n_before - len(merged):,}, '
          f'{(n_before - len(merged))/n_before*100:.2f}%)')
    
    # Save merged bond lengths
    merged.to_csv(os.path.join(args.out, 'bond_lengths.csv'), index=False)
    print(f'  Saved bond_lengths.csv')
    
    # ── Step 3: Engineer features ─────────────────────────────────────────
    print('\n[3/5] Engineering Block A / Block B features...')
    merged = engineer_features(merged)
    
    # Report feature distributions
    for col in BLOCK_A_COLS + BLOCK_B_COLS:
        vc = merged[col].value_counts()
        print(f'  {col}: {len(vc)} categories, '
              f'top={vc.index[0]} (n={vc.iloc[0]:,})')
    
    # ── Step 4: ANOVA variance decomposition ──────────────────────────────
    print('\n[4/5] Running ANOVA variance decomposition...')
    anova_results = []
    for bond in BOND_NAMES:
        print(f'  Analyzing {BOND_LABELS[bond]}...')
        res = run_anova_decomposition(merged, bond, BLOCK_A_COLS, BLOCK_B_COLS)
        anova_results.append(res)
        print(f'    R²: A={res.get("R2_A",0):.4f}, '
              f'B={res.get("R2_B",0):.4f}, '
              f'AB={res.get("R2_AB",0):.4f}')
    
    # ── Diagnostic: continuous predictor correlations ──────────────────────
    print('\n  [Diagnostic] Pearson r with continuous features:')
    cont_features = ['hb_best_e', 'hb_n_strong', 'bfactor_ca', 'sc_mass',
                     'contact_count_8A', 'burial_proxy',
                     'angle_NCaC', 'angle_CaCN', 'angle_CNCa',
                     'phi_deg', 'psi_deg', 'omega_deg',
                     'chi1_rad', 'sc_lever_arm']
    for bond in BOND_NAMES:
        print(f'  {BOND_LABELS[bond]}:')
        corrs = []
        for feat in cont_features:
            if feat in merged.columns:
                mask = merged[bond].notna() & merged[feat].notna()
                if mask.sum() > 100:
                    r, p = sp_stats.pearsonr(merged.loc[mask, bond],
                                             merged.loc[mask, feat])
                    corrs.append((feat, r, p))
        corrs.sort(key=lambda x: -abs(x[1]))
        for feat, r, p in corrs[:8]:
            print(f'    {feat:20s}  r={r:+.4f}  p={p:.2e}  r²={r**2:.4f}')
    
    # ── Step 5: Subgrouping convergence ───────────────────────────────────
    print('\n[5/5] Running recursive subgrouping...')
    all_split_cols = BLOCK_A_COLS + BLOCK_B_COLS
    convergence_data = {}
    leaf_data = {}
    
    for bond in BOND_NAMES:
        print(f'  Subgrouping {BOND_LABELS[bond]}...')
        conv, stats = recursive_subgroup(
            merged, bond, all_split_cols,
            min_leaf=args.min_leaf, max_depth=args.max_depth)
        convergence_data[bond] = conv
        leaf_data[bond] = stats
        if conv:
            print(f'    Final: {conv[-1][1]} leaves, '
                  f'resid_var={conv[-1][2]:.6f} Å²')
        if stats:
            ns = [s['n'] for s in stats]
            print(f'    Leaf n: min={min(ns)}, '
                  f'median={np.median(ns):.0f}, max={max(ns)}')
    
    # ── Output ────────────────────────────────────────────────────────────
    print('\nGenerating plots and report...')
    plot_regime_summary(anova_results, args.out)
    plot_ss_decomposition(anova_results, args.out)
    plot_convergence(convergence_data, args.out)
    plot_leaf_histograms(leaf_data, args.out)
    write_report(anova_results, convergence_data, leaf_data, args.out)
    
    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.1f}s. Results in {args.out}/')
    print(f'  bond_lengths.csv           — {len(merged):,} rows')
    print(f'  variance_decomposition.txt — full ANOVA report')
    print(f'  regime_summary.png         — R² bar chart')
    print(f'  ss_decomposition.png       — SS partition stacked bars')
    print(f'  convergence_curves.png     — residual var vs depth')
    print(f'  leaf_histograms.png        — leaf n distribution')


if __name__ == '__main__':
    main()