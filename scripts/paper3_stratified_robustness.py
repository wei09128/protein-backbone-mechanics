#!/usr/bin/env python3
"""
paper3_stratified_robustness.py — Resolution + refinement-software stratification
================================================================================
Reviewer 1's explicit requirement: "The author should at least repeat the
analysis on high/ultrahigh-resolution structures (e.g., <=1.5 A), report
refinement software/restraint libraries where possible. Without these
controls, the ANOVA/GAM coupling values are not reliably interpretable."

This reuses anova2_decomposition() VERBATIM from
paper3_03_coupling_decomposition.py (copied, not reimplemented, to avoid
introducing a new bug into an already-correct calculation) and reruns it
stratified by:
  1. Resolution: <=1.5 A (ultrahigh) vs 1.5-2.0 A (standard)
  2. Refinement software: PHENIX vs REFMAC (parsed from REMARK 3 headers,
     same method used for the Paper 4/MechLib confound check)

Requires: resolutions.csv (PDB resolution per structure, already used for
Paper 4's Fig 9) and access to the same pdb_cache/ directory.

Usage:
  python paper3_stratified_robustness.py \
      --csv features_lj_FINAL_CLEAN.csv --resolutions resolutions.csv \
      --pdb_dir ./pdb_cache --bin_size 10
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

OBSERVABLES = ['tau_deg', 'angle_NCaC', 'angle_C_CA_CB', 'angle_N_CA_CB',
               'omega_deg', 'bond_CA_C', 'bond_N_CA', 'bond_C_N']


def anova2_decomposition(df, phi_col, psi_col, value_col, bin_size=10):
    """VERBATIM copy from paper3_03_coupling_decomposition.py -- do not
    modify the core calculation, only how/where it's called from."""
    sub = df[[phi_col, psi_col, value_col]].dropna().copy()
    if len(sub) < 100:
        return None

    phi_bins = np.arange(-180, 180 + bin_size, bin_size)
    psi_bins = np.arange(-180, 180 + bin_size, bin_size)

    sub['phi_bin'] = pd.cut(sub[phi_col], phi_bins, labels=False, right=False)
    sub['psi_bin'] = pd.cut(sub[psi_col], psi_bins, labels=False, right=False)
    sub = sub.dropna(subset=['phi_bin', 'psi_bin'])
    sub['phi_bin'] = sub['phi_bin'].astype(int)
    sub['psi_bin'] = sub['psi_bin'].astype(int)

    if len(sub) < 100:
        return None

    f0 = sub[value_col].mean()
    var_total = sub[value_col].var()

    cell_stats = sub.groupby(['phi_bin', 'psi_bin'])[value_col].agg(['mean', 'count', 'var'])
    cell_stats.columns = ['cell_mean', 'cell_count', 'cell_var']

    MIN_COUNT = 5
    cell_stats = cell_stats[cell_stats['cell_count'] >= MIN_COUNT]
    if len(cell_stats) < 10:
        return None

    phi_marginal = (cell_stats.reset_index().groupby('phi_bin')
                     .apply(lambda g: np.average(g['cell_mean'], weights=g['cell_count']),
                            include_groups=False))
    psi_marginal = (cell_stats.reset_index().groupby('psi_bin')
                     .apply(lambda g: np.average(g['cell_mean'], weights=g['cell_count']),
                            include_groups=False))

    phi_effect = phi_marginal - f0
    psi_effect = psi_marginal - f0

    coupling = cell_stats[['cell_mean', 'cell_count']].copy()
    coupling['phi_effect'] = coupling.index.get_level_values('phi_bin').map(phi_effect)
    coupling['psi_effect'] = coupling.index.get_level_values('psi_bin').map(psi_effect)
    coupling['additive_prediction'] = f0 + coupling['phi_effect'] + coupling['psi_effect']
    coupling['coupling_residual'] = coupling['cell_mean'] - coupling['additive_prediction']
    coupling = coupling.dropna()
    if len(coupling) < 10:
        return None

    weights = coupling['cell_count'].values
    phi_vals = coupling['phi_effect'].values
    psi_vals = coupling['psi_effect'].values
    coupling_vals = coupling['coupling_residual'].values

    def wvar(x, w):
        mu = np.average(x, weights=w)
        return np.average((x - mu) ** 2, weights=w)

    var_phi_marginal = wvar(phi_vals, weights)
    var_psi_marginal = wvar(psi_vals, weights)
    var_coupling = wvar(coupling_vals, weights)

    var_sys = var_phi_marginal + var_psi_marginal + var_coupling
    if var_sys < 1e-12:
        eta2_coupling = 0.0
    else:
        eta2_coupling = var_coupling / var_sys

    return {'eta2_coupling': eta2_coupling, 'n_total': len(sub), 'n_cells': len(coupling)}


def classify_refinement_software(pdb_path):
    program_text = ""
    try:
        with open(pdb_path, errors='ignore') as fh:
            for line in fh:
                if line.startswith('REMARK   3   PROGRAM'):
                    program_text += line[19:].strip() + " "
                elif line.startswith('CRYST1'):
                    break
    except Exception:
        return 'UNKNOWN'
    text = program_text.upper()
    if not text.strip():
        return 'UNKNOWN'
    if 'PHENIX' in text:
        return 'PHENIX'
    if 'REFMAC' in text:
        return 'REFMAC'
    if 'CNS' in text or 'X-PLOR' in text:
        return 'CNS/X-PLOR'
    if 'BUSTER' in text:
        return 'BUSTER'
    if 'TNT' in text:
        return 'TNT'
    return 'OTHER'


def report_stratum(label, df, bin_size):
    print(f"\n  --- {label} (n_residues={len(df):,}, n_structures={df['pdb_id'].nunique():,}) ---")
    for col in OBSERVABLES:
        if col not in df.columns:
            continue
        res = anova2_decomposition(df, 'phi_deg', 'psi_deg', col, bin_size)
        if res is None:
            print(f"    {col:20s}: insufficient data")
            continue
        print(f"    {col:20s}: eta2_coupling = {res['eta2_coupling']:.1%}  "
              f"(n={res['n_total']:,}, cells={res['n_cells']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--resolutions', required=True)
    ap.add_argument('--pdb_dir', required=True)
    ap.add_argument('--bin_size', type=int, default=10)
    args = ap.parse_args()

    print("Loading data...")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"  {len(df):,} residues, {df['pdb_id'].nunique()} structures")

    res_df = pd.read_csv(args.resolutions)
    res_col = [c for c in res_df.columns if 'resol' in c.lower()][0]
    id_col = [c for c in res_df.columns if 'pdb' in c.lower() or 'id' in c.lower()][0]
    res_map = dict(zip(res_df[id_col], res_df[res_col]))
    df['resolution'] = df['pdb_id'].map(res_map)
    print(f"  Resolution matched for {df['resolution'].notna().mean()*100:.1f}% of residues")

    print("Classifying refinement software from PDB headers...")
    pdb_ids = df['pdb_id'].unique()
    software_map = {}
    for k, pdb_id in enumerate(pdb_ids, 1):
        pdb_path = Path(args.pdb_dir) / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            pdb_path = Path(args.pdb_dir) / f"{pdb_id.lower()}.pdb"
        software_map[pdb_id] = (classify_refinement_software(pdb_path)
                                if pdb_path.exists() else 'UNKNOWN')
        if k % 2000 == 0:
            print(f"  [{k}/{len(pdb_ids)}]")
    df['refinement_sw'] = df['pdb_id'].map(software_map)

    print(f"\n{'='*70}")
    print("  1. FULL DATASET (baseline, for comparison)")
    print(f"{'='*70}")
    report_stratum("Full dataset", df, args.bin_size)

    print(f"\n{'='*70}")
    print("  2. BY RESOLUTION (Reviewer 1's explicit requirement)")
    print(f"{'='*70}")
    ultrahigh = df[df['resolution'] <= 1.5]
    standard = df[(df['resolution'] > 1.5) & (df['resolution'] <= 2.0)]
    report_stratum("Ultrahigh resolution (<=1.5 A)", ultrahigh, args.bin_size)
    report_stratum("Standard resolution (1.5-2.0 A)", standard, args.bin_size)

    print(f"\n{'='*70}")
    print("  3. BY REFINEMENT SOFTWARE (Reviewer 1's explicit requirement)")
    print(f"{'='*70}")
    for sw in ['REFMAC', 'PHENIX', 'CNS/X-PLOR', 'BUSTER', 'TNT']:
        group = df[df['refinement_sw'] == sw]
        if group['pdb_id'].nunique() < 20:
            print(f"\n  --- {sw}: too few structures, skipping ---")
            continue
        report_stratum(sw, group, args.bin_size)

    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print("  If eta2_coupling is SIMILAR across resolution bins and across")
    print("  REFMAC vs PHENIX, that argues coupling is a real geometric")
    print("  regularity, not a resolution- or restraint-library-driven")
    print("  artifact. If PHENIX (CDL-capable) shows notably higher coupling")
    print("  than REFMAC, or ultrahigh-resolution shows notably different")
    print("  coupling than standard-resolution, that is evidence the signal")
    print("  is at least partly confound-driven and needs to be reported as")
    print("  such, per Reviewer 1.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
