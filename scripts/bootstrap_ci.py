#!/usr/bin/env python3
"""
bootstrap_ci.py — Bootstrap confidence intervals for MechLib strain reductions
=================================================================================
Resamples by STRUCTURE (pdb_id), not by residue, since residues within the
same structure are not independent observations (shared crystal quality,
shared fold, shared refinement). Resampling at the residue level would
understate the true uncertainty.

Usage:
  python bootstrap_ci.py --csv features_lj_FINAL_CLEAN.csv --mode protein --out ./bootstrap/
  python bootstrap_ci.py --csv dna_features_v3_clean.csv --mode dna --out ./bootstrap/
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

AMBER = {
    'tau_deg':        {'eq': 110.10, 'k': 63.0,  'unit': 'deg'},  # verified: ff14SB.xml
    'angle_N_CA_CB':  {'eq': 109.70, 'k': 63.0,  'unit': 'deg'},  # verified: ff14SB.xml
    'angle_C_CA_CB':  {'eq': 111.10, 'k': 63.0,  'unit': 'deg'},  # verified: ff14SB.xml
    'angle_CaCN':     {'eq': 116.6, 'k': 70.0,  'unit': 'deg'},
    'angle_CNCa':     {'eq': 121.9, 'k': 50.0,  'unit': 'deg'},
    'angle_CA_C_O':   {'eq': 120.4, 'k': 80.0,  'unit': 'deg'},
    'bond_N_CA':      {'eq': 1.449,  'k': 337.0, 'unit': 'Å'},  # verified: ff14SB.xml
    'bond_CA_C':      {'eq': 1.522, 'k': 317.0, 'unit': 'Å'},
    'bond_C_O':       {'eq': 1.229, 'k': 570.0, 'unit': 'Å'},
    'bond_C_N_next':  {'eq': 1.335, 'k': 490.0, 'unit': 'Å'},
    'bond_CA_CB':     {'eq': 1.526, 'k': 317.0, 'unit': 'Å'},
}

AMBER_DNA = {
    'bond_P_O5':        {'eq': 1.610,  'k': 230.0, 'unit': 'Å'},
    'bond_O5_C5':       {'eq': 1.410,  'k': 320.0, 'unit': 'Å'},
    'bond_C5_C4':       {'eq': 1.526,  'k': 310.0, 'unit': 'Å'},
    'bond_C4_O4':       {'eq': 1.410,  'k': 320.0, 'unit': 'Å'},
    'bond_C4_C3':       {'eq': 1.526,  'k': 310.0, 'unit': 'Å'},
    'bond_C3_O3':       {'eq': 1.410,  'k': 320.0, 'unit': 'Å'},
    'bond_C3_C2':       {'eq': 1.526,  'k': 310.0, 'unit': 'Å'},
    'bond_C2_C1':       {'eq': 1.526,  'k': 310.0, 'unit': 'Å'},
    'bond_C1_O4':       {'eq': 1.410,  'k': 320.0, 'unit': 'Å'},
    'bond_C1_N':        {'eq': 1.475,  'k': 337.0, 'unit': 'Å'},
    'bond_O3_Pnext':     {'eq': 1.610,  'k': 230.0, 'unit': 'Å'},
    'angle_O5_P_O3prev': {'eq': 102.60, 'k': 45.0,  'unit': 'deg'},
    'angle_OP1_P_OP2':   {'eq': 119.90, 'k': 140.0, 'unit': 'deg'},
    'angle_C5_O5_P':     {'eq': 120.50, 'k': 100.0, 'unit': 'deg'},
    'angle_C5_C4_O4':    {'eq': 109.50, 'k': 50.0,  'unit': 'deg'},
    'angle_C5_C4_C3':    {'eq': 109.50, 'k': 40.0,  'unit': 'deg'},
    'angle_O4_C4_C3':    {'eq': 109.50, 'k': 50.0,  'unit': 'deg'},
    'angle_C4_C3_O3':    {'eq': 109.50, 'k': 50.0,  'unit': 'deg'},
    'angle_C4_C3_C2':    {'eq': 109.50, 'k': 40.0,  'unit': 'deg'},
    'angle_O3_C3_C2':    {'eq': 109.50, 'k': 50.0,  'unit': 'deg'},
    'angle_C3_C2_C1':    {'eq': 109.50, 'k': 40.0,  'unit': 'deg'},
    'angle_C2_C1_O4':    {'eq': 109.50, 'k': 50.0,  'unit': 'deg'},
    'angle_C1_O4_C4':    {'eq': 109.50, 'k': 60.0,  'unit': 'deg'},
    'angle_O4_C1_N':     {'eq': 109.50, 'k': 50.0,  'unit': 'deg'},
    'angle_C2_C1_N':     {'eq': 109.50, 'k': 50.0,  'unit': 'deg'},
    'angle_C3_O3_Pnext': {'eq': 120.50, 'k': 100.0, 'unit': 'deg'},
}


def build_protein_strain(df, min_count=10):
    """Precompute per-residue AMBER and library strain (protein)."""
    phi_bins = np.arange(-180, 190, 10)
    psi_bins = np.arange(-180, 190, 10)
    sub = df.copy()
    sub['pb'] = pd.cut(sub['phi_deg'], phi_bins, labels=False, right=False)
    sub['qb'] = pd.cut(sub['psi_deg'], psi_bins, labels=False, right=False)
    sub = sub.dropna(subset=['pb', 'qb'])
    sub['pb'] = sub['pb'].astype(int); sub['qb'] = sub['qb'].astype(int)

    geo_cols = [c for c in AMBER if c in sub.columns]
    total_a = np.zeros(len(sub))
    total_l = np.zeros(len(sub))
    for col in geo_cols:
        k = AMBER[col]['k']
        conv = np.pi / 180.0 if AMBER[col]['unit'] == 'deg' else 1.0
        eq = AMBER[col]['eq']
        valid = sub[col].notna()
        sv = sub.loc[valid]
        lib_mean = sv.groupby(['res_name', 'pb', 'qb'])[col].transform('mean')
        counts = sv.groupby(['res_name', 'pb', 'qb'])[col].transform('count')
        pooled_mean = sv.groupby(['pb', 'qb'])[col].transform('mean')
        lib_eq = np.where(counts >= min_count, lib_mean, pooled_mean)
        ea = np.zeros(len(sub)); el = np.zeros(len(sub))
        ea[valid.values] = 0.5 * k * ((sv[col] - eq) * conv) ** 2
        el[valid.values] = 0.5 * k * ((sv[col] - lib_eq) * conv) ** 2
        total_a += ea; total_l += el
    sub['_total_a'] = total_a
    sub['_total_l'] = total_l
    return sub


def build_dna_strain(df, min_count=10):
    """Precompute per-nucleotide AMBER and library strain (DNA)."""
    sub = df[df['res_name'].isin(['DA', 'DC', 'DG', 'DT'])].copy()
    sub = sub.dropna(subset=['bi_bii', 'pucker_class'])
    sub = sub[sub['bi_bii'] != 'undefined']
    geo_cols = [c for c in AMBER_DNA if c in sub.columns]

    global_mean = sub[geo_cols].mean()
    total_a = np.zeros(len(sub))
    total_l = np.zeros(len(sub))
    for col in geo_cols:
        k = AMBER_DNA[col]['k']
        conv = np.pi / 180.0 if AMBER_DNA[col]['unit'] == 'deg' else 1.0
        eq = AMBER_DNA[col]['eq']
        valid = sub[col].notna()
        sv = sub.loc[valid]
        lib_mean = sv.groupby(['res_name', 'bi_bii', 'pucker_class'])[col].transform('mean')
        counts = sv.groupby(['res_name', 'bi_bii', 'pucker_class'])[col].transform('count')
        pooled_mean = sv.groupby(['bi_bii', 'pucker_class'])[col].transform('mean')
        lib_eq = np.where(counts >= min_count, lib_mean, pooled_mean)
        ea = np.zeros(len(sub)); el = np.zeros(len(sub))
        ea[valid.values] = 0.5 * k * ((sv[col] - eq) * conv) ** 2
        el[valid.values] = 0.5 * k * ((sv[col] - lib_eq) * conv) ** 2
        total_a += ea; total_l += el
    sub['_total_a'] = total_a
    sub['_total_l'] = total_l
    return sub


def bootstrap_reduction(sub, id_col, n_boot=2000, use_median=False, seed=42):
    """
    Bootstrap by resampling STRUCTURES (not residues) with replacement.
    Returns array of n_boot reduction percentages.
    """
    rng = np.random.default_rng(seed)
    structures = sub[id_col].unique()
    n_struct = len(structures)

    # Pre-group row indices by structure for fast resampling
    groups = sub.groupby(id_col).indices  # dict: pdb_id -> array of row positions

    reductions = np.empty(n_boot)
    a_vals = sub['_total_a'].values
    l_vals = sub['_total_l'].values

    for b in range(n_boot):
        sample_structs = rng.choice(structures, size=n_struct, replace=True)
        idx = np.concatenate([groups[s] for s in sample_structs])
        a = a_vals[idx]; l = l_vals[idx]
        if use_median:
            am, lm = np.median(a), np.median(l)
        else:
            am, lm = np.mean(a), np.mean(l)
        reductions[b] = 100 * (am - lm) / am

    return reductions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--mode', choices=['protein', 'dna'], required=True)
    ap.add_argument('--out', default='./bootstrap')
    ap.add_argument('--n_boot', type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"  {len(df):,} rows")

    if args.mode == 'protein':
        sub = build_protein_strain(df)
        id_col = 'pdb_id'
        use_median = False
        label = "Protein vs AMBER ff14SB (mean)"
    else:
        sub = build_dna_strain(df)
        id_col = 'pdb_id'
        use_median = True
        label = "DNA vs AMBER parm10 (median)"

    print(f"Bootstrapping ({args.n_boot} resamples, resampling by {id_col})...")
    reductions = bootstrap_reduction(sub, id_col, n_boot=args.n_boot, use_median=use_median)

    point_estimate = reductions.mean()  # bootstrap mean as center (close to point estimate)
    ci_lo, ci_hi = np.percentile(reductions, [2.5, 97.5])

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Point estimate (bootstrap mean): {point_estimate:.1f}%")
    print(f"  95% CI: [{ci_lo:.1f}%, {ci_hi:.1f}%]")
    print(f"  Bootstrap std: {reductions.std():.2f}")
    print(f"{'='*60}")

    np.save(os.path.join(args.out, f'bootstrap_{args.mode}.npy'), reductions)
    print(f"Saved raw bootstrap distribution to {args.out}/bootstrap_{args.mode}.npy")


if __name__ == '__main__':
    main()
