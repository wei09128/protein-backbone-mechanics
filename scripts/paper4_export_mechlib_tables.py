#!/usr/bin/env python3
"""
export_mechlib_tables.py — Export the final MechLib correction libraries
============================================================================
Produces the actual downloadable MechLib-Protein and MechLib-DNA
correction tables: (residue/base, phi_bin/BI-BII, psi_bin/pucker) ->
corrected local equilibrium value for every backbone bond/angle.

This is the actual "library" deliverable implied by the paper's title --
distinct from the analysis scripts, which only compute strain-reduction
statistics using this library internally without ever writing it out.

Usage:
  python export_mechlib_tables.py --protein_csv features_lj_FINAL_CLEAN.csv \\
      --dna_csv dna_features_v3_clean.csv --out ./mechlib_tables/
"""

import argparse
import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

PROTEIN_COLS = [
    'tau_deg', 'angle_N_CA_CB', 'angle_C_CA_CB', 'angle_CaCN', 'angle_CNCa',
    'angle_CA_C_O', 'bond_N_CA', 'bond_CA_C', 'bond_C_O', 'bond_C_N_next',
    'bond_CA_CB',
]

DNA_COLS = [
    'bond_P_O5', 'bond_O5_C5', 'bond_C5_C4', 'bond_C4_O4', 'bond_C4_C3',
    'bond_C3_O3', 'bond_C3_C2', 'bond_C2_C1', 'bond_C1_O4', 'bond_C1_N',
    'bond_O3_Pnext', 'angle_O5_P_O3prev', 'angle_OP1_P_OP2', 'angle_C5_O5_P',
    'angle_C5_C4_O4', 'angle_C5_C4_C3', 'angle_O4_C4_C3', 'angle_C4_C3_O3',
    'angle_C4_C3_C2', 'angle_O3_C3_C2', 'angle_C3_C2_C1', 'angle_C2_C1_O4',
    'angle_C1_O4_C4', 'angle_O4_C1_N', 'angle_C2_C1_N', 'angle_C3_O3_Pnext',
]


def export_protein_library(csv_path, out_path, bin_size=10, min_count=10):
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)

    phi_bins = np.arange(-180, 180 + bin_size, bin_size)
    psi_bins = np.arange(-180, 180 + bin_size, bin_size)
    phi_centers = phi_bins[:-1] + bin_size / 2
    psi_centers = psi_bins[:-1] + bin_size / 2

    sub = df.copy()
    sub['pb'] = pd.cut(sub['phi_deg'], phi_bins, labels=False, right=False)
    sub['qb'] = pd.cut(sub['psi_deg'], psi_bins, labels=False, right=False)
    sub = sub.dropna(subset=['pb', 'qb'])
    sub['pb'] = sub['pb'].astype(int)
    sub['qb'] = sub['qb'].astype(int)

    geo_cols = [c for c in PROTEIN_COLS if c in sub.columns]
    rows = []
    for (res, pb, qb), grp in sub.groupby(['res_name', 'pb', 'qb']):
        if len(grp) < min_count:
            continue
        row = {
            'res_name': res,
            'phi_bin_center': phi_centers[pb],
            'psi_bin_center': psi_centers[qb],
            'n_residues': len(grp),
        }
        for col in geo_cols:
            vals = grp[col].dropna()
            row[col] = round(vals.mean(), 4) if len(vals) else np.nan
        rows.append(row)

    lib_df = pd.DataFrame(rows)
    lib_df.to_csv(out_path, index=False)
    print(f"  Protein library: {len(lib_df)} (residue, phi, psi) entries -> {out_path}")
    return lib_df


def export_dna_library(csv_path, out_path, min_count=10):
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df['res_name'].isin(['DA', 'DC', 'DG', 'DT'])].copy()
    df = df.dropna(subset=['bi_bii', 'pucker_class'])
    df = df[df['bi_bii'] != 'undefined']

    geo_cols = [c for c in DNA_COLS if c in df.columns]
    rows = []
    for (res, bibii, pucker), grp in df.groupby(['res_name', 'bi_bii', 'pucker_class']):
        if len(grp) < min_count:
            continue
        row = {
            'res_name': res,
            'bi_bii': bibii,
            'pucker_class': pucker,
            'n_nucleotides': len(grp),
        }
        for col in geo_cols:
            vals = grp[col].dropna()
            row[col] = round(vals.mean(), 4) if len(vals) else np.nan
        rows.append(row)

    lib_df = pd.DataFrame(rows)
    lib_df.to_csv(out_path, index=False)
    print(f"  DNA library: {len(lib_df)} (base, BI/BII, pucker) entries -> {out_path}")
    return lib_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protein_csv', default=None)
    ap.add_argument('--dna_csv', default=None)
    ap.add_argument('--out', default='./mechlib_tables')
    ap.add_argument('--min_count', type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.protein_csv:
        export_protein_library(
            args.protein_csv,
            os.path.join(args.out, 'MechLib_protein_library.csv'),
            min_count=args.min_count)

    if args.dna_csv:
        export_dna_library(
            args.dna_csv,
            os.path.join(args.out, 'MechLib_DNA_library.csv'),
            min_count=args.min_count)

    print("\nDone. These CSVs are the actual usable MechLib correction tables:")
    print("  - Look up by (res_name, phi_bin_center, psi_bin_center) for protein")
    print("  - Look up by (res_name, bi_bii, pucker_class) for DNA")
    print("  - Fall back to a pooled/global mean (see analysis scripts) for any")
    print("    combination not present in these tables (insufficient data).")


if __name__ == '__main__':
    main()
