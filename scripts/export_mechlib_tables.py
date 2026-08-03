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
import json
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

CHI1_COLS = ['angle_N_CA_CB', 'angle_C_CA_CB']

# Must exactly match backbone_geometry_library.py's _KEY_MAP -- these are the
# suffixed keys GeometryLibrary._get_value() looks for inside each cell.
JSON_KEY_MAP = {
    'tau_deg':        'tau_deg_eq',
    'angle_N_CA_CB':  'angle_N_CA_CB_eq',
    'angle_C_CA_CB':  'angle_C_CA_CB_eq',
    'angle_CaCN':     'angle_CaCN_eq',
    'angle_CNCa':     'angle_CNCa_eq',
    'angle_CA_C_O':   'angle_CA_C_O_eq',
    'bond_N_CA':      'bond_N_CA_eq',
    'bond_CA_C':      'bond_CA_C_eq',
    'bond_C_O':       'bond_C_O_eq',
    'bond_C_N_next':  'bond_C_N_next_eq',
    'bond_CA_CB':     'bond_CA_CB_eq',
}


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


def protein_csv_to_json(csv_path, json_path, min_count_all=30):
    """Convert the flat protein-library CSV into the nested JSON that
    GeometryLibrary actually loads: {res_name: {phi_bin: {psi_bin: {param_eq: val}}}}.

    Also synthesizes the 'ALL' pooled fallback (residue-agnostic mean per
    phi/psi bin) that GeometryLibrary._lookup_cell falls back to when a
    specific residue's cell is missing -- the flat CSV alone has no such
    entry, so without this conversion every under-populated (residue, bin)
    combination would silently fall straight through to hardcoded AMBER
    defaults instead of the pooled library value.
    """
    print(f"Converting {csv_path} -> {json_path} ...")
    df = pd.read_csv(csv_path)
    lib = {}

    # per-residue cells
    for _, row in df.iterrows():
        res = row['res_name']
        pk = str(int(round(row['phi_bin_center'])))
        qk = str(int(round(row['psi_bin_center'])))
        cell = {}
        for col, jkey in JSON_KEY_MAP.items():
            if col in row and pd.notna(row[col]):
                cell[jkey] = float(row[col])
        lib.setdefault(res, {}).setdefault(pk, {})[qk] = cell

    # pooled 'ALL' fallback, weighted by n_residues per (phi_bin, psi_bin)
    all_cells = {}
    for (pb, qb), grp in df.groupby(['phi_bin_center', 'psi_bin_center']):
        n_total = grp['n_residues'].sum()
        if n_total < min_count_all:
            continue
        pk, qk = str(int(round(pb))), str(int(round(qb)))
        cell = {}
        for col, jkey in JSON_KEY_MAP.items():
            if col in grp.columns:
                w = grp['n_residues']
                vals = grp[col]
                mask = vals.notna()
                if mask.sum():
                    cell[jkey] = float(np.average(vals[mask], weights=w[mask]))
        all_cells.setdefault(pk, {})[qk] = cell
    lib['ALL'] = all_cells

    with open(json_path, 'w') as f:
        json.dump(lib, f, indent=1)
    n_res_cells = sum(len(v) for k, v in lib.items() if k != 'ALL')
    print(f"  Wrote {json_path}: {len(lib)-1} residues, "
          f"{n_res_cells} phi-bin groups, "
          f"{len(lib['ALL'])} pooled ALL phi-bin groups")
    return lib


def export_chi1_library(csv_path, out_csv, out_json, bin_size=20, min_count=10):
    """Export the chi1-dependent Cbeta-angle correction table (Paper III
    finding: chi1 sets *direction*, not just amplitude, of N-Ca-Cb / C-Ca-Cb
    deformation). Deliberately coarser phi/psi binning than the main
    protein library (default 20 deg vs 10 deg) -- the extra rotamer
    dimension (g-/t/g+) thins out cell populations fast, and GeometryLibrary
    .get_chi1_correction() has no bin-level fallback (only returns None if
    the exact cell is missing, safely skipping the chi1 refinement), so
    coarser bins matter more here than a lower min_count would.

    Expects columns: res_name, phi_deg, psi_deg, chi1_rotamer (g-/t/g+),
    angle_N_CA_CB, angle_C_CA_CB. Adjust the column names below if your
    chi1 feature table differs.
    """
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)

    bins = np.arange(-180, 180 + bin_size, bin_size)
    centers = bins[:-1] + bin_size / 2

    df = df.dropna(subset=['phi_deg', 'psi_deg', 'chi1_rotamer'])
    df['pb'] = pd.cut(df['phi_deg'], bins, labels=False, right=False)
    df['qb'] = pd.cut(df['psi_deg'], bins, labels=False, right=False)
    df = df.dropna(subset=['pb', 'qb'])
    df['pb'] = df['pb'].astype(int)
    df['qb'] = df['qb'].astype(int)

    geo_cols = [c for c in CHI1_COLS if c in df.columns]
    rows = []
    for (res, pb, qb, rot), grp in df.groupby(['res_name', 'pb', 'qb', 'chi1_rotamer']):
        if len(grp) < min_count:
            continue
        row = {
            'res_name': res,
            'phi_bin_center': centers[pb],
            'psi_bin_center': centers[qb],
            'chi1_rotamer': rot,
            'n_residues': len(grp),
        }
        for col in geo_cols:
            vals = grp[col].dropna()
            row[col] = round(vals.mean(), 4) if len(vals) else np.nan
        rows.append(row)

    lib_df = pd.DataFrame(rows)
    lib_df.to_csv(out_csv, index=False)
    print(f"  Chi1 library CSV: {len(lib_df)} (residue, phi, psi, rotamer) "
          f"entries -> {out_csv}  [bin_size={bin_size} deg]")

    # nested JSON: res -> phi_bin -> psi_bin -> rotamer -> {angle_..._eq}
    chi1_lib = {}
    for _, row in lib_df.iterrows():
        res = row['res_name']
        pk = str(int(round(row['phi_bin_center'])))
        qk = str(int(round(row['psi_bin_center'])))
        rot = row['chi1_rotamer']
        cell = {}
        if 'angle_N_CA_CB' in row and pd.notna(row['angle_N_CA_CB']):
            cell['angle_N_CA_CB_eq'] = float(row['angle_N_CA_CB'])
        if 'angle_C_CA_CB' in row and pd.notna(row['angle_C_CA_CB']):
            cell['angle_C_CA_CB_eq'] = float(row['angle_C_CA_CB'])
        chi1_lib.setdefault(res, {}).setdefault(pk, {}).setdefault(qk, {})[rot] = cell

    with open(out_json, 'w') as f:
        json.dump(chi1_lib, f, indent=1)
    print(f"  Chi1 library JSON -> {out_json}")
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
    ap.add_argument('--chi1_csv', default=None,
                     help='Feature table with res_name, phi_deg, psi_deg, '
                          'chi1_rotamer, angle_N_CA_CB, angle_C_CA_CB (Paper III)')
    ap.add_argument('--out', default='./mechlib_tables')
    ap.add_argument('--min_count', type=int, default=10)
    ap.add_argument('--chi1_bin_size', type=int, default=20,
                     help='Coarser than the main library default (10 deg) -- '
                          'the rotamer split thins out cells fast')
    ap.add_argument('--no_json', action='store_true',
                     help='Skip writing the nested JSON GeometryLibrary consumes '
                          '(CSV only)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.protein_csv:
        export_protein_library(
            args.protein_csv,
            os.path.join(args.out, 'MechLib_protein_library.csv'),
            min_count=args.min_count)
        if not args.no_json:
            protein_csv_to_json(
                os.path.join(args.out, 'MechLib_protein_library.csv'),
                os.path.join(args.out, 'constants_library.json'))

    if args.chi1_csv:
        export_chi1_library(
            args.chi1_csv,
            os.path.join(args.out, 'MechLib_protein_chi1_library.csv'),
            os.path.join(args.out, 'constants_chi1.json'),
            bin_size=args.chi1_bin_size,
            min_count=args.min_count)

    if args.dna_csv:
        export_dna_library(
            args.dna_csv,
            os.path.join(args.out, 'MechLib_DNA_library.csv'),
            min_count=args.min_count)

    print("\nDone. Outputs in", args.out + ":")
    print("  - MechLib_protein_library.csv   (flat, human-readable)")
    print("  - constants_library.json        (nested, what GeometryLibrary loads)")
    print("  - MechLib_protein_chi1_library.csv / constants_chi1.json  (if --chi1_csv given)")
    print("  - MechLib_DNA_library.csv        (if --dna_csv given)")
    print("  constants_library.json and constants_chi1.json must sit next to")
    print("  backbone_geometry_library.py, or be passed explicitly via")
    print("  library_path= / chi1_path=.")


if __name__ == '__main__':
    main()
