#!/usr/bin/env python3
"""
blind_validation.py — Temporally blind train/test validation for MechLib
=============================================================================
Builds the MechLib correction library using ONLY structures released before
a cutoff date, then evaluates strain reduction on structures released AFTER
that date -- which the library has never seen. This directly answers the
"of course local means fit local data" overfitting objection: if MechLib's
correction still reduces strain on genuinely unseen future depositions, the
library is capturing a real, generalizable geometric regularity rather than
memorizing training-set noise.

Usage:
  python blind_validation.py --csv features_lj_FINAL_CLEAN.csv \\
      --dates release_dates.csv --cutoff 2020-01-01 --out ./blind_validation/
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

AMBER = {
    'tau_deg':        {'eq': 110.10, 'k': 63.0,  'unit': 'deg'},
    'angle_N_CA_CB':  {'eq': 109.70, 'k': 80.0,  'unit': 'deg'},
    'angle_C_CA_CB':  {'eq': 111.10, 'k': 63.0,  'unit': 'deg'},
    'angle_CaCN':     {'eq': 116.6, 'k': 70.0,  'unit': 'deg'},
    'angle_CNCa':     {'eq': 121.9, 'k': 50.0,  'unit': 'deg'},
    'angle_CA_C_O':   {'eq': 120.4, 'k': 80.0,  'unit': 'deg'},
    'bond_N_CA':      {'eq': 1.449,  'k': 337.0, 'unit': 'Å'},
    'bond_CA_C':      {'eq': 1.522, 'k': 317.0, 'unit': 'Å'},
    'bond_C_O':       {'eq': 1.229, 'k': 570.0, 'unit': 'Å'},
    'bond_C_N_next':  {'eq': 1.335, 'k': 490.0, 'unit': 'Å'},
    'bond_CA_CB':     {'eq': 1.526, 'k': 317.0, 'unit': 'Å'},
}


def bin_phi_psi(df):
    phi_bins = np.arange(-180, 190, 10)
    psi_bins = np.arange(-180, 190, 10)
    sub = df.copy()
    sub['pb'] = pd.cut(sub['phi_deg'], phi_bins, labels=False, right=False)
    sub['qb'] = pd.cut(sub['psi_deg'], psi_bins, labels=False, right=False)
    sub = sub.dropna(subset=['pb', 'qb'])
    sub['pb'] = sub['pb'].astype(int); sub['qb'] = sub['qb'].astype(int)
    return sub


def build_library_from_train(train, geo_cols, min_count=10):
    """Build (res_name, pb, qb) -> mean lookup dict from TRAIN data only."""
    lib = {}
    pooled = {}
    global_mean = train[geo_cols].mean()

    for (res, pb, qb), grp in train.groupby(['res_name', 'pb', 'qb']):
        if len(grp) >= min_count:
            lib[(res, pb, qb)] = grp[geo_cols].mean()
    for (pb, qb), grp in train.groupby(['pb', 'qb']):
        pooled[(pb, qb)] = grp[geo_cols].mean()
    return lib, pooled, global_mean


def lookup(lib, pooled, global_mean, res, pb, qb, col):
    key = (res, pb, qb)
    if key in lib and not np.isnan(lib[key][col]):
        return lib[key][col]
    pkey = (pb, qb)
    if pkey in pooled and not np.isnan(pooled[pkey][col]):
        return pooled[pkey][col]
    return global_mean[col]


def compute_strain(data, geo_cols, eq_lookup_fn=None, fixed=False):
    """
    Compute total strain per row. If fixed=True, uses fixed AMBER eq.
    Otherwise uses eq_lookup_fn(res, pb, qb, col) for the local library value.
    """
    total = np.zeros(len(data))
    for col in geo_cols:
        k = AMBER[col]['k']
        conv = np.pi / 180.0 if AMBER[col]['unit'] == 'deg' else 1.0
        obs = data[col].values
        valid = ~np.isnan(obs)

        if fixed:
            eq = np.full(len(data), AMBER[col]['eq'])
        else:
            eq = np.array([
                eq_lookup_fn(data['res_name'].iloc[i], data['pb'].iloc[i],
                             data['qb'].iloc[i], col)
                if valid[i] else AMBER[col]['eq']
                for i in range(len(data))
            ])

        e = np.zeros(len(data))
        e[valid] = 0.5 * k * ((obs[valid] - eq[valid]) * conv) ** 2
        total += e
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--dates', required=True)
    ap.add_argument('--cutoff', default='2020-01-01')
    ap.add_argument('--out', default='./blind_validation')
    ap.add_argument('--min_count', type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(args.csv, low_memory=False)
    dates = pd.read_csv(args.dates)
    dates['release_date'] = pd.to_datetime(dates['release_date'])
    cutoff = pd.Timestamp(args.cutoff)

    df = df.merge(dates, on='pdb_id', how='left')
    print(f"  {len(df):,} rows, {df['pdb_id'].nunique()} structures")
    print(f"  {df['release_date'].isna().sum()} rows missing release date (excluded)")
    df = df.dropna(subset=['release_date'])

    sub = bin_phi_psi(df)
    geo_cols = [c for c in AMBER if c in sub.columns]

    train = sub[sub['release_date'] < cutoff].copy()
    test = sub[sub['release_date'] >= cutoff].copy()
    print(f"\n  TRAIN: {train['pdb_id'].nunique()} structures, {len(train):,} residues "
          f"(released before {args.cutoff})")
    print(f"  TEST : {test['pdb_id'].nunique()} structures, {len(test):,} residues "
          f"(released on/after {args.cutoff}, NEVER seen by the library)")

    print("\nBuilding MechLib library from TRAIN only...")
    lib, pooled, global_mean = build_library_from_train(train, geo_cols, args.min_count)
    print(f"  {len(lib)} (res,phi,psi) bins built from training data")

    eq_fn = lambda res, pb, qb, col: lookup(lib, pooled, global_mean, res, pb, qb, col)

    print("\nEvaluating on held-out TEST structures...")
    test_amber = compute_strain(test, geo_cols, fixed=True)
    test_lib = compute_strain(test, geo_cols, eq_lookup_fn=eq_fn, fixed=False)

    mean_a, mean_l = test_amber.mean(), test_lib.mean()
    reduction = 100 * (mean_a - mean_l) / mean_a

    print(f"\n{'='*60}")
    print(f"  BLIND TEST RESULT (structures never seen during library construction)")
    print(f"  Mean AMBER strain (test) : {mean_a:.5f}")
    print(f"  Mean Library strain (test): {mean_l:.5f}")
    print(f"  Reduction on unseen structures: {reduction:.1f}%")
    print(f"{'='*60}")

    # For comparison: in-sample reduction on TRAIN itself
    train_amber = compute_strain(train, geo_cols, fixed=True)
    train_lib = compute_strain(train, geo_cols, eq_lookup_fn=eq_fn, fixed=False)
    train_reduction = 100 * (train_amber.mean() - train_lib.mean()) / train_amber.mean()
    print(f"\n  (For reference, in-sample reduction on TRAIN: {train_reduction:.1f}%)")

    with open(os.path.join(args.out, 'blind_validation_summary.txt'), 'w') as f:
        f.write(f"Cutoff date: {args.cutoff}\n")
        f.write(f"Train: {train['pdb_id'].nunique()} structures, {len(train)} residues\n")
        f.write(f"Test: {test['pdb_id'].nunique()} structures, {len(test)} residues\n")
        f.write(f"In-sample (train) reduction: {train_reduction:.1f}%\n")
        f.write(f"Blind (test) reduction: {reduction:.1f}%\n")
    print(f"\nSaved summary to {args.out}/blind_validation_summary.txt")


if __name__ == '__main__':
    main()
