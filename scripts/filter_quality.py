#!/usr/bin/env python3
"""
filter_quality.py — Post-collection quality filter for backbone features
========================================================================
Applies per-residue quality filters to the raw features CSV produced by
paper2_features_collector.py, matching the original Paper 1 dataset.

Filters applied:
  1. B-factor sliding window mean <= threshold (default 30)
     Window of ±N residues (default N=3, i.e. 7-residue window)
     Applied per chain (resets at chain breaks)
  2. phi/psi must both be defined (not NaN)
  3. Residue must be standard amino acid (in the 20 canonical)
  4. Optional: minimum residues per chain after filtering

Usage:
  python filter_quality.py --input features_raw.csv --output features.csv
  python filter_quality.py --input features_raw.csv --output features.csv \\
      --bfactor 30 --window 3 --min_chain_len 10

Author: Wei (Cvek Lab, LSUHSC)
"""

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path

_STANDARD_AA = {
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'
}

def parse_args():
    ap = argparse.ArgumentParser(
        description='Quality-filter backbone features CSV')
    ap.add_argument('--input',  required=True, help='Raw features CSV')
    ap.add_argument('--output', required=True, help='Filtered output CSV')
    ap.add_argument('--bfactor', type=float, default=30.0,
                    help='B-factor sliding window threshold (default: 30)')
    ap.add_argument('--window',  type=int,   default=3,
                    help='Half-width of B-factor sliding window (default: 3)')
    ap.add_argument('--min_chain_len', type=int, default=10,
                    help='Min residues per chain after filtering (default: 10)')
    return ap.parse_args()


def sliding_bfactor_ok(bf_values, threshold, half_win):
    """
    Returns boolean mask: True where the sliding window mean B-factor
    is <= threshold. Window is ±half_win residues (clamped at ends).
    """
    n = len(bf_values)
    ok = np.ones(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - half_win)
        hi = min(n, i + half_win + 1)
        window_vals = [v for v in bf_values[lo:hi] if not np.isnan(v)]
        if window_vals:
            ok[i] = np.mean(window_vals) <= threshold
        else:
            ok[i] = False
    return ok


def main():
    args = parse_args()
    t0 = time.time()

    print("=" * 70)
    print("filter_quality.py")
    print("=" * 70)
    print(f"  Input    : {args.input}")
    print(f"  Output   : {args.output}")
    print(f"  B-factor : <= {args.bfactor}  (window ±{args.window})")

    print("\n[1] Loading CSV...")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"  {len(df):,} rows loaded")

    # ── Filter 1: standard amino acid ────────────────────────────────────────
    mask_aa = df['res_name'].isin(_STANDARD_AA)
    print(f"\n[2] Standard AA filter: {mask_aa.sum():,} / {len(df):,} pass")
    df = df[mask_aa].copy()

    # ── Filter 2: phi and psi defined ────────────────────────────────────────
    mask_angles = df['phi_deg'].notna() & df['psi_deg'].notna()
    print(f"[3] phi/psi defined   : {mask_angles.sum():,} / {len(df):,} pass")
    df = df[mask_angles].copy()

    # ── Filter 3: B-factor sliding window ────────────────────────────────────
    print(f"[4] B-factor filter   : window ±{args.window}, threshold {args.bfactor}...")

    # Group by pdb_id + chain, apply sliding window within each chain
    bf_ok_indices = []
    group_cols = ['pdb_id', 'chain'] if 'chain' in df.columns else ['pdb_id']

    for _, grp in df.groupby(group_cols, sort=False):
        bf_vals = grp['bfactor_ca'].fillna(0.0).values
        ok_mask = sliding_bfactor_ok(bf_vals, args.bfactor, args.window)
        bf_ok_indices.extend(grp.index[ok_mask].tolist())

    df = df.loc[bf_ok_indices].copy()
    print(f"  After B-factor filter: {len(df):,} residues")

    # ── Filter 4: minimum chain length ───────────────────────────────────────
    print(f"[5] Min chain length  : >= {args.min_chain_len} residues per chain...")
    before = len(df)
    chain_counts = df.groupby(group_cols).size()
    valid_chains = chain_counts[chain_counts >= args.min_chain_len].index

    if len(group_cols) == 2:
        df = df[df.set_index(group_cols).index.isin(valid_chains)].copy()
    else:
        df = df[df['pdb_id'].isin(valid_chains)].copy()

    print(f"  After chain filter : {len(df):,} residues "
          f"(removed {before - len(df):,})")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_chains = df.groupby(group_cols).ngroups
    n_pdbs   = df['pdb_id'].nunique()

    print(f"\n[6] Writing output...")
    df.to_csv(args.output, index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Residues : {len(df):,}")
    print(f"  Chains   : {n_chains:,}")
    print(f"  PDBs     : {n_pdbs:,}")
    print(f"  Written  : {args.output}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
