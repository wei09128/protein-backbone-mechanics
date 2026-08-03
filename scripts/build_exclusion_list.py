#!/usr/bin/env python3
"""
build_exclusion_list.py — Cross-reference clash candidates with real tau values
==================================================================================
scan_all_close_contacts.py flags ANY atom pair under 1.5 Å, which includes
many genuine, harmless hydrogen bonds (backbone O...H at 1.2-1.5 Å is normal
secondary-structure geometry, not a defect). This script cross-references
those candidates against the ACTUAL computed tau_phi_lj/tau_psi_lj magnitudes
in features_lj.csv, and only flags residues whose real torque values exceed
a physically-motivated threshold — i.e. residues where the close contact
demonstrably DID produce a catastrophic force, not just any residue that
happens to sit near a normal H-bond distance.

Usage:
  python build_exclusion_list.py \\
      --clashes clash_exclusions.csv \\
      --features features_lj.csv \\
      --threshold 10000 \\
      --out final_exclusions.csv
"""

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clashes', required=True)
    ap.add_argument('--features', required=True)
    ap.add_argument('--threshold', type=float, default=10000.0,
                    help='Only flag residues whose |tau_phi_lj| or |tau_psi_lj| '
                         'exceeds this value (default: 10000)')
    ap.add_argument('--out', default='final_exclusions.csv')
    args = ap.parse_args()

    print(f"Loading clash candidates from {args.clashes} ...")
    clashes = pd.read_csv(args.clashes)
    print(f"  {len(clashes)} candidate pairs")

    # Every residue mentioned in the clash list (either side of the pair)
    # is a CANDIDATE — we still need to check its real tau value.
    candidates = pd.concat([
        clashes[['pdb_id', 'res_idx_a']].rename(columns={'res_idx_a': 'res_idx'}),
        clashes[['pdb_id', 'res_idx_b']].rename(columns={'res_idx_b': 'res_idx'}),
    ], ignore_index=True).drop_duplicates()
    print(f"  {len(candidates)} unique (pdb_id, res_idx) candidate residues")

    print(f"\nLoading {args.features} (this may take a moment)...")
    feat_cols = ['pdb_id', 'res_idx', 'res_name', 'tau_phi_lj', 'tau_psi_lj']
    features = pd.read_csv(args.features, usecols=feat_cols)
    print(f"  {len(features):,} total rows")

    # Join candidates against their actual computed tau values
    merged = candidates.merge(features, on=['pdb_id', 'res_idx'], how='left')
    merged['max_abs_tau'] = merged[['tau_phi_lj', 'tau_psi_lj']].abs().max(axis=1)

    print(f"\n{'='*60}")
    print(f"Candidate residues by actual computed |tau| magnitude:")
    print(f"{'='*60}")
    print(merged[['pdb_id', 'res_idx', 'res_name', 'tau_phi_lj', 'tau_psi_lj',
                  'max_abs_tau']].sort_values('max_abs_tau', ascending=False).head(30).to_string(index=False))

    # ── Final exclusion: only residues that CLEAR the threshold ─────────
    final = merged[merged['max_abs_tau'] > args.threshold].copy()
    final = final.sort_values('max_abs_tau', ascending=False)

    print(f"\n{'='*60}")
    print(f"FINAL EXCLUSION SUMMARY (threshold = {args.threshold})")
    print(f"{'='*60}")
    print(f"  Candidates from distance scan : {len(candidates)}")
    print(f"  Confirmed genuine (|tau| > {args.threshold}) : {len(final)}")
    print(f"  Rejected as harmless (likely real H-bonds)   : "
          f"{len(candidates) - len(final)}")
    n_pdbs = final['pdb_id'].nunique()
    print(f"  Unique PDB structures needing exclusion      : {n_pdbs}")

    final[['pdb_id', 'res_idx', 'res_name', 'tau_phi_lj', 'tau_psi_lj']].to_csv(
        args.out, index=False)
    print(f"\nWritten final exclusion list to {args.out}")


if __name__ == '__main__':
    main()
