#!/usr/bin/env python3
"""
refmac_only_sensitivity.py — Confound-minimized strain reduction estimate
================================================================================
Recomputes MechLib's headline strain-reduction result restricted to
REFMAC-refined structures only (the largest non-PHENIX category, and REFMAC
does not implement Berkholz CDL-based conformation-dependent restraints).
This serves as a sensitivity check reported alongside the full-dataset
38.7% figure: if the REFMAC-only estimate remains substantial and
statistically robust, that argues the effect is not primarily an artifact
of CDL-based refinement restraints in PHENIX/TNT-refined structures.

Reports:
  1. Overall REFMAC-only reduction (matches Fig 1 overall stat)
  2. By secondary-structure class (matches Fig 1A)
  3. By residue type (matches Fig 1C)

Usage:
  python refmac_only_sensitivity.py \
      --csv features_lj_FINAL_CLEAN.csv --pdb_dir ./pdb_cache --n_boot 2000
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

AMBER = {
    'tau_deg':        {'eq': 110.10, 'k': 63.0,  'unit': 'deg'},
    'angle_N_CA_CB':  {'eq': 109.70, 'k': 63.0,  'unit': 'deg'},
    'angle_C_CA_CB':  {'eq': 111.10, 'k': 63.0,  'unit': 'deg'},
    'angle_CaCN':     {'eq': 116.6,  'k': 70.0,  'unit': 'deg'},
    'angle_CNCa':     {'eq': 121.9,  'k': 50.0,  'unit': 'deg'},
    'angle_CA_C_O':   {'eq': 120.4,  'k': 80.0,  'unit': 'deg'},
    'bond_N_CA':      {'eq': 1.449,  'k': 337.0, 'unit': 'A'},
    'bond_CA_C':      {'eq': 1.522,  'k': 317.0, 'unit': 'A'},
    'bond_C_O':       {'eq': 1.229,  'k': 570.0, 'unit': 'A'},
    'bond_C_N_next':  {'eq': 1.335,  'k': 490.0, 'unit': 'A'},
    'bond_CA_CB':     {'eq': 1.526,  'k': 317.0, 'unit': 'A'},
}


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
    if 'CNS' in text or 'X-PLOR' in text or 'XPLOR' in text:
        return 'CNS/X-PLOR'
    if 'BUSTER' in text:
        return 'BUSTER'
    if 'TNT' in text:
        return 'TNT'
    return 'OTHER'


def assign_ss(phi, psi):
    # PPII checked BEFORE beta -- PPII's range (phi -100..-40, psi 100..180)
    # is a strict subset of beta's range (phi -180..-40, psi>=80), so beta's
    # check must come second or it silently absorbs every PPII residue.
    if -100 <= phi <= -30 and -80 <= psi <= 10:
        return 'alphaR'
    if -100 <= phi <= -40 and 100 <= psi <= 180:
        return 'PPII'
    if -180 <= phi <= -40 and (psi >= 80 or psi <= -170):
        return 'beta'
    if 20 <= phi <= 100 and -20 <= psi <= 100:
        return 'alphaL'
    return 'coil'


def bootstrap_reduction(a_vals, l_vals, pdb_ids, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'a': a_vals, 'l': l_vals, 'pdb_id': pdb_ids})
    groups_a = df.groupby('pdb_id')['a'].apply(list).to_dict()
    groups_l = df.groupby('pdb_id')['l'].apply(list).to_dict()
    structs = df['pdb_id'].unique()
    n_struct = len(structs)
    reductions = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(structs, size=n_struct, replace=True)
        a = np.concatenate([groups_a[s] for s in sample])
        l = np.concatenate([groups_l[s] for s in sample])
        reductions[b] = 100 * (a.mean() - l.mean()) / a.mean()
    return reductions


def report_reduction(label, group, n_boot):
    n_struct = group['pdb_id'].nunique()
    if n_struct < 20 or len(group) < 50:
        print(f"  {label:20s}: n_structures={n_struct} n_residues={len(group)} "
              f"(too few, skipping)")
        return
    reductions = bootstrap_reduction(
        group['_total_a'].values, group['_total_l'].values,
        group['pdb_id'].values, n_boot)
    pe = reductions.mean()
    ci_lo, ci_hi = np.percentile(reductions, [2.5, 97.5])
    print(f"  {label:20s}: n_structures={n_struct:5d}  n_residues={len(group):8,}  "
          f"reduction={pe:5.1f}%  95% CI=[{ci_lo:.1f}%, {ci_hi:.1f}%]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--pdb_dir', required=True)
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--min_count', type=int, default=10)
    args = ap.parse_args()

    print("Loading data...")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"  {len(df):,} residues, {df['pdb_id'].nunique()} structures")

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
    df['ss_class'] = [assign_ss(p, q) for p, q in zip(df['phi_deg'], df['psi_deg'])]

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
        lib_eq = np.where(counts >= args.min_count, lib_mean, pooled_mean)
        ea = np.zeros(len(sub)); el = np.zeros(len(sub))
        ea[valid.values] = 0.5 * k * ((sv[col] - eq) * conv) ** 2
        el[valid.values] = 0.5 * k * ((sv[col] - lib_eq) * conv) ** 2
        total_a += ea; total_l += el
    sub['_total_a'] = total_a
    sub['_total_l'] = total_l

    refmac = sub[sub['refinement_sw'] == 'REFMAC']

    print(f"\n{'='*70}")
    print("  1. OVERALL: REFMAC-only vs full dataset (both computed identically)")
    print(f"  ({args.n_boot} bootstrap resamples, resampled by structure)")
    print(f"{'='*70}")
    report_reduction('Full dataset', sub, args.n_boot)
    report_reduction('REFMAC only', refmac, args.n_boot)

    print(f"\n{'='*70}")
    print("  2. REFMAC-only, BY SECONDARY STRUCTURE (cf. Fig 1A)")
    print(f"{'='*70}")
    for ss in ['alphaR', 'beta', 'PPII', 'alphaL', 'coil']:
        report_reduction(ss, refmac[refmac['ss_class'] == ss], args.n_boot)

    print(f"\n{'='*70}")
    print("  3. REFMAC-only, BY RESIDUE TYPE (cf. Fig 1C)")
    print(f"{'='*70}")
    res_counts = refmac['res_name'].value_counts()
    for res in res_counts.index:
        report_reduction(res, refmac[refmac['res_name'] == res], args.n_boot)

    print(f"\n{'='*70}")
    print("Use the OVERALL REFMAC-only number as the confound-minimized")
    print("sensitivity estimate to report alongside the full-dataset 38.7%.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
