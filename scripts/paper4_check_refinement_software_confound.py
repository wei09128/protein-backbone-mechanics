#!/usr/bin/env python3
"""
check_refinement_software_confound.py — Tests the CDL-restraint confound
================================================================================
Reviewer concern (raised on a companion manuscript, applies directly here):
Berkholz's conformation-dependent library (CDL) has been implemented as
active refinement restraints in PHENIX and TNT. If a meaningful fraction of
structures in this dataset were refined using CDL-based restraints, their
backbone geometry may already be (phi,psi)-dependent BY CONSTRUCTION of the
refinement process itself -- meaning MechLib's strain reduction could partly
reflect rediscovering the restraint library's own assumption, not an
independent physical property of protein backbones.

This script parses refinement-software identity directly from the REMARK 3
section of each PDB file already in pdb_cache/ (standard PDB format, no new
API calls needed), then stratifies MechLib's strain-reduction result by
refinement software category to test whether PHENIX-refined structures
(where CDL restraints are more likely to have been active, especially in
recent versions) show systematically different reduction than non-PHENIX
structures.

IMPORTANT CAVEATS (read before interpreting results):
  1. This is a rough proxy, not a perfect control: PHENIX refinement does NOT
     guarantee CDL restraints were actually enabled/used for a given
     structure -- CDL restraint usage depends on PHENIX version and specific
     refinement settings, which are not directly recoverable from REMARK 3
     alone in most depositions.
  2. If PHENIX-refined structures show LOWER strain reduction (already close
     to their local conformation-specific equilibrium before MechLib even
     runs), that would be consistent with the confound. If reduction is
     SIMILAR across all refinement software, that argues against the
     restraint library explaining the effect on its own.
  3. This is a necessary first-pass check, not a definitive resolution --
     a fully rigorous test would require actual restraint-library metadata
     per structure, which is not reliably available at this scale.

Usage:
  python check_refinement_software_confound.py \
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
    """Parse REMARK 3 for refinement program name. Returns a normalized
    category string, or 'UNKNOWN' if not found/not parseable."""
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

    print("Classifying refinement software from PDB headers (REMARK 3)...")
    pdb_ids = df['pdb_id'].unique()
    software_map = {}
    for k, pdb_id in enumerate(pdb_ids, 1):
        pdb_path = Path(args.pdb_dir) / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            pdb_path = Path(args.pdb_dir) / f"{pdb_id.lower()}.pdb"
        if not pdb_path.exists():
            software_map[pdb_id] = 'UNKNOWN'
            continue
        software_map[pdb_id] = classify_refinement_software(pdb_path)
        if k % 1000 == 0:
            print(f"  [{k}/{len(pdb_ids)}]")

    df['refinement_sw'] = df['pdb_id'].map(software_map)
    print("\nRefinement software distribution (by structure):")
    sw_counts = pd.Series(software_map).value_counts()
    print(sw_counts.to_string())

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

    print(f"\n{'='*70}")
    print("  STRAIN REDUCTION STRATIFIED BY REFINEMENT SOFTWARE")
    print(f"  ({args.n_boot} bootstrap resamples, resampled by structure)")
    print(f"{'='*70}")
    for sw in sw_counts.index:
        group = sub[sub['refinement_sw'] == sw]
        n_struct = group['pdb_id'].nunique()
        if n_struct < 20:
            print(f"  {sw:12s}: n_structures={n_struct} (too few for reliable bootstrap, skipping)")
            continue
        reductions = bootstrap_reduction(
            group['_total_a'].values, group['_total_l'].values,
            group['pdb_id'].values, args.n_boot)
        pe = reductions.mean()
        ci_lo, ci_hi = np.percentile(reductions, [2.5, 97.5])
        print(f"  {sw:12s}: n_structures={n_struct:5d}  n_residues={len(group):8,}  "
              f"reduction={pe:5.1f}%  95% CI=[{ci_lo:.1f}%, {ci_hi:.1f}%]")

    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print("  If PHENIX shows NOTABLY LOWER reduction than REFMAC/CNS/other,")
    print("  that is consistent with the CDL-restraint confound (PHENIX")
    print("  structures may already be closer to conformation-dependent")
    print("  equilibria before MechLib runs). If reduction is SIMILAR across")
    print("  all software categories, that argues against the restraint")
    print("  library being the primary driver of the observed effect.")
    print("  Remember: PHENIX usage does not guarantee CDL restraints were")
    print("  actually active for any given structure -- this is a proxy,")
    print("  not a definitive test. See script docstring for caveats.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
