#!/usr/bin/env python3
"""
paper2_09_angle_resolution_stratified.py
==========================================

Reviewer 1, Comment 5 asks: do the reported basin-dependent bond-ANGLE
deviations (tau, omega, angle_N_CA_CB, angle_C_CA_CB) survive when the
dataset is progressively restricted to higher-resolution structures, or
do they shrink toward zero / flip sign the way the carbohydrate-structure
precedent did (DOI: 10.1038/nchembio.1798)?

This is a DIFFERENT question from paper2_08_bond_length_scoping.py /
paper2_08_resolution_stratified.py, which stratify BOND LENGTHS
(N-Ca, Ca-C, C=O) for Paper 3. This script stratifies the four Paper 2
bond ANGLES by resolution and checks sign + magnitude stability across
basins -- the actual analysis Comment 5's response describes.

Two ways to supply per-structure resolution:
  1. --pdb_dir <dir>            : parse REMARK 2 from each PDB header
                                   (slow-ish, but self-contained)
  2. --resolution_csv <path>    : reuse an existing pdb_id,resolution CSV,
                                   e.g. pdb_resolutions.csv already
                                   produced by paper2_08_resolution_stratified.py
                                   for Paper 3 -- same metadata, no need
                                   to regenerate it.

Usage
-----
    python paper2_09_angle_resolution_stratified.py \\
        --features features_lj_v3_FINAL_CLEAN.csv \\
        --resolution_csv pdb_resolutions.csv \\
        --out ./resolution_stratified/

    # or, if you don't have pdb_resolutions.csv cached:
    python paper2_09_angle_resolution_stratified.py \\
        --features features_lj_v3_FINAL_CLEAN.csv \\
        --pdb_dir /path/to/pdb_cache \\
        --out ./resolution_stratified/
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Cbeta-dependent angles (angle_N_CA_CB, angle_C_CA_CB) are undefined for
# glycine (no Cbeta) and are conventionally excluded for proline (fixed-ring
# geometry) -- this is the 18-residue set used for Table 1 / Fig. 2.
_RES18_NO_GLY_PRO = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','HIS','ILE','LEU',
                     'LYS','MET','PHE','SER','THR','TRP','TYR','VAL']

# tau and omega do not involve Cbeta at all, so glycine has a perfectly
# well-defined value for both and should NOT be dropped here -- excluding it
# was the bug that made every angle collapse to the same (smaller) n at each
# cutoff. Proline retains its usual exclusion (ring-constrained phi).
_RES19_NO_PRO = _RES18_NO_GLY_PRO + ['GLY']

_REGIONS = {
    'α':    {'phi': (-80, -40),  'psi': (-60, -20)},
    'β':    {'phi': (-150, -90), 'psi': (100, 160)},
    'PPII': {'phi': (-90, -60),  'psi': (120, 160)},
    'αL':   {'phi': (40, 80),    'psi': (20, 80)},
}
_REGION_ORDER = ['α', 'β', 'PPII', 'αL']

# (feature column, is-circular, display label, residue set to use)
_ANGLES = [
    ('tau_deg',            False, 'τ (N-Cα-C)', _RES19_NO_PRO),
    ('omega_measured_deg', True,  'ω (peptide)', _RES19_NO_PRO),
    ('angle_N_CA_CB',      False, '∠N-Cα-Cβ',    _RES18_NO_GLY_PRO),
    ('angle_C_CA_CB',      False, '∠C-Cα-Cβ',    _RES18_NO_GLY_PRO),
]

# Table S2 compares the loosest cutoff (<=3.5A, labeled as such in the
# caption even though it is numerically identical to the full <=2.0A
# dataset -- everything was already collected at <=2.0A) against the
# strictest (<=1.0A). Intermediate cutoffs (1.5, 1.2) aren't part of S2
# and have been dropped; add them back if you want the finer-grained
# supplementary breakdown instead.
LOOSE_CUTOFF = 2.0
LOOSE_LABEL = '<=3.5 Å'
TIGHT_CUTOFF = 1.0
STRENGTHENED_THRESHOLD_PCT = 40.0

MIN_N_PER_CELL = 30  # minimum residues per (angle, basin, cutoff) cell


def _circ_median_deg(v):
    v = np.asarray(v, dtype=float)
    rad = np.radians(v)
    mean = np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad))))
    shifted = ((v - mean + 180) % 360) - 180
    return ((np.median(shifted) + mean + 180) % 360) - 180


def _circ_diff(a, b):
    return (a - b + 180) % 360 - 180


def extract_resolution(pdb_path):
    """Extract resolution (Angstrom) from a PDB file's REMARK 2 header line."""
    try:
        with open(pdb_path) as fh:
            for line in fh:
                if line.startswith(('ATOM', 'HETATM', 'MODEL')):
                    break
                if not line.startswith('REMARK   2'):
                    continue
                upper = line.upper()
                if 'RESOLUTION' not in upper:
                    continue
                if 'NOT APPLICABLE' in upper:
                    return None
                after = upper.split('RESOLUTION')[1]
                m = re.search(r'(\d+\.\d+)', after)
                if m:
                    val = float(m.group(1))
                    if 0.3 <= val <= 15.0:
                        return val
    except Exception:
        pass
    return None


def build_resolution_table(pdb_dir, verbose=False):
    pdb_files = sorted(Path(pdb_dir).glob('*.pdb'))
    records = []
    for i, p in enumerate(pdb_files):
        res = extract_resolution(p)
        if res is not None:
            records.append({'pdb_id': p.stem.lower(), 'resolution': res})
        if verbose and (i + 1) % 500 == 0:
            print(f"  scanned {i+1}/{len(pdb_files)}")
    df = pd.DataFrame(records)
    print(f"  Resolution extracted for {len(df):,}/{len(pdb_files):,} PDBs")
    return df


def prepare_delta(df, angle, circular, residue_set):
    """Per-residue Delta = angle - median(angle | residue type), matching
    the Methods 'Per-residue deviation' convention used throughout Paper 2."""
    d = df.dropna(subset=[angle, 'phi_deg', 'psi_deg', 'res_name']).copy()
    d = d[d['res_name'].isin(residue_set)]
    if circular:
        medians = d.groupby('res_name')[angle].apply(lambda g: _circ_median_deg(g.values))
        d['delta'] = _circ_diff(d[angle].values, d['res_name'].map(medians).values)
    else:
        medians = d.groupby('res_name')[angle].median()
        d['delta'] = d[angle] - d['res_name'].map(medians)
    return d


def region_mean(d, region, circular):
    p, q = _REGIONS[region]['phi'], _REGIONS[region]['psi']
    m = d['phi_deg'].between(p[0], p[1]) & d['psi_deg'].between(q[0], q[1])
    sub = d.loc[m, 'delta'].values
    n = len(sub)
    if n == 0:
        return float('nan'), 0
    if circular:
        rad = np.radians(sub)
        mean = np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad))))
        return mean, n
    return float(np.mean(sub)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True,
                     help='features_lj_v3_FINAL_CLEAN.csv (or equivalent)')
    ap.add_argument('--resolution_csv', default=None,
                     help='Existing pdb_id,resolution CSV (reuse from Paper 3 scripts if available)')
    ap.add_argument('--pdb_dir', default=None,
                     help='Directory of PDB files, used only if --resolution_csv not given')
    ap.add_argument('--out', default='./resolution_stratified')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    if not args.resolution_csv and not args.pdb_dir:
        print("ERROR: supply either --resolution_csv or --pdb_dir")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {args.features} ...")
    feat = pd.read_csv(args.features)
    print(f"  {len(feat):,} rows")
    if 'pdb_id' not in feat.columns:
        print("ERROR: features CSV has no 'pdb_id' column -- cannot merge resolution.")
        sys.exit(1)
    feat['pdb_id'] = feat['pdb_id'].astype(str).str.lower()

    if args.resolution_csv:
        print(f"Loading cached resolution table {args.resolution_csv} ...")
        res_df = pd.read_csv(args.resolution_csv)
        if 'pdb_id' not in res_df.columns or 'resolution' not in res_df.columns:
            print(f"ERROR: {args.resolution_csv} must have 'pdb_id' and "
                  f"'resolution' columns; found: {list(res_df.columns)}")
            sys.exit(1)
    else:
        pdb_dir_path = Path(args.pdb_dir)
        if not pdb_dir_path.is_dir():
            print(f"ERROR: --pdb_dir '{args.pdb_dir}' is not a directory "
                  f"(check the path -- this looks like a placeholder or typo).")
            sys.exit(1)
        n_pdb_files = len(list(pdb_dir_path.glob('*.pdb')))
        if n_pdb_files == 0:
            print(f"ERROR: no *.pdb files found in '{args.pdb_dir}'. "
                  f"Point --pdb_dir at the directory your other scripts "
                  f"(e.g. filter_quality.py) actually read PDB files from.")
            sys.exit(1)
        print(f"Extracting resolution from PDB headers in {args.pdb_dir} "
              f"({n_pdb_files:,} .pdb files found) ...")
        res_df = build_resolution_table(args.pdb_dir, verbose=args.verbose)
        if len(res_df) == 0:
            print("ERROR: found .pdb files but extracted zero resolutions -- "
                  "check that these files have standard 'REMARK   2 RESOLUTION.' "
                  "header lines (e.g. not stripped-down/ATOM-only files).")
            sys.exit(1)
    res_df['pdb_id'] = res_df['pdb_id'].astype(str).str.lower()
    res_df = res_df.drop_duplicates('pdb_id')

    merged = feat.merge(res_df, on='pdb_id', how='inner')
    print(f"  Merged with resolution: {len(merged):,} / {len(feat):,} rows "
          f"({len(merged)/len(feat)*100:.1f}%)")
    n_structs_total = merged['pdb_id'].nunique()
    print(f"  Distinct structures with resolution: {n_structs_total:,}")

    # Only the two Table S2 endpoints are needed: the loose cutoff (labeled
    # <=3.5A in the caption, numerically <=2.0A since that's the full
    # dataset) and the tight cutoff (<=1.0A).
    print("\n" + "=" * 78)
    print("STRUCTURE / RESIDUE COUNTS PER CUTOFF")
    print("=" * 78)
    cutoff_frames = {}
    for cutoff, label in [(LOOSE_CUTOFF, LOOSE_LABEL), (TIGHT_CUTOFF, f'<={TIGHT_CUTOFF:.1f} Å')]:
        sub = merged[merged['resolution'] <= cutoff]
        cutoff_frames[cutoff] = sub
        n_struct = sub['pdb_id'].nunique()
        print(f"  {label} : {n_struct:,} structures, {len(sub):,} residue-rows")

    # ------------------------------------------------------------------
    # Per-angle, per-basin Delta at the loose vs. tight cutoff, using
    # each angle's own residue set (see _ANGLES above) -- this is what
    # produces a different n for tau/omega vs. the Cbeta-coupled angles,
    # matching how Table S2 was originally reported.
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"TABLE S2: {LOOSE_LABEL} vs. <={TIGHT_CUTOFF:.1f} Å, BY ANGLE × BASIN")
    print("=" * 100)

    s2_rows = []
    n_sign_flip = 0

    for col, circ, label, residue_set in _ANGLES:
        d_loose = prepare_delta(cutoff_frames[LOOSE_CUTOFF], col, circ, residue_set)
        d_tight = prepare_delta(cutoff_frames[TIGHT_CUTOFF], col, circ, residue_set)
        print(f"\n{label}  (residue set: {len(residue_set)} residues)")
        for reg in _REGION_ORDER:
            m_loose, n_loose = region_mean(d_loose, reg, circ)
            m_tight, n_tight = region_mean(d_tight, reg, circ)

            if n_loose < MIN_N_PER_CELL or n_tight < MIN_N_PER_CELL \
                    or not np.isfinite(m_loose) or not np.isfinite(m_tight):
                print(f"  {reg:<6s} skipped (n < {MIN_N_PER_CELL} at one cutoff)")
                continue

            same_sign = (np.sign(m_tight) == np.sign(m_loose)) or abs(m_loose) < 1e-6
            if not same_sign:
                n_sign_flip += 1
            pct_change = (m_tight - m_loose) / m_loose * 100.0 if abs(m_loose) > 1e-9 else float('nan')
            verdict = 'Strengthened' if abs(pct_change) >= STRENGTHENED_THRESHOLD_PCT else 'Robust'
            flag = '' if same_sign else '  <-- SIGN FLIP'
            print(f"  {reg:<6s} n({LOOSE_LABEL})={n_loose:>8,}  Δ={m_loose:>+7.3f}   "
                  f"n(<={TIGHT_CUTOFF:.1f}Å)={n_tight:>7,}  Δ={m_tight:>+7.3f}   "
                  f"{pct_change:>+6.0f}%  {verdict}{flag}")

            s2_rows.append(dict(
                angle=label, basin=reg,
                n_loose=n_loose, delta_loose=round(m_loose, 3),
                n_tight=n_tight, delta_tight=round(m_tight, 3),
                pct_change=round(pct_change, 1), verdict=verdict,
                same_sign=same_sign,
            ))

    n_weakened = sum(1 for r in s2_rows if r['verdict'] == 'Robust' and abs(r['pct_change']) < STRENGTHENED_THRESHOLD_PCT and r['pct_change'] < 0)
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Rows: {len(s2_rows)}")
    print(f"  Sign flips: {n_sign_flip}")
    print(f"  Strengthened (|% change| >= {STRENGTHENED_THRESHOLD_PCT:.0f}%): "
          f"{sum(1 for r in s2_rows if r['verdict'] == 'Strengthened')}")
    print(f"  Robust (|% change| < {STRENGTHENED_THRESHOLD_PCT:.0f}%): "
          f"{sum(1 for r in s2_rows if r['verdict'] == 'Robust')}")
    if n_sign_flip == 0:
        print("  -> Consistent with claim: no sign flips at higher resolution "
              "(opposite of carbohydrate failure mode).")
    else:
        print("  -> Sign flip(s) detected -- check rows above before using in "
              "the response letter or Table S2.")

    out_csv = os.path.join(args.out, 'table_s2.csv')
    pd.DataFrame(s2_rows).to_csv(out_csv, index=False)
    print(f"\nTable S2 saved: {out_csv}")


if __name__ == '__main__':
    main()