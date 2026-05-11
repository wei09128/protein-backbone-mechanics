#!/usr/bin/env python3
"""
Paper 4 — Library Patch: Fix ω, Flag Unreliable Coupling
==========================================================

Fixes two issues identified in validation:

  1. ω WRAPPING: raw ω values near ±180° average to ~0° when mixed.
     Fix: compute circular mean, and also store omega_dev = |180 - |ω||
     (deviation from planarity) as the NeRF-relevant quantity.

  2. UNRELIABLE COUPLING: bond C=O and C-N coupling corrections are
     statistically unreliable (bootstrap frac_reliable < 0.50).
     Fix: set coupling to 0.0 for flagged cells, add reliability
     metadata to the library.

Also adds:
  - 'reliable' flag per coupling correction
  - 'omega_dev_eq' (deviation from planarity, always positive)
  - Cleaned omega_deg_eq using circular mean

Usage:
  python paper4_04_patch_library.py \
      --csv /mnt/f/Protein_Folding/v8_g/p3.csv \
      --library_dir ./paper4_library/ \
      --bootstrap ./paper4_library/validation_bootstrap.csv \
      --out ./paper4_library_v2/ \
      --bin_size 10

Author: Wei (Cvek Lab, LSUS)
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

kT = 0.596  # kcal/mol at 300K


# ══════════════════════════════════════════════════════════════════════════════
# Circular mean for ω
# ══════════════════════════════════════════════════════════════════════════════

def circular_mean_deg(angles_deg):
    """Compute circular mean of angles in degrees.
    
    Handles the ±180° wrapping correctly:
    mean([179, -179]) = 180, not 0.
    """
    if len(angles_deg) == 0:
        return np.nan
    rads = np.radians(angles_deg)
    sin_mean = np.mean(np.sin(rads))
    cos_mean = np.mean(np.cos(rads))
    return np.degrees(np.arctan2(sin_mean, cos_mean))


def circular_std_deg(angles_deg):
    """Circular standard deviation in degrees."""
    if len(angles_deg) < 2:
        return np.nan
    rads = np.radians(angles_deg)
    R = np.sqrt(np.mean(np.sin(rads))**2 + np.mean(np.cos(rads))**2)
    R = min(R, 1.0)  # numerical safety
    if R < 1e-10:
        return 180.0
    return np.degrees(np.sqrt(-2 * np.log(R)))


# ══════════════════════════════════════════════════════════════════════════════
# Recompute ω statistics per cell
# ══════════════════════════════════════════════════════════════════════════════

def recompute_omega(df, bin_size=10, min_count=10):
    """Recompute ω statistics using circular mean + planarity deviation.
    
    Returns DataFrame with columns:
        phi_bin, psi_bin, phi_center, psi_center,
        omega_circ_mean, omega_circ_std, omega_dev_mean, omega_dev_std, n
    """
    phi_bins = np.arange(-180, 180 + bin_size, bin_size)
    psi_bins = np.arange(-180, 180 + bin_size, bin_size)
    phi_centers = phi_bins[:-1] + bin_size / 2
    psi_centers = psi_bins[:-1] + bin_size / 2
    
    sub = df[['phi_deg', 'psi_deg', 'omega_deg']].dropna().copy()
    sub['phi_bin'] = pd.cut(sub['phi_deg'], phi_bins, labels=False, right=False)
    sub['psi_bin'] = pd.cut(sub['psi_deg'], psi_bins, labels=False, right=False)
    sub = sub.dropna(subset=['phi_bin', 'psi_bin'])
    sub['phi_bin'] = sub['phi_bin'].astype(int)
    sub['psi_bin'] = sub['psi_bin'].astype(int)
    
    # Planarity deviation: always positive, no wrapping issue
    sub['omega_dev'] = (180.0 - sub['omega_deg'].abs()).abs()
    
    rows = []
    for (pb, qb), group in sub.groupby(['phi_bin', 'psi_bin']):
        if len(group) < min_count:
            continue
        
        omega_vals = group['omega_deg'].values
        dev_vals = group['omega_dev'].values
        
        rows.append({
            'phi_bin': int(pb),
            'psi_bin': int(qb),
            'phi_center': phi_centers[int(pb)],
            'psi_center': psi_centers[int(qb)],
            'omega_circ_mean': round(circular_mean_deg(omega_vals), 3),
            'omega_circ_std': round(circular_std_deg(omega_vals), 3),
            'omega_dev_mean': round(dev_vals.mean(), 3),
            'omega_dev_std': round(dev_vals.std(), 3),
            'omega_n': len(group),
            'omega_frac_cis': round((np.abs(omega_vals) < 90).mean(), 4),
        })
    
    return pd.DataFrame(rows)


def recompute_omega_by_class(df, bin_size=10, min_count=10):
    """Same but per amino acid."""
    results = {}
    if 'res_name' not in df.columns:
        return results
    
    for aa in df['res_name'].unique():
        mask = df['res_name'] == aa
        if mask.sum() < 500:
            continue
        omega_df = recompute_omega(df[mask], bin_size, min_count)
        if len(omega_df) > 0:
            results[aa] = omega_df
    
    # Also ALL
    results['ALL'] = recompute_omega(df, bin_size, min_count)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Patch the library
# ══════════════════════════════════════════════════════════════════════════════

def patch_library(lib_json, omega_data, bootstrap_df, unreliable_cols):
    """Patch the library JSON with fixed ω and reliability flags.
    
    Args:
        lib_json: the loaded constants_library.json dict
        omega_data: dict of class -> DataFrame with corrected ω
        bootstrap_df: DataFrame from validation_bootstrap.csv
        unreliable_cols: list of column names whose coupling is unreliable
    
    Returns: patched lib_json
    """
    patched = {}
    
    for cls_name, phi_dict in lib_json.items():
        patched[cls_name] = {}
        
        omega_cls = omega_data.get(cls_name, omega_data.get('ALL'))
        
        for phi_key, psi_dict in phi_dict.items():
            patched[cls_name][phi_key] = {}
            
            for psi_key, cell in psi_dict.items():
                new_cell = dict(cell)
                
                # ── Fix ω ────────────────────────────────────────────
                # Replace omega_deg_eq with circular mean
                if omega_cls is not None:
                    phi_val = int(phi_key)
                    psi_val = int(psi_key)
                    
                    match = omega_cls[
                        (omega_cls['phi_center'] == phi_val) &
                        (omega_cls['psi_center'] == psi_val)
                    ]
                    
                    if len(match) > 0:
                        row = match.iloc[0]
                        new_cell['omega_deg_eq'] = float(row['omega_circ_mean'])
                        new_cell['omega_deg_std'] = float(row['omega_circ_std'])
                        new_cell['omega_dev_eq'] = float(row['omega_dev_mean'])
                        new_cell['omega_dev_std'] = float(row['omega_dev_std'])
                        new_cell['omega_frac_cis'] = float(row['omega_frac_cis'])
                        
                        # Recompute ω spring constant from circular std
                        circ_std_rad = np.radians(row['omega_circ_std'])
                        if circ_std_rad > 1e-4:
                            new_cell['omega_deg_k'] = round(kT / circ_std_rad**2, 2)
                    else:
                        # No omega data for this cell — use 180°
                        new_cell['omega_deg_eq'] = 180.0
                        new_cell['omega_dev_eq'] = 0.0
                
                # ── Flag unreliable coupling ─────────────────────────
                for col in unreliable_cols:
                    coup_key = f'{col}_coupling'
                    if coup_key in new_cell:
                        new_cell[coup_key] = 0.0
                        new_cell[f'{col}_coupling_reliable'] = False
                
                # Mark reliable ones
                reliable_cols = [c for c in new_cell.keys() 
                                if c.endswith('_coupling') and 
                                not c.endswith('_coupling_reliable')]
                for c in reliable_cols:
                    base = c.replace('_coupling', '')
                    flag_key = f'{base}_coupling_reliable'
                    if flag_key not in new_cell:
                        new_cell[flag_key] = True
                
                patched[cls_name][phi_key][psi_key] = new_cell
    
    return patched


# ══════════════════════════════════════════════════════════════════════════════
# Also patch the flat CSV
# ══════════════════════════════════════════════════════════════════════════════

def patch_csv(csv_path, omega_all, unreliable_cols, out_path):
    """Patch the constants_library.csv with fixed ω and zeroed coupling."""
    df = pd.read_csv(csv_path)
    
    # Merge corrected ω
    if omega_all is not None and len(omega_all) > 0:
        omega_merge = omega_all[['phi_center', 'psi_center', 
                                  'omega_circ_mean', 'omega_circ_std',
                                  'omega_dev_mean', 'omega_dev_std',
                                  'omega_frac_cis']].copy()
        
        # Drop old omega columns
        for col in ['omega_deg_eq', 'omega_deg_std', 'omega_deg_median']:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        df = df.merge(omega_merge, on=['phi_center', 'psi_center'], how='left')
        df = df.rename(columns={
            'omega_circ_mean': 'omega_deg_eq',
            'omega_circ_std': 'omega_deg_std',
        })
    
    # Zero unreliable coupling
    for col in unreliable_cols:
        coup_col = f'{col}_coupling'
        if coup_col in df.columns:
            df[coup_col] = 0.0
    
    df.to_csv(out_path, index=False)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

def build_report(omega_all, unreliable_cols, lib_v1_path, lib_v2_path):
    R = []
    R.append("=" * 78)
    R.append("Paper 4 — Library Patch Report")
    R.append("=" * 78)
    
    R.append("\n  CHANGES APPLIED:")
    R.append("")
    
    # ω fix
    R.append("  [1] ω WRAPPING FIX")
    R.append("  " + "─" * 60)
    if omega_all is not None:
        # Compare old vs new
        old_mean = 26.1  # from the library report
        new_vals = omega_all['omega_circ_mean'].dropna()
        new_mean = new_vals.mean()
        R.append(f"    Old omega_deg_eq (arithmetic mean):  ~{old_mean:.1f}°  ← WRONG")
        R.append(f"    New omega_deg_eq (circular mean):    {new_mean:.1f}°  ← CORRECT")
        R.append(f"    Cells with corrected ω: {len(new_vals)}")
        R.append(f"    ω deviation from planarity:")
        dev = omega_all['omega_dev_mean'].dropna()
        R.append(f"      mean |180-|ω|| = {dev.mean():.2f}°")
        R.append(f"      max  |180-|ω|| = {dev.max():.2f}°")
        R.append(f"    Fraction cis (|ω|<90°): {omega_all['omega_frac_cis'].mean()*100:.2f}%")
    
    # Unreliable coupling
    R.append(f"\n  [2] UNRELIABLE COUPLING ZEROED")
    R.append("  " + "─" * 60)
    R.append(f"    Columns with coupling set to 0.0:")
    for col in unreliable_cols:
        R.append(f"      - {col}")
    R.append(f"    Reason: bootstrap frac_reliable < 0.50")
    R.append(f"    These observables still have (φ,ψ)-dependent equilibrium")
    R.append(f"    values — only the coupling CORRECTION is zeroed.")
    
    # Reliability flags
    R.append(f"\n  [3] RELIABILITY METADATA ADDED")
    R.append("  " + "─" * 60)
    R.append(f"    Each coupling correction now has a '_reliable' boolean flag.")
    R.append(f"    NeRF should check: if not reliable, use marginal mean.")
    
    R.append(f"\n  OUTPUT FILES:")
    R.append(f"    {lib_v2_path}/constants_library.json  ← PATCHED (use this)")
    R.append(f"    {lib_v2_path}/constants_library.csv   ← PATCHED flat table")
    R.append(f"    {lib_v2_path}/omega_corrected.csv     ← per-cell ω statistics")
    
    R.append(f"\n  The v1 library in {lib_v1_path}/ is preserved unchanged.")
    
    return '\n'.join(R)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Paper 4 — Patch library (fix ω, flag unreliable coupling)')
    ap.add_argument('--csv', required=True,
                    help='Raw per-residue CSV (p3.csv)')
    ap.add_argument('--library_dir', required=True,
                    help='Directory with constants_library.json/csv')
    ap.add_argument('--bootstrap', default=None,
                    help='Path to validation_bootstrap.csv')
    ap.add_argument('--out', required=True,
                    help='Output directory for patched library')
    ap.add_argument('--bin_size', type=int, default=10)
    ap.add_argument('--unreliable_threshold', type=float, default=0.50,
                    help='Bootstrap frac_reliable below this → zero coupling')
    args = ap.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    
    # ── Load ─────────────────────────────────────────────────────────────
    print(f"[1/5] Loading data...")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"  {len(df):,} rows")
    
    lib_json_path = os.path.join(args.library_dir, 'constants_library.json')
    with open(lib_json_path) as f:
        lib_json = json.load(f)
    print(f"  Library: {len(lib_json)} classes")
    
    # Load bootstrap results
    boot_df = None
    unreliable_cols = []
    if args.bootstrap and os.path.exists(args.bootstrap):
        boot_df = pd.read_csv(args.bootstrap)
        print(f"  Bootstrap: {len(boot_df)} cells")
        
        # Determine unreliable columns
        for col in boot_df['geo_col'].unique():
            sub = boot_df[boot_df['geo_col'] == col]
            tested = len(sub[sub['status'].isin(['RELIABLE', 'UNRELIABLE'])])
            reliable = len(sub[sub['status'] == 'RELIABLE'])
            frac = reliable / tested if tested > 0 else 0
            if frac < args.unreliable_threshold:
                unreliable_cols.append(col)
                print(f"  UNRELIABLE: {col} ({frac:.0%} reliable)")
    else:
        # Default unreliable columns based on validation results
        unreliable_cols = ['bond_C_O', 'bond_C_N_next']
        print(f"  No bootstrap file — using default unreliable: {unreliable_cols}")
    
    # ── Recompute ω ──────────────────────────────────────────────────────
    print(f"[2/5] Recomputing ω with circular mean...")
    omega_data = recompute_omega_by_class(df, args.bin_size)
    omega_all = omega_data.get('ALL')
    if omega_all is not None:
        print(f"  {len(omega_all)} cells with corrected ω")
        print(f"  Mean circular ω: {omega_all['omega_circ_mean'].mean():.1f}°")
        print(f"  Mean |180-|ω||: {omega_all['omega_dev_mean'].mean():.2f}°")
        
        # Save
        omega_path = os.path.join(args.out, 'omega_corrected.csv')
        omega_all.to_csv(omega_path, index=False)
    
    # ── Patch JSON ───────────────────────────────────────────────────────
    print(f"[3/5] Patching JSON library...")
    patched_json = patch_library(lib_json, omega_data, boot_df, unreliable_cols)
    
    json_path = os.path.join(args.out, 'constants_library.json')
    with open(json_path, 'w') as f:
        json.dump(patched_json, f, indent=2)
    print(f"  Saved {json_path} ({os.path.getsize(json_path)/1024:.0f} KB)")
    
    # ── Patch CSV ────────────────────────────────────────────────────────
    print(f"[4/5] Patching CSV library...")
    csv_in = os.path.join(args.library_dir, 'constants_library.csv')
    csv_out = os.path.join(args.out, 'constants_library.csv')
    patched_csv = patch_csv(csv_in, omega_all, unreliable_cols, csv_out)
    print(f"  Saved {csv_out}")
    
    # Copy per-AA CSVs (unpatched — ω fix is in the JSON)
    for fname in os.listdir(args.library_dir):
        if fname.startswith('constants_') and fname.endswith('.csv') and fname != 'constants_library.csv':
            src = os.path.join(args.library_dir, fname)
            dst = os.path.join(args.out, fname)
            if not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)
    
    # Copy chi1 library
    for fname in ['constants_chi1.json', 'constants_chi1.csv']:
        src = os.path.join(args.library_dir, fname)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, os.path.join(args.out, fname))
    
    # ── Report ───────────────────────────────────────────────────────────
    print(f"[5/5] Report...")
    report = build_report(omega_all, unreliable_cols, args.library_dir, args.out)
    
    report_path = os.path.join(args.out, 'patch_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(report)
    
    # ── Verify ───────────────────────────────────────────────────────────
    print(f"\n  VERIFICATION:")
    
    # Check ω in patched library
    all_lib = patched_json.get('ALL', {})
    omega_vals = []
    for phi_d in all_lib.values():
        for psi_d in phi_d.values():
            ov = psi_d.get('omega_deg_eq')
            if ov is not None:
                omega_vals.append(ov)
    if omega_vals:
        omega_arr = np.array(omega_vals)
        print(f"    Patched ω: mean={omega_arr.mean():.1f}°  "
              f"std={omega_arr.std():.1f}°  "
              f"range=[{omega_arr.min():.1f}, {omega_arr.max():.1f}]")
        print(f"    Fraction near ±180°: {(np.abs(omega_arr) > 170).mean()*100:.1f}%")
    
    # Check zeroed coupling
    for col in unreliable_cols:
        coup_key = f'{col}_coupling'
        coup_vals = []
        for phi_d in all_lib.values():
            for psi_d in phi_d.values():
                cv = psi_d.get(coup_key, None)
                if cv is not None:
                    coup_vals.append(cv)
        if coup_vals:
            print(f"    {col} coupling: all zero = {all(v == 0 for v in coup_vals)}")
    
    print(f"\n  Done in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()