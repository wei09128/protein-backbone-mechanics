#!/usr/bin/env python3
"""
Paper 4 — Fair Benchmarks: Strain Energy + Temporal Split
==========================================================

Addresses the circularity concern: the library was derived from PDB
structures, so testing it against PDB structures is partially circular.

Two fair benchmarks:

  [A] STRAIN ENERGY BENCHMARK
      For each residue, compute the harmonic strain energy:
          E_strain = Σ ½ k_i (θ_PDB,i - θ_eq,i)²
      summed over all geometry observables.
      
      If the library's θ_eq consistently produces lower strain than
      AMBER's θ_eq, it means AMBER imposes phantom strain on real
      structures — a physically meaningful claim.

  [B] TEMPORAL SPLIT BENCHMARK
      Train the library on structures deposited before a cutoff date.
      Test on structures deposited AFTER the cutoff.
      If the library's MAE on the holdout set matches training, the
      coupling is universal physics, not a dataset artifact.

      Since we don't have deposition dates in the CSV, we use a PDB
      ID-based split: first 80% of sorted PDB IDs = train, last 20% = test.
      PDB IDs are roughly chronological (1xxx before 2xxx before 3xxx, etc.)

Usage:
  python paper4_05_fair_benchmarks.py \
      --csv /mnt/f/Protein_Folding/v8_g/p3.csv \
      --out ./paper4_fair/ --bin_size 10

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

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

AMBER = {
    'tau_deg':        {'eq': 111.1, 'k': 63.0,  'unit': 'deg'},
    'angle_N_CA_CB':  {'eq': 110.1, 'k': 63.0,  'unit': 'deg'},
    'angle_C_CA_CB':  {'eq': 110.1, 'k': 63.0,  'unit': 'deg'},
    'angle_CaCN':     {'eq': 116.6, 'k': 70.0,  'unit': 'deg'},
    'angle_CNCa':     {'eq': 121.9, 'k': 50.0,  'unit': 'deg'},
    'angle_CA_C_O':   {'eq': 120.4, 'k': 80.0,  'unit': 'deg'},
    'bond_N_CA':      {'eq': 1.458, 'k': 337.0, 'unit': 'Å'},
    'bond_CA_C':      {'eq': 1.522, 'k': 317.0, 'unit': 'Å'},
    'bond_C_O':       {'eq': 1.229, 'k': 570.0, 'unit': 'Å'},
    'bond_C_N_next':  {'eq': 1.335, 'k': 490.0, 'unit': 'Å'},
    'bond_CA_CB':     {'eq': 1.526, 'k': 317.0, 'unit': 'Å'},
}

kT = 0.596  # kcal/mol at 300K


# ══════════════════════════════════════════════════════════════════════════════
# Library extraction (self-contained, no external dependency)
# ══════════════════════════════════════════════════════════════════════════════

def build_library_from_df(df, bin_size=10, min_count=10):
    """Build a quick library from a DataFrame.
    
    Returns nested dict: res_name -> phi_key -> psi_key -> {col_eq: value}
    """
    phi_bins = np.arange(-180, 180 + bin_size, bin_size)
    psi_bins = np.arange(-180, 180 + bin_size, bin_size)
    phi_centers = phi_bins[:-1] + bin_size / 2
    psi_centers = psi_bins[:-1] + bin_size / 2
    
    sub = df.copy()
    sub['phi_bin'] = pd.cut(sub['phi_deg'], phi_bins, labels=False, right=False)
    sub['psi_bin'] = pd.cut(sub['psi_deg'], psi_bins, labels=False, right=False)
    sub = sub.dropna(subset=['phi_bin', 'psi_bin'])
    sub['phi_bin'] = sub['phi_bin'].astype(int)
    sub['psi_bin'] = sub['psi_bin'].astype(int)
    
    geo_cols = [c for c in AMBER if c in sub.columns]
    
    lib = {}
    
    # Build per-residue and ALL
    for group_name in list(sub['res_name'].unique()) + ['ALL']:
        if group_name == 'ALL':
            grp = sub
        else:
            grp = sub[sub['res_name'] == group_name]
        
        cell_means = grp.groupby(['phi_bin', 'psi_bin'])[geo_cols].agg(['mean', 'count'])
        
        lib[group_name] = {}
        for (pb, qb), row in cell_means.iterrows():
            # Check min count for any column
            counts = [row[(c, 'count')] for c in geo_cols if (c, 'count') in row.index]
            if not counts or max(counts) < min_count:
                continue
            
            phi_key = str(int(phi_centers[int(pb)]))
            psi_key = str(int(psi_centers[int(qb)]))
            
            if phi_key not in lib[group_name]:
                lib[group_name][phi_key] = {}
            
            entry = {}
            for c in geo_cols:
                if (c, 'mean') in row.index and not np.isnan(row[(c, 'mean')]):
                    entry[c] = float(row[(c, 'mean')])
            
            lib[group_name][phi_key][psi_key] = entry
    
    return lib


def lookup(lib, phi, psi, res_name, param, bin_size=10):
    """Look up a value from the library."""
    half = bin_size / 2.0
    centers = np.arange(-180 + half, 180 + half, bin_size)
    
    phi_wrapped = ((phi + 180) % 360) - 180
    idx = int(np.round((phi_wrapped - centers[0]) / bin_size))
    idx = max(0, min(idx, len(centers) - 1))
    phi_key = str(int(centers[idx]))
    
    psi_wrapped = ((psi + 180) % 360) - 180
    idx = int(np.round((psi_wrapped - centers[0]) / bin_size))
    idx = max(0, min(idx, len(centers) - 1))
    psi_key = str(int(centers[idx]))
    
    for cls in [res_name, 'ALL']:
        if cls in lib:
            cell = lib[cls].get(phi_key, {}).get(psi_key)
            if cell and param in cell:
                return cell[param]
    
    return AMBER[param]['eq']


# ══════════════════════════════════════════════════════════════════════════════
# [A] Strain Energy Benchmark
# ══════════════════════════════════════════════════════════════════════════════

def strain_energy_benchmark(df, lib, out_dir):
    """Compute per-residue strain energy for AMBER vs Library."""
    R = []
    R.append("=" * 78)
    R.append("[A] STRAIN ENERGY BENCHMARK")
    R.append("=" * 78)
    R.append("""
  For each residue, compute:
      E_strain = Σ_i ½ k_i (θ_PDB,i - θ_eq,i)²
  
  using AMBER's fixed θ_eq vs the library's (φ,ψ,res)-dependent θ_eq.
  Both use the SAME spring constants (AMBER k_i) for fair comparison.
  
  Lower strain = better equilibrium values.
  "Phantom strain" = energy AMBER assigns to a structure that is
  actually at its natural equilibrium geometry.
""")
    
    phi = df['phi_deg'].values
    psi = df['psi_deg'].values
    res = df['res_name'].values
    
    geo_cols = [c for c in AMBER if c in df.columns]
    
    # Compute strain per residue
    strain_amber = np.zeros(len(df))
    strain_lib = np.zeros(len(df))
    
    for col in geo_cols:
        observed = df[col].values
        valid = ~np.isnan(observed)
        
        k = AMBER[col]['k']
        amber_eq = AMBER[col]['eq']
        unit = AMBER[col]['unit']
        
        # Convert to radians for angular quantities
        if unit == 'deg':
            conv = np.pi / 180.0
        else:
            conv = 1.0
        
        # AMBER strain
        diff_amber = (observed - amber_eq) * conv
        e_amber = 0.5 * k * diff_amber**2
        e_amber[~valid] = 0.0
        strain_amber += e_amber
        
        # Library strain
        lib_eq = np.array([lookup(lib, phi[i], psi[i], res[i], col) 
                           if valid[i] and not np.isnan(phi[i]) and not np.isnan(psi[i])
                           else amber_eq
                           for i in range(len(df))])
        diff_lib = (observed - lib_eq) * conv
        e_lib = 0.5 * k * diff_lib**2
        e_lib[~valid] = 0.0
        strain_lib += e_lib
    
    # Summary
    R.append(f"  Residues: {len(df):,}")
    R.append(f"  Observables used: {len(geo_cols)}")
    R.append("")
    
    R.append(f"  {'Metric':>30s}  {'AMBER':>12s}  {'Library':>12s}  {'Δ':>12s}")
    R.append("  " + "─" * 70)
    
    mean_a = strain_amber.mean()
    mean_l = strain_lib.mean()
    R.append(f"  {'Mean total strain (kcal/mol)':>30s}  {mean_a:>12.4f}  {mean_l:>12.4f}  {mean_a-mean_l:>+12.4f}")
    
    med_a = np.median(strain_amber)
    med_l = np.median(strain_lib)
    R.append(f"  {'Median total strain':>30s}  {med_a:>12.4f}  {med_l:>12.4f}  {med_a-med_l:>+12.4f}")
    
    p95_a = np.percentile(strain_amber, 95)
    p95_l = np.percentile(strain_lib, 95)
    R.append(f"  {'95th percentile':>30s}  {p95_a:>12.4f}  {p95_l:>12.4f}  {p95_a-p95_l:>+12.4f}")
    
    frac_lib_lower = (strain_lib < strain_amber).mean()
    R.append(f"  {'% residues lib < AMBER':>30s}  {'':>12s}  {frac_lib_lower*100:>12.1f}%")
    
    reduction_pct = (mean_a - mean_l) / mean_a * 100
    R.append(f"  {'Strain reduction':>30s}  {'':>12s}  {'':>12s}  {reduction_pct:>+12.1f}%")
    
    # Per-observable strain
    R.append(f"\n  Per-observable mean strain (kcal/mol):")
    R.append(f"  {'Observable':>20s}  {'AMBER':>10s}  {'Library':>10s}  {'Δ':>10s}  {'%':>8s}")
    R.append("  " + "─" * 62)
    
    for col in geo_cols:
        observed = df[col].values
        valid = ~np.isnan(observed)
        k = AMBER[col]['k']
        unit = AMBER[col]['unit']
        conv = np.pi / 180.0 if unit == 'deg' else 1.0
        
        diff_a = (observed[valid] - AMBER[col]['eq']) * conv
        ea = np.mean(0.5 * k * diff_a**2)
        
        lib_eq = np.array([lookup(lib, phi[i], psi[i], res[i], col)
                           if not np.isnan(phi[i]) and not np.isnan(psi[i])
                           else AMBER[col]['eq']
                           for i in range(len(df)) if valid[i]])
        diff_l = (observed[valid] - lib_eq) * conv
        el = np.mean(0.5 * k * diff_l**2)
        
        pct = (ea - el) / ea * 100 if ea > 0 else 0
        R.append(f"  {col:>20s}  {ea:>10.4f}  {el:>10.4f}  {ea-el:>+10.4f}  {pct:>+8.1f}%")
    
    # Per-secondary-structure
    if 'ss_bin' in df.columns:
        ss_labels = {0: 'αR', 1: 'β', 2: 'PPII', 3: '3₁₀', 4: 'coil', 5: 'αL'}
        R.append(f"\n  Per-SS mean total strain (kcal/mol):")
        R.append(f"  {'SS':>8s}  {'n':>9s}  {'AMBER':>10s}  {'Library':>10s}  {'Δ':>10s}  {'%':>8s}")
        
        for ss_val in sorted(ss_labels.keys()):
            mask = df['ss_bin'].values == ss_val
            if mask.sum() < 100:
                continue
            ea = strain_amber[mask].mean()
            el = strain_lib[mask].mean()
            pct = (ea - el) / ea * 100 if ea > 0 else 0
            R.append(f"  {ss_labels[ss_val]:>8s}  {mask.sum():>9,}  "
                     f"{ea:>10.4f}  {el:>10.4f}  {ea-el:>+10.4f}  {pct:>+8.1f}%")
    
    # Interpretation
    R.append(f"""
  INTERPRETATION:
    AMBER assigns {mean_a:.3f} kcal/mol of strain per residue on average.
    The library reduces this to {mean_l:.3f} kcal/mol ({reduction_pct:+.1f}%).
    This means AMBER's fixed constants impose {mean_a-mean_l:.3f} kcal/mol
    of PHANTOM STRAIN per residue — energy that arises from using the wrong
    equilibrium value, not from real structural deformation.
    
    Over a 100-residue protein, total phantom strain = {(mean_a-mean_l)*100:.1f} kcal/mol.
    This is comparable to {(mean_a-mean_l)*100/1.5:.0f} hydrogen bonds.
""")
    
    # Save per-residue strains
    df_out = pd.DataFrame({
        'strain_amber': strain_amber,
        'strain_library': strain_lib,
        'strain_reduction': strain_amber - strain_lib,
    })
    df_out.to_csv(os.path.join(out_dir, 'strain_per_residue.csv'), index=False)
    
    return '\n'.join(R)


# ══════════════════════════════════════════════════════════════════════════════
# [B] Temporal Split Benchmark
# ══════════════════════════════════════════════════════════════════════════════

def temporal_split_benchmark(df, out_dir, bin_size=10, train_frac=0.8):
    """Train/test split by PDB ID (proxy for deposition date)."""
    R = []
    R.append("\n" + "=" * 78)
    R.append("[B] TEMPORAL SPLIT BENCHMARK (OUT-OF-SAMPLE)")
    R.append("=" * 78)
    
    if 'pdb_id' not in df.columns:
        R.append("  pdb_id column not found — skipping")
        return '\n'.join(R)
    
    # Sort PDB IDs (roughly chronological: 1xxx < 2xxx < 3xxx...)
    pdb_ids = sorted(df['pdb_id'].str.strip().str.lower().unique())
    n_pdbs = len(pdb_ids)
    split_idx = int(n_pdbs * train_frac)
    
    train_ids = set(pdb_ids[:split_idx])
    test_ids = set(pdb_ids[split_idx:])
    
    df_lower = df.copy()
    df_lower['pdb_lower'] = df_lower['pdb_id'].str.strip().str.lower()
    
    train_mask = df_lower['pdb_lower'].isin(train_ids)
    test_mask = df_lower['pdb_lower'].isin(test_ids)
    
    df_train = df[train_mask]
    df_test = df[test_mask]
    
    R.append(f"\n  Total PDBs: {n_pdbs}")
    R.append(f"  Train PDBs: {len(train_ids)} ({train_frac*100:.0f}%)")
    R.append(f"  Test PDBs:  {len(test_ids)} ({(1-train_frac)*100:.0f}%)")
    R.append(f"  Train residues: {len(df_train):,}")
    R.append(f"  Test residues:  {len(df_test):,}")
    R.append(f"\n  Train ID range: {min(train_ids)} – {max(train_ids)}")
    R.append(f"  Test ID range:  {min(test_ids)} – {max(test_ids)}")
    
    # Build library from TRAIN set only
    R.append(f"\n  Building library from train set only...")
    lib_train = build_library_from_df(df_train, bin_size)
    
    n_classes = len(lib_train)
    n_cells_all = len(lib_train.get('ALL', {}))
    R.append(f"  Train library: {n_classes} classes, ~{n_cells_all} cells (ALL)")
    
    # Evaluate on TEST set
    R.append(f"\n  Evaluating on test set (NEVER SEEN by library)...")
    
    geo_cols = [c for c in AMBER if c in df_test.columns]
    phi_test = df_test['phi_deg'].values
    psi_test = df_test['psi_deg'].values
    res_test = df_test['res_name'].values
    
    R.append(f"\n  {'Observable':>20s}  {'MAE_AMB':>8s}  {'MAE_train':>9s}  {'MAE_test':>8s}  "
             f"{'%_train':>8s}  {'%_test':>8s}  {'Gap':>6s}")
    R.append("  " + "─" * 72)
    
    summary_rows = []
    
    for col in geo_cols:
        amber_eq = AMBER[col]['eq']
        
        # Train MAE (library on its own data)
        obs_train = df_train[col].values
        valid_train = ~np.isnan(obs_train) & ~np.isnan(df_train['phi_deg'].values)
        if valid_train.sum() < 100:
            continue
        
        phi_tr = df_train['phi_deg'].values
        psi_tr = df_train['psi_deg'].values
        res_tr = df_train['res_name'].values
        
        lib_pred_train = np.array([
            lookup(lib_train, phi_tr[i], psi_tr[i], res_tr[i], col)
            if valid_train[i] else amber_eq
            for i in range(len(df_train))
        ])
        
        mae_amber_train = np.mean(np.abs(obs_train[valid_train] - amber_eq))
        mae_lib_train = np.mean(np.abs(obs_train[valid_train] - lib_pred_train[valid_train]))
        
        # Test MAE (library on unseen data)
        obs_test = df_test[col].values
        valid_test = ~np.isnan(obs_test) & ~np.isnan(phi_test)
        if valid_test.sum() < 100:
            continue
        
        lib_pred_test = np.array([
            lookup(lib_train, phi_test[i], psi_test[i], res_test[i], col)
            if valid_test[i] else amber_eq
            for i in range(len(df_test))
        ])
        
        mae_amber_test = np.mean(np.abs(obs_test[valid_test] - amber_eq))
        mae_lib_test = np.mean(np.abs(obs_test[valid_test] - lib_pred_test[valid_test]))
        
        pct_train = (mae_amber_train - mae_lib_train) / mae_amber_train * 100
        pct_test = (mae_amber_test - mae_lib_test) / mae_amber_test * 100
        gap = pct_train - pct_test  # positive = overfit
        
        R.append(f"  {col:>20s}  {mae_amber_test:>8.4f}  {mae_lib_train:>9.4f}  "
                 f"{mae_lib_test:>8.4f}  {pct_train:>+8.1f}%  {pct_test:>+8.1f}%  "
                 f"{gap:>+6.1f}")
        
        summary_rows.append({
            'observable': col,
            'mae_amber': mae_amber_test,
            'mae_lib_train': mae_lib_train,
            'mae_lib_test': mae_lib_test,
            'improvement_train_pct': pct_train,
            'improvement_test_pct': pct_test,
            'overfit_gap': gap,
        })
    
    # Summary
    if summary_rows:
        avg_train = np.mean([r['improvement_train_pct'] for r in summary_rows])
        avg_test = np.mean([r['improvement_test_pct'] for r in summary_rows])
        avg_gap = np.mean([r['overfit_gap'] for r in summary_rows])
        
        R.append("  " + "─" * 72)
        R.append(f"  {'AVERAGE':>20s}  {'':>8s}  {'':>9s}  {'':>8s}  "
                 f"{avg_train:>+8.1f}%  {avg_test:>+8.1f}%  {avg_gap:>+6.1f}")
        
        R.append(f"""
  INTERPRETATION:
    %_train = improvement on data the library was built from
    %_test  = improvement on UNSEEN data (out-of-sample)
    Gap     = train% - test% (positive = overfitting)
    
    If %_test ≈ %_train (gap < 2%): library captures universal physics
    If %_test << %_train (gap > 5%): library is overfitting to training data
    If %_test is negative: library HURTS on unseen data (bad sign)
    
    Average train improvement: {avg_train:+.1f}%
    Average test improvement:  {avg_test:+.1f}%
    Average overfit gap:       {avg_gap:+.1f}%
""")
        
        if avg_gap < 2:
            R.append("  VERDICT: ✓ Minimal overfitting — library captures universal geometry")
        elif avg_gap < 5:
            R.append("  VERDICT: ~ Modest overfitting — library is largely generalizable")
        else:
            R.append("  VERDICT: ✗ Significant overfitting — coupling may be dataset-specific")
    
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(out_dir, 'temporal_split_results.csv'), index=False)
    
    return '\n'.join(R)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Paper 4 — Fair benchmarks (strain energy + temporal split)')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='./paper4_fair')
    ap.add_argument('--bin_size', type=int, default=10)
    ap.add_argument('--train_frac', type=float, default=0.8)
    args = ap.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    
    print(f"[1/4] Reading {args.csv}...")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"  {len(df):,} rows")
    
    # Build full library for strain energy
    print(f"[2/4] Building full library for strain benchmark...")
    lib_full = build_library_from_df(df, args.bin_size)
    print(f"  {len(lib_full)} classes")
    
    # [A] Strain energy
    print(f"[3/4] Strain energy benchmark...")
    report_a = strain_energy_benchmark(df, lib_full, args.out)
    
    # [B] Temporal split
    print(f"[4/4] Temporal split benchmark...")
    report_b = temporal_split_benchmark(df, args.out, args.bin_size, args.train_frac)
    
    # Combined report
    report = report_a + "\n" + report_b
    
    report_path = os.path.join(args.out, 'fair_benchmark_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n  Saved {report_path}")
    print(report)
    
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()