#!/usr/bin/env python3
"""
Paper 4 — Local Geometry Benchmark
====================================

The full-chain NeRF benchmark (script 02) is dominated by error
accumulation, not constant accuracy. This script measures what matters:
does the library predict LOCAL geometry better than AMBER?

Metrics (per-residue, no error accumulation):
  1. Bond length MAE: |predicted - observed| for N-Cα, Cα-C, C-N, C=O
  2. Bond angle MAE: |predicted - observed| for τ, ∠N-Cα-Cβ, ∠C-Cα-Cβ, etc.
  3. Local triplet RMSD: rebuild just N[i], CA[i], C[i] from the PREVIOUS
     residue's PDB coordinates, compare to PDB positions
  4. Stratify by secondary structure and residue type

For each residue:
  - AMBER predicts: bond_NCA = 1.458 Å, tau = 111.1° (always)
  - Library predicts: bond_NCA = f(φ,ψ,res), tau = f(φ,ψ,res)
  - PDB truth: the actual measured value

Usage:
  python paper4_03_local_benchmark.py \
      --library ./paper4_library/constants_library.json \
      --csv /mnt/f/Protein_Folding/v8_g/p3.csv \
      --out ./paper4_local_benchmark/

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
# AMBER reference constants
# ══════════════════════════════════════════════════════════════════════════════

AMBER = {
    'tau_deg':        111.1,
    'angle_N_CA_CB':  110.1,
    'angle_C_CA_CB':  110.1,
    'angle_CaCN':     116.6,
    'angle_CNCa':     121.9,
    'angle_CA_C_O':   120.4,
    'bond_N_CA':      1.458,
    'bond_CA_C':      1.522,
    'bond_C_O':       1.229,
    'bond_C_N_next':  1.335,
    'bond_CA_CB':     1.526,
}

# Library JSON key mapping
LIB_MAP = {
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


# ══════════════════════════════════════════════════════════════════════════════
# Library lookup
# ══════════════════════════════════════════════════════════════════════════════

class QuickLibrary:
    """Fast library lookup for vectorized benchmark."""
    
    def __init__(self, json_path, bin_size=10):
        with open(json_path) as f:
            self._lib = json.load(f)
        self._bin_size = bin_size
        self._half = bin_size / 2.0
        self._centers = np.arange(-180 + self._half, 180 + self._half, bin_size)
    
    def _bin_key(self, angle):
        angle = ((angle + 180) % 360) - 180
        idx = int(np.round((angle - self._centers[0]) / self._bin_size))
        idx = max(0, min(idx, len(self._centers) - 1))
        return str(int(self._centers[idx]))
    
    def get(self, phi, psi, res_name, param):
        """Get a single parameter value. Returns None if not found."""
        pk = self._bin_key(phi)
        qk = self._bin_key(psi)
        lib_key = LIB_MAP.get(param)
        if lib_key is None:
            return None
        
        # Try specific residue
        for cls in [res_name, 'ALL']:
            if cls in self._lib:
                cell = self._lib[cls].get(pk, {}).get(qk)
                if cell and lib_key in cell:
                    return cell[lib_key]
        return None


def vectorized_lookup(lib, phi_arr, psi_arr, res_arr, param):
    """Vectorized library lookup for an entire column."""
    n = len(phi_arr)
    result = np.full(n, np.nan)
    amber_val = AMBER.get(param, np.nan)
    
    for i in range(n):
        if np.isnan(phi_arr[i]) or np.isnan(psi_arr[i]):
            result[i] = amber_val
            continue
        val = lib.get(phi_arr[i], psi_arr[i], res_arr[i], param)
        result[i] = val if val is not None else amber_val
    
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(df, lib, out_dir):
    """Run the local geometry benchmark."""
    
    R = []
    R.append("=" * 78)
    R.append("Paper 4 — Local Geometry Benchmark: Library vs AMBER")
    R.append("=" * 78)
    R.append(f"\n  Residues: {len(df):,}")
    
    phi = df['phi_deg'].values
    psi = df['psi_deg'].values
    res = df['res_name'].values
    
    # ── Per-observable comparison ─────────────────────────────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 1: PER-OBSERVABLE MAE (AMBER vs LIBRARY)")
    R.append("━" * 78)
    
    header = (f"  {'Observable':>20s}  {'Unit':>4s}  {'AMBER val':>9s}  "
              f"{'MAE_AMBER':>10s}  {'MAE_LIB':>10s}  {'Improv':>8s}  {'%':>7s}")
    R.append(header)
    R.append("  " + "─" * (len(header.strip())))
    
    results_summary = []
    
    for param, amber_val in AMBER.items():
        if param not in df.columns:
            continue
        
        observed = df[param].values
        valid = ~np.isnan(observed)
        if valid.sum() < 100:
            continue
        
        # AMBER prediction: constant for all residues
        amber_pred = np.full(len(df), amber_val)
        
        # Library prediction: (φ,ψ,res)-dependent
        lib_pred = vectorized_lookup(lib, phi, psi, res, param)
        
        # MAE
        mae_amber = np.mean(np.abs(amber_pred[valid] - observed[valid]))
        mae_lib = np.mean(np.abs(lib_pred[valid] - observed[valid]))
        
        improvement = mae_amber - mae_lib
        pct = improvement / mae_amber * 100 if mae_amber > 0 else 0
        
        unit = '°' if 'deg' in param or 'angle' in param else 'Å'
        
        R.append(f"  {param:>20s}  {unit:>4s}  {amber_val:>9.3f}  "
                 f"{mae_amber:>10.4f}  {mae_lib:>10.4f}  "
                 f"{improvement:>+8.4f}  {pct:>+7.1f}%")
        
        results_summary.append({
            'observable': param,
            'unit': unit,
            'amber_constant': amber_val,
            'mae_amber': mae_amber,
            'mae_lib': mae_lib,
            'improvement': improvement,
            'improvement_pct': pct,
            'n': valid.sum(),
        })
    
    # ── Stratified by secondary structure ────────────────────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 2: IMPROVEMENT BY SECONDARY STRUCTURE")
    R.append("━" * 78)
    
    ss_labels = {0: 'αR', 1: 'β', 2: 'PPII', 3: '3₁₀', 4: 'coil', 5: 'αL'}
    key_params = ['tau_deg', 'bond_N_CA', 'bond_CA_C', 'angle_N_CA_CB']
    
    for param in key_params:
        if param not in df.columns:
            continue
        amber_val = AMBER[param]
        R.append(f"\n  [{param}]")
        header2 = f"    {'SS':>6s}  {'n':>9s}  {'MAE_AMB':>8s}  {'MAE_LIB':>8s}  {'Δ':>8s}  {'%':>7s}"
        R.append(header2)
        
        lib_pred = vectorized_lookup(lib, phi, psi, res, param)
        observed = df[param].values
        
        if 'ss_bin' in df.columns:
            for ss_val in sorted(ss_labels.keys()):
                mask = (df['ss_bin'].values == ss_val) & ~np.isnan(observed)
                if mask.sum() < 50:
                    continue
                mae_a = np.mean(np.abs(amber_val - observed[mask]))
                mae_l = np.mean(np.abs(lib_pred[mask] - observed[mask]))
                delta = mae_a - mae_l
                pct = delta / mae_a * 100 if mae_a > 0 else 0
                R.append(f"    {ss_labels[ss_val]:>6s}  {mask.sum():>9,}  "
                         f"{mae_a:>8.4f}  {mae_l:>8.4f}  "
                         f"{delta:>+8.4f}  {pct:>+7.1f}%")
    
    # ── Stratified by residue type ───────────────────────────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 3: τ IMPROVEMENT BY RESIDUE TYPE")
    R.append("━" * 78)
    
    if 'tau_deg' in df.columns:
        observed_tau = df['tau_deg'].values
        lib_tau = vectorized_lookup(lib, phi, psi, res, 'tau_deg')
        amber_tau = AMBER['tau_deg']
        
        aa_results = []
        for aa in sorted(df['res_name'].unique()):
            mask = (res == aa) & ~np.isnan(observed_tau)
            if mask.sum() < 100:
                continue
            mae_a = np.mean(np.abs(amber_tau - observed_tau[mask]))
            mae_l = np.mean(np.abs(lib_tau[mask] - observed_tau[mask]))
            delta = mae_a - mae_l
            pct = delta / mae_a * 100 if mae_a > 0 else 0
            aa_results.append({
                'res': aa, 'n': mask.sum(),
                'mae_amber': mae_a, 'mae_lib': mae_l,
                'improvement': delta, 'pct': pct
            })
        
        # Sort by improvement
        aa_results.sort(key=lambda x: x['pct'], reverse=True)
        
        header3 = f"  {'Res':>4s}  {'n':>9s}  {'MAE_AMB':>8s}  {'MAE_LIB':>8s}  {'Δ':>8s}  {'%':>7s}"
        R.append(header3)
        R.append("  " + "─" * (len(header3.strip())))
        
        for r in aa_results:
            R.append(f"  {r['res']:>4s}  {r['n']:>9,}  "
                     f"{r['mae_amber']:>8.4f}  {r['mae_lib']:>8.4f}  "
                     f"{r['improvement']:>+8.4f}  {r['pct']:>+7.1f}%")
    
    # ── Distribution analysis ────────────────────────────────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 4: ERROR DISTRIBUTIONS")
    R.append("━" * 78)
    
    for param in ['tau_deg', 'bond_N_CA', 'bond_CA_C']:
        if param not in df.columns:
            continue
        observed = df[param].values
        valid = ~np.isnan(observed)
        lib_pred = vectorized_lookup(lib, phi, psi, res, param)
        
        err_amber = np.abs(AMBER[param] - observed[valid])
        err_lib = np.abs(lib_pred[valid] - observed[valid])
        
        # What fraction of residues does library improve?
        better = (err_lib < err_amber).sum()
        worse = (err_lib > err_amber).sum()
        same = (err_lib == err_amber).sum()
        total = valid.sum()
        
        R.append(f"\n  {param}:")
        R.append(f"    Library better: {better:>9,} ({better/total*100:.1f}%)")
        R.append(f"    Library worse:  {worse:>9,} ({worse/total*100:.1f}%)")
        R.append(f"    Equal:          {same:>9,} ({same/total*100:.1f}%)")
        
        # Percentile comparison
        R.append(f"    AMBER  error percentiles: "
                 f"p25={np.percentile(err_amber,25):.4f}  "
                 f"p50={np.percentile(err_amber,50):.4f}  "
                 f"p75={np.percentile(err_amber,75):.4f}  "
                 f"p95={np.percentile(err_amber,95):.4f}")
        R.append(f"    Library error percentiles: "
                 f"p25={np.percentile(err_lib,25):.4f}  "
                 f"p50={np.percentile(err_lib,50):.4f}  "
                 f"p75={np.percentile(err_lib,75):.4f}  "
                 f"p95={np.percentile(err_lib,95):.4f}")
    
    # ── With vs without coupling ─────────────────────────────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 5: EFFECT OF COUPLING CORRECTIONS")
    R.append("━" * 78)
    R.append("  (Library equilibrium values already include cell-mean coupling.)")
    R.append("  The question: does using the per-class (residue-specific) library")
    R.append("  beat using the ALL (pooled) library?")
    
    # Compare: specific-AA lookup vs ALL-only lookup
    for param in ['tau_deg', 'angle_N_CA_CB']:
        if param not in df.columns:
            continue
        observed = df[param].values
        valid = ~np.isnan(observed)
        
        # ALL-only lookup
        lib_all = np.full(len(df), np.nan)
        for i in range(len(df)):
            if np.isnan(phi[i]) or np.isnan(psi[i]):
                lib_all[i] = AMBER[param]
                continue
            val = lib.get(phi[i], psi[i], 'ALL_ONLY', param)  # force ALL
            # Manual ALL lookup
            pk = lib._bin_key(phi[i])
            qk = lib._bin_key(psi[i])
            cell = lib._lib.get('ALL', {}).get(pk, {}).get(qk)
            if cell and LIB_MAP[param] in cell:
                lib_all[i] = cell[LIB_MAP[param]]
            else:
                lib_all[i] = AMBER[param]
        
        # Specific-AA lookup (already computed)
        lib_spec = vectorized_lookup(lib, phi, psi, res, param)
        
        mae_all = np.mean(np.abs(lib_all[valid] - observed[valid]))
        mae_spec = np.mean(np.abs(lib_spec[valid] - observed[valid]))
        mae_amber = np.mean(np.abs(AMBER[param] - observed[valid]))
        
        R.append(f"\n  {param}:")
        R.append(f"    MAE AMBER:         {mae_amber:.4f}")
        R.append(f"    MAE Library (ALL):  {mae_all:.4f}  ({(mae_amber-mae_all)/mae_amber*100:+.1f}% vs AMBER)")
        R.append(f"    MAE Library (AA):   {mae_spec:.4f}  ({(mae_amber-mae_spec)/mae_amber*100:+.1f}% vs AMBER)")
        R.append(f"    Gain from AA-specific: {(mae_all-mae_spec)/mae_all*100:+.2f}%")
    
    # ── Summary verdict ──────────────────────────────────────────────────
    R.append("\n" + "━" * 78)
    R.append("VERDICT")
    R.append("━" * 78)
    
    all_improvements = [r['improvement_pct'] for r in results_summary if r['improvement_pct'] != 0]
    n_improved = sum(1 for p in all_improvements if p > 0)
    n_total = len(all_improvements)
    
    R.append(f"\n  Observables where library beats AMBER: {n_improved}/{n_total}")
    
    if results_summary:
        best = max(results_summary, key=lambda x: x['improvement_pct'])
        worst = min(results_summary, key=lambda x: x['improvement_pct'])
        R.append(f"  Best improvement:  {best['observable']} ({best['improvement_pct']:+.1f}%)")
        R.append(f"  Worst:             {worst['observable']} ({worst['improvement_pct']:+.1f}%)")
    
    return '\n'.join(R), results_summary


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def make_plots(df, lib, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    phi = df['phi_deg'].values
    psi = df['psi_deg'].values
    res = df['res_name'].values
    
    for param in ['tau_deg', 'bond_N_CA']:
        if param not in df.columns:
            continue
        observed = df[param].values
        valid = ~np.isnan(observed)
        lib_pred = vectorized_lookup(lib, phi, psi, res, param)
        amber_val = AMBER[param]
        
        err_amber = amber_val - observed[valid]
        err_lib = lib_pred[valid] - observed[valid]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        ax = axes[0]
        bins = np.linspace(-3, 3, 80) if 'deg' in param else np.linspace(-0.03, 0.03, 80)
        ax.hist(err_amber, bins=bins, alpha=0.5, label=f'AMBER (MAE={np.mean(np.abs(err_amber)):.4f})',
                color='#B2182B', density=True)
        ax.hist(err_lib, bins=bins, alpha=0.5, label=f'Library (MAE={np.mean(np.abs(err_lib)):.4f})',
                color='#2166AC', density=True)
        ax.axvline(0, color='black', lw=0.5)
        ax.set_xlabel(f'Prediction error [{"\u00B0" if "deg" in param else "\u00C5"}]')
        ax.set_ylabel('Density')
        ax.set_title(f'{param}: Error distribution')
        ax.legend(fontsize=8)
        
        # Scatter: AMBER error vs Library error
        ax = axes[1]
        if len(err_amber) > 20000:
            idx = np.random.choice(len(err_amber), 20000, replace=False)
        else:
            idx = np.arange(len(err_amber))
        ax.scatter(np.abs(err_amber[idx]), np.abs(err_lib[idx]),
                  s=1, alpha=0.05, c='#333333')
        lim = max(np.percentile(np.abs(err_amber), 99),
                  np.percentile(np.abs(err_lib), 99))
        ax.plot([0, lim], [0, lim], 'r--', lw=1, label='Equal')
        ax.set_xlabel(f'|AMBER error|')
        ax.set_ylabel(f'|Library error|')
        ax.set_title(f'{param}: Points below diagonal = library wins')
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect('equal')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'local_benchmark_{param}.png'), dpi=150)
        plt.close()
        print(f"  Saved local_benchmark_{param}.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Paper 4 — Local geometry benchmark')
    ap.add_argument('--library', required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='./paper4_local_benchmark')
    args = ap.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    
    print(f"[1/4] Loading library...")
    lib = QuickLibrary(args.library)
    
    print(f"[2/4] Reading {args.csv}...")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"  {len(df):,} rows")
    
    print(f"[3/4] Running benchmark...")
    report, summary = run_benchmark(df, lib, args.out)
    
    report_path = os.path.join(args.out, 'local_benchmark_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(report)
    
    pd.DataFrame(summary).to_csv(
        os.path.join(args.out, 'local_benchmark_summary.csv'), index=False)
    
    print(f"\n[4/4] Plots...")
    make_plots(df, lib, args.out)
    
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()