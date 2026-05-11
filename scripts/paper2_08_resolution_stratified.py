#!/usr/bin/env python3
"""
Paper 3 — Resolution-Stratified Bond Length Analysis
=====================================================

The scoping analysis showed R² < 7% for all bonds across all features.
Hypothesis: refinement restraints (Engh & Huber targets) dominate bond
length distributions at typical resolution (1.5–2.5 Å), masking genuine
physical variation that would be visible in ultra-high-resolution structures.

This script tests that hypothesis by:
  1. Extracting resolution from PDB headers
  2. Merging with bond_lengths.csv (output of the scoping analysis)
  3. Re-running ANOVA and Pearson correlations within resolution bins
  4. Plotting R² vs resolution to see if the physical signal emerges

If R²(local) rises sharply at sub-1.2 Å → Paper 3 is alive.
If R²(local) stays flat even at sub-1.0 Å → bond lengths are invariant.

Usage:
  python paper3_resolution_stratified.py \
      --bonds ./scoping/bond_lengths.csv \
      --pdb_dir /path/to/pdb_cache \
      --out ./scoping_resolution/
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# Reuse feature engineering from the scoping script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from paper3_bond_length_scoping import (
        engineer_features, run_anova_decomposition,
        BLOCK_A_COLS, BLOCK_B_COLS, BOND_NAMES, BOND_LABELS,
    )
    HAVE_SCOPING = True
except ImportError:
    HAVE_SCOPING = False
    print("WARNING: paper3_bond_length_scoping.py not found in same directory.")
    print("         Will use inline feature engineering (reduced feature set).")


# ══════════════════════════════════════════════════════════════════════════════
# Resolution extraction from PDB headers
# ══════════════════════════════════════════════════════════════════════════════

def extract_resolution(pdb_path):
    """
    Extract resolution from a PDB file header.
    
    Looks for REMARK 2 lines containing "RESOLUTION" and a numeric value.
    Returns resolution in Angstroms, or None if not found (e.g., NMR structures).
    """
    import re
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
                # Extract first decimal number after "RESOLUTION"
                after = upper.split('RESOLUTION')[1]
                m = re.search(r'(\d+\.\d+)', after)
                if m:
                    val = float(m.group(1))
                    if 0.3 <= val <= 15.0:
                        return val
    except Exception:
        pass
    return None


def extract_all_resolutions(pdb_dir, max_pdbs=None, verbose=False):
    """Extract resolution for all PDB files in a directory."""
    pdb_files = sorted(Path(pdb_dir).glob('*.pdb'))
    if max_pdbs:
        pdb_files = pdb_files[:max_pdbs]
    
    records = []
    n_found = n_missing = 0
    
    for i, pdb_path in enumerate(pdb_files):
        pdb_id = pdb_path.stem.lower()
        res = extract_resolution(pdb_path)
        if res is not None:
            records.append({'pdb_id': pdb_id, 'resolution': res})
            n_found += 1
        else:
            n_missing += 1
        
        if verbose and (i + 1) % 500 == 0:
            print(f'  Scanned {i+1}/{len(pdb_files)} PDBs, '
                  f'{n_found} with resolution, {n_missing} without')
    
    print(f'  Resolution extracted: {n_found} PDBs with resolution, '
          f'{n_missing} without (NMR/other)')
    
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# Inline feature engineering (fallback if scoping script not importable)
# ══════════════════════════════════════════════════════════════════════════════

if not HAVE_SCOPING:
    BOND_NAMES = ['bond_NCa', 'bond_CaC', 'bond_CO']
    BOND_LABELS = {'bond_NCa': 'N–Cα', 'bond_CaC': 'Cα–C', 'bond_CO': 'C=O'}
    BLOCK_A_COLS = ['res_name', 'basin', 'chi1_class', 'ss_class']
    BLOCK_B_COLS = ['hb_co_status', 'packing_bin', 'burial_class']


# ══════════════════════════════════════════════════════════════════════════════
# Resolution-stratified analysis
# ══════════════════════════════════════════════════════════════════════════════

# Resolution bins: ultra-high, high, medium, low
RES_BINS = [
    ('sub-1.0',  0.0, 1.0),
    ('1.0-1.2',  1.0, 1.2),
    ('1.2-1.5',  1.2, 1.5),
    ('1.5-2.0',  1.5, 2.0),
    ('2.0-2.5',  2.0, 2.5),
    ('2.5+',     2.5, 99.0),
]


def run_stratified_analysis(df, block_a_cols, block_b_cols):
    """
    Run ANOVA and continuous correlations within each resolution bin.
    
    Returns:
      results: list of dicts, one per (bond × resolution_bin)
    """
    results = []
    
    for bin_label, lo, hi in RES_BINS:
        mask = df['resolution'].between(lo, hi, inclusive='left')
        if bin_label == '2.5+':
            mask = df['resolution'] >= lo
        subset = df[mask]
        n = len(subset)
        
        if n < 500:
            print(f'  {bin_label:8s}  n={n:>8,}  [SKIP — too few residues]')
            continue
        
        print(f'  {bin_label:8s}  n={n:>8,}')
        
        for bond in BOND_NAMES:
            label = BOND_LABELS[bond]
            
            # ANOVA (categorical)
            res = run_anova_decomposition(subset, bond, block_a_cols, block_b_cols)
            
            # Continuous correlations — top features
            cont_features = ['hb_best_e', 'hb_n_strong', 'bfactor_ca', 'sc_mass',
                             'angle_NCaC', 'angle_CaCN', 'angle_CNCa',
                             'phi_deg', 'psi_deg', 'omega_deg',
                             'sc_lever_arm', 'chi1_rad']
            
            best_cont_r2 = 0.0
            best_cont_feat = ''
            cont_corrs = {}
            for feat in cont_features:
                if feat in subset.columns:
                    m = subset[bond].notna() & subset[feat].notna()
                    if m.sum() > 100:
                        r, p = sp_stats.pearsonr(subset.loc[m, bond],
                                                 subset.loc[m, feat])
                        cont_corrs[feat] = {'r': r, 'r2': r**2, 'p': p}
                        if r**2 > best_cont_r2:
                            best_cont_r2 = r**2
                            best_cont_feat = feat
            
            # Per-residue conditional variance (how tight are the within-group SDs?)
            if 'res_name' in subset.columns:
                grp = subset.groupby('res_name')[bond]
                per_res_std = grp.std().dropna()
                per_res_n = grp.count()
            else:
                per_res_std = pd.Series()
                per_res_n = pd.Series()
            
            # Bond length descriptive stats in this bin
            vals = subset[bond].dropna()
            
            results.append({
                'res_bin': bin_label,
                'res_lo': lo,
                'res_hi': hi,
                'bond': bond,
                'bond_label': label,
                'n': n,
                'n_bond': len(vals),
                'mean': vals.mean(),
                'std': vals.std(),
                'R2_A': res.get('R2_A', np.nan),
                'R2_B': res.get('R2_B', np.nan),
                'R2_AB': res.get('R2_AB', np.nan),
                'best_cont_feat': best_cont_feat,
                'best_cont_r2': best_cont_r2,
                'cont_corrs': cont_corrs,
                'per_res_std_mean': per_res_std.mean() if len(per_res_std) > 0 else np.nan,
                'per_res_std_min': per_res_std.min() if len(per_res_std) > 0 else np.nan,
                'per_res_std_max': per_res_std.max() if len(per_res_std) > 0 else np.nan,
            })
            
            print(f'    {label:6s}  R²_A={res.get("R2_A",0):.4f}  '
                  f'R²_B={res.get("R2_B",0):.4f}  '
                  f'R²_AB={res.get("R2_AB",0):.4f}  '
                  f'σ={vals.std():.4f}  '
                  f'best_cont={best_cont_feat}({best_cont_r2:.4f})')
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_r2_vs_resolution(results, out_dir):
    """Main diagnostic plot: R²(local) vs resolution bin for each bond."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=False)
    
    for ax, bond in zip(axes, BOND_NAMES):
        label = BOND_LABELS[bond]
        bond_data = [r for r in results if r['bond'] == bond]
        if not bond_data:
            ax.set_title(label)
            continue
        
        bins = [r['res_bin'] for r in bond_data]
        r2_a = [r['R2_A'] for r in bond_data]
        r2_b = [r['R2_B'] for r in bond_data]
        r2_ab = [r['R2_AB'] for r in bond_data]
        ns = [r['n'] for r in bond_data]
        
        x = np.arange(len(bins))
        w = 0.25
        
        bars_a = ax.bar(x - w, r2_a, w, label='Block A (local)', color='#4C72B0')
        bars_b = ax.bar(x, r2_b, w, label='Block B (env)', color='#DD8452')
        bars_ab = ax.bar(x + w, r2_ab, w, label='A+B', color='#55A868')
        
        # Annotate with R² values
        for i, (a, b, ab) in enumerate(zip(r2_a, r2_b, r2_ab)):
            ax.text(i - w, a + 0.002, f'{a:.3f}', ha='center', va='bottom',
                    fontsize=7, rotation=45)
            ax.text(i + w, ab + 0.002, f'{ab:.3f}', ha='center', va='bottom',
                    fontsize=7, rotation=45)
        
        # Secondary axis: sample size
        ax2 = ax.twinx()
        ax2.plot(x, ns, 'k--', alpha=0.3, marker='s', markersize=4)
        ax2.set_ylabel('n residues', fontsize=8, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=30, ha='right', fontsize=9)
        ax.set_xlabel('Resolution (Å)')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_ylabel('R²')
    
    axes[0].legend(fontsize=8, loc='upper right')
    fig.suptitle('Does Physical Signal Emerge at High Resolution?',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'r2_vs_resolution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved r2_vs_resolution.png')


def plot_std_vs_resolution(results, out_dir):
    """Bond length σ vs resolution — does variance increase when restraints loosen?"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    
    for ax, bond in zip(axes, BOND_NAMES):
        label = BOND_LABELS[bond]
        bond_data = [r for r in results if r['bond'] == bond]
        if not bond_data:
            ax.set_title(label)
            continue
        
        bins = [r['res_bin'] for r in bond_data]
        stds = [r['std'] for r in bond_data]
        per_res_mean = [r['per_res_std_mean'] for r in bond_data]
        per_res_min = [r['per_res_std_min'] for r in bond_data]
        per_res_max = [r['per_res_std_max'] for r in bond_data]
        
        x = np.arange(len(bins))
        
        ax.bar(x, stds, 0.4, label='Overall σ', color='#4C72B0', alpha=0.7)
        ax.errorbar(x + 0.25, per_res_mean,
                    yerr=[np.array(per_res_mean) - np.array(per_res_min),
                          np.array(per_res_max) - np.array(per_res_mean)],
                    fmt='o', color='#C44E52', markersize=5,
                    label='Per-residue σ (mean ± range)')
        
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=30, ha='right', fontsize=9)
        ax.set_xlabel('Resolution (Å)')
        ax.set_ylabel('σ (Å)')
        ax.set_title(label, fontsize=12, fontweight='bold')
    
    axes[0].legend(fontsize=8)
    fig.suptitle('Bond Length Dispersion vs Crystallographic Resolution',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'std_vs_resolution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved std_vs_resolution.png')


def plot_top_correlations(results, out_dir):
    """Heatmap: top continuous feature r² per bond × resolution bin."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Build matrix: features × (bond, res_bin)
    all_feats = set()
    for r in results:
        all_feats.update(r.get('cont_corrs', {}).keys())
    all_feats = sorted(all_feats)
    
    if not all_feats:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, bond in zip(axes, BOND_NAMES):
        label = BOND_LABELS[bond]
        bond_data = [r for r in results if r['bond'] == bond]
        if not bond_data:
            continue
        
        bins = [r['res_bin'] for r in bond_data]
        
        # Build r² matrix
        matrix = np.zeros((len(all_feats), len(bins)))
        for j, r in enumerate(bond_data):
            for i, feat in enumerate(all_feats):
                if feat in r.get('cont_corrs', {}):
                    matrix[i, j] = r['cont_corrs'][feat]['r2']
        
        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0,
                       vmax=max(0.05, matrix.max()))
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(bins, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(all_feats)))
        ax.set_yticklabels(all_feats, fontsize=7)
        ax.set_title(label, fontsize=11, fontweight='bold')
        
        # Annotate cells
        for i in range(len(all_feats)):
            for j in range(len(bins)):
                val = matrix[i, j]
                if val > 0.005:
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                            fontsize=6, color='white' if val > 0.02 else 'black')
        
        plt.colorbar(im, ax=ax, shrink=0.6, label='r²')
    
    fig.suptitle('Continuous Feature r² by Resolution Bin', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'correlation_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved correlation_heatmap.png')


# ══════════════════════════════════════════════════════════════════════════════
# Text report
# ══════════════════════════════════════════════════════════════════════════════

def write_stratified_report(results, res_df, out_dir):
    """Write a human-readable resolution-stratified analysis report."""
    path = os.path.join(out_dir, 'resolution_analysis.txt')
    
    with open(path, 'w') as fh:
        fh.write('=' * 72 + '\n')
        fh.write('RESOLUTION-STRATIFIED BOND LENGTH ANALYSIS\n')
        fh.write('=' * 72 + '\n\n')
        
        # Resolution distribution
        fh.write('Resolution distribution of dataset:\n')
        for label, lo, hi in RES_BINS:
            if label == '2.5+':
                n = (res_df['resolution'] >= lo).sum()
            else:
                n = res_df['resolution'].between(lo, hi, inclusive='left').sum()
            fh.write(f'  {label:8s}: {n:>6,} PDBs\n')
        fh.write(f'  Total:    {len(res_df):>6,} PDBs with resolution\n')
        fh.write(f'  Mean resolution: {res_df["resolution"].mean():.2f} Å\n')
        fh.write(f'  Median resolution: {res_df["resolution"].median():.2f} Å\n\n')
        
        # Results table
        fh.write('─' * 72 + '\n')
        fh.write(f'{"Bin":8s} {"Bond":6s} {"n":>9s} {"σ(Å)":>7s} '
                 f'{"R²_A":>7s} {"R²_B":>7s} {"R²_AB":>7s} '
                 f'{"Best cont":>15s} {"r²":>7s}\n')
        fh.write('─' * 72 + '\n')
        
        for r in results:
            fh.write(f'{r["res_bin"]:8s} {r["bond_label"]:6s} '
                     f'{r["n"]:>9,} {r["std"]:>7.4f} '
                     f'{r["R2_A"]:>7.4f} {r["R2_B"]:>7.4f} '
                     f'{r.get("R2_AB", 0):>7.4f} '
                     f'{r["best_cont_feat"]:>15s} '
                     f'{r["best_cont_r2"]:>7.4f}\n')
        
        fh.write('─' * 72 + '\n\n')
        
        # Per-bond trend analysis
        fh.write('TREND ANALYSIS\n')
        fh.write('─' * 40 + '\n')
        for bond in BOND_NAMES:
            label = BOND_LABELS[bond]
            bond_data = [r for r in results if r['bond'] == bond]
            if len(bond_data) < 2:
                continue
            
            r2_values = [r['R2_A'] for r in bond_data]
            std_values = [r['std'] for r in bond_data]
            res_midpoints = [(r['res_lo'] + min(r['res_hi'], 5.0)) / 2
                             for r in bond_data]
            
            # Trend: does R² increase as resolution improves (decreases)?
            if len(res_midpoints) >= 3:
                slope, intercept, r_val, p_val, se = sp_stats.linregress(
                    res_midpoints, r2_values)
                fh.write(f'\n  {label}:\n')
                fh.write(f'    R²_A range: {min(r2_values):.4f} – {max(r2_values):.4f}\n')
                fh.write(f'    R²_A trend vs resolution: slope={slope:.4f} '
                         f'({"+" if slope > 0 else "–"} with worse resolution)\n')
                fh.write(f'    σ range: {min(std_values):.4f} – {max(std_values):.4f} Å\n')
                
                # Key diagnostic
                best_r2 = max(r2_values)
                best_bin = bond_data[r2_values.index(best_r2)]['res_bin']
                worst_r2 = min(r2_values)
                worst_bin = bond_data[r2_values.index(worst_r2)]['res_bin']
                
                ratio = best_r2 / worst_r2 if worst_r2 > 0 else float('inf')
                fh.write(f'    Best R²_A: {best_r2:.4f} in bin {best_bin}\n')
                fh.write(f'    Worst R²_A: {worst_r2:.4f} in bin {worst_bin}\n')
                fh.write(f'    Ratio: {ratio:.1f}x\n')
                
                if ratio > 3.0 and slope < 0:
                    fh.write(f'    → SIGNAL EMERGING: R² increases {ratio:.1f}x '
                             f'at high resolution\n')
                    fh.write(f'      Paper 3 interpretation: refinement restraints '
                             f'mask real variation\n')
                elif ratio > 1.5 and slope < 0:
                    fh.write(f'    → MODERATE TREND: {ratio:.1f}x improvement. '
                             f'Signal present but weak.\n')
                else:
                    fh.write(f'    → FLAT: R² does not depend on resolution. '
                             f'Bond length is intrinsically invariant.\n')
        
        fh.write('\n' + '=' * 72 + '\n')
        fh.write('VERDICT\n')
        fh.write('─' * 40 + '\n')
        
        # Automated verdict
        nca_data = [r for r in results if r['bond'] == 'bond_NCa']
        if nca_data:
            best_nca = max(r['R2_A'] for r in nca_data)
            if best_nca > 0.15:
                fh.write('  Paper 3 is ALIVE.\n')
                fh.write('  Physical variation in bond lengths is real but masked\n')
                fh.write('  by refinement restraints at typical resolution.\n')
                fh.write('  Story: "Engh & Huber parameters are self-fulfilling."\n')
            elif best_nca > 0.08:
                fh.write('  Paper 3 is MARGINAL.\n')
                fh.write('  Some signal exists at high resolution but it may not\n')
                fh.write('  be strong enough for a standalone paper. Consider\n')
                fh.write('  folding the resolution finding into Paper 4.\n')
            else:
                fh.write('  Paper 3 is DEAD (as originally conceived).\n')
                fh.write('  Bond lengths are genuinely stiff — one k per bond,\n')
                fh.write('  independent of local context. Skip to Paper 4.\n')
                fh.write('  \n')
                fh.write('  BUT: the negative result itself is publishable if\n')
                fh.write('  you frame it as "bond lengths are the one degree of\n')
                fh.write('  freedom where AMBER gets it right." That contrast\n')
                fh.write('  with dihedrals (Paper 1) and angles (Paper 2) is\n')
                fh.write('  itself informative.\n')
    
    print(f'  Saved resolution_analysis.txt')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Resolution-stratified bond length analysis for Paper 3')
    ap.add_argument('--bonds', required=True,
                    help='Path to bond_lengths.csv (from scoping analysis)')
    ap.add_argument('--pdb_dir', required=True,
                    help='Directory containing PDB files')
    ap.add_argument('--out', default='./scoping_resolution',
                    help='Output directory')
    ap.add_argument('--max_pdbs', type=int, default=None,
                    help='Limit PDBs for resolution extraction')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    
    # ── Step 1: Extract resolutions ───────────────────────────────────────
    print('\n[1/4] Extracting resolution from PDB headers...')
    res_df = extract_all_resolutions(args.pdb_dir, args.max_pdbs, args.verbose)
    
    if len(res_df) == 0:
        print('ERROR: no resolutions extracted.')
        sys.exit(1)
    
    print(f'  Resolution stats:')
    print(f'    mean={res_df["resolution"].mean():.2f} Å, '
          f'median={res_df["resolution"].median():.2f} Å')
    print(f'    range=[{res_df["resolution"].min():.2f}, '
          f'{res_df["resolution"].max():.2f}]')
    
    # Distribution across bins
    for label, lo, hi in RES_BINS:
        if label == '2.5+':
            n = (res_df['resolution'] >= lo).sum()
        else:
            n = res_df['resolution'].between(lo, hi, inclusive='left').sum()
        print(f'    {label:8s}: {n:>5,} PDBs')
    
    # Save resolution data
    res_df.to_csv(os.path.join(args.out, 'pdb_resolutions.csv'), index=False)
    
    # ── Step 2: Load and merge bond lengths ───────────────────────────────
    print('\n[2/4] Loading bond_lengths.csv and merging with resolution...')
    bonds_df = pd.read_csv(args.bonds)
    print(f'  bond_lengths.csv: {len(bonds_df):,} rows')
    
    # Normalize pdb_id
    bonds_df['pdb_id'] = bonds_df['pdb_id'].astype(str).str.lower()
    res_df['pdb_id'] = res_df['pdb_id'].astype(str).str.lower()
    
    merged = pd.merge(bonds_df, res_df, on='pdb_id', how='inner')
    print(f'  After merge: {len(merged):,} rows '
          f'({len(merged)/len(bonds_df)*100:.1f}%)')
    
    if len(merged) < 1000:
        print('ERROR: too few merged rows. Check pdb_id alignment.')
        sys.exit(1)
    
    # ── Step 3: Engineer features (if not already present) ────────────────
    # Check if features were already engineered in bond_lengths.csv
    need_engineering = 'basin' not in merged.columns
    if need_engineering:
        print('\n[3/4] Engineering features...')
        merged = engineer_features(merged)
    else:
        print('\n[3/4] Features already present in bond_lengths.csv, skipping...')
    
    # ── Step 4: Resolution-stratified ANOVA ───────────────────────────────
    print('\n[4/4] Running resolution-stratified analysis...')
    results = run_stratified_analysis(merged, BLOCK_A_COLS, BLOCK_B_COLS)
    
    # ── Outputs ───────────────────────────────────────────────────────────
    print('\nGenerating plots and report...')
    plot_r2_vs_resolution(results, args.out)
    plot_std_vs_resolution(results, args.out)
    plot_top_correlations(results, args.out)
    write_stratified_report(results, res_df, args.out)
    
    # Summary table to stdout
    print('\n' + '=' * 72)
    print('RESOLUTION-STRATIFIED R² SUMMARY')
    print('─' * 72)
    print(f'{"Bin":8s} {"Bond":6s} {"n":>9s} {"σ(Å)":>7s} '
          f'{"R²_A":>7s} {"R²_B":>7s} {"R²_AB":>7s}')
    print('─' * 72)
    for r in results:
        print(f'{r["res_bin"]:8s} {r["bond_label"]:6s} '
              f'{r["n"]:>9,} {r["std"]:>7.4f} '
              f'{r["R2_A"]:>7.4f} {r["R2_B"]:>7.4f} '
              f'{r.get("R2_AB", 0):>7.4f}')
    print('─' * 72)
    
    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.1f}s. Results in {args.out}/')


if __name__ == '__main__':
    main()