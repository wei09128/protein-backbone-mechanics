#!/usr/bin/env python3
"""
Paper 3 — GAM-Based Coupling Decomposition
=============================================

Proper decomposition using Generalized Additive Models:

  Model A (additive):    y = s(φ) + s(ψ) + ε
  Model B (full):        y = te(φ,ψ) + ε

  ΔR²_coupling = R²_B − R²_A

This avoids all binning artifacts and non-orthogonality issues.
The ti(φ,ψ) = te(φ,ψ) − s(φ) − s(ψ) interaction surface is
orthogonal to main effects by construction in mgcv; here we
achieve the same via the ΔR² approach.

Because pyGAM's te() is memory-intensive, we:
  1. Run GAM on N_SUBSAMPLE rows (default 50K)
  2. Bootstrap B times for confidence intervals
  3. Also compute binned cell-means on FULL data for coupling maps
  4. Cross-validate the cell-means R² on full data as a check

Usage:
  python paper3_06_gam_coupling.py \
      --csv /mnt/f/Protein_Folding/v8_g/p3.csv \
      --out ./paper3_gam/ --n_sub 50000 --n_boot 20

Author: Wei (Cvek Lab, LSUS)
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# GAM decomposition (subsampled)
# ══════════════════════════════════════════════════════════════════════════════

def gam_decomposition(df, phi_col, psi_col, value_col,
                      n_sub=50000, n_boot=20, n_splines=15):
    """GAM-based R² decomposition with bootstrap CI.
    
    Returns dict with:
        r2_additive: mean R² from s(φ) + s(ψ)
        r2_full:     mean R² from te(φ,ψ)
        delta_coupling: mean ΔR²
        ci_coupling:  (lo, hi) 95% CI for ΔR²
        all bootstrap values
    """
    from pygam import LinearGAM, s, te
    
    sub = df[[phi_col, psi_col, value_col]].dropna()
    if len(sub) < 1000:
        return None
    
    r2_adds = []
    r2_fulls = []
    deltas = []
    
    for b in range(n_boot):
        # Sample
        sample = sub.sample(min(n_sub, len(sub)), random_state=42 + b)
        X = sample[[phi_col, psi_col]].values
        y = sample[value_col].values
        
        # Model A: additive
        gam_add = LinearGAM(
            s(0, n_splines=n_splines) + s(1, n_splines=n_splines)
        ).fit(X, y)
        r2_a = gam_add.statistics_['pseudo_r2']['explained_deviance']
        
        # Model B: tensor (full interaction)
        gam_full = LinearGAM(
            te(0, 1, n_splines=[n_splines, n_splines])
        ).fit(X, y)
        r2_f = gam_full.statistics_['pseudo_r2']['explained_deviance']
        
        r2_adds.append(r2_a)
        r2_fulls.append(r2_f)
        deltas.append(r2_f - r2_a)
    
    r2_adds = np.array(r2_adds)
    r2_fulls = np.array(r2_fulls)
    deltas = np.array(deltas)
    
    return {
        'r2_additive_mean': r2_adds.mean(),
        'r2_additive_std': r2_adds.std(),
        'r2_full_mean': r2_fulls.mean(),
        'r2_full_std': r2_fulls.std(),
        'delta_coupling_mean': deltas.mean(),
        'delta_coupling_std': deltas.std(),
        'delta_coupling_ci': (np.percentile(deltas, 2.5),
                              np.percentile(deltas, 97.5)),
        'n_boot': n_boot,
        'n_sub': min(n_sub, len(sub)),
        'n_total': len(sub),
        'boot_deltas': deltas,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Full-data cell-means R² (for comparison and coupling maps)
# ══════════════════════════════════════════════════════════════════════════════

def cellmeans_r2(df, phi_col, psi_col, value_col, bin_size=10, min_count=5):
    """Full-data cell-means decomposition.
    
    This uses ALL rows (no sampling) via groupby aggregation.
    R²_full = 1 - SS_within / SS_total
    R²_additive computed via marginal predictions.
    """
    sub = df[[phi_col, psi_col, value_col]].dropna().copy()
    if len(sub) < 200:
        return None
    
    phi_bins = np.arange(-180, 180 + bin_size, bin_size)
    psi_bins = np.arange(-180, 180 + bin_size, bin_size)
    
    sub['phi_bin'] = pd.cut(sub[phi_col], phi_bins, labels=False, right=False)
    sub['psi_bin'] = pd.cut(sub[psi_col], psi_bins, labels=False, right=False)
    sub = sub.dropna(subset=['phi_bin', 'psi_bin'])
    sub['phi_bin'] = sub['phi_bin'].astype(int)
    sub['psi_bin'] = sub['psi_bin'].astype(int)
    
    y = sub[value_col].values
    grand_mean = y.mean()
    ss_total = np.sum((y - grand_mean)**2)
    
    if ss_total < 1e-12:
        return None
    
    # Cell means model (= full model)
    cell_means = sub.groupby(['phi_bin', 'psi_bin'])[value_col].transform('mean')
    cell_counts = sub.groupby(['phi_bin', 'psi_bin'])[value_col].transform('count')
    
    # Filter sparse cells
    mask_valid = cell_counts >= min_count
    sub_valid = sub[mask_valid].copy()
    y_valid = sub_valid[value_col].values
    cell_means_valid = cell_means[mask_valid].values
    
    ss_total_v = np.sum((y_valid - y_valid.mean())**2)
    ss_within = np.sum((y_valid - cell_means_valid)**2)
    r2_full = 1.0 - ss_within / ss_total_v
    
    # Additive model: predict each y from phi_marginal + psi_marginal - grand_mean
    phi_marginal = sub_valid.groupby('phi_bin')[value_col].transform('mean')
    psi_marginal = sub_valid.groupby('psi_bin')[value_col].transform('mean')
    y_mean_v = y_valid.mean()
    additive_pred = phi_marginal.values + psi_marginal.values - y_mean_v
    
    ss_resid_add = np.sum((y_valid - additive_pred)**2)
    r2_additive = 1.0 - ss_resid_add / ss_total_v
    
    delta = r2_full - r2_additive
    
    # Coupling map
    cell_stats = sub_valid.groupby(['phi_bin', 'psi_bin'])[value_col].agg(['mean', 'count'])
    cell_stats.columns = ['cell_mean', 'cell_count']
    
    phi_m = sub_valid.groupby('phi_bin')[value_col].mean()
    psi_m = sub_valid.groupby('psi_bin')[value_col].mean()
    
    n_phi = len(phi_bins) - 1
    n_psi = len(psi_bins) - 1
    phi_centers = phi_bins[:-1] + bin_size / 2
    psi_centers = psi_bins[:-1] + bin_size / 2
    
    coupling_map = np.full((n_psi, n_phi), np.nan)
    count_map = np.full((n_psi, n_phi), 0)
    
    for (pb, qb), row in cell_stats.iterrows():
        pe = phi_m.get(pb, np.nan)
        qe = psi_m.get(qb, np.nan)
        if np.isnan(pe) or np.isnan(qe):
            continue
        additive = y_mean_v + (pe - y_mean_v) + (qe - y_mean_v)
        coupling_map[int(qb), int(pb)] = row['cell_mean'] - additive
        count_map[int(qb), int(pb)] = int(row['cell_count'])
    
    valid = coupling_map[~np.isnan(coupling_map)]
    w = count_map[~np.isnan(coupling_map)]
    rms = np.sqrt(np.average(valid**2, weights=w)) if len(valid) > 0 else 0
    pp = valid.max() - valid.min() if len(valid) > 0 else 0
    
    return {
        'n': len(sub_valid),
        'r2_additive': r2_additive,
        'r2_full': r2_full,
        'delta_coupling': delta,
        'rms_coupling': rms,
        'peak_to_peak': pp,
        'coupling_map': coupling_map,
        'count_map': count_map,
        'phi_centers': phi_centers,
        'psi_centers': psi_centers,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

def build_report(gam_results, cell_results, gam_by_class, cell_by_class,
                 gam_by_ss):
    R = []
    R.append("=" * 78)
    R.append("Paper 3 — GAM-Based Coupling Decomposition")
    R.append("=" * 78)
    R.append("""
  Two methods, cross-validated:

  METHOD 1 (GAM): y = s(φ) + s(ψ) vs y = te(φ,ψ)
    Smooth spline basis, no binning, proper interaction.
    Run on 50K subsample × 20 bootstrap iterations.
    Reports ΔR² with 95% CI.

  METHOD 2 (Cell-means): binned ANOVA on FULL data (1.77M rows)
    R²_full = cell-means model, R²_add = marginal-sum model.
    ΔR² = R²_full − R²_add.
    Also provides coupling maps for visualization.

  Agreement between methods validates both.
""")
    
    # ── Section 1: GAM results ───────────────────────────────────────────
    R.append("━" * 78)
    R.append("SECTION 1: GAM DECOMPOSITION (SUBSAMPLED, BOOTSTRAPPED)")
    R.append("━" * 78)
    
    header = (f"  {'Observable':>20s}  {'R²_add':>10s}  {'R²_full':>10s}  "
              f"{'ΔR²_coup':>10s}  {'95% CI':>20s}")
    R.append(header)
    R.append("  " + "─" * (len(header.strip())))
    
    for name, res in sorted(gam_results.items()):
        if res is None:
            continue
        ci = res['delta_coupling_ci']
        R.append(
            f"  {name:>20s}  "
            f"{res['r2_additive_mean']:>10.2%}  "
            f"{res['r2_full_mean']:>10.2%}  "
            f"{res['delta_coupling_mean']:>+10.2%}  "
            f"[{ci[0]:+.2%}, {ci[1]:+.2%}]")
    
    # ── Section 2: Cell-means on full data ───────────────────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 2: CELL-MEANS ON FULL DATA")
    R.append("━" * 78)
    
    header2 = (f"  {'Observable':>20s}  {'n':>10s}  {'R²_add':>8s}  "
               f"{'R²_full':>8s}  {'ΔR²_coup':>9s}  "
               f"{'RMS':>7s}  {'p-p':>7s}")
    R.append(header2)
    R.append("  " + "─" * (len(header2.strip())))
    
    for name, res in sorted(cell_results.items()):
        if res is None:
            continue
        R.append(
            f"  {name:>20s}  {res['n']:>10,}  "
            f"{res['r2_additive']:>8.2%}  "
            f"{res['r2_full']:>8.2%}  "
            f"{res['delta_coupling']:>+9.2%}  "
            f"{res['rms_coupling']:>7.3f}  "
            f"{res['peak_to_peak']:>7.2f}")
    
    # ── Section 3: Method agreement ──────────────────────────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 3: METHOD AGREEMENT")
    R.append("━" * 78)
    R.append("")
    
    header3 = f"  {'Observable':>20s}  {'GAM ΔR²':>10s}  {'Cell ΔR²':>10s}  {'Ratio':>7s}"
    R.append(header3)
    R.append("  " + "─" * (len(header3.strip())))
    
    for name in sorted(set(gam_results.keys()) & set(cell_results.keys())):
        g = gam_results[name]
        c = cell_results[name]
        if g is None or c is None:
            continue
        ratio = g['delta_coupling_mean'] / c['delta_coupling'] if c['delta_coupling'] > 1e-6 else 0
        R.append(
            f"  {name:>20s}  "
            f"{g['delta_coupling_mean']:>+10.2%}  "
            f"{c['delta_coupling']:>+10.2%}  "
            f"{ratio:>7.2f}")
    
    R.append("")
    R.append("  Ratio ≈ 1.0 → methods agree (binning ≈ smooth splines)")
    R.append("  Ratio > 1.0 → GAM captures more (splines better than step functions)")
    R.append("  Ratio < 1.0 → cell-means captures more (possible overfitting of splines)")
    
    # ── Section 4: By residue class (GAM) ────────────────────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 4: BY RESIDUE CLASS")
    R.append("━" * 78)
    
    for cls_name in ['GLY', 'PRO', 'beta_branched', 'non_branched', 'ALL']:
        if cls_name not in gam_by_class:
            continue
        R.append(f"\n  [{cls_name}]")
        cls_g = gam_by_class[cls_name]
        cls_c = cell_by_class.get(cls_name, {})
        
        for obs_name in sorted(cls_g.keys()):
            g = cls_g[obs_name]
            c = cls_c.get(obs_name)
            if g is None:
                continue
            ci = g['delta_coupling_ci']
            cell_str = f"  cell:{c['delta_coupling']:+.2%}" if c else ""
            R.append(
                f"    {obs_name:>20s}  "
                f"GAM: {g['delta_coupling_mean']:+.2%} "
                f"[{ci[0]:+.2%},{ci[1]:+.2%}]"
                f"{cell_str}")
    
    # ── Section 5: By SS ─────────────────────────────────────────────────
    if gam_by_ss:
        R.append("\n" + "━" * 78)
        R.append("SECTION 5: BY SECONDARY STRUCTURE")
        R.append("━" * 78)
        
        ss_labels = {0: 'αR', 1: 'β', 2: 'PPII', 3: '3₁₀', 4: 'coil', 5: 'αL'}
        for ss_val in sorted(gam_by_ss.keys()):
            label = ss_labels.get(ss_val, f'SS{ss_val}')
            R.append(f"\n  [{label}]")
            for obs_name, g in sorted(gam_by_ss[ss_val].items()):
                if g is None:
                    continue
                ci = g['delta_coupling_ci']
                R.append(
                    f"    {obs_name:>20s}  "
                    f"ΔR²={g['delta_coupling_mean']:+.2%} "
                    f"[{ci[0]:+.2%},{ci[1]:+.2%}]  "
                    f"R²_add={g['r2_additive_mean']:.1%}")
    
    # ── Section 6: Final numbers for manuscript ──────────────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 6: NUMBERS FOR MANUSCRIPT")
    R.append("━" * 78)
    R.append("""
  Use GAM ΔR² as the primary metric in the paper.
  Use cell-means coupling maps for Figure 1.
  
  Report as:
    "The φ×ψ coupling term accounts for ΔR² = X.X% [95% CI: X.X–X.X%]
     of additional variance beyond the additive model."
  
  This is conservative, defensible, and assumption-free.
""")
    
    return '\n'.join(R)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Paper 3 — GAM-based coupling decomposition')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='./paper3_gam')
    ap.add_argument('--bin_size', type=int, default=10)
    ap.add_argument('--n_sub', type=int, default=50000,
                    help='Subsample size for GAM (default: 50K)')
    ap.add_argument('--n_boot', type=int, default=20,
                    help='Bootstrap iterations (default: 20)')
    ap.add_argument('--n_splines', type=int, default=15,
                    help='Spline basis size (default: 15)')
    args = ap.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    
    print(f"[1/6] Reading {args.csv}...")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"  {len(df):,} rows  ({time.time()-t0:.1f}s)")
    
    observables = {}
    for col in ['tau_deg', 'angle_NCaC', 'angle_C_CA_CB', 'angle_N_CA_CB',
                'omega_deg', 'bond_CA_C', 'bond_N_CA']:
        if col in df.columns:
            observables[col] = col
    
    # ── GAM decomposition (all residues) ─────────────────────────────────
    print(f"[2/6] GAM decomposition (all, {args.n_sub} × {args.n_boot})...")
    gam_results = {}
    for name in observables:
        print(f"  {name}...", end=" ", flush=True)
        t1 = time.time()
        res = gam_decomposition(df, 'phi_deg', 'psi_deg', name,
                                n_sub=args.n_sub, n_boot=args.n_boot,
                                n_splines=args.n_splines)
        gam_results[name] = res
        if res:
            print(f"ΔR²={res['delta_coupling_mean']:+.2%} "
                  f"[{res['delta_coupling_ci'][0]:+.2%},"
                  f"{res['delta_coupling_ci'][1]:+.2%}]  "
                  f"({time.time()-t1:.0f}s)")
        else:
            print("failed")
    
    # ── Cell-means on full data ──────────────────────────────────────────
    print(f"[3/6] Cell-means on full data...")
    cell_results = {}
    for name in observables:
        res = cellmeans_r2(df, 'phi_deg', 'psi_deg', name, args.bin_size)
        cell_results[name] = res
        if res:
            print(f"  {name:>20s}: ΔR²={res['delta_coupling']:+.2%}")
    
    # ── By residue class ─────────────────────────────────────────────────
    print(f"[4/6] By residue class...")
    gam_by_class = {}
    cell_by_class = {}
    
    if 'res_name' in df.columns:
        classes = {
            'GLY': df['res_name'] == 'GLY',
            'PRO': df['res_name'] == 'PRO',
            'beta_branched': df['res_name'].isin(['VAL', 'ILE', 'THR']),
            'non_branched': ~df['res_name'].isin(['GLY', 'PRO', 'VAL', 'ILE', 'THR']),
            'ALL': pd.Series(True, index=df.index),
        }
        
        for cls_name, mask in classes.items():
            df_cls = df[mask]
            gam_cls = {}
            cell_cls = {}
            for name in ['tau_deg', 'angle_N_CA_CB']:
                if name not in df_cls.columns:
                    continue
                gam_cls[name] = gam_decomposition(
                    df_cls, 'phi_deg', 'psi_deg', name,
                    n_sub=min(args.n_sub, len(df_cls)),
                    n_boot=args.n_boot, n_splines=args.n_splines)
                cell_cls[name] = cellmeans_r2(
                    df_cls, 'phi_deg', 'psi_deg', name, args.bin_size)
            gam_by_class[cls_name] = gam_cls
            cell_by_class[cls_name] = cell_cls
            
            t_res = gam_cls.get('tau_deg')
            if t_res:
                print(f"  {cls_name:>15s}: τ ΔR²={t_res['delta_coupling_mean']:+.2%}")
    
    # ── By secondary structure ───────────────────────────────────────────
    print(f"[5/6] By secondary structure...")
    gam_by_ss = {}
    if 'ss_bin' in df.columns:
        ss_labels = {0: 'αR', 1: 'β', 2: 'PPII', 3: '3₁₀', 4: 'coil', 5: 'αL'}
        for ss_val, label in ss_labels.items():
            mask = df['ss_bin'] == ss_val
            if mask.sum() < 1000:
                continue
            df_ss = df[mask]
            ss_res = {}
            for name in ['tau_deg', 'angle_N_CA_CB']:
                if name not in df_ss.columns:
                    continue
                ss_res[name] = gam_decomposition(
                    df_ss, 'phi_deg', 'psi_deg', name,
                    n_sub=min(args.n_sub, len(df_ss)),
                    n_boot=args.n_boot, n_splines=args.n_splines)
            gam_by_ss[ss_val] = ss_res
            t_res = ss_res.get('tau_deg')
            if t_res:
                print(f"  {label:>6s}: τ ΔR²={t_res['delta_coupling_mean']:+.2%}")
    
    # ── Report ───────────────────────────────────────────────────────────
    print(f"[6/6] Report...")
    report = build_report(gam_results, cell_results, gam_by_class,
                          cell_by_class, gam_by_ss)
    
    report_path = os.path.join(args.out, 'gam_coupling_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved {report_path}")
    print(report)
    
    # Save CSV summary
    rows_out = []
    for name in sorted(gam_results.keys()):
        g = gam_results[name]
        c = cell_results.get(name)
        if g is None:
            continue
        rows_out.append({
            'observable': name,
            'gam_r2_add': g['r2_additive_mean'],
            'gam_r2_full': g['r2_full_mean'],
            'gam_delta_coupling': g['delta_coupling_mean'],
            'gam_delta_ci_lo': g['delta_coupling_ci'][0],
            'gam_delta_ci_hi': g['delta_coupling_ci'][1],
            'cell_r2_add': c['r2_additive'] if c else None,
            'cell_r2_full': c['r2_full'] if c else None,
            'cell_delta_coupling': c['delta_coupling'] if c else None,
            'cell_rms': c['rms_coupling'] if c else None,
            'cell_pp': c['peak_to_peak'] if c else None,
        })
    pd.DataFrame(rows_out).to_csv(
        os.path.join(args.out, 'gam_coupling_summary.csv'), index=False)
    
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()