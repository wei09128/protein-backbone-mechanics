#!/usr/bin/env python3
"""
Paper 3 — Coupling Structure Decomposition
============================================

Central question: Where does separability fail in backbone conformational
energy, and what is the correct coupling structure?

Classical force fields assume:
    E(φ,ψ) = E_φ(φ) + E_ψ(ψ)                        [SEPARABLE]

Reality (from PDB statistics):
    E(φ,ψ) = E_φ(φ) + E_ψ(ψ) + E_coupling(φ,ψ)     [COUPLED]

This script performs formal ANOVA-style decomposition of backbone
observables f(φ,ψ) into:

    f(φ,ψ) = f₀ + Δf_φ(φ) + Δf_ψ(ψ) + Δf_φψ(φ,ψ) + ε

where:
    f₀           = grand mean
    Δf_φ(φ)      = marginal φ effect (row mean - grand mean)
    Δf_ψ(ψ)      = marginal ψ effect (col mean - grand mean)
    Δf_φψ(φ,ψ)   = coupling residual (what separability misses)
    ε             = within-cell variance (noise + higher-order effects)

The key metric is:
    η² = Var(Δf_φψ) / Var(f - f₀)
    = fraction of systematic variance due to coupling

If η² is large → separability fails → classical FF is wrong
If η² is small → separability holds → classical FF is adequate

We compute this for:
    - τ (bond angle N-Cα-C)
    - ∠C-Cα-Cβ and ∠N-Cα-Cβ (Cβ bond angles)
    - ω (peptide planarity)
    - k_eff proxy (local curvature of population density)
    - All above stratified by residue class (Gly, Pro, β-branched, other)

Then we map WHERE the coupling is strongest on the Ramachandran plane.

Usage:
  python paper3_03_coupling_decomposition.py \\
      --csv /mnt/f/Protein_Folding/v6_GeometryDeformation/features.csv \\
      --out ./paper3_coupling/ --bin_size 10

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
# Core decomposition
# ══════════════════════════════════════════════════════════════════════════════

def anova2_decomposition(df, phi_col, psi_col, value_col, bin_size=10):
    """Two-way additive decomposition of value(φ,ψ).
    
    Returns a dict with:
        - grand_mean: f₀
        - phi_effect: Series indexed by φ-bin, values = Δf_φ(φ)
        - psi_effect: Series indexed by ψ-bin, values = Δf_ψ(ψ)
        - coupling: DataFrame indexed by (φ-bin, ψ-bin), values = Δf_φψ
        - cell_means: DataFrame of mean(f) per (φ,ψ) cell
        - cell_counts: DataFrame of count per (φ,ψ) cell
        - var_total: total variance of f
        - var_phi: variance explained by φ marginal
        - var_psi: variance explained by ψ marginal
        - var_coupling: variance of coupling residual
        - var_residual: within-cell variance
        - eta2_phi, eta2_psi, eta2_coupling: fraction of systematic variance
        - n_total: total observations
        - n_cells: number of populated cells
    """
    sub = df[[phi_col, psi_col, value_col]].dropna().copy()
    if len(sub) < 100:
        return None
    
    # Bin into grid
    phi_bins = np.arange(-180, 180 + bin_size, bin_size)
    psi_bins = np.arange(-180, 180 + bin_size, bin_size)
    
    sub['phi_bin'] = pd.cut(sub[phi_col], phi_bins, labels=False, right=False)
    sub['psi_bin'] = pd.cut(sub[psi_col], psi_bins, labels=False, right=False)
    sub = sub.dropna(subset=['phi_bin', 'psi_bin'])
    sub['phi_bin'] = sub['phi_bin'].astype(int)
    sub['psi_bin'] = sub['psi_bin'].astype(int)
    
    if len(sub) < 100:
        return None
    
    # Grand mean
    f0 = sub[value_col].mean()
    var_total = sub[value_col].var()
    
    # Cell means and counts
    cell_stats = sub.groupby(['phi_bin', 'psi_bin'])[value_col].agg(['mean', 'count', 'var'])
    cell_stats.columns = ['cell_mean', 'cell_count', 'cell_var']
    
    # Minimum count filter per cell
    MIN_COUNT = 5
    cell_stats = cell_stats[cell_stats['cell_count'] >= MIN_COUNT]
    
    if len(cell_stats) < 10:
        return None
    
    # Marginal means (weighted by cell count for proper ANOVA)
    # φ marginal: mean across all ψ for each φ-bin
    phi_marginal = (
        cell_stats.reset_index()
        .groupby('phi_bin')
        .apply(lambda g: np.average(g['cell_mean'], weights=g['cell_count']),
               include_groups=False)
    )
    
    # ψ marginal: mean across all φ for each ψ-bin
    psi_marginal = (
        cell_stats.reset_index()
        .groupby('psi_bin')
        .apply(lambda g: np.average(g['cell_mean'], weights=g['cell_count']),
               include_groups=False)
    )
    
    # Effects (deviations from grand mean)
    phi_effect = phi_marginal - f0
    psi_effect = psi_marginal - f0
    
    # Coupling: cell_mean - f0 - phi_effect - psi_effect
    coupling = cell_stats[['cell_mean', 'cell_count']].copy()
    coupling['phi_effect'] = coupling.index.get_level_values('phi_bin').map(phi_effect)
    coupling['psi_effect'] = coupling.index.get_level_values('psi_bin').map(psi_effect)
    coupling['additive_prediction'] = f0 + coupling['phi_effect'] + coupling['psi_effect']
    coupling['coupling_residual'] = coupling['cell_mean'] - coupling['additive_prediction']
    
    # Drop cells where we couldn't compute marginals
    coupling = coupling.dropna()
    
    if len(coupling) < 10:
        return None
    
    # Variance decomposition (weighted by cell counts)
    weights = coupling['cell_count'].values
    total_w = weights.sum()
    
    # Variance of the additive components (between-cell)
    phi_vals = coupling['phi_effect'].values
    psi_vals = coupling['psi_effect'].values
    coupling_vals = coupling['coupling_residual'].values
    cell_mean_vals = coupling['cell_mean'].values
    
    # Weighted variance = Σ w_i (x_i - x̄_w)² / Σ w_i
    def wvar(x, w):
        mu = np.average(x, weights=w)
        return np.average((x - mu)**2, weights=w)
    
    var_between_cells = wvar(cell_mean_vals - f0, weights)
    var_phi_marginal = wvar(phi_vals, weights)
    var_psi_marginal = wvar(psi_vals, weights)
    var_coupling = wvar(coupling_vals, weights)
    
    # Within-cell variance (average of cell variances, weighted)
    valid_var = coupling.join(cell_stats[['cell_var']]).dropna()
    var_within = np.average(valid_var['cell_var'].values,
                           weights=valid_var['cell_count'].values)
    
    # η² = fraction of between-cell variance
    var_sys = var_phi_marginal + var_psi_marginal + var_coupling
    if var_sys < 1e-12:
        eta2_phi = eta2_psi = eta2_coupling = 0.0
    else:
        eta2_phi = var_phi_marginal / var_sys
        eta2_psi = var_psi_marginal / var_sys
        eta2_coupling = var_coupling / var_sys
    
    # Also compute fraction of TOTAL variance
    if var_total < 1e-12:
        frac_phi = frac_psi = frac_coupling = frac_within = 0.0
    else:
        frac_phi = var_phi_marginal / var_total
        frac_psi = var_psi_marginal / var_total
        frac_coupling = var_coupling / var_total
        frac_within = var_within / var_total
    
    # Convert bin indices back to degrees for interpretability
    phi_centers = phi_bins[:-1] + bin_size / 2
    psi_centers = psi_bins[:-1] + bin_size / 2
    
    # Build coupling map as 2D array
    n_phi = len(phi_bins) - 1
    n_psi = len(psi_bins) - 1
    coupling_map = np.full((n_psi, n_phi), np.nan)
    count_map = np.full((n_psi, n_phi), 0)
    additive_map = np.full((n_psi, n_phi), np.nan)
    observed_map = np.full((n_psi, n_phi), np.nan)
    
    for (pb, qb), row in coupling.iterrows():
        if 0 <= int(pb) < n_phi and 0 <= int(qb) < n_psi:
            coupling_map[int(qb), int(pb)] = row['coupling_residual']
            count_map[int(qb), int(pb)] = int(row['cell_count'])
            additive_map[int(qb), int(pb)] = row['additive_prediction']
            observed_map[int(qb), int(pb)] = row['cell_mean']
    
    return {
        'grand_mean': f0,
        'phi_effect': phi_effect,
        'psi_effect': psi_effect,
        'coupling_df': coupling,
        'coupling_map': coupling_map,
        'count_map': count_map,
        'additive_map': additive_map,
        'observed_map': observed_map,
        'phi_centers': phi_centers,
        'psi_centers': psi_centers,
        'var_total': var_total,
        'var_phi': var_phi_marginal,
        'var_psi': var_psi_marginal,
        'var_coupling': var_coupling,
        'var_within': var_within,
        'eta2_phi': eta2_phi,
        'eta2_psi': eta2_psi,
        'eta2_coupling': eta2_coupling,
        'frac_phi': frac_phi,
        'frac_psi': frac_psi,
        'frac_coupling': frac_coupling,
        'frac_within': frac_within,
        'n_total': len(sub),
        'n_cells': len(coupling),
        'bin_size': bin_size,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Extended decomposition: f(φ, ψ, χ₁) three-way
# ══════════════════════════════════════════════════════════════════════════════

def threeway_decomposition(df, phi_col, psi_col, chi1_col, value_col, 
                           phi_bin_size=20, psi_bin_size=20):
    """Three-way decomposition: f(φ, ψ, χ₁) = f₀ + Δφ + Δψ + Δχ₁ 
       + Δφψ + Δφχ₁ + Δψχ₁ + Δφψχ₁
    
    χ₁ is binned into 3 rotamer states: gauche+ (0-120), trans (120-240), gauche- (240-360).
    We report variance fractions for each term.
    """
    sub = df[[phi_col, psi_col, chi1_col, value_col]].dropna().copy()
    if len(sub) < 200:
        return None
    
    # Bin φ, ψ
    phi_bins = np.arange(-180, 180 + phi_bin_size, phi_bin_size)
    psi_bins = np.arange(-180, 180 + psi_bin_size, psi_bin_size)
    sub['phi_bin'] = pd.cut(sub[phi_col], phi_bins, labels=False, right=False)
    sub['psi_bin'] = pd.cut(sub[psi_col], psi_bins, labels=False, right=False)
    
    # Bin χ₁ into 3 rotamers
    chi1_vals = sub[chi1_col].values.copy()
    # Convert radians to degrees if needed (chi1_rad is in [-π, π])
    if chi1_col == 'chi1_rad' or np.nanmax(np.abs(chi1_vals)) < 7:
        chi1_vals = np.degrees(chi1_vals)
    chi1_vals = chi1_vals % 360
    sub['chi1_bin'] = pd.cut(chi1_vals, bins=[0, 120, 240, 360], 
                             labels=['g+', 't', 'g-'], right=False)
    
    sub = sub.dropna(subset=['phi_bin', 'psi_bin', 'chi1_bin'])
    if len(sub) < 200:
        return None
    
    sub['phi_bin'] = sub['phi_bin'].astype(int)
    sub['psi_bin'] = sub['psi_bin'].astype(int)
    
    f0 = sub[value_col].mean()
    var_total = sub[value_col].var()
    
    # Cell means for 3D grid
    cells = sub.groupby(['phi_bin', 'psi_bin', 'chi1_bin'])[value_col].agg(['mean', 'count'])
    cells = cells[cells['count'] >= 3]
    
    if len(cells) < 20:
        return None
    
    # Marginals
    phi_m = sub.groupby('phi_bin')[value_col].mean()
    psi_m = sub.groupby('psi_bin')[value_col].mean()
    chi1_m = sub.groupby('chi1_bin')[value_col].mean()
    
    # 2-way marginals
    phi_psi_m = sub.groupby(['phi_bin', 'psi_bin'])[value_col].mean()
    phi_chi1_m = sub.groupby(['phi_bin', 'chi1_bin'])[value_col].mean()
    psi_chi1_m = sub.groupby(['psi_bin', 'chi1_bin'])[value_col].mean()
    
    # Variance explained by each factor (Type I sequential, approximate)
    # Use R² from linear models as proxy
    from sklearn.linear_model import LinearRegression
    
    y = sub[value_col].values
    
    # Encode factors
    phi_dum = pd.get_dummies(sub['phi_bin'], prefix='p').values
    psi_dum = pd.get_dummies(sub['psi_bin'], prefix='q').values
    chi1_dum = pd.get_dummies(sub['chi1_bin'], prefix='x').values
    
    def r2(X, y):
        if X.shape[1] == 0:
            return 0.0
        lr = LinearRegression().fit(X, y)
        return max(0, lr.score(X, y))
    
    r2_phi = r2(phi_dum, y)
    r2_psi = r2(psi_dum, y)
    r2_chi1 = r2(chi1_dum, y)
    r2_phi_psi = r2(np.hstack([phi_dum, psi_dum]), y)
    r2_phi_chi1 = r2(np.hstack([phi_dum, chi1_dum]), y)
    r2_psi_chi1 = r2(np.hstack([psi_dum, chi1_dum]), y)
    r2_all_additive = r2(np.hstack([phi_dum, psi_dum, chi1_dum]), y)
    
    # Interaction terms (cross products)
    phi_psi_inter = (phi_dum[:, :, None] * psi_dum[:, None, :]).reshape(len(y), -1)
    phi_chi1_inter = (phi_dum[:, :, None] * chi1_dum[:, None, :]).reshape(len(y), -1)
    psi_chi1_inter = (psi_dum[:, :, None] * chi1_dum[:, None, :]).reshape(len(y), -1)
    
    X_with_phipsi = np.hstack([phi_dum, psi_dum, chi1_dum, phi_psi_inter])
    X_with_all_2way = np.hstack([phi_dum, psi_dum, chi1_dum, 
                                  phi_psi_inter, phi_chi1_inter, psi_chi1_inter])
    
    r2_with_phipsi_coupling = r2(X_with_phipsi, y)
    r2_with_all_2way = r2(X_with_all_2way, y)
    
    return {
        'n': len(sub),
        'var_total': var_total,
        'r2_phi': r2_phi,
        'r2_psi': r2_psi,
        'r2_chi1': r2_chi1,
        'r2_phi_psi_additive': r2_phi_psi,
        'r2_all_additive': r2_all_additive,
        'r2_with_phipsi_coupling': r2_with_phipsi_coupling,
        'r2_with_all_2way': r2_with_all_2way,
        'delta_r2_phipsi_coupling': r2_with_phipsi_coupling - r2_all_additive,
        'delta_r2_chi1_couplings': r2_with_all_2way - r2_with_phipsi_coupling,
        'chi1_marginals': chi1_m.to_dict(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

def build_report(results_2way, results_3way, results_by_class):
    R = []
    R.append("=" * 78)
    R.append("Paper 3 — Coupling Structure of Backbone Conformational Geometry")
    R.append("=" * 78)
    
    # ── Section 1: Two-way decomposition (all residues) ──────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 1: TWO-WAY DECOMPOSITION  f(φ,ψ) = f₀ + Δf_φ + Δf_ψ + Δf_φψ")
    R.append("━" * 78)
    R.append("")
    R.append("  The coupling term Δf_φψ is what SEPARABLE force fields MISS.")
    R.append("  η²_coupling = fraction of systematic (between-cell) variance")
    R.append("  frac_coupling = fraction of TOTAL variance (incl. within-cell noise)")
    R.append("")
    
    header = (f"  {'Observable':>20s}  {'n':>8s}  {'cells':>5s}  "
              f"{'η²_φ':>7s}  {'η²_ψ':>7s}  {'η²_φψ':>7s}  "
              f"{'frac_φ':>7s}  {'frac_ψ':>7s}  {'frac_φψ':>8s}  {'frac_ε':>7s}")
    R.append(header)
    R.append("  " + "─" * len(header.strip()))
    
    for name, res in sorted(results_2way.items()):
        if res is None:
            continue
        R.append(
            f"  {name:>20s}  {res['n_total']:>8,}  {res['n_cells']:>5d}  "
            f"{res['eta2_phi']:>7.1%}  {res['eta2_psi']:>7.1%}  {res['eta2_coupling']:>7.1%}  "
            f"{res['frac_phi']:>7.1%}  {res['frac_psi']:>7.1%}  {res['frac_coupling']:>8.1%}  "
            f"{res['frac_within']:>7.1%}")
    
    # Interpretation
    R.append("")
    for name, res in sorted(results_2way.items()):
        if res is None:
            continue
        if res['eta2_coupling'] > 0.20:
            verdict = "STRONG coupling — separability FAILS"
        elif res['eta2_coupling'] > 0.10:
            verdict = "MODERATE coupling — separability partially fails"
        elif res['eta2_coupling'] > 0.05:
            verdict = "WEAK coupling — separability mostly holds"
        else:
            verdict = "NEGLIGIBLE coupling — fully separable"
        R.append(f"  {name:>20s}: {verdict}  (η²_φψ = {res['eta2_coupling']:.1%})")
    
    # ── Section 1b: Coupling magnitude ───────────────────────────────────
    R.append("")
    R.append("  COUPLING MAGNITUDE (peak-to-peak Δf_φψ):")
    for name, res in sorted(results_2way.items()):
        if res is None:
            continue
        cmap = res['coupling_map']
        valid = cmap[~np.isnan(cmap)]
        if len(valid) > 0:
            pp = valid.max() - valid.min()
            rms = np.sqrt(np.mean(valid**2))
            R.append(f"  {name:>20s}:  peak-to-peak = {pp:.3f}°  "
                     f"RMS = {rms:.3f}°  range [{valid.min():.3f}, {valid.max():.3f}]")
    
    # ── Section 2: Per-residue-class decomposition ───────────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 2: COUPLING BY RESIDUE CLASS")
    R.append("━" * 78)
    R.append("")
    R.append("  Does coupling strength depend on residue identity?")
    R.append("  β-branched residues may have stronger coupling due to")
    R.append("  sidechain–backbone geometric constraints.")
    R.append("")
    
    for cls_name in ['GLY', 'PRO', 'beta_branched', 'non_branched_other', 'ALL']:
        if cls_name not in results_by_class:
            continue
        R.append(f"  [{cls_name}]")
        cls_res = results_by_class[cls_name]
        
        header2 = (f"    {'Observable':>20s}  {'n':>8s}  "
                   f"{'η²_φ':>7s}  {'η²_ψ':>7s}  {'η²_φψ':>7s}  "
                   f"{'p-p coupling':>13s}")
        R.append(header2)
        
        for obs_name, res in sorted(cls_res.items()):
            if res is None:
                continue
            cmap = res['coupling_map']
            valid = cmap[~np.isnan(cmap)]
            pp = (valid.max() - valid.min()) if len(valid) > 0 else 0
            R.append(
                f"    {obs_name:>20s}  {res['n_total']:>8,}  "
                f"{res['eta2_phi']:>7.1%}  {res['eta2_psi']:>7.1%}  {res['eta2_coupling']:>7.1%}  "
                f"{pp:>12.3f}°")
        R.append("")
    
    # ── Section 3: Three-way decomposition ───────────────────────────────
    R.append("━" * 78)
    R.append("SECTION 3: THREE-WAY DECOMPOSITION  f(φ, ψ, χ₁)")
    R.append("━" * 78)
    R.append("")
    R.append("  Does χ₁ add independent information? Does it COUPLE with φ/ψ?")
    R.append("")
    
    if results_3way:
        header3 = (f"  {'Observable':>20s}  {'n':>7s}  "
                   f"{'R²_add':>7s}  {'R²+φψ':>7s}  {'R²+all2w':>8s}  "
                   f"{'Δ(φψ)':>7s}  {'Δ(χ₁×)':>8s}")
        R.append(header3)
        R.append("  " + "─" * len(header3.strip()))
        
        for name, res in sorted(results_3way.items()):
            if res is None:
                continue
            R.append(
                f"  {name:>20s}  {res['n']:>7,}  "
                f"{res['r2_all_additive']:>7.1%}  "
                f"{res['r2_with_phipsi_coupling']:>7.1%}  "
                f"{res['r2_with_all_2way']:>8.1%}  "
                f"{res['delta_r2_phipsi_coupling']:>+7.1%}  "
                f"{res['delta_r2_chi1_couplings']:>+8.1%}")
        
        R.append("")
        R.append("  R²_add = R² from φ + ψ + χ₁ (all additive, no interactions)")
        R.append("  R²+φψ  = R² after adding φ×ψ interaction terms")
        R.append("  R²+all2w = R² after adding ALL two-way interactions")
        R.append("  Δ(φψ) = R² gain from φ×ψ coupling (the separability failure)")
        R.append("  Δ(χ₁×) = additional R² gain from χ₁ couplings")
    
    # ── Section 4: Where does coupling localise? ─────────────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 4: WHERE DOES COUPLING LOCALISE ON THE RAMACHANDRAN PLANE?")
    R.append("━" * 78)
    R.append("")
    
    # For τ, find the cells with strongest coupling
    tau_res = results_2way.get('tau_deg')
    if tau_res is not None:
        cmap = tau_res['coupling_map']
        phi_c = tau_res['phi_centers']
        psi_c = tau_res['psi_centers']
        
        # Find top 10 cells by |coupling|
        cells_ranked = []
        for qi in range(cmap.shape[0]):
            for pi in range(cmap.shape[1]):
                if not np.isnan(cmap[qi, pi]):
                    cnt = tau_res['count_map'][qi, pi]
                    if cnt >= 10:
                        cells_ranked.append((
                            abs(cmap[qi, pi]), cmap[qi, pi],
                            phi_c[pi], psi_c[qi], cnt,
                            tau_res['observed_map'][qi, pi],
                            tau_res['additive_map'][qi, pi],
                        ))
        
        cells_ranked.sort(reverse=True)
        
        R.append("  TOP 15 CELLS BY |Δτ_coupling| (where separability fails most for τ):")
        R.append(f"    {'φ':>6s} {'ψ':>6s}  {'n':>6s}  {'⟨τ⟩_obs':>8s}  "
                 f"{'τ_additive':>10s}  {'Δτ_coupling':>12s}")
        R.append(f"    {'─'*6} {'─'*6}  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*12}")
        
        for i, (absv, val, phi, psi, n, obs, add) in enumerate(cells_ranked[:15]):
            R.append(f"    {phi:>6.0f} {psi:>6.0f}  {n:>6d}  {obs:>8.2f}°  "
                     f"{add:>10.2f}°  {val:>+12.3f}°")
        
        # Basin summary of coupling
        R.append("")
        R.append("  COUPLING BY BASIN (mean |Δτ_coupling|):")
        for basin_name, phi_range, psi_range in [
            ('αR core', (-80, -50), (-50, -20)),
            ('αR edge', (-110, -40), (-70, 0)),
            ('β core', (-150, -100), (110, 160)),
            ('β edge', (-170, -70), (80, 180)),
            ('PPII', (-85, -55), (120, 165)),
            ('αL', (30, 90), (10, 70)),
            ('bridge αR→β', (-100, -60), (0, 60)),
            ('φ≈-60 wall', (-70, -50), (-180, 180)),
            ('ψ≈-40 wall', (-180, 0), (-50, -30)),
        ]:
            mask_phi = (phi_c >= phi_range[0]) & (phi_c <= phi_range[1])
            mask_psi = (psi_c >= psi_range[0]) & (psi_c <= psi_range[1])
            
            region = cmap[np.ix_(mask_psi, mask_phi)]
            valid = region[~np.isnan(region)]
            if len(valid) >= 3:
                R.append(f"    {basin_name:20s}  n_cells={len(valid):>3d}  "
                         f"⟨|Δτ|⟩={np.mean(np.abs(valid)):.3f}°  "
                         f"max={np.max(np.abs(valid)):.3f}°  "
                         f"sign_bias={np.mean(valid):+.3f}°")
    
    # ── Section 5: Paper 1 connection ────────────────────────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 5: CONNECTION TO PAPER 1 STERIC DISCONTINUITIES")
    R.append("━" * 78)
    R.append("""
  Paper 1 found two orthogonal steric discontinuities:
    • Vertical wall in k_φ at φ ≈ -60°
    • Horizontal wall in k_ψ at ψ ≈ -40°
  
  If these walls are signatures of φ×ψ coupling, then Δf_φψ should show:
    • Sign change across φ = -60° (coupling flips direction)
    • Sign change across ψ = -40°
    • Maximum |Δf_φψ| at the intersection (αR region)
  
  CHECK the basin coupling values above:
    • Is coupling strongest near the αR region?
    • Does the coupling change sign between αR and β?
    • Is the "bridge" region (φ ≈ -80, ψ ≈ 0-60) anomalous?
""")
    
    # ── Section 6: Formula ───────────────────────────────────────────────
    R.append("━" * 78)
    R.append("SECTION 6: THE CORRECT ENERGY FORMULA")
    R.append("━" * 78)
    R.append("""
  Classical (separable):
    E(φ,ψ) = E_φ(φ) + E_ψ(ψ) + E_τ(τ₀) + E_nonlocal
  
  Corrected (this work):
    E(φ,ψ) = E_φ(φ) + E_ψ(ψ) + E_φψ(φ,ψ) + E_τ(τ|φ,ψ) + E_nonlocal
  
  where:
    E_φψ(φ,ψ) = coupling correction (from Δf_φψ maps above)
    E_τ(τ|φ,ψ) = bond-angle energy depends on WHICH (φ,ψ) bin
                  (not a global spring constant — Paper 2's finding)
  
  The QM result (from n→π* and HC analysis):
    E_QM_corrections ≈ 0 (already encoded in E_φψ geometry)
    → The coupling term IS the QM contribution, expressed classically
""")
    
    return '\n'.join(R)


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def make_plots(results_2way, results_by_class, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        print("  matplotlib not available")
        return
    
    # ── Plot 1: Coupling maps for all observables ────────────────────────
    obs_list = [k for k in ['tau_deg', 'angle_C_CA_CB', 'angle_N_CA_CB', 
                             'omega_deg']
                if k in results_2way and results_2way[k] is not None]
    
    if obs_list:
        n_obs = len(obs_list)
        fig, axes = plt.subplots(2, n_obs, figsize=(5 * n_obs, 10))
        if n_obs == 1:
            axes = axes.reshape(2, 1)
        
        for col_idx, obs in enumerate(obs_list):
            res = results_2way[obs]
            phi_c = res['phi_centers']
            psi_c = res['psi_centers']
            
            # Top row: observed mean
            ax = axes[0, col_idx]
            obs_map = res['observed_map']
            valid = obs_map[~np.isnan(obs_map)]
            if len(valid) > 0:
                vmin, vmax = np.percentile(valid, [2, 98])
                im = ax.pcolormesh(phi_c, psi_c, obs_map,
                                   cmap='RdBu_r', vmin=vmin, vmax=vmax,
                                   shading='auto')
                plt.colorbar(im, ax=ax, label='°')
            ax.set_title(f'⟨{obs}⟩ observed')
            ax.set_xlabel('φ')
            ax.set_ylabel('ψ')
            ax.set_xlim(-180, 180)
            ax.set_ylim(-180, 180)
            ax.set_aspect('equal')
            
            # Bottom row: coupling residual
            ax = axes[1, col_idx]
            cmap_data = res['coupling_map']
            valid = cmap_data[~np.isnan(cmap_data)]
            if len(valid) > 0:
                vlim = max(abs(np.percentile(valid, 2)),
                          abs(np.percentile(valid, 98)))
                if vlim < 0.01:
                    vlim = 1.0
                im = ax.pcolormesh(phi_c, psi_c, cmap_data,
                                   cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                                   shading='auto')
                plt.colorbar(im, ax=ax, label='°')
            ax.set_title(f'Δ{obs}_coupling (φ×ψ)')
            ax.set_xlabel('φ')
            ax.set_ylabel('ψ')
            ax.set_xlim(-180, 180)
            ax.set_ylim(-180, 180)
            ax.set_aspect('equal')
            
            # Mark Paper 1 discontinuities
            ax.axvline(-60, color='lime', ls='--', lw=1, alpha=0.7)
            ax.axhline(-40, color='lime', ls='--', lw=1, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'coupling_maps.png'), dpi=150)
        plt.close()
        print(f"  Saved coupling_maps.png")
    
    # ── Plot 2: Marginal effects (φ and ψ profiles) ─────────────────────
    if 'tau_deg' in results_2way and results_2way['tau_deg'] is not None:
        res = results_2way['tau_deg']
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # φ marginal
        ax = axes[0]
        phi_eff = res['phi_effect']
        phi_c = res['phi_centers']
        ax.plot(phi_c[phi_eff.index.astype(int)], phi_eff.values, 'b.-')
        ax.axhline(0, color='gray', ls='-', lw=0.5)
        ax.axvline(-60, color='red', ls='--', alpha=0.5, label='φ = -60°')
        ax.set_xlabel('φ [°]')
        ax.set_ylabel('Δτ_φ [°]')
        ax.set_title('φ marginal effect on τ')
        ax.legend()
        
        # ψ marginal
        ax = axes[1]
        psi_eff = res['psi_effect']
        psi_c = res['psi_centers']
        ax.plot(psi_c[psi_eff.index.astype(int)], psi_eff.values, 'r.-')
        ax.axhline(0, color='gray', ls='-', lw=0.5)
        ax.axhline(0, color='gray', ls='-', lw=0.5)
        ax.axvline(-40, color='blue', ls='--', alpha=0.5, label='ψ = -40°')
        ax.set_xlabel('ψ [°]')
        ax.set_ylabel('Δτ_ψ [°]')
        ax.set_title('ψ marginal effect on τ')
        ax.legend()
        
        # Variance pie chart
        ax = axes[2]
        fracs = [res['frac_phi'], res['frac_psi'], res['frac_coupling'],
                 res['frac_within']]
        labels = [f"φ marginal\n({fracs[0]:.1%})",
                  f"ψ marginal\n({fracs[1]:.1%})",
                  f"φ×ψ coupling\n({fracs[2]:.1%})",
                  f"within-cell\n({fracs[3]:.1%})"]
        colors = ['#2166AC', '#B2182B', '#7A0177', '#CCCCCC']
        ax.pie(fracs, labels=labels, colors=colors, autopct='',
               startangle=90, textprops={'fontsize': 9})
        ax.set_title('Variance decomposition of τ')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'tau_decomposition.png'), dpi=150)
        plt.close()
        print(f"  Saved tau_decomposition.png")
    
    # ── Plot 3: η² comparison across residue classes ─────────────────────
    if results_by_class:
        classes = [c for c in ['GLY', 'PRO', 'beta_branched', 'non_branched_other', 'ALL']
                   if c in results_by_class]
        obs = 'tau_deg'
        
        if all(obs in results_by_class[c] and results_by_class[c][obs] is not None 
               for c in classes):
            fig, ax = plt.subplots(figsize=(8, 5))
            
            x = np.arange(len(classes))
            eta_phi = [results_by_class[c][obs]['eta2_phi'] for c in classes]
            eta_psi = [results_by_class[c][obs]['eta2_psi'] for c in classes]
            eta_coup = [results_by_class[c][obs]['eta2_coupling'] for c in classes]
            
            w = 0.25
            ax.bar(x - w, eta_phi, w, label='η²_φ', color='#2166AC')
            ax.bar(x, eta_psi, w, label='η²_ψ', color='#B2182B')
            ax.bar(x + w, eta_coup, w, label='η²_φψ coupling', color='#7A0177')
            
            ax.set_xticks(x)
            ax.set_xticklabels(classes, fontsize=9)
            ax.set_ylabel('η² (fraction of systematic variance)')
            ax.set_title('Coupling strength by residue class (τ)')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'coupling_by_class.png'), dpi=150)
            plt.close()
            print(f"  Saved coupling_by_class.png")
    
    # ── Plot 4: Save coupling maps as CSV for later use ──────────────────
    for obs_name, res in results_2way.items():
        if res is None:
            continue
        # Save coupling map
        cmap = res['coupling_map']
        phi_c = res['phi_centers']
        psi_c = res['psi_centers']
        
        rows_out = []
        for qi in range(cmap.shape[0]):
            for pi in range(cmap.shape[1]):
                if not np.isnan(cmap[qi, pi]):
                    rows_out.append({
                        'phi': phi_c[pi], 'psi': psi_c[qi],
                        'coupling': cmap[qi, pi],
                        'observed': res['observed_map'][qi, pi],
                        'additive': res['additive_map'][qi, pi],
                        'count': res['count_map'][qi, pi],
                    })
        pd.DataFrame(rows_out).to_csv(
            os.path.join(out_dir, f'{obs_name}_coupling_map.csv'), index=False)
    print(f"  Saved coupling map CSVs")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Paper 3 — Coupling structure decomposition')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='./paper3_coupling')
    ap.add_argument('--bin_size', type=int, default=10,
                    help='Ramachandran bin size in degrees (default: 10)')
    args = ap.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    
    # ── Read CSV ─────────────────────────────────────────────────────────
    print(f"[1/5] Reading {args.csv}...")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns  ({time.time()-t0:.1f}s)")
    
    # ── Two-way decomposition (all residues) ─────────────────────────────
    print(f"[2/5] Two-way decomposition (all residues)...")
    
    observables = {}
    for col in ['tau_deg', 'angle_NCaC', 'angle_C_CA_CB', 'angle_N_CA_CB',
                'omega_deg', 'bond_CA_C', 'bond_N_CA', 'bond_C_N']:
        if col in df.columns:
            observables[col] = col
    
    results_2way = {}
    for name, col in observables.items():
        res = anova2_decomposition(df, 'phi_deg', 'psi_deg', col, args.bin_size)
        results_2way[name] = res
        if res:
            print(f"  {name:>20s}: η²_coupling = {res['eta2_coupling']:.1%}  "
                  f"(frac_total = {res['frac_coupling']:.1%})")
    
    # ── Per-class decomposition ──────────────────────────────────────────
    print(f"[3/5] Per-class decomposition...")
    
    # Define classes
    if 'res_name' in df.columns:
        classes = {
            'GLY': df['res_name'] == 'GLY',
            'PRO': df['res_name'] == 'PRO',
            'beta_branched': df['res_name'].isin(['VAL', 'ILE', 'THR']),
            'non_branched_other': ~df['res_name'].isin(['GLY', 'PRO', 'VAL', 'ILE', 'THR']),
            'ALL': pd.Series(True, index=df.index),
        }
    else:
        classes = {'ALL': pd.Series(True, index=df.index)}
    
    results_by_class = {}
    for cls_name, mask in classes.items():
        df_cls = df[mask]
        cls_results = {}
        for name, col in observables.items():
            res = anova2_decomposition(df_cls, 'phi_deg', 'psi_deg', col, args.bin_size)
            cls_results[name] = res
        results_by_class[cls_name] = cls_results
        
        tau_res = cls_results.get('tau_deg')
        if tau_res:
            print(f"  {cls_name:>20s}: τ η²_coupling = {tau_res['eta2_coupling']:.1%}  "
                  f"(n={tau_res['n_total']:,})")
    
    # ── Three-way decomposition (φ, ψ, χ₁) ──────────────────────────────
    print(f"[4/5] Three-way decomposition (φ, ψ, χ₁)...")
    
    results_3way = {}
    chi1_col = None
    for c in ['chi1_deg', 'chi1', 'chi1_rad']:
        if c in df.columns:
            chi1_col = c
            break
    
    if chi1_col:
        # Only for residues that have χ₁ (exclude Gly, Ala, and has_chi1==0 placeholders)
        if 'res_name' in df.columns:
            mask = ~df['res_name'].isin(['GLY', 'ALA'])
            if 'has_chi1' in df.columns:
                mask = mask & (df['has_chi1'] == 1)
            df_chi = df[mask].copy()
        else:
            df_chi = df.copy()
        
        for name, col in observables.items():
            if col in df_chi.columns:
                res = threeway_decomposition(
                    df_chi, 'phi_deg', 'psi_deg', chi1_col, col,
                    phi_bin_size=20, psi_bin_size=20)
                results_3way[name] = res
                if res:
                    print(f"  {name:>20s}: R²_add={res['r2_all_additive']:.1%}  "
                          f"Δ(φψ)={res['delta_r2_phipsi_coupling']:+.1%}  "
                          f"Δ(χ₁×)={res['delta_r2_chi1_couplings']:+.1%}")
    else:
        print("  χ₁ column not found — skipping 3-way decomposition")
    
    # ── Report and plots ─────────────────────────────────────────────────
    print(f"[5/5] Report and plots...")
    
    report = build_report(results_2way, results_3way, results_by_class)
    
    report_path = os.path.join(args.out, 'coupling_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved {report_path}")
    print(report)
    
    make_plots(results_2way, results_by_class, args.out)
    
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()