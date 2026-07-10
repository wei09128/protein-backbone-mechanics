#!/usr/bin/env python3
"""
Paper 3 — Steric Wall Sign-Change Analysis
=============================================

Paper 1 found two orthogonal steric discontinuities:
  • Vertical wall in k_φ at φ ≈ -60°
  • Horizontal wall in k_ψ at ψ ≈ -40°

Paper 3 script 03 showed that φ×ψ coupling is strongest AT these walls.
This script tests the specific prediction: if the walls are COUPLING
BOUNDARIES, then Δf_φψ must change sign across them.

Concretely:
  1. For the φ-wall: compare ⟨Δτ_coupling⟩ at φ < -60° vs φ > -60°
     holding ψ constant. If the sign flips → the wall is a coupling node.
  
  2. For the ψ-wall: compare ⟨Δτ_coupling⟩ at ψ < -40° vs ψ > -40°
     holding φ constant.
  
  3. Scan the wall position: find the φ* and ψ* that MAXIMIZE the sign
     change. If the optimal positions match Paper 1's walls, that's the
     confirmation.

  4. Test all observables (τ, ∠C-Cα-Cβ, ∠N-Cα-Cβ, bond lengths).

  5. Stratify by residue class to check if the walls shift for
     Gly/Pro/β-branched.

Usage:
  python paper3_04_wall_sign_change.py \
      --coupling_dir ./paper3_coupling/ \
      --csv /mnt/f/Protein_Folding/v8_g/p3.csv \
      --out ./paper3_walls/ --bin_size 10

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
# Coupling decomposition (inline — reused from script 03)
# ══════════════════════════════════════════════════════════════════════════════

def compute_coupling_map(df, phi_col, psi_col, value_col, bin_size=10,
                         min_count=5):
    """Compute the coupling residual map Δf_φψ(φ,ψ).
    
    Returns: (coupling_map, count_map, phi_centers, psi_centers, grand_mean)
    All as numpy arrays. coupling_map[ψ_idx, φ_idx] = Δf_coupling.
    """
    sub = df[[phi_col, psi_col, value_col]].dropna().copy()
    if len(sub) < 100:
        return None

    phi_bins = np.arange(-180, 180 + bin_size, bin_size)
    psi_bins = np.arange(-180, 180 + bin_size, bin_size)
    phi_centers = phi_bins[:-1] + bin_size / 2
    psi_centers = psi_bins[:-1] + bin_size / 2

    sub['phi_bin'] = pd.cut(sub[phi_col], phi_bins, labels=False, right=False)
    sub['psi_bin'] = pd.cut(sub[psi_col], psi_bins, labels=False, right=False)
    sub = sub.dropna(subset=['phi_bin', 'psi_bin'])
    sub['phi_bin'] = sub['phi_bin'].astype(int)
    sub['psi_bin'] = sub['psi_bin'].astype(int)

    f0 = sub[value_col].mean()

    cell_stats = sub.groupby(['phi_bin', 'psi_bin'])[value_col].agg(['mean', 'count'])
    cell_stats.columns = ['cell_mean', 'cell_count']
    cell_stats = cell_stats[cell_stats['cell_count'] >= min_count]

    if len(cell_stats) < 10:
        return None

    phi_marginal = (cell_stats.reset_index()
                    .groupby('phi_bin')
                    .apply(lambda g: np.average(g['cell_mean'],
                                                weights=g['cell_count']),
                           include_groups=False))
    psi_marginal = (cell_stats.reset_index()
                    .groupby('psi_bin')
                    .apply(lambda g: np.average(g['cell_mean'],
                                                weights=g['cell_count']),
                           include_groups=False))

    n_phi = len(phi_bins) - 1
    n_psi = len(psi_bins) - 1
    coupling_map = np.full((n_psi, n_phi), np.nan)
    count_map = np.full((n_psi, n_phi), 0)

    for (pb, qb), row in cell_stats.iterrows():
        phi_eff = phi_marginal.get(pb, np.nan)
        psi_eff = psi_marginal.get(qb, np.nan)
        if np.isnan(phi_eff) or np.isnan(psi_eff):
            continue
        additive = f0 + (phi_eff - f0) + (psi_eff - f0)
        coupling_map[int(qb), int(pb)] = row['cell_mean'] - additive
        count_map[int(qb), int(pb)] = int(row['cell_count'])

    return coupling_map, count_map, phi_centers, psi_centers, f0


# ══════════════════════════════════════════════════════════════════════════════
# Wall sign-change analysis
# ══════════════════════════════════════════════════════════════════════════════

def wall_sign_change(coupling_map, count_map, phi_centers, psi_centers,
                     wall_axis, wall_position, min_cells=3):
    """Test sign change of coupling across a wall.
    
    Args:
        wall_axis: 'phi' or 'psi' — which axis the wall is perpendicular to
        wall_position: the wall location in degrees
    
    Returns dict with sign-change statistics.
    """
    if wall_axis == 'phi':
        # Split columns: left (φ < wall) vs right (φ > wall)
        left_idx = np.where(phi_centers < wall_position)[0]
        right_idx = np.where(phi_centers >= wall_position)[0]
        
        # For each ψ row, compute mean coupling on each side
        left_vals, right_vals, psi_vals = [], [], []
        for qi in range(len(psi_centers)):
            left_cells = [(coupling_map[qi, pi], count_map[qi, pi])
                         for pi in left_idx
                         if not np.isnan(coupling_map[qi, pi])
                         and count_map[qi, pi] >= 5]
            right_cells = [(coupling_map[qi, pi], count_map[qi, pi])
                          for pi in right_idx
                          if not np.isnan(coupling_map[qi, pi])
                          and count_map[qi, pi] >= 5]
            
            if len(left_cells) >= min_cells and len(right_cells) >= min_cells:
                lw = np.array([c[1] for c in left_cells])
                lv = np.array([c[0] for c in left_cells])
                rw = np.array([c[1] for c in right_cells])
                rv = np.array([c[0] for c in right_cells])
                left_vals.append(np.average(lv, weights=lw))
                right_vals.append(np.average(rv, weights=rw))
                psi_vals.append(psi_centers[qi])
        
    else:  # wall_axis == 'psi'
        below_idx = np.where(psi_centers < wall_position)[0]
        above_idx = np.where(psi_centers >= wall_position)[0]
        
        left_vals, right_vals, psi_vals = [], [], []
        for pi in range(len(phi_centers)):
            below_cells = [(coupling_map[qi, pi], count_map[qi, pi])
                          for qi in below_idx
                          if not np.isnan(coupling_map[qi, pi])
                          and count_map[qi, pi] >= 5]
            above_cells = [(coupling_map[qi, pi], count_map[qi, pi])
                          for qi in above_idx
                          if not np.isnan(coupling_map[qi, pi])
                          and count_map[qi, pi] >= 5]
            
            if len(below_cells) >= min_cells and len(above_cells) >= min_cells:
                bw = np.array([c[1] for c in below_cells])
                bv = np.array([c[0] for c in below_cells])
                aw = np.array([c[1] for c in above_cells])
                av = np.array([c[0] for c in above_cells])
                left_vals.append(np.average(bv, weights=bw))
                right_vals.append(np.average(av, weights=aw))
                psi_vals.append(phi_centers[pi])
    
    if len(left_vals) < 3:
        return None
    
    left_arr = np.array(left_vals)
    right_arr = np.array(right_vals)
    diff = right_arr - left_arr
    
    # Sign change: how many rows/columns flip sign?
    n_flip = np.sum(np.sign(left_arr) != np.sign(right_arr))
    
    # Mean values on each side
    mean_left = np.mean(left_arr)
    mean_right = np.mean(right_arr)
    
    # Paired t-test
    t_stat, p_val = sp_stats.ttest_rel(left_arr, right_arr)
    
    # Effect size (Cohen's d)
    d_pool = np.sqrt((np.var(left_arr) + np.var(right_arr)) / 2)
    cohens_d = (mean_right - mean_left) / d_pool if d_pool > 1e-10 else 0.0
    
    return {
        'wall_axis': wall_axis,
        'wall_position': wall_position,
        'n_slices': len(left_vals),
        'mean_left': mean_left,
        'mean_right': mean_right,
        'mean_diff': np.mean(diff),
        'n_sign_flip': n_flip,
        'frac_sign_flip': n_flip / len(left_vals),
        't_stat': t_stat,
        'p_val': p_val,
        'cohens_d': cohens_d,
    }


def scan_wall_position(coupling_map, count_map, phi_centers, psi_centers,
                       wall_axis, scan_range, min_cells=3):
    """Scan wall position to find the one that maximizes sign change."""
    results = []
    for pos in scan_range:
        res = wall_sign_change(coupling_map, count_map, phi_centers,
                               psi_centers, wall_axis, pos, min_cells)
        if res is not None:
            results.append(res)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Cross-wall profile extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_cross_wall_profiles(coupling_map, count_map, phi_centers,
                                 psi_centers, wall_axis, wall_pos):
    """Extract coupling profiles perpendicular to the wall.
    
    Returns list of (slice_position, values_before, values_after) tuples.
    """
    profiles = []
    
    if wall_axis == 'phi':
        # For selected ψ values, extract coupling vs φ profile
        for qi in range(len(psi_centers)):
            vals = []
            for pi in range(len(phi_centers)):
                if not np.isnan(coupling_map[qi, pi]) and count_map[qi, pi] >= 5:
                    vals.append((phi_centers[pi], coupling_map[qi, pi],
                                count_map[qi, pi]))
            if len(vals) >= 5:
                profiles.append((psi_centers[qi], vals))
    else:
        for pi in range(len(phi_centers)):
            vals = []
            for qi in range(len(psi_centers)):
                if not np.isnan(coupling_map[qi, pi]) and count_map[qi, pi] >= 5:
                    vals.append((psi_centers[qi], coupling_map[qi, pi],
                                count_map[qi, pi]))
            if len(vals) >= 5:
                profiles.append((phi_centers[pi], vals))
    
    return profiles


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

def build_report(results_all, scan_results, profile_results, 
                 results_by_class):
    R = []
    R.append("=" * 78)
    R.append("Paper 3 — Steric Wall Sign-Change Analysis")
    R.append("=" * 78)
    
    # ── Section 1: Sign change at Paper 1 wall positions ─────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 1: SIGN CHANGE AT PAPER 1 WALL POSITIONS")
    R.append("━" * 78)
    R.append("")
    R.append("  Paper 1 walls: φ* = -60°, ψ* = -40°")
    R.append("  If these are coupling boundaries, Δf_φψ flips sign across them.")
    R.append("")
    
    header = (f"  {'Observable':>20s}  {'Wall':>5s}  {'Pos':>5s}  "
              f"{'⟨left⟩':>8s}  {'⟨right⟩':>8s}  {'Δ':>8s}  "
              f"{'flip%':>6s}  {'t':>7s}  {'p':>10s}  {'d':>7s}")
    R.append(header)
    R.append("  " + "─" * (len(header.strip())))
    
    for obs_name, res_list in sorted(results_all.items()):
        for res in res_list:
            star = '***' if res['p_val'] < 0.001 else '**' if res['p_val'] < 0.01 else '*' if res['p_val'] < 0.05 else ''
            R.append(
                f"  {obs_name:>20s}  {res['wall_axis']:>5s}  "
                f"{res['wall_position']:>5.0f}  "
                f"{res['mean_left']:>+8.4f}  {res['mean_right']:>+8.4f}  "
                f"{res['mean_diff']:>+8.4f}  "
                f"{res['frac_sign_flip']:>5.0%}  "
                f"{res['t_stat']:>+7.2f}  {res['p_val']:>10.2e}  "
                f"{res['cohens_d']:>+7.3f} {star}")
    
    # ── Section 2: Optimal wall position scan ────────────────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 2: OPTIMAL WALL POSITION (maximize |sign change|)")
    R.append("━" * 78)
    R.append("")
    R.append("  Scanning wall position to find φ* and ψ* that maximize")
    R.append("  the coupling sign change. If optimal ≈ Paper 1 values,")
    R.append("  the walls are confirmed as coupling boundaries.")
    R.append("")
    
    for obs_name, scans in sorted(scan_results.items()):
        R.append(f"  [{obs_name}]")
        for wall_axis in ['phi', 'psi']:
            axis_scans = [s for s in scans if s['wall_axis'] == wall_axis]
            if not axis_scans:
                continue
            
            # Find position with maximum |mean_diff|
            best = max(axis_scans, key=lambda s: abs(s['mean_diff']))
            # Find position with maximum |t_stat|
            best_t = max(axis_scans, key=lambda s: abs(s['t_stat']))
            # Find position with maximum sign flip fraction
            best_flip = max(axis_scans, key=lambda s: s['frac_sign_flip'])
            
            R.append(f"    {wall_axis}-wall:")
            R.append(f"      max |Δ|:       {wall_axis}* = {best['wall_position']:>6.0f}°  "
                     f"|Δ| = {abs(best['mean_diff']):.4f}°")
            R.append(f"      max |t|:       {wall_axis}* = {best_t['wall_position']:>6.0f}°  "
                     f"|t| = {abs(best_t['t_stat']):.2f}")
            R.append(f"      max flip%:     {wall_axis}* = {best_flip['wall_position']:>6.0f}°  "
                     f"flip = {best_flip['frac_sign_flip']:.0%}")
            
            # Comparison with Paper 1
            paper1_pos = -60.0 if wall_axis == 'phi' else -40.0
            paper1_scan = [s for s in axis_scans
                          if abs(s['wall_position'] - paper1_pos) < 1]
            if paper1_scan:
                p1 = paper1_scan[0]
                R.append(f"      Paper 1 pos:   {wall_axis}* = {paper1_pos:>6.0f}°  "
                         f"|Δ| = {abs(p1['mean_diff']):.4f}°  "
                         f"|t| = {abs(p1['t_stat']):.2f}")
                
                # Is Paper 1 position within 10° of optimal?
                if abs(best['wall_position'] - paper1_pos) <= 15:
                    R.append(f"      → CONFIRMED: optimal within 15° of Paper 1")
                else:
                    R.append(f"      → SHIFTED: optimal differs from Paper 1 by "
                             f"{abs(best['wall_position'] - paper1_pos):.0f}°")
        R.append("")
    
    # ── Section 3: Per-class wall positions ──────────────────────────────
    R.append("━" * 78)
    R.append("SECTION 3: WALL POSITION BY RESIDUE CLASS")
    R.append("━" * 78)
    R.append("")
    R.append("  Do the walls shift for Gly/Pro/β-branched?")
    R.append("")
    
    for cls_name, cls_scans in sorted(results_by_class.items()):
        R.append(f"  [{cls_name}]")
        for wall_axis in ['phi', 'psi']:
            axis_scans = [s for s in cls_scans if s['wall_axis'] == wall_axis]
            if not axis_scans:
                continue
            best = max(axis_scans, key=lambda s: abs(s['mean_diff']))
            paper1_pos = -60.0 if wall_axis == 'phi' else -40.0
            shift = best['wall_position'] - paper1_pos
            R.append(f"    {wall_axis}-wall: optimal = {best['wall_position']:>6.0f}°  "
                     f"(Δ from Paper 1: {shift:+.0f}°)  "
                     f"|Δcoupling| = {abs(best['mean_diff']):.4f}°")
        R.append("")
    
    # ── Section 4: Cross-wall profiles ───────────────────────────────────
    R.append("━" * 78)
    R.append("SECTION 4: CROSS-WALL COUPLING PROFILES")
    R.append("━" * 78)
    R.append("")
    R.append("  Coupling values along φ at selected ψ slices (crossing φ-wall):")
    R.append("  Look for sign change near φ = -60°")
    R.append("")
    
    if 'tau_deg' in profile_results:
        phi_profiles = profile_results['tau_deg'].get('phi', [])
        # Show profiles for ψ near key basins
        target_psi = [-25, -45, 135, 155]  # αR, wall, β, PPII
        for target in target_psi:
            # Find closest profile
            closest = min(phi_profiles, 
                         key=lambda p: abs(p[0] - target),
                         default=None)
            if closest is None or abs(closest[0] - target) > 10:
                continue
            psi_val, vals = closest
            R.append(f"  ψ ≈ {psi_val:.0f}°:")
            line = "    "
            for phi_v, coup_v, cnt in vals:
                marker = " ←" if abs(phi_v - (-60)) < 6 else ""
                line += f"  φ={phi_v:+4.0f}:{coup_v:+.2f}{marker}"
            R.append(line)
        R.append("")
        
        R.append("  Coupling values along ψ at selected φ slices (crossing ψ-wall):")
        R.append("  Look for sign change near ψ = -40°")
        R.append("")
        
        psi_profiles = profile_results['tau_deg'].get('psi', [])
        target_phi = [-65, -85, -125, -155]
        for target in target_phi:
            closest = min(psi_profiles,
                         key=lambda p: abs(p[0] - target),
                         default=None)
            if closest is None or abs(closest[0] - target) > 10:
                continue
            phi_val, vals = closest
            R.append(f"  φ ≈ {phi_val:.0f}°:")
            line = "    "
            for psi_v, coup_v, cnt in vals:
                marker = " ←" if abs(psi_v - (-40)) < 6 else ""
                line += f"  ψ={psi_v:+4.0f}:{coup_v:+.2f}{marker}"
            R.append(line)
    
    # ── Section 5: Verdict ───────────────────────────────────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 5: VERDICT")
    R.append("━" * 78)
    R.append("""
  The steric walls are coupling boundaries if ALL of:
    ✓ Coupling changes sign across φ = -60° and ψ = -40°
    ✓ The sign change is statistically significant (p < 0.01)
    ✓ Optimal wall positions match Paper 1 (within 15°)
    ✓ The effect is universal (present in all residue classes)
    ✓ Gly shows STRONGER coupling (no sidechain to constrain)
    ✓ Pro shows WEAKER coupling (rigid ring constrains φ)
  
  If confirmed → Paper 3 title:
    "The coupling structure of backbone conformational geometry:
     why separable force fields fail at steric boundaries"
  
  Paper 3 formula:
    E(φ,ψ) = E_φ(φ) + E_ψ(ψ) + E_φψ(φ,ψ)
    where E_φψ is the coupling correction map provided as
    supplementary data (a lookup table for force-field correction).
""")
    
    return '\n'.join(R)


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def make_plots(scan_results, profile_results, coupling_maps, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    # ── Plot 1: Wall position scan ───────────────────────────────────────
    for obs_name in ['tau_deg', 'angle_N_CA_CB']:
        if obs_name not in scan_results:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for ax, wall_axis, paper1_pos, label in [
            (axes[0], 'phi', -60, 'φ-wall'),
            (axes[1], 'psi', -40, 'ψ-wall'),
        ]:
            scans = [s for s in scan_results[obs_name]
                    if s['wall_axis'] == wall_axis]
            if not scans:
                continue
            
            positions = [s['wall_position'] for s in scans]
            diffs = [abs(s['mean_diff']) for s in scans]
            t_stats = [abs(s['t_stat']) for s in scans]
            
            ax.plot(positions, diffs, 'b.-', label='|Δ coupling|')
            ax.axvline(paper1_pos, color='red', ls='--', lw=2,
                      label=f'Paper 1: {paper1_pos}°')
            
            best = max(scans, key=lambda s: abs(s['mean_diff']))
            ax.axvline(best['wall_position'], color='green', ls=':',
                      lw=2, label=f'Optimal: {best["wall_position"]:.0f}°')
            
            ax.set_xlabel(f'{wall_axis} wall position [°]')
            ax.set_ylabel(f'|Δ coupling| for {obs_name} [°]')
            ax.set_title(f'{label} scan — {obs_name}')
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'wall_scan_{obs_name}.png'), dpi=150)
        plt.close()
        print(f"  Saved wall_scan_{obs_name}.png")
    
    # ── Plot 2: Cross-wall profiles for τ ────────────────────────────────
    if 'tau_deg' in profile_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # φ-profiles at different ψ
        ax = axes[0]
        phi_profs = profile_results['tau_deg'].get('phi', [])
        target_psi = [-25, -45, 135, 155]
        colors = ['#2166AC', '#B2182B', '#4DAF4A', '#984EA3']
        for target, color in zip(target_psi, colors):
            closest = min(phi_profs, key=lambda p: abs(p[0] - target),
                         default=None)
            if closest is None or abs(closest[0] - target) > 10:
                continue
            psi_val, vals = closest
            x = [v[0] for v in vals]
            y = [v[1] for v in vals]
            ax.plot(x, y, '.-', color=color, label=f'ψ≈{psi_val:.0f}°',
                   markersize=4)
        ax.axvline(-60, color='red', ls='--', lw=1.5, alpha=0.7)
        ax.axhline(0, color='gray', ls='-', lw=0.5)
        ax.set_xlabel('φ [°]')
        ax.set_ylabel('Δτ_coupling [°]')
        ax.set_title('τ coupling across φ-wall at different ψ')
        ax.legend(fontsize=8)
        
        # ψ-profiles at different φ
        ax = axes[1]
        psi_profs = profile_results['tau_deg'].get('psi', [])
        target_phi = [-65, -85, -125, -155]
        for target, color in zip(target_phi, colors):
            closest = min(psi_profs, key=lambda p: abs(p[0] - target),
                         default=None)
            if closest is None or abs(closest[0] - target) > 10:
                continue
            phi_val, vals = closest
            x = [v[0] for v in vals]
            y = [v[1] for v in vals]
            ax.plot(x, y, '.-', color=color, label=f'φ≈{phi_val:.0f}°',
                   markersize=4)
        ax.axvline(-40, color='red', ls='--', lw=1.5, alpha=0.7)
        ax.axhline(0, color='gray', ls='-', lw=0.5)
        ax.set_xlabel('ψ [°]')
        ax.set_ylabel('Δτ_coupling [°]')
        ax.set_title('τ coupling across ψ-wall at different φ')
        ax.legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'cross_wall_profiles.png'), dpi=150)
        plt.close()
        print(f"  Saved cross_wall_profiles.png")
    
    # ── Plot 3: Coupling map with walls overlaid ─────────────────────────
    if 'tau_deg' in coupling_maps:
        cmap_data, cnt_data, phi_c, psi_c, _ = coupling_maps['tau_deg']
        
        fig, ax = plt.subplots(figsize=(8, 7))
        valid = cmap_data[~np.isnan(cmap_data)]
        if len(valid) > 0:
            vlim = np.percentile(np.abs(valid), 95)
            if vlim < 0.1:
                vlim = 1.0
            im = ax.pcolormesh(phi_c, psi_c, cmap_data,
                               cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                               shading='auto')
            plt.colorbar(im, ax=ax, label='Δτ_coupling [°]')
        
        # Paper 1 walls
        ax.axvline(-60, color='lime', ls='--', lw=2, label='φ-wall (-60°)')
        ax.axhline(-40, color='cyan', ls='--', lw=2, label='ψ-wall (-40°)')
        
        # Mark basin centers
        for name, phi, psi in [('αR', -63, -43), ('β', -120, 135),
                                ('PPII', -75, 145), ('αL', 57, 47)]:
            ax.plot(phi, psi, 'k*', markersize=10)
            ax.annotate(name, (phi, psi), textcoords="offset points",
                       xytext=(8, 8), fontsize=9, fontweight='bold')
        
        ax.set_xlabel('φ [°]')
        ax.set_ylabel('ψ [°]')
        ax.set_title('τ coupling map with Paper 1 steric walls')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_aspect('equal')
        ax.legend(loc='lower right', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'coupling_with_walls.png'), dpi=150)
        plt.close()
        print(f"  Saved coupling_with_walls.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Paper 3 — Steric wall sign-change analysis')
    ap.add_argument('--csv', required=True, help='features CSV (p3.csv)')
    ap.add_argument('--out', default='./paper3_walls')
    ap.add_argument('--bin_size', type=int, default=10)
    args = ap.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    
    # ── Read CSV ─────────────────────────────────────────────────────────
    print(f"[1/5] Reading {args.csv}...")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"  {len(df):,} rows  ({time.time()-t0:.1f}s)")
    
    # ── Compute coupling maps ────────────────────────────────────────────
    print(f"[2/5] Computing coupling maps...")
    
    observables = {}
    for col in ['tau_deg', 'angle_NCaC', 'angle_C_CA_CB', 'angle_N_CA_CB',
                'omega_deg', 'bond_CA_C', 'bond_N_CA']:
        if col in df.columns:
            observables[col] = col
    
    coupling_maps = {}
    for name, col in observables.items():
        result = compute_coupling_map(df, 'phi_deg', 'psi_deg', col,
                                       args.bin_size)
        if result is not None:
            coupling_maps[name] = result
            print(f"  {name}: computed")
    
    # ── Sign-change tests at Paper 1 positions ───────────────────────────
    print(f"[3/5] Testing sign change at Paper 1 walls...")
    
    results_all = {}
    for name, (cmap, cnt, phi_c, psi_c, f0) in coupling_maps.items():
        tests = []
        for axis, pos in [('phi', -60.0), ('psi', -40.0)]:
            res = wall_sign_change(cmap, cnt, phi_c, psi_c, axis, pos)
            if res is not None:
                tests.append(res)
                print(f"  {name:>20s}  {axis}-wall at {pos:>4.0f}°: "
                      f"Δ={res['mean_diff']:+.4f}  flip={res['frac_sign_flip']:.0%}  "
                      f"p={res['p_val']:.2e}")
        results_all[name] = tests
    
    # ── Scan wall positions ──────────────────────────────────────────────
    print(f"[4/5] Scanning wall positions...")
    
    scan_results = {}
    for name, (cmap, cnt, phi_c, psi_c, f0) in coupling_maps.items():
        scans = []
        # Scan φ-wall from -120 to 0
        phi_scan = scan_wall_position(cmap, cnt, phi_c, psi_c, 'phi',
                                       np.arange(-120, 1, 5))
        scans.extend(phi_scan)
        # Scan ψ-wall from -100 to 20
        psi_scan = scan_wall_position(cmap, cnt, phi_c, psi_c, 'psi',
                                       np.arange(-100, 21, 5))
        scans.extend(psi_scan)
        scan_results[name] = scans
        
        # Report optimal
        for axis in ['phi', 'psi']:
            axis_s = [s for s in scans if s['wall_axis'] == axis]
            if axis_s:
                best = max(axis_s, key=lambda s: abs(s['mean_diff']))
                print(f"  {name:>20s}  {axis}-wall optimal: {best['wall_position']:>6.0f}°  "
                      f"|Δ|={abs(best['mean_diff']):.4f}")
    
    # ── Cross-wall profiles ──────────────────────────────────────────────
    profile_results = {}
    for name in ['tau_deg', 'angle_N_CA_CB']:
        if name in coupling_maps:
            cmap, cnt, phi_c, psi_c, f0 = coupling_maps[name]
            phi_profs = extract_cross_wall_profiles(cmap, cnt, phi_c, psi_c,
                                                     'phi', -60)
            psi_profs = extract_cross_wall_profiles(cmap, cnt, phi_c, psi_c,
                                                     'psi', -40)
            profile_results[name] = {'phi': phi_profs, 'psi': psi_profs}
    
    # ── Per-class wall scan (τ only) ─────────────────────────────────────
    results_by_class = {}
    if 'res_name' in df.columns:
        classes = {
            'GLY': df['res_name'] == 'GLY',
            'PRO': df['res_name'] == 'PRO',
            'beta_branched': df['res_name'].isin(['VAL', 'ILE', 'THR']),
            'non_branched': ~df['res_name'].isin(['GLY', 'PRO', 'VAL', 'ILE', 'THR']),
        }
        for cls_name, mask in classes.items():
            df_cls = df[mask]
            result = compute_coupling_map(df_cls, 'phi_deg', 'psi_deg',
                                           'tau_deg', args.bin_size)
            if result is None:
                continue
            cmap, cnt, phi_c, psi_c, f0 = result
            scans = []
            scans.extend(scan_wall_position(cmap, cnt, phi_c, psi_c, 'phi',
                                            np.arange(-120, 1, 5)))
            scans.extend(scan_wall_position(cmap, cnt, phi_c, psi_c, 'psi',
                                            np.arange(-100, 21, 5)))
            results_by_class[cls_name] = scans
            
            for axis in ['phi', 'psi']:
                axis_s = [s for s in scans if s['wall_axis'] == axis]
                if axis_s:
                    best = max(axis_s, key=lambda s: abs(s['mean_diff']))
                    print(f"  {cls_name:>15s}  τ {axis}-wall: {best['wall_position']:>6.0f}°")
    
    # ── Report ───────────────────────────────────────────────────────────
    print(f"[5/5] Report and plots...")
    
    report = build_report(results_all, scan_results, profile_results,
                          results_by_class)
    
    report_path = os.path.join(args.out, 'wall_sign_change_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved {report_path}")
    print(report)
    
    make_plots(scan_results, profile_results, coupling_maps, args.out)
    
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()