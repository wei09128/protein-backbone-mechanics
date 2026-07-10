#!/usr/bin/env python3
"""
Paper 3 — Comprehensive Supplementary Analysis
================================================

Fills the remaining gaps for a complete Paper 3 manuscript:

  [A] Resolution dependence of coupling
      — Does η²_coupling increase at high resolution?
      — If yes: refinement restraints suppress coupling at low res
      — If flat: coupling is robust across resolution ranges

  [B] All-20 residue coupling profiles
      — Which amino acids have strongest/weakest coupling?
      — Correlation with sidechain properties (mass, branching, aromaticity)

  [C] ω cleanup
      — Filter |ω| > 150° (remove cis-peptides) and recompute

  [D] Coupling as a function of secondary structure (DSSP-like)
      — Does coupling differ in helices vs sheets vs coil?

  [E] Coupling magnitude in physical units
      — Convert Δf_φψ from degrees to kcal/mol using harmonic approximation
      — E_coupling ≈ ½ k (Δθ)² where k is from Paper 2's spring constants

Usage:
  python paper3_05_comprehensive.py \
      --csv /mnt/f/Protein_Folding/v8_g/p3_filtered.csv \
      --out ./paper3_comprehensive/ --bin_size 10

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
# Coupling decomposition (reused from script 03)
# ══════════════════════════════════════════════════════════════════════════════

def compute_coupling_stats(df, phi_col, psi_col, value_col, bin_size=10,
                           min_count=5):
    """Compute coupling η² and related stats. Returns dict or None."""
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

    if len(sub) < 200:
        return None

    f0 = sub[value_col].mean()
    var_total = sub[value_col].var()

    cell_stats = sub.groupby(['phi_bin', 'psi_bin'])[value_col].agg(
        ['mean', 'count', 'var'])
    cell_stats.columns = ['cell_mean', 'cell_count', 'cell_var']
    cell_stats = cell_stats[cell_stats['cell_count'] >= min_count]

    if len(cell_stats) < 10:
        return None

    phi_marginal = (cell_stats.reset_index().groupby('phi_bin')
                    .apply(lambda g: np.average(g['cell_mean'],
                                                weights=g['cell_count']),
                           include_groups=False))
    psi_marginal = (cell_stats.reset_index().groupby('psi_bin')
                    .apply(lambda g: np.average(g['cell_mean'],
                                                weights=g['cell_count']),
                           include_groups=False))

    coupling = cell_stats[['cell_mean', 'cell_count']].copy()
    coupling['phi_eff'] = coupling.index.get_level_values('phi_bin').map(
        phi_marginal) - f0
    coupling['psi_eff'] = coupling.index.get_level_values('psi_bin').map(
        psi_marginal) - f0
    coupling['coupling'] = coupling['cell_mean'] - f0 - coupling['phi_eff'] - coupling['psi_eff']
    coupling = coupling.dropna()

    if len(coupling) < 10:
        return None

    w = coupling['cell_count'].values

    def wvar(x, ww):
        mu = np.average(x, weights=ww)
        return np.average((x - mu)**2, weights=ww)

    v_phi = wvar(coupling['phi_eff'].values, w)
    v_psi = wvar(coupling['psi_eff'].values, w)
    v_coup = wvar(coupling['coupling'].values, w)
    v_sys = v_phi + v_psi + v_coup

    valid_var = coupling.join(cell_stats[['cell_var']]).dropna()
    v_within = np.average(valid_var['cell_var'].values,
                          weights=valid_var['cell_count'].values)

    eta2_c = v_coup / v_sys if v_sys > 1e-12 else 0
    frac_c = v_coup / var_total if var_total > 1e-12 else 0

    coup_vals = coupling['coupling'].values
    pp = coup_vals.max() - coup_vals.min() if len(coup_vals) > 0 else 0
    rms = np.sqrt(np.mean(coup_vals**2)) if len(coup_vals) > 0 else 0

    return {
        'n': len(sub), 'n_cells': len(coupling),
        'eta2_phi': v_phi / v_sys if v_sys > 1e-12 else 0,
        'eta2_psi': v_psi / v_sys if v_sys > 1e-12 else 0,
        'eta2_coupling': eta2_c,
        'frac_coupling': frac_c,
        'frac_within': v_within / var_total if var_total > 1e-12 else 0,
        'peak_to_peak': pp, 'rms_coupling': rms,
        'var_total': var_total,
    }


# ══════════════════════════════════════════════════════════════════════════════
# [A] Resolution dependence
# ══════════════════════════════════════════════════════════════════════════════

def resolution_analysis(df, out_dir, bin_size=10):
    """Compute coupling η² in resolution bins."""
    R = []
    R.append("\n" + "═" * 78)
    R.append("[A] RESOLUTION DEPENDENCE OF COUPLING")
    R.append("═" * 78)

    # Check if resolution column exists
    res_col = None
    for c in ['resolution', 'res_A', 'resolution_A']:
        if c in df.columns:
            res_col = c
            break

    if res_col is None:
        R.append("  Resolution column not found — skipping.")
        R.append("  (Need 'resolution' column in CSV)")
        # If no resolution, use bfactor_ca as a proxy for data quality
        if 'bfactor_ca' in df.columns:
            R.append("\n  Using bfactor_ca as data-quality proxy instead:")
            bf_bins = [(0, 10), (10, 15), (15, 20), (20, 25), (25, 30)]
            observables = ['tau_deg', 'angle_N_CA_CB', 'angle_C_CA_CB']

            header = f"  {'B-factor':>12s}  {'n':>9s}"
            for obs in observables:
                header += f"  {'η²_' + obs[:8]:>12s}"
            R.append(header)

            for lo, hi in bf_bins:
                mask = (df['bfactor_ca'] >= lo) & (df['bfactor_ca'] < hi)
                sub = df[mask]
                line = f"  {lo:>4.0f}–{hi:<4.0f} Å²  {mask.sum():>9,}"
                for obs in observables:
                    if obs in sub.columns:
                        res = compute_coupling_stats(sub, 'phi_deg', 'psi_deg',
                                                      obs, bin_size)
                        if res:
                            line += f"  {res['eta2_coupling']:>12.1%}"
                        else:
                            line += f"  {'n/a':>12s}"
                    else:
                        line += f"  {'—':>12s}"
                R.append(line)
        return '\n'.join(R)

    R.append(f"  Using column: {res_col}")

    res_bins = [(0.0, 1.2), (1.2, 1.5), (1.5, 1.8), (1.8, 2.0),
                (2.0, 2.5), (2.5, 3.0)]
    observables = ['tau_deg', 'angle_N_CA_CB', 'angle_C_CA_CB',
                   'bond_CA_C', 'omega_deg']

    header = f"  {'Resolution':>12s}  {'n':>9s}"
    for obs in observables:
        header += f"  {'η²_' + obs[:8]:>12s}"
    R.append(header)
    R.append("  " + "─" * (len(header.strip())))

    for lo, hi in res_bins:
        mask = (df[res_col] >= lo) & (df[res_col] < hi)
        sub = df[mask]
        line = f"  {lo:.1f}–{hi:.1f} Å     {mask.sum():>9,}"
        for obs in observables:
            if obs in sub.columns:
                res = compute_coupling_stats(sub, 'phi_deg', 'psi_deg',
                                              obs, bin_size)
                if res:
                    line += f"  {res['eta2_coupling']:>12.1%}"
                else:
                    line += f"  {'n/a':>12s}"
            else:
                line += f"  {'—':>12s}"
        R.append(line)

    R.append("")
    R.append("  Interpretation:")
    R.append("    η² increases at high res → refinement restraints suppress coupling")
    R.append("    η² flat across res → coupling is robust (intrinsic to geometry)")

    return '\n'.join(R)


# ══════════════════════════════════════════════════════════════════════════════
# [B] All-20 residue coupling profiles
# ══════════════════════════════════════════════════════════════════════════════

def per_residue_analysis(df, out_dir, bin_size=10):
    """Compute coupling η² for each of the 20 amino acids."""
    R = []
    R.append("\n" + "═" * 78)
    R.append("[B] PER-RESIDUE COUPLING PROFILES (ALL 20 AMINO ACIDS)")
    R.append("═" * 78)

    if 'res_name' not in df.columns:
        R.append("  res_name column not found")
        return '\n'.join(R)

    aa_order = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO',
                'PHE', 'TYR', 'TRP', 'SER', 'THR', 'CYS',
                'MET', 'ASP', 'ASN', 'GLU', 'GLN', 'LYS',
                'ARG', 'HIS']

    observables = ['tau_deg', 'angle_N_CA_CB', 'angle_C_CA_CB']

    # Sidechain properties for correlation
    sc_props = {
        'GLY': {'mass': 0, 'branched': 0, 'aromatic': 0, 'polar': 0, 'charged': 0, 'n_chi': 0},
        'ALA': {'mass': 15, 'branched': 0, 'aromatic': 0, 'polar': 0, 'charged': 0, 'n_chi': 0},
        'VAL': {'mass': 43, 'branched': 1, 'aromatic': 0, 'polar': 0, 'charged': 0, 'n_chi': 1},
        'LEU': {'mass': 57, 'branched': 0, 'aromatic': 0, 'polar': 0, 'charged': 0, 'n_chi': 2},
        'ILE': {'mass': 57, 'branched': 1, 'aromatic': 0, 'polar': 0, 'charged': 0, 'n_chi': 2},
        'PRO': {'mass': 42, 'branched': 0, 'aromatic': 0, 'polar': 0, 'charged': 0, 'n_chi': 0},
        'PHE': {'mass': 91, 'branched': 0, 'aromatic': 1, 'polar': 0, 'charged': 0, 'n_chi': 2},
        'TYR': {'mass': 107, 'branched': 0, 'aromatic': 1, 'polar': 1, 'charged': 0, 'n_chi': 2},
        'TRP': {'mass': 130, 'branched': 0, 'aromatic': 1, 'polar': 0, 'charged': 0, 'n_chi': 2},
        'SER': {'mass': 31, 'branched': 0, 'aromatic': 0, 'polar': 1, 'charged': 0, 'n_chi': 1},
        'THR': {'mass': 45, 'branched': 1, 'aromatic': 0, 'polar': 1, 'charged': 0, 'n_chi': 1},
        'CYS': {'mass': 47, 'branched': 0, 'aromatic': 0, 'polar': 1, 'charged': 0, 'n_chi': 1},
        'MET': {'mass': 75, 'branched': 0, 'aromatic': 0, 'polar': 0, 'charged': 0, 'n_chi': 3},
        'ASP': {'mass': 58, 'branched': 0, 'aromatic': 0, 'polar': 0, 'charged': 1, 'n_chi': 2},
        'ASN': {'mass': 58, 'branched': 0, 'aromatic': 0, 'polar': 1, 'charged': 0, 'n_chi': 2},
        'GLU': {'mass': 72, 'branched': 0, 'aromatic': 0, 'polar': 0, 'charged': 1, 'n_chi': 3},
        'GLN': {'mass': 72, 'branched': 0, 'aromatic': 0, 'polar': 1, 'charged': 0, 'n_chi': 3},
        'LYS': {'mass': 72, 'branched': 0, 'aromatic': 0, 'polar': 0, 'charged': 1, 'n_chi': 4},
        'ARG': {'mass': 100, 'branched': 0, 'aromatic': 0, 'polar': 0, 'charged': 1, 'n_chi': 4},
        'HIS': {'mass': 81, 'branched': 0, 'aromatic': 1, 'polar': 1, 'charged': 0, 'n_chi': 2},
    }

    # Compute per-residue
    results = {}
    for obs in observables:
        R.append(f"\n  [{obs}]")
        header = f"  {'Res':>4s}  {'n':>9s}  {'η²_φ':>7s}  {'η²_ψ':>7s}  {'η²_φψ':>7s}  {'p-p':>7s}  {'RMS':>7s}  {'frac_ε':>7s}"
        R.append(header)
        R.append("  " + "─" * (len(header.strip())))

        obs_results = []
        for aa in aa_order:
            mask = df['res_name'] == aa
            if mask.sum() < 200:
                continue
            sub = df[mask]
            if obs not in sub.columns:
                continue
            # Skip Cβ angles for GLY
            if obs in ('angle_N_CA_CB', 'angle_C_CA_CB') and aa == 'GLY':
                continue
            res = compute_coupling_stats(sub, 'phi_deg', 'psi_deg',
                                          obs, bin_size)
            if res is None:
                continue

            R.append(f"  {aa:>4s}  {res['n']:>9,}  {res['eta2_phi']:>7.1%}  "
                     f"{res['eta2_psi']:>7.1%}  {res['eta2_coupling']:>7.1%}  "
                     f"{res['peak_to_peak']:>7.2f}  {res['rms_coupling']:>7.3f}  "
                     f"{res['frac_within']:>7.1%}")

            obs_results.append({
                'aa': aa, 'eta2_coupling': res['eta2_coupling'],
                'peak_to_peak': res['peak_to_peak'],
                'rms_coupling': res['rms_coupling'],
                **sc_props.get(aa, {}),
            })

        results[obs] = obs_results

    # Correlation with sidechain properties
    R.append("\n  CORRELATION: η²_coupling vs sidechain properties")
    R.append("  " + "─" * 60)

    for obs in observables:
        if obs not in results or len(results[obs]) < 10:
            continue
        obs_df = pd.DataFrame(results[obs])
        R.append(f"\n  [{obs}]")
        for prop in ['mass', 'branched', 'aromatic', 'n_chi']:
            if prop in obs_df.columns:
                r, p = sp_stats.pearsonr(obs_df[prop], obs_df['eta2_coupling'])
                star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                R.append(f"    r(η²_coupling, {prop:>10s}) = {r:+.3f}  p={p:.3f} {star}")

    # Save per-residue table
    all_rows = []
    for obs, obs_res in results.items():
        for row in obs_res:
            row['observable'] = obs
            all_rows.append(row)
    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            os.path.join(out_dir, 'per_residue_coupling.csv'), index=False)

    return '\n'.join(R)


# ══════════════════════════════════════════════════════════════════════════════
# [C] ω cleanup
# ══════════════════════════════════════════════════════════════════════════════

def omega_cleanup_analysis(df, out_dir, bin_size=10):
    """Recompute ω coupling after filtering cis-peptides."""
    R = []
    R.append("\n" + "═" * 78)
    R.append("[C] ω COUPLING — CLEANED (|ω| > 150° only)")
    R.append("═" * 78)

    if 'omega_deg' not in df.columns:
        R.append("  omega_deg not found")
        return '\n'.join(R)

    # Filter trans-only
    omega_col = 'omega_deg'
    if 'omega_measured_deg' in df.columns:
        omega_col = 'omega_measured_deg'

    trans = df[df[omega_col].abs() > 150].copy()
    R.append(f"  Total residues: {len(df):,}")
    R.append(f"  Trans (|ω|>150°): {len(trans):,}  ({len(trans)/len(df)*100:.1f}%)")
    R.append(f"  Cis removed: {len(df)-len(trans):,}")

    # Compute |Δω| = deviation from 180°
    trans['omega_dev'] = 180.0 - trans[omega_col].abs()

    # Coupling on ω deviation
    R.append("\n  Coupling on |180° - ω| (peptide planarity deviation):")
    res_raw = compute_coupling_stats(df, 'phi_deg', 'psi_deg',
                                      omega_col, bin_size)
    res_trans = compute_coupling_stats(trans, 'phi_deg', 'psi_deg',
                                        omega_col, bin_size)
    res_dev = compute_coupling_stats(trans, 'phi_deg', 'psi_deg',
                                      'omega_dev', bin_size)

    if res_raw:
        R.append(f"    Raw ω (all):      η²_coupling = {res_raw['eta2_coupling']:.1%}  "
                 f"p-p = {res_raw['peak_to_peak']:.1f}°")
    if res_trans:
        R.append(f"    Trans ω only:     η²_coupling = {res_trans['eta2_coupling']:.1%}  "
                 f"p-p = {res_trans['peak_to_peak']:.1f}°")
    if res_dev:
        R.append(f"    |180°-ω| (dev):   η²_coupling = {res_dev['eta2_coupling']:.1%}  "
                 f"p-p = {res_dev['peak_to_peak']:.2f}°  "
                 f"RMS = {res_dev['rms_coupling']:.3f}°")

    return '\n'.join(R)


# ══════════════════════════════════════════════════════════════════════════════
# [D] Secondary structure dependence
# ══════════════════════════════════════════════════════════════════════════════

def ss_analysis(df, out_dir, bin_size=10):
    """Coupling by secondary structure bin."""
    R = []
    R.append("\n" + "═" * 78)
    R.append("[D] COUPLING BY SECONDARY STRUCTURE")
    R.append("═" * 78)

    if 'ss_bin' not in df.columns:
        R.append("  ss_bin column not found")
        return '\n'.join(R)

    ss_labels = {0: 'αR', 1: 'β', 2: 'PPII', 3: '3₁₀', 4: 'coil', 5: 'αL'}
    observables = ['tau_deg', 'angle_N_CA_CB', 'angle_C_CA_CB']

    header = f"  {'SS':>6s}  {'n':>9s}"
    for obs in observables:
        header += f"  {'η²_' + obs[:8]:>12s}  {'p-p':>7s}"
    R.append(header)
    R.append("  " + "─" * (len(header.strip())))

    for ss_val in sorted(ss_labels.keys()):
        mask = df['ss_bin'] == ss_val
        if mask.sum() < 500:
            continue
        sub = df[mask]
        label = ss_labels.get(ss_val, f'SS{ss_val}')
        line = f"  {label:>6s}  {mask.sum():>9,}"

        for obs in observables:
            if obs in sub.columns:
                # For within-basin analysis, use finer bins
                res = compute_coupling_stats(sub, 'phi_deg', 'psi_deg',
                                              obs, bin_size)
                if res:
                    line += f"  {res['eta2_coupling']:>12.1%}  {res['peak_to_peak']:>7.2f}"
                else:
                    line += f"  {'n/a':>12s}  {'n/a':>7s}"
            else:
                line += f"  {'—':>12s}  {'—':>7s}"
        R.append(line)

    return '\n'.join(R)


# ══════════════════════════════════════════════════════════════════════════════
# [E] Coupling magnitude in energy units
# ══════════════════════════════════════════════════════════════════════════════

def energy_conversion(df, out_dir, bin_size=10):
    """Convert coupling Δθ to approximate energy (kcal/mol)."""
    R = []
    R.append("\n" + "═" * 78)
    R.append("[E] COUPLING MAGNITUDE IN ENERGY UNITS")
    R.append("═" * 78)

    R.append("""
  Approximate conversion using harmonic spring constants from Paper 2:
    E_coupling ≈ ½ k (Δθ_coupling)²

  Spring constants (from Paper 2 / AMBER ff14SB):
    k_τ     ≈ 63 kcal/mol/rad²  (bond angle N-Cα-C)
    k_NCaCB ≈ 63 kcal/mol/rad²  (bond angle N-Cα-Cβ)
    k_CCaCB ≈ 63 kcal/mol/rad²  (bond angle C-Cα-Cβ)
    k_ω     ≈ 10.5 kcal/mol/rad² (improper for ω planarity)
""")

    spring_constants = {
        'tau_deg': 63.0,
        'angle_NCaC': 63.0,
        'angle_N_CA_CB': 63.0,
        'angle_C_CA_CB': 63.0,
        'omega_deg': 10.5,
        'bond_CA_C': 317.0 * 100,  # kcal/mol/Å² → convert p-p from Å
        'bond_N_CA': 337.0 * 100,
    }

    observables = ['tau_deg', 'angle_N_CA_CB', 'angle_C_CA_CB', 'omega_deg']

    R.append(f"  {'Observable':>20s}  {'RMS Δθ':>8s}  {'p-p Δθ':>8s}  "
             f"{'E_rms':>10s}  {'E_p-p':>10s}")
    R.append("  " + "─" * 62)

    for obs in observables:
        if obs not in df.columns:
            continue
        res = compute_coupling_stats(df, 'phi_deg', 'psi_deg', obs, bin_size)
        if res is None:
            continue

        k = spring_constants.get(obs, 63.0)
        rms_rad = np.radians(res['rms_coupling'])
        pp_rad = np.radians(res['peak_to_peak'])

        e_rms = 0.5 * k * rms_rad**2
        e_pp = 0.5 * k * pp_rad**2

        R.append(f"  {obs:>20s}  {res['rms_coupling']:>8.3f}°  "
                 f"{res['peak_to_peak']:>8.2f}°  "
                 f"{e_rms:>10.4f}  {e_pp:>10.3f}")
        R.append(f"  {'':>20s}  {'':>8s}  {'':>8s}  "
                 f"{'kcal/mol':>10s}  {'kcal/mol':>10s}")

    R.append("""
  Context:
    kT at 300K ≈ 0.6 kcal/mol
    Typical H-bond ≈ 1-3 kcal/mol
    If E_coupling(RMS) > 0.1 kcal/mol → energetically significant
    If E_coupling(p-p) > 0.5 kcal/mol → comparable to H-bond
""")

    return '\n'.join(R)


# ══════════════════════════════════════════════════════════════════════════════
# [F] Neighbour coupling: does residue i±1 identity affect coupling?
# ══════════════════════════════════════════════════════════════════════════════

def neighbour_analysis(df, out_dir, bin_size=10):
    """Test if neighbouring residue identity modulates coupling."""
    R = []
    R.append("\n" + "═" * 78)
    R.append("[F] NEIGHBOUR EFFECTS ON COUPLING")
    R.append("═" * 78)
    R.append("")
    R.append("  Does the identity of residue i±1 modulate the coupling")
    R.append("  at residue i? Test using sc_mass_nm1/np1 as proxy.")
    R.append("")

    if 'sc_mass_nm1' not in df.columns or 'sc_mass_np1' not in df.columns:
        R.append("  Neighbour mass columns not found — skipping")
        return '\n'.join(R)

    # Stratify by neighbour mass
    for direction, col in [('i-1', 'sc_mass_nm1'), ('i+1', 'sc_mass_np1')]:
        R.append(f"\n  Neighbour {direction} mass → coupling at i:")
        
        # Tertiles of neighbour mass
        q33 = df[col].quantile(0.33)
        q67 = df[col].quantile(0.67)

        bins_def = [
            ('light', df[col] <= q33),
            ('medium', (df[col] > q33) & (df[col] <= q67)),
            ('heavy', df[col] > q67),
        ]

        for obs in ['tau_deg', 'angle_N_CA_CB']:
            if obs not in df.columns:
                continue
            R.append(f"    {obs}:")
            for label, mask in bins_def:
                sub = df[mask]
                res = compute_coupling_stats(sub, 'phi_deg', 'psi_deg',
                                              obs, bin_size)
                if res:
                    R.append(f"      {label:>8s} (n={mask.sum():>8,}): "
                             f"η²_coupling = {res['eta2_coupling']:.1%}  "
                             f"p-p = {res['peak_to_peak']:.2f}°")

    return '\n'.join(R)


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def make_plots(df, out_dir, bin_size=10):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    aa_order = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO',
                'PHE', 'TYR', 'TRP', 'SER', 'THR', 'CYS',
                'MET', 'ASP', 'ASN', 'GLU', 'GLN', 'LYS',
                'ARG', 'HIS']

    # ── Plot: Per-residue η² bar chart ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, obs in zip(axes, ['tau_deg', 'angle_N_CA_CB', 'angle_C_CA_CB']):
        eta_vals, labels, colors = [], [], []
        for aa in aa_order:
            if obs in ('angle_N_CA_CB', 'angle_C_CA_CB') and aa == 'GLY':
                continue
            mask = df['res_name'] == aa
            if mask.sum() < 200:
                continue
            res = compute_coupling_stats(df[mask], 'phi_deg', 'psi_deg',
                                          obs, bin_size)
            if res is None:
                continue
            eta_vals.append(res['eta2_coupling'])
            labels.append(aa)
            if aa in ('VAL', 'ILE', 'THR'):
                colors.append('#EF6548')
            elif aa == 'GLY':
                colors.append('#41AB5D')
            elif aa == 'PRO':
                colors.append('#807DBA')
            else:
                colors.append('#4292C6')

        x = range(len(eta_vals))
        ax.bar(x, [v * 100 for v in eta_vals], color=colors,
               edgecolor='#333333', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=45)
        ax.set_ylabel('η²_coupling (%)')
        ax.set_title(f'{obs}')
        ax.axhline(15, color='gray', ls='--', lw=0.8, alpha=0.5)

    plt.suptitle('φ×ψ coupling strength by residue type\n'
                 '(red=β-branched, green=Gly, purple=Pro)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'per_residue_eta2.png'), dpi=150)
    plt.close()
    print(f"  Saved per_residue_eta2.png")

    # ── Plot: Variance decomposition stacked bar ─────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    obs_list = ['tau_deg', 'angle_N_CA_CB', 'angle_C_CA_CB',
                'bond_CA_C', 'bond_N_CA', 'omega_deg']
    obs_labels = ['τ (N-Cα-C)', '∠N-Cα-Cβ', '∠C-Cα-Cβ',
                  'bond Cα-C', 'bond N-Cα', 'ω']

    frac_phi, frac_psi, frac_coup, frac_eps = [], [], [], []
    valid_labels = []

    for obs, label in zip(obs_list, obs_labels):
        if obs not in df.columns:
            continue
        res = compute_coupling_stats(df, 'phi_deg', 'psi_deg', obs, bin_size)
        if res is None:
            continue
        total_sys = res['eta2_phi'] + res['eta2_psi'] + res['eta2_coupling']
        # Normalize to fraction of total variance
        f_sys = 1.0 - res['frac_within']
        frac_phi.append(res['eta2_phi'] * f_sys)
        frac_psi.append(res['eta2_psi'] * f_sys)
        frac_coup.append(res['eta2_coupling'] * f_sys)
        frac_eps.append(res['frac_within'])
        valid_labels.append(label)

    x = np.arange(len(valid_labels))
    w = 0.6
    ax.bar(x, frac_phi, w, label='φ marginal', color='#2166AC')
    ax.bar(x, frac_psi, w, bottom=frac_phi, label='ψ marginal', color='#B2182B')
    ax.bar(x, frac_coup, w, bottom=[a+b for a, b in zip(frac_phi, frac_psi)],
           label='φ×ψ coupling', color='#7A0177')
    ax.bar(x, frac_eps, w,
           bottom=[a+b+c for a, b, c in zip(frac_phi, frac_psi, frac_coup)],
           label='within-cell (noise)', color='#CCCCCC')

    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels, fontsize=9)
    ax.set_ylabel('Fraction of total variance')
    ax.set_title('Variance decomposition of backbone observables')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'variance_stacked.png'), dpi=150)
    plt.close()
    print(f"  Saved variance_stacked.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Paper 3 — Comprehensive supplementary analysis')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='./paper3_comprehensive')
    ap.add_argument('--bin_size', type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    print(f"[1/7] Reading {args.csv}...")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns  ({time.time()-t0:.1f}s)")

    sections = []

    print(f"[2/7] Resolution analysis...")
    sections.append(resolution_analysis(df, args.out, args.bin_size))

    print(f"[3/7] Per-residue analysis...")
    sections.append(per_residue_analysis(df, args.out, args.bin_size))

    print(f"[4/7] ω cleanup...")
    sections.append(omega_cleanup_analysis(df, args.out, args.bin_size))

    print(f"[5/7] Secondary structure analysis...")
    sections.append(ss_analysis(df, args.out, args.bin_size))

    print(f"[6/7] Energy conversion...")
    sections.append(energy_conversion(df, args.out, args.bin_size))

    print(f"[7/7] Neighbour analysis...")
    sections.append(neighbour_analysis(df, args.out, args.bin_size))

    # ── Compile report ───────────────────────────────────────────────────
    report = "=" * 78 + "\n"
    report += "Paper 3 — Comprehensive Supplementary Analysis\n"
    report += "=" * 78 + "\n"
    report += '\n'.join(sections)

    report_path = os.path.join(args.out, 'comprehensive_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved {report_path}")
    print(report)

    # ── Plots ────────────────────────────────────────────────────────────
    print(f"\nGenerating plots...")
    make_plots(df, args.out, args.bin_size)

    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()