#!/usr/bin/env python3
"""
Paper 4 — Geometry Constant Library Extraction
================================================

Extracts a complete, structured library of backbone geometry constants
from the PDB-derived features.csv. This library replaces the fixed
AMBER/Engh-Huber constants used in NeRF builders with (φ,ψ)-dependent
lookup tables that capture the coupling physics from Papers 1–3.

Library structure:
  constants[residue_class][phi_bin][psi_bin] = {
      # Equilibrium geometry (cell means)
      'tau_eq':          111.2,   # ∠N-Cα-C equilibrium [deg]
      'angle_NCaCB_eq':  110.5,   # ∠N-Cα-Cβ [deg]
      'angle_CCaCB_eq':  110.1,   # ∠C-Cα-Cβ [deg]
      'bond_NCA_eq':     1.458,   # N-Cα bond [Å]
      'bond_CAC_eq':     1.524,   # Cα-C bond [Å]
      'bond_CO_eq':      1.231,   # C=O bond [Å]
      'bond_CN_eq':      1.329,   # C-N peptide bond [Å]
      'omega_eq':        179.5,   # ω [deg]

      # Spring constants (from within-cell variance)
      'k_tau':           63.0,    # kcal/mol/rad²
      'k_NCaCB':         55.0,
      'k_CCaCB':         60.0,

      # Coupling corrections (Paper 3)
      'delta_tau':       +0.45,   # Δτ_φψ coupling [deg]
      'delta_NCaCB':     -0.12,
      'delta_CCaCB':     +0.08,

      # Statistics
      'n':               1234,    # observations in this cell
      'sigma_tau':       1.8,     # std dev of τ [deg]
  }

Residue classes:
  GLY, PRO, VAL, ILE, THR (β-branched individually),
  ALA, LEU, ... (remaining 15 as individual entries)

χ₁ sublibrary (for Cβ angles only):
  constants_chi1[residue][phi_bin][psi_bin][chi1_rotamer] = {
      'angle_NCaCB_eq': ...,
      'angle_CCaCB_eq': ...,
  }

Output formats:
  - constants_library.json     (for Python/JS consumption)
  - constants_library.csv      (flat table for inspection)
  - constants_chi1.json        (χ₁-dependent sublibrary)
  - library_summary.txt        (human-readable report)
  - amber_comparison.csv       (library vs AMBER ff14SB)

Usage:
  python paper4_01_constant_library.py \
      --csv /mnt/f/Protein_Folding/v8_g/p3.csv \
      --out ./paper4_library/ --bin_size 10

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
# AMBER ff14SB reference constants (for comparison)
# ══════════════════════════════════════════════════════════════════════════════

AMBER_REF = {
    'tau_eq': 111.1,          # ∠N-Cα-C
    'angle_NCaCB_eq': 110.1,  # ∠N-Cα-Cβ
    'angle_CCaCB_eq': 110.1,  # ∠C-Cα-Cβ
    'bond_NCA_eq': 1.458,     # N-Cα
    'bond_CAC_eq': 1.522,     # Cα-C
    'bond_CO_eq': 1.229,      # C=O
    'bond_CN_eq': 1.335,      # C-N (peptide)
    'omega_eq': 180.0,
    'k_tau': 63.0,            # kcal/mol/rad²
    'k_NCaCB': 63.0,
    'k_CCaCB': 63.0,
    'k_bond_NCA': 337.0,      # kcal/mol/Å²
    'k_bond_CAC': 317.0,
    'k_bond_CO': 570.0,
    'k_bond_CN': 490.0,
}

# Boltzmann constant × 300K in kcal/mol
kT = 0.596  # kcal/mol at 300K


# ══════════════════════════════════════════════════════════════════════════════
# Column mapping
# ══════════════════════════════════════════════════════════════════════════════

GEOMETRY_COLS = {
    'tau_deg':         {'amber_eq': 111.1, 'amber_k': 63.0,  'unit': 'deg',  'label': 'τ (N-Cα-C)'},
    'angle_N_CA_CB':   {'amber_eq': 110.1, 'amber_k': 63.0,  'unit': 'deg',  'label': '∠N-Cα-Cβ'},
    'angle_C_CA_CB':   {'amber_eq': 110.1, 'amber_k': 63.0,  'unit': 'deg',  'label': '∠C-Cα-Cβ'},
    'angle_CaCN':      {'amber_eq': 116.6, 'amber_k': 70.0,  'unit': 'deg',  'label': '∠Cα-C-N'},
    'angle_CNCa':      {'amber_eq': 121.9, 'amber_k': 50.0,  'unit': 'deg',  'label': '∠C-N-Cα'},
    'angle_CA_C_O':    {'amber_eq': 120.4, 'amber_k': 80.0,  'unit': 'deg',  'label': '∠Cα-C=O'},
    'bond_N_CA':       {'amber_eq': 1.458, 'amber_k': 337.0, 'unit': 'Å',    'label': 'N-Cα'},
    'bond_CA_C':       {'amber_eq': 1.522, 'amber_k': 317.0, 'unit': 'Å',    'label': 'Cα-C'},
    'bond_C_O':        {'amber_eq': 1.229, 'amber_k': 570.0, 'unit': 'Å',    'label': 'C=O'},
    'bond_C_N_next':   {'amber_eq': 1.335, 'amber_k': 490.0, 'unit': 'Å',    'label': 'C-N'},
    'bond_CA_CB':      {'amber_eq': 1.526, 'amber_k': 317.0, 'unit': 'Å',    'label': 'Cα-Cβ'},
    'omega_deg':       {'amber_eq': 180.0, 'amber_k': 10.5,  'unit': 'deg',  'label': 'ω'},
}

# All 20 amino acids as individual entries
ALL_AA = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO',
          'PHE', 'TYR', 'TRP', 'SER', 'THR', 'CYS',
          'MET', 'ASP', 'ASN', 'GLU', 'GLN', 'LYS',
          'ARG', 'HIS']

# Residue classes for compact library
RESIDUE_CLASSES = {
    'GLY': ['GLY'],
    'PRO': ['PRO'],
    'VAL': ['VAL'],
    'ILE': ['ILE'],
    'THR': ['THR'],
    'ALA': ['ALA'],
    'LEU_like': ['LEU', 'MET'],
    'aromatic': ['PHE', 'TYR', 'TRP', 'HIS'],
    'small_polar': ['SER', 'CYS', 'ASN', 'ASP'],
    'long_polar': ['GLU', 'GLN', 'LYS', 'ARG'],
}


# ══════════════════════════════════════════════════════════════════════════════
# Library extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_cell_stats(df, phi_col, psi_col, bin_size=10, min_count=10):
    """Extract per-cell statistics for all geometry columns.
    
    Returns DataFrame with one row per (phi_bin, psi_bin) cell.
    """
    phi_bins = np.arange(-180, 180 + bin_size, bin_size)
    psi_bins = np.arange(-180, 180 + bin_size, bin_size)
    phi_centers = phi_bins[:-1] + bin_size / 2
    psi_centers = psi_bins[:-1] + bin_size / 2

    df = df.copy()
    df['phi_bin'] = pd.cut(df[phi_col], phi_bins, labels=False, right=False)
    df['psi_bin'] = pd.cut(df[psi_col], psi_bins, labels=False, right=False)
    df = df.dropna(subset=['phi_bin', 'psi_bin'])
    df['phi_bin'] = df['phi_bin'].astype(int)
    df['psi_bin'] = df['psi_bin'].astype(int)

    # Identify which geometry columns exist
    geo_cols = [c for c in GEOMETRY_COLS if c in df.columns]

    # Group by cell
    grouped = df.groupby(['phi_bin', 'psi_bin'])

    rows = []
    for (pb, qb), group in grouped:
        if len(group) < min_count:
            continue

        row = {
            'phi_center': phi_centers[pb],
            'psi_center': psi_centers[qb],
            'phi_bin': int(pb),
            'psi_bin': int(qb),
            'n': len(group),
        }

        for col in geo_cols:
            vals = group[col].dropna()
            if len(vals) < 5:
                continue
            
            mean_val = vals.mean()
            std_val = vals.std()
            median_val = vals.median()

            row[f'{col}_eq'] = round(mean_val, 4)
            row[f'{col}_median'] = round(median_val, 4)
            row[f'{col}_std'] = round(std_val, 4)
            row[f'{col}_n'] = len(vals)

            # Spring constant from variance: k = kT / σ²
            # For angles: convert σ from degrees to radians
            info = GEOMETRY_COLS[col]
            if info['unit'] == 'deg':
                sigma_rad = np.radians(std_val)
                if sigma_rad > 1e-6:
                    k_empirical = kT / (sigma_rad ** 2)
                else:
                    k_empirical = 999.0
            else:  # Å
                if std_val > 1e-6:
                    k_empirical = kT / (std_val ** 2)
                else:
                    k_empirical = 99999.0
            row[f'{col}_k'] = round(k_empirical, 2)

            # Deviation from AMBER
            row[f'{col}_delta_amber'] = round(mean_val - info['amber_eq'], 4)

        rows.append(row)

    return pd.DataFrame(rows)

def safe_weighted_avg(g, val_col, weight_col):
    # Filter for non-null values
    mask = g[val_col].notna()
    weights = g.loc[mask, weight_col]
    values = g.loc[mask, val_col]
    
    # Check if we have data and if weights actually sum to something > 0
    if weights.sum() > 0:
        return np.average(values, weights=weights)
    else:
        return np.nan

def compute_coupling_corrections(df, cell_stats, phi_col, psi_col,
                                  bin_size=10):
    """Compute coupling correction Δf_φψ for each cell."""
    phi_bins = np.arange(-180, 180 + bin_size, bin_size)
    psi_bins = np.arange(-180, 180 + bin_size, bin_size)

    geo_cols = [c for c in GEOMETRY_COLS if f'{c}_eq' in cell_stats.columns]

    for col in geo_cols:
        eq_col = f'{col}_eq'
        n_col = f'{col}_n'
        coup_col = f'{col}_coupling'

        if eq_col not in cell_stats.columns:
            continue

        # Grand mean
        valid = cell_stats[[eq_col, 'n']].dropna()
        if len(valid) == 0:
            continue
        grand_mean = np.average(valid[eq_col], weights=valid['n'])

        # Marginals
        phi_marginal = (cell_stats.groupby('phi_bin')
                        .apply(lambda g: safe_weighted_avg(g, eq_col, 'n'), 
                               include_groups=False))
        
        psi_marginal = (cell_stats.groupby('psi_bin')
                        .apply(lambda g: safe_weighted_avg(g, eq_col, 'n'), 
                               include_groups=False))

        # Coupling = cell_mean - grand_mean - phi_effect - psi_effect
        couplings = []
        for idx, row in cell_stats.iterrows():
            if pd.isna(row.get(eq_col)):
                couplings.append(np.nan)
                continue
            phi_eff = phi_marginal.get(row['phi_bin'], np.nan)
            psi_eff = psi_marginal.get(row['psi_bin'], np.nan)
            if np.isnan(phi_eff) or np.isnan(psi_eff):
                couplings.append(np.nan)
            else:
                additive = grand_mean + (phi_eff - grand_mean) + (psi_eff - grand_mean)
                couplings.append(round(row[eq_col] - additive, 4))

        cell_stats[coup_col] = couplings

    return cell_stats


def extract_chi1_sublibrary(df, bin_size=20, min_count=20):
    """Extract χ₁-dependent Cβ angle corrections.
    
    Uses coarser bins (20°) since we're splitting by 3 rotamers.
    """
    chi1_col = None
    for c in ['chi1_rad', 'chi1_deg', 'chi1']:
        if c in df.columns:
            chi1_col = c
            break
    if chi1_col is None:
        return None

    # Filter to residues with real χ₁
    if 'has_chi1' in df.columns:
        df = df[df['has_chi1'] == 1].copy()
    else:
        df = df[~df['res_name'].isin(['GLY', 'ALA'])].copy()

    # Convert to degrees if needed
    chi1_vals = df[chi1_col].values.copy()
    if np.nanmax(np.abs(chi1_vals)) < 7:  # radians
        chi1_vals = np.degrees(chi1_vals)
    chi1_vals = chi1_vals % 360

    df['chi1_rotamer'] = pd.cut(chi1_vals, bins=[0, 120, 240, 360],
                                 labels=['g+', 't', 'g-'], right=False)

    phi_bins = np.arange(-180, 180 + bin_size, bin_size)
    psi_bins = np.arange(-180, 180 + bin_size, bin_size)
    phi_centers = phi_bins[:-1] + bin_size / 2
    psi_centers = psi_bins[:-1] + bin_size / 2

    df['phi_bin'] = pd.cut(df['phi_deg'], phi_bins, labels=False, right=False)
    df['psi_bin'] = pd.cut(df['psi_deg'], psi_bins, labels=False, right=False)
    df = df.dropna(subset=['phi_bin', 'psi_bin', 'chi1_rotamer'])

    angle_cols = [c for c in ['angle_N_CA_CB', 'angle_C_CA_CB', 'bond_CA_CB']
                  if c in df.columns]

    grouped = df.groupby(['res_name', 'phi_bin', 'psi_bin', 'chi1_rotamer'])

    rows = []
    for (res, pb, qb, rot), group in grouped:
        if len(group) < min_count:
            continue
        row = {
            'res_name': res,
            'phi_center': phi_centers[int(pb)],
            'psi_center': psi_centers[int(qb)],
            'chi1_rotamer': rot,
            'n': len(group),
        }
        for col in angle_cols:
            vals = group[col].dropna()
            if len(vals) >= 5:
                row[f'{col}_eq'] = round(vals.mean(), 4)
                row[f'{col}_std'] = round(vals.std(), 4)
        rows.append(row)

    return pd.DataFrame(rows) if rows else None


# ══════════════════════════════════════════════════════════════════════════════
# JSON export
# ══════════════════════════════════════════════════════════════════════════════

def to_nested_json(cell_stats, geo_cols):
    """Convert flat DataFrame to nested dict: phi -> psi -> {values}."""
    lib = {}
    for _, row in cell_stats.iterrows():
        phi_key = str(int(row['phi_center']))
        psi_key = str(int(row['psi_center']))

        if phi_key not in lib:
            lib[phi_key] = {}

        entry = {'n': int(row['n'])}
        for col in geo_cols:
            for suffix in ['_eq', '_std', '_k', '_coupling', '_delta_amber']:
                key = f'{col}{suffix}'
                if key in row and not pd.isna(row[key]):
                    entry[key] = float(row[key])

        lib[phi_key][psi_key] = entry

    return lib


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

def build_report(cell_stats_all, cell_stats_by_class, chi1_lib, geo_cols):
    R = []
    R.append("=" * 78)
    R.append("Paper 4 — Geometry Constant Library")
    R.append("=" * 78)

    # ── Summary ──────────────────────────────────────────────────────────
    R.append(f"\n  Total cells (ALL): {len(cell_stats_all):,}")
    R.append(f"  Geometry columns: {len(geo_cols)}")
    R.append(f"  Residue classes: {len(cell_stats_by_class)}")
    if chi1_lib is not None:
        R.append(f"  χ₁ sublibrary entries: {len(chi1_lib):,}")

    # ── Global statistics ────────────────────────────────────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 1: GLOBAL EQUILIBRIUM VALUES vs AMBER ff14SB")
    R.append("━" * 78)

    header = f"  {'Observable':>20s}  {'AMBER':>8s}  {'PDB mean':>8s}  {'PDB σ':>8s}  {'Δ(PDB-AMBER)':>13s}  {'k_PDB':>8s}  {'k_AMBER':>8s}"
    R.append(header)
    R.append("  " + "─" * (len(header.strip())))

    for col in geo_cols:
        eq_col = f'{col}_eq'
        std_col = f'{col}_std'
        k_col = f'{col}_k'
        if eq_col not in cell_stats_all.columns:
            continue

        info = GEOMETRY_COLS[col]
        valid = cell_stats_all[cell_stats_all[eq_col].notna()]
        if len(valid) == 0:
            continue

        w = valid['n'].values
        pdb_mean = np.average(valid[eq_col], weights=w)
        pdb_std = np.sqrt(np.average((valid[eq_col] - pdb_mean)**2, weights=w))
        pdb_k = np.average(valid[k_col].dropna(), weights=valid.loc[valid[k_col].notna(), 'n']) if k_col in valid.columns else 0
        delta = pdb_mean - info['amber_eq']

        R.append(f"  {info['label']:>20s}  {info['amber_eq']:>8.3f}  {pdb_mean:>8.3f}  "
                 f"{pdb_std:>8.3f}  {delta:>+13.4f}  {pdb_k:>8.1f}  {info['amber_k']:>8.1f}")

    # ── Per-class summary ────────────────────────────────────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 2: PER-RESIDUE-CLASS EQUILIBRIUM τ")
    R.append("━" * 78)

    if 'tau_deg_eq' in cell_stats_all.columns:
        R.append(f"  {'Class':>15s}  {'n_cells':>8s}  {'⟨τ⟩':>8s}  {'σ(τ)':>8s}  {'⟨k_τ⟩':>8s}")
        R.append("  " + "─" * 50)

        for cls_name in sorted(cell_stats_by_class.keys()):
            cs = cell_stats_by_class[cls_name]
            if 'tau_deg_eq' not in cs.columns:
                continue
            valid = cs[cs['tau_deg_eq'].notna()]
            if len(valid) < 5:
                continue
            w = valid['n'].values
            mu = np.average(valid['tau_deg_eq'], weights=w)
            sig = np.sqrt(np.average((valid['tau_deg_eq'] - mu)**2, weights=w))
            k_avg = np.average(valid['tau_deg_k'].dropna(),
                               weights=valid.loc[valid['tau_deg_k'].notna(), 'n']) if 'tau_deg_k' in valid.columns else 0

            R.append(f"  {cls_name:>15s}  {len(valid):>8d}  {mu:>8.2f}  {sig:>8.3f}  {k_avg:>8.1f}")

    # ── Coupling correction summary ──────────────────────────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 3: COUPLING CORRECTION MAGNITUDES")
    R.append("━" * 78)

    for col in geo_cols:
        coup_col = f'{col}_coupling'
        if coup_col not in cell_stats_all.columns:
            continue
        valid = cell_stats_all[coup_col].dropna()
        if len(valid) < 10:
            continue
        info = GEOMETRY_COLS[col]
        R.append(f"  {info['label']:>20s}:  RMS = {np.sqrt(np.mean(valid**2)):.4f}  "
                 f"p-p = [{valid.min():.3f}, {valid.max():.3f}]  "
                 f"range = {valid.max()-valid.min():.3f}")

    # ── χ₁ sublibrary ────────────────────────────────────────────────────
    if chi1_lib is not None and len(chi1_lib) > 0:
        R.append("\n" + "━" * 78)
        R.append("SECTION 4: χ₁ SUBLIBRARY SUMMARY")
        R.append("━" * 78)

        for col in ['angle_N_CA_CB', 'angle_C_CA_CB']:
            eq_col = f'{col}_eq'
            if eq_col not in chi1_lib.columns:
                continue
            R.append(f"\n  {col}:")
            for rot in ['g+', 't', 'g-']:
                sub = chi1_lib[chi1_lib['chi1_rotamer'] == rot]
                if eq_col in sub.columns:
                    vals = sub[eq_col].dropna()
                    if len(vals) > 0:
                        R.append(f"    {rot:>3s}: ⟨eq⟩ = {vals.mean():.2f}°  "
                                 f"σ = {vals.std():.3f}°  n_cells = {len(vals)}")

    # ── Library usage ────────────────────────────────────────────────────
    R.append("\n" + "━" * 78)
    R.append("SECTION 5: HOW TO USE THE LIBRARY")
    R.append("━" * 78)
    R.append("""
  Python usage:
    import json
    with open('constants_library.json') as f:
        lib = json.load(f)
    
    # Look up τ equilibrium for φ=-65°, ψ=-45° (αR), GLY class
    phi_key = '-65'
    psi_key = '-45'
    tau_eq = lib['GLY'][phi_key][psi_key]['tau_deg_eq']
    tau_k  = lib['GLY'][phi_key][psi_key]['tau_deg_k']
    
    # With coupling correction
    tau_corrected = tau_eq + lib['GLY'][phi_key][psi_key].get('tau_deg_coupling', 0)
    
    # For NeRF reconstruction
    bond_NCA = lib['ALL'][phi_key][psi_key]['bond_N_CA_eq']
    bond_CAC = lib['ALL'][phi_key][psi_key]['bond_CA_C_eq']

  Fallback: if a specific (φ,ψ) cell is missing, use the 'ALL' class
  or the nearest populated cell.
""")

    return '\n'.join(R)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Paper 4 — Extract geometry constant library')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='./paper4_library')
    ap.add_argument('--bin_size', type=int, default=10)
    ap.add_argument('--min_count', type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    # ── Read CSV ─────────────────────────────────────────────────────────
    print(f"[1/6] Reading {args.csv}...")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns  ({time.time()-t0:.1f}s)")

    geo_cols = [c for c in GEOMETRY_COLS if c in df.columns]
    print(f"  Geometry columns found: {len(geo_cols)}")

    # ── Extract ALL-residue library ──────────────────────────────────────
    print(f"[2/6] Extracting ALL-residue cell statistics...")
    cell_stats_all = extract_cell_stats(df, 'phi_deg', 'psi_deg',
                                         args.bin_size, args.min_count)
    cell_stats_all = compute_coupling_corrections(
        df, cell_stats_all, 'phi_deg', 'psi_deg', args.bin_size)
    print(f"  {len(cell_stats_all)} cells")

    # ── Per-residue-class libraries ──────────────────────────────────────
    print(f"[3/6] Per-residue libraries...")
    cell_stats_by_class = {'ALL': cell_stats_all}

    if 'res_name' in df.columns:
        # Individual amino acids
        for aa in ALL_AA:
            mask = df['res_name'] == aa
            if mask.sum() < 500:
                continue
            cs = extract_cell_stats(df[mask], 'phi_deg', 'psi_deg',
                                     args.bin_size, args.min_count)
            if len(cs) > 10:
                cs = compute_coupling_corrections(
                    df[mask], cs, 'phi_deg', 'psi_deg', args.bin_size)
                cell_stats_by_class[aa] = cs
                print(f"  {aa:>4s}: {len(cs)} cells")

    # ── χ₁ sublibrary ────────────────────────────────────────────────────
    print(f"[4/6] χ₁ sublibrary...")
    chi1_lib = extract_chi1_sublibrary(df, bin_size=20, min_count=20)
    if chi1_lib is not None:
        print(f"  {len(chi1_lib):,} entries")
    else:
        print(f"  χ₁ column not found")

    # ── Export ────────────────────────────────────────────────────────────
    print(f"[5/6] Exporting...")

    # JSON library (nested: class -> phi -> psi -> values)
    json_lib = {}
    for cls_name, cs in cell_stats_by_class.items():
        json_lib[cls_name] = to_nested_json(cs, geo_cols)

    json_path = os.path.join(args.out, 'constants_library.json')
    with open(json_path, 'w') as f:
        json.dump(json_lib, f, indent=2)
    print(f"  Saved {json_path} ({os.path.getsize(json_path)/1024:.0f} KB)")

    # Flat CSV (for inspection)
    csv_path = os.path.join(args.out, 'constants_library.csv')
    cell_stats_all.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    # Per-class CSVs
    for cls_name, cs in cell_stats_by_class.items():
        if cls_name == 'ALL':
            continue
        cls_path = os.path.join(args.out, f'constants_{cls_name}.csv')
        cs.to_csv(cls_path, index=False)

    # χ₁ sublibrary
    if chi1_lib is not None:
        chi1_json = {}
        for _, row in chi1_lib.iterrows():
            res = row['res_name']
            phi_k = str(int(row['phi_center']))
            psi_k = str(int(row['psi_center']))
            rot = row['chi1_rotamer']
            if res not in chi1_json:
                chi1_json[res] = {}
            if phi_k not in chi1_json[res]:
                chi1_json[res][phi_k] = {}
            if psi_k not in chi1_json[res][phi_k]:
                chi1_json[res][phi_k][psi_k] = {}
            entry = {'n': int(row['n'])}
            for col in ['angle_N_CA_CB', 'angle_C_CA_CB', 'bond_CA_CB']:
                for suf in ['_eq', '_std']:
                    k = f'{col}{suf}'
                    if k in row and not pd.isna(row[k]):
                        entry[k] = float(row[k])
            chi1_json[res][phi_k][psi_k][rot] = entry

        chi1_path = os.path.join(args.out, 'constants_chi1.json')
        with open(chi1_path, 'w') as f:
            json.dump(chi1_json, f, indent=2)
        print(f"  Saved {chi1_path}")

        chi1_csv_path = os.path.join(args.out, 'constants_chi1.csv')
        chi1_lib.to_csv(chi1_csv_path, index=False)
        print(f"  Saved {chi1_csv_path}")

    # AMBER comparison
    amber_rows = []
    for _, row in cell_stats_all.iterrows():
        arow = {
            'phi': row['phi_center'], 'psi': row['psi_center'], 'n': row['n']
        }
        for col in geo_cols:
            eq_col = f'{col}_eq'
            k_col = f'{col}_k'
            da_col = f'{col}_delta_amber'
            if eq_col in row:
                arow[f'{col}_pdb'] = row.get(eq_col)
                arow[f'{col}_amber'] = GEOMETRY_COLS[col]['amber_eq']
                arow[f'{col}_delta'] = row.get(da_col)
                arow[f'{col}_k_pdb'] = row.get(k_col)
                arow[f'{col}_k_amber'] = GEOMETRY_COLS[col]['amber_k']
        amber_rows.append(arow)
    amber_df = pd.DataFrame(amber_rows)
    amber_path = os.path.join(args.out, 'amber_comparison.csv')
    amber_df.to_csv(amber_path, index=False)
    print(f"  Saved {amber_path}")

    # ── Report ───────────────────────────────────────────────────────────
    print(f"[6/6] Report...")
    report = build_report(cell_stats_all, cell_stats_by_class, chi1_lib,
                          geo_cols)

    report_path = os.path.join(args.out, 'library_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved {report_path}")
    print(report)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Library extraction complete")
    print(f"  Output: {args.out}/")
    print(f"    constants_library.json  — main library (nested JSON)")
    print(f"    constants_library.csv   — flat table (ALL residues)")
    print(f"    constants_<AA>.csv      — per-amino-acid tables")
    if chi1_lib is not None:
        print(f"    constants_chi1.json    — χ₁-dependent Cβ angles")
    print(f"    amber_comparison.csv   — PDB vs AMBER comparison")
    print(f"    library_report.txt     — summary report")
    print(f"  Done in {time.time()-t0:.0f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()