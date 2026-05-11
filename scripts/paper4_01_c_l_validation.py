#!/usr/bin/env python3
"""
Paper 4 — Geometry Library Validation
======================================

Three validation approaches, in priority order:

  1. Engh & Huber (2001) comparison
     Global PDB means vs. the crystallographic gold standard.
     Flags any observable whose global mean deviates by more than
     one Engh-Huber σ — those entries need re-examination before
     the library is used in NeRF.

  2. τ by secondary structure
     αR / β-strand / PPII / GLY-αR cells are compared against
     published reference values (Lovell 2003, Berkholz 2009).
     Validates that the (φ,ψ) coupling signal is real and correctly
     signed, independent of any external validation set.

  3. Bootstrap confidence intervals on coupling corrections
     For every populated (φ,ψ) cell, 500 bootstrap resamples
     estimate the sampling uncertainty on the coupling correction.
     Cells where |correction| < 2×σ_bootstrap are flagged as
     statistically unreliable; the caller should fall back to the
     marginal mean for those cells.

Output:
  validation_eh_comparison.csv     — observable-level E&H diff table
  validation_ss_tau.csv            — secondary-structure τ table
  validation_bootstrap.csv         — per-cell bootstrap results
  validation_report.txt            — human-readable summary
  validation_flags.json            — machine-readable pass/fail dict

Usage:
  python paper4_02_validation.py \
      --csv  /mnt/f/Protein_Folding/v8_g/p3.csv \
      --lib  ./paper4_library/constants_library.csv \
      --out  ./paper4_library/

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
# 1. ENGH & HUBER 2001 REFERENCE VALUES
#    Source: Engh & Huber, "Structure quality and target parameters",
#            International Tables for Crystallography Vol. F, 2001.
#    These are derived from ultra-high-resolution (<1.0 Å) structures,
#    so they are the closest thing to "true" small-molecule geometry.
#
#    Columns: (amber_eq, eh_mean, eh_sigma, unit, label)
#    eh_sigma is the population σ from Table 1 of E&H 2001.
# ══════════════════════════════════════════════════════════════════════════════

EH_REF = {
    # Angles [degrees]
    'tau_deg':       dict(eh_mean=111.2, eh_sigma=2.8,  unit='deg',  label='τ (N-Cα-C)'),
    'angle_N_CA_CB': dict(eh_mean=110.5, eh_sigma=1.7,  unit='deg',  label='∠N-Cα-Cβ'),
    'angle_C_CA_CB': dict(eh_mean=110.1, eh_sigma=1.9,  unit='deg',  label='∠C-Cα-Cβ'),
    'angle_CaCN':    dict(eh_mean=116.2, eh_sigma=2.0,  unit='deg',  label='∠Cα-C-N'),
    'angle_CNCa':    dict(eh_mean=121.7, eh_sigma=1.8,  unit='deg',  label='∠C-N-Cα'),
    'angle_CA_C_O':  dict(eh_mean=120.8, eh_sigma=1.7,  unit='deg',  label='∠Cα-C=O'),
    # Bonds [Å]
    'bond_N_CA':     dict(eh_mean=1.459, eh_sigma=0.020, unit='Å',   label='N-Cα'),
    'bond_CA_C':     dict(eh_mean=1.525, eh_sigma=0.021, unit='Å',   label='Cα-C'),
    'bond_C_O':      dict(eh_mean=1.229, eh_sigma=0.019, unit='Å',   label='C=O'),
    'bond_C_N_next': dict(eh_mean=1.336, eh_sigma=0.023, unit='Å',   label='C-N'),
    'bond_CA_CB':    dict(eh_mean=1.530, eh_sigma=0.020, unit='Å',   label='Cα-Cβ'),
}

# ══════════════════════════════════════════════════════════════════════════════
# 2. SECONDARY STRUCTURE τ REFERENCE VALUES
#    Source: Lovell et al. 2003 (Proteins 50:437) Table 1;
#            Berkholz et al. 2009 (Structure 17:1316) for PPII.
#    These are φ/ψ-region means, not per-structure values.
# ══════════════════════════════════════════════════════════════════════════════

SS_TAU_REF = {
    'alpha_R': dict(
        phi_range=(-90, -40),  psi_range=(-70, -10),
        tau_ref=111.6, tau_ref_sigma=0.3,
        source='Lovell 2003',
        label='αR helix',
    ),
    'beta_strand': dict(
        phi_range=(-160, -90), psi_range=(90, 170),
        tau_ref=110.4, tau_ref_sigma=0.4,
        source='Lovell 2003',
        label='β-strand',
    ),
    'PPII': dict(
        phi_range=(-90, -50),  psi_range=(120, 180),
        tau_ref=111.0, tau_ref_sigma=0.5,
        source='Berkholz 2009',
        label='PPII helix',
    ),
    'gly_alpha_R': dict(
        phi_range=(-90, -40),  psi_range=(-70, -10),
        tau_ref=113.1, tau_ref_sigma=0.5,
        source='Lovell 2003 (GLY)',
        label='GLY αR',
        residue='GLY',
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def weighted_mean_std(values, weights):
    """Weighted mean and population std."""
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[mask], w[mask]
    if len(v) == 0:
        return np.nan, np.nan
    mu = np.average(v, weights=w)
    var = np.average((v - mu) ** 2, weights=w)
    return mu, np.sqrt(var)


def cells_in_region(cell_stats, phi_range, psi_range):
    """Return rows of cell_stats whose centres fall in the given ranges."""
    mask = (
        (cell_stats['phi_center'] >= phi_range[0]) &
        (cell_stats['phi_center'] <= phi_range[1]) &
        (cell_stats['psi_center'] >= psi_range[0]) &
        (cell_stats['psi_center'] <= psi_range[1])
    )
    return cell_stats[mask]


# ══════════════════════════════════════════════════════════════════════════════
# Validation 1 — Engh & Huber
# ══════════════════════════════════════════════════════════════════════════════

def validate_engh_huber(cell_stats):
    """Compare global PDB means to Engh & Huber 2001 reference values.

    Returns
    -------
    rows : list[dict]
        One row per observable.
    flags : dict
        {observable: 'PASS'|'WARN'|'FAIL'}
        PASS  : |Δ| < 0.5 σ_EH
        WARN  : 0.5 σ_EH ≤ |Δ| < 1.0 σ_EH
        FAIL  : |Δ| ≥ 1.0 σ_EH
    """
    rows = []
    flags = {}

    for col, ref in EH_REF.items():
        eq_col = f'{col}_eq'
        if eq_col not in cell_stats.columns:
            continue

        valid = cell_stats[[eq_col, 'n']].dropna()
        if len(valid) < 5:
            continue

        pdb_mean, pdb_std = weighted_mean_std(valid[eq_col], valid['n'])
        delta = pdb_mean - ref['eh_mean']
        n_sigma = abs(delta) / ref['eh_sigma']

        if n_sigma < 0.5:
            status = 'PASS'
        elif n_sigma < 1.0:
            status = 'WARN'
        else:
            status = 'FAIL'

        rows.append({
            'observable': col,
            'label':      ref['label'],
            'unit':       ref['unit'],
            'eh_mean':    ref['eh_mean'],
            'eh_sigma':   ref['eh_sigma'],
            'pdb_mean':   round(pdb_mean, 4),
            'pdb_std':    round(pdb_std, 4),
            'delta':      round(delta, 4),
            'n_sigma_EH': round(n_sigma, 3),
            'status':     status,
        })
        flags[col] = status

    return pd.DataFrame(rows), flags


# ══════════════════════════════════════════════════════════════════════════════
# Validation 2 — τ by secondary structure
# ══════════════════════════════════════════════════════════════════════════════

def validate_ss_tau(cell_stats, cell_stats_by_class):
    """Compare per-region τ means to published reference values.

    Returns
    -------
    rows : list[dict]
    flags : dict
    """
    rows = []
    flags = {}

    for ss_name, ref in SS_TAU_REF.items():
        # Choose correct cell_stats frame
        residue = ref.get('residue')
        if residue:
            cs = cell_stats_by_class.get(residue, cell_stats)
        else:
            cs = cell_stats

        if 'tau_deg_eq' not in cs.columns:
            continue

        region = cells_in_region(cs, ref['phi_range'], ref['psi_range'])
        valid = region[['tau_deg_eq', 'n']].dropna()

        if len(valid) < 3:
            rows.append({
                'region': ss_name, 'label': ref['label'],
                'tau_ref': ref['tau_ref'], 'tau_ref_sigma': ref['tau_ref_sigma'],
                'tau_pdb': np.nan, 'tau_pdb_spread': np.nan,
                'delta': np.nan, 'n_cells': 0, 'n_obs': 0,
                'status': 'MISSING',
            })
            flags[ss_name] = 'MISSING'
            continue

        tau_pdb, tau_spread = weighted_mean_std(valid['tau_deg_eq'], valid['n'])
        delta = tau_pdb - ref['tau_ref']
        # flag based on published σ + spread
        threshold = ref['tau_ref_sigma'] + tau_spread / 2
        status = 'PASS' if abs(delta) < threshold else 'WARN' if abs(delta) < 2 * threshold else 'FAIL'

        rows.append({
            'region':         ss_name,
            'label':          ref['label'],
            'source':         ref['source'],
            'tau_ref':        ref['tau_ref'],
            'tau_ref_sigma':  ref['tau_ref_sigma'],
            'tau_pdb':        round(tau_pdb, 3),
            'tau_pdb_spread': round(tau_spread, 3),
            'delta':          round(delta, 4),
            'n_cells':        len(valid),
            'n_obs':          int(valid['n'].sum()),
            'status':         status,
        })
        flags[ss_name] = status

    # Monotonicity check: β-strand τ < αR τ (well-established physics)
    ss_res = {r['region']: r for r in rows if not np.isnan(r.get('tau_pdb', np.nan))}
    if 'beta_strand' in ss_res and 'alpha_R' in ss_res:
        beta_tau = ss_res['beta_strand']['tau_pdb']
        alpha_tau = ss_res['alpha_R']['tau_pdb']
        mono_ok = beta_tau < alpha_tau
        rows.append({
            'region': 'monotonicity_check',
            'label': 'β < αR (physics constraint)',
            'source': 'Lovell 2003',
            'tau_ref': np.nan, 'tau_ref_sigma': np.nan,
            'tau_pdb': alpha_tau - beta_tau,  # should be > 0
            'tau_pdb_spread': np.nan,
            'delta': np.nan,
            'n_cells': 0, 'n_obs': 0,
            'status': 'PASS' if mono_ok else 'FAIL',
        })
        flags['monotonicity_beta_lt_alphaR'] = 'PASS' if mono_ok else 'FAIL'

    return pd.DataFrame(rows), flags


# ══════════════════════════════════════════════════════════════════════════════
# Validation 3 — Bootstrap confidence intervals on coupling corrections
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_coupling(df, phi_col, psi_col, geo_col, bin_size,
                       phi_center, psi_center,
                       n_boot=500, seed=42):
    """Bootstrap the coupling correction for one (φ,ψ) cell.

    Strategy:
      - Pull all observations in the cell from the raw data.
      - Resample with replacement n_boot times.
      - For each resample, recompute the coupling correction using the
        same grand-mean / marginal decomposition as the main library.
      - Return the bootstrap std of the coupling correction.

    Note: this uses the *full-data* grand mean and marginals as fixed
    reference (not re-estimated per bootstrap), which is the correct
    approach when the cell is small relative to the total dataset —
    it isolates sampling uncertainty in the cell mean rather than
    re-estimating global structure each time.
    """
    rng = np.random.default_rng(seed)
    half = bin_size / 2

    phi_lo = phi_center - half
    phi_hi = phi_center + half
    psi_lo = psi_center - half
    psi_hi = psi_center + half

    mask = (
        (df[phi_col] >= phi_lo) & (df[phi_col] < phi_hi) &
        (df[psi_col] >= psi_lo) & (df[psi_col] < psi_hi) &
        df[geo_col].notna()
    )
    cell_data = df.loc[mask, geo_col].values

    if len(cell_data) < 20:
        return np.nan, len(cell_data)

    # Fixed reference: grand mean from full column
    grand_mean = df[geo_col].mean()

    # Marginal means (fixed from full data)
    phi_bins = np.arange(-180, 180 + bin_size, bin_size)
    psi_bins = np.arange(-180, 180 + bin_size, bin_size)
    df2 = df[[phi_col, psi_col, geo_col]].dropna()
    df2 = df2.copy()
    df2['pb'] = pd.cut(df2[phi_col], phi_bins, labels=False, right=False)
    df2['qb'] = pd.cut(df2[psi_col], psi_bins, labels=False, right=False)

    phi_bin_idx = int((phi_center + 180) / bin_size)
    psi_bin_idx = int((psi_center + 180) / bin_size)

    phi_marg = df2[df2['pb'] == phi_bin_idx][geo_col].mean()
    psi_marg = df2[df2['qb'] == psi_bin_idx][geo_col].mean()

    if np.isnan(phi_marg) or np.isnan(psi_marg):
        return np.nan, len(cell_data)

    additive = grand_mean + (phi_marg - grand_mean) + (psi_marg - grand_mean)

    # Bootstrap
    boot_corrections = []
    for _ in range(n_boot):
        sample = rng.choice(cell_data, size=len(cell_data), replace=True)
        cell_mean_boot = sample.mean()
        boot_corrections.append(cell_mean_boot - additive)

    return np.std(boot_corrections), len(cell_data)


def validate_bootstrap(df, cell_stats, geo_cols,
                       bin_size=10, n_boot=500, max_cells=2000):
    """Run bootstrap validation on coupling corrections.

    Only runs on cells that (a) have a coupling correction column and
    (b) have n >= 20 raw observations.

    Returns
    -------
    result_df : pd.DataFrame
        One row per (phi_center, psi_center, geo_col).
    flags : dict
        Fraction of cells that are statistically reliable, per geo_col.
    """
    rows = []
    flags = {}

    # Subsample cells if too many (for speed)
    work_cells = cell_stats.copy()
    if len(work_cells) > max_cells:
        work_cells = work_cells.sample(max_cells, random_state=42)

    for col in geo_cols:
        coup_col = f'{col}_coupling'
        if coup_col not in cell_stats.columns:
            continue
        if col not in df.columns:
            continue

        n_reliable = 0
        n_tested = 0

        for _, crow in work_cells.iterrows():
            coup_val = crow.get(coup_col)
            if pd.isna(coup_val):
                continue

            boot_sigma, cell_n = bootstrap_coupling(
                df, 'phi_deg', 'psi_deg', col,
                bin_size, crow['phi_center'], crow['psi_center'],
                n_boot=n_boot,
            )

            if np.isnan(boot_sigma):
                status = 'INSUFFICIENT'
            elif abs(coup_val) >= 2 * boot_sigma:
                status = 'RELIABLE'
                n_reliable += 1
            else:
                status = 'UNRELIABLE'
            n_tested += 1

            rows.append({
                'phi_center':    crow['phi_center'],
                'psi_center':    crow['psi_center'],
                'geo_col':       col,
                'coupling_val':  round(coup_val, 4),
                'boot_sigma':    round(boot_sigma, 4) if not np.isnan(boot_sigma) else np.nan,
                'snr':           round(abs(coup_val) / boot_sigma, 2) if (not np.isnan(boot_sigma) and boot_sigma > 0) else np.nan,
                'cell_n':        cell_n,
                'status':        status,
            })

        frac_reliable = n_reliable / n_tested if n_tested > 0 else np.nan
        flags[col] = {
            'n_tested':      n_tested,
            'n_reliable':    n_reliable,
            'frac_reliable': round(frac_reliable, 3) if not np.isnan(frac_reliable) else None,
        }

    return pd.DataFrame(rows), flags


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

def build_validation_report(eh_df, ss_df, boot_flags, all_flags):
    R = []
    R.append("=" * 78)
    R.append("Paper 4 — Geometry Library Validation Report")
    R.append("=" * 78)

    # ── 1. Engh & Huber ───────────────────────────────────────────────────
    R.append("\n" + "━" * 78)
    R.append("VALIDATION 1: ENGH & HUBER 2001 COMPARISON")
    R.append("━" * 78)
    R.append(f"  {'Observable':>20s}  {'E&H':>7s}  {'PDB':>7s}  {'Δ':>8s}  {'|Δ|/σ_EH':>9s}  {'Status':>8s}")
    R.append("  " + "─" * 66)

    for _, row in eh_df.iterrows():
        R.append(
            f"  {row['label']:>20s}  {row['eh_mean']:>7.3f}  {row['pdb_mean']:>7.3f}  "
            f"{row['delta']:>+8.4f}  {row['n_sigma_EH']:>9.3f}  {row['status']:>8s}"
        )

    eh_pass = (eh_df['status'] == 'PASS').sum()
    eh_warn = (eh_df['status'] == 'WARN').sum()
    eh_fail = (eh_df['status'] == 'FAIL').sum()
    R.append(f"\n  Summary: {eh_pass} PASS  {eh_warn} WARN  {eh_fail} FAIL  "
             f"(of {len(eh_df)} observables)")
    R.append("""
  Thresholds:
    PASS : |Δ| < 0.5 σ_EH  — library mean within ½ E&H uncertainty
    WARN : |Δ| < 1.0 σ_EH  — marginal; investigate but not disqualifying
    FAIL : |Δ| ≥ 1.0 σ_EH  — global mean differs from crystallographic ref
                              by more than one E&H population σ
""")

    # ── 2. Secondary structure τ ──────────────────────────────────────────
    R.append("━" * 78)
    R.append("VALIDATION 2: τ BY SECONDARY STRUCTURE")
    R.append("━" * 78)
    R.append(f"  {'Region':>18s}  {'τ_ref':>7s}  {'τ_PDB':>7s}  {'Δ':>7s}  "
             f"{'N_cells':>8s}  {'N_obs':>8s}  {'Source':>16s}  {'Status':>8s}")
    R.append("  " + "─" * 84)

    for _, row in ss_df.iterrows():
        if np.isnan(row.get('tau_ref', np.nan)):
            R.append(f"  {row['label']:>18s}  {'—':>7s}  {row['tau_pdb']:>7.3f}  {'—':>7s}  "
                     f"{'—':>8s}  {'—':>8s}  {'—':>16s}  {row['status']:>8s}")
        else:
            R.append(
                f"  {row['label']:>18s}  {row['tau_ref']:>7.3f}  {row['tau_pdb']:>7.3f}  "
                f"{row['delta']:>+7.4f}  {int(row['n_cells']):>8d}  {int(row['n_obs']):>8d}  "
                f"{row.get('source',''):>16s}  {row['status']:>8s}"
            )

    R.append("""
  Physics constraint check:
    β-strand τ < αR τ is required by the pyramidalization model (Paper 2).
    FAIL here means the coupling signal has the wrong sign — data issue.

  Note: PPII WARN/FAIL is expected for datasets with <5% PPII content
  because the region is sparsely sampled.
""")

    # ── 3. Bootstrap ─────────────────────────────────────────────────────
    R.append("━" * 78)
    R.append("VALIDATION 3: BOOTSTRAP CONFIDENCE INTERVALS ON COUPLING CORRECTIONS")
    R.append("━" * 78)

    if boot_flags:
        R.append(f"  {'Observable':>20s}  {'N_tested':>9s}  {'N_reliable':>11s}  {'Frac_reliable':>14s}")
        R.append("  " + "─" * 60)
        for col, info in boot_flags.items():
            fr = info['frac_reliable']
            R.append(f"  {col:>20s}  {info['n_tested']:>9d}  {info['n_reliable']:>11d}  "
                     f"{fr if fr is not None else '—':>14}")
        R.append("""
  A cell's coupling correction is RELIABLE when |correction| ≥ 2×σ_bootstrap.
  Cells flagged UNRELIABLE should use the marginal mean in NeRF reconstruction.
  Frac_reliable < 0.50 for a column means its coupling map is mostly noise
  at the current bin size — consider increasing bin_size to 20°.
""")
    else:
        R.append("  Bootstrap skipped (no coupling columns found).\n")

    # ── Overall verdict ───────────────────────────────────────────────────
    R.append("━" * 78)
    R.append("OVERALL VERDICT")
    R.append("━" * 78)

    all_pass = all(v == 'PASS' for v in all_flags.values() if v in ('PASS', 'FAIL', 'WARN'))
    any_fail = any(v == 'FAIL' for v in all_flags.values())

    for k, v in all_flags.items():
        R.append(f"  {k:<40s}  {v}")

    R.append("")
    if any_fail:
        R.append("  *** LIBRARY HAS FAILURES — review before NeRF integration ***")
    elif all_pass:
        R.append("  ✓ All checks pass — library is ready for NeRF integration")
    else:
        R.append("  ~ Warnings present — library is usable; investigate flagged cells")

    return '\n'.join(R)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='Paper 4 — Library Validation')
    ap.add_argument('--csv',      required=True,  help='Raw per-residue CSV (p3.csv)')
    ap.add_argument('--lib',      required=True,  help='constants_library.csv (ALL residues)')
    ap.add_argument('--lib_dir',  default=None,   help='Directory with per-AA CSVs (optional)')
    ap.add_argument('--out',      default='./paper4_library')
    ap.add_argument('--bin_size', type=int, default=10)
    ap.add_argument('--n_boot',   type=int, default=500,
                    help='Bootstrap resamples per cell (default 500)')
    ap.add_argument('--max_boot_cells', type=int, default=500,
                    help='Max cells to bootstrap per observable (speed limit)')
    ap.add_argument('--skip_bootstrap', action='store_true',
                    help='Skip bootstrap (fast mode for debugging)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"[1/5] Loading raw data from {args.csv}...")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"  {len(df):,} rows")

    print(f"[2/5] Loading library from {args.lib}...")
    cell_stats = pd.read_csv(args.lib)
    print(f"  {len(cell_stats):,} cells")

    # Per-class cell stats (optional)
    cell_stats_by_class = {'ALL': cell_stats}
    if args.lib_dir:
        for aa in ['GLY', 'ALA', 'VAL', 'ILE', 'LEU', 'PRO',
                   'PHE', 'TYR', 'TRP', 'SER', 'THR', 'CYS',
                   'MET', 'ASP', 'ASN', 'GLU', 'GLN', 'LYS', 'ARG', 'HIS']:
            path = os.path.join(args.lib_dir, f'constants_{aa}.csv')
            if os.path.exists(path):
                cell_stats_by_class[aa] = pd.read_csv(path)

    geo_cols = [c.replace('_eq', '') for c in cell_stats.columns if c.endswith('_eq')]
    print(f"  Geometry columns: {geo_cols}")

    # ── Validation 1: Engh & Huber ────────────────────────────────────────
    print(f"[3/5] Engh & Huber comparison...")
    eh_df, eh_flags = validate_engh_huber(cell_stats)
    eh_path = os.path.join(args.out, 'validation_eh_comparison.csv')
    eh_df.to_csv(eh_path, index=False)
    print(f"  Saved {eh_path}")
    for _, row in eh_df.iterrows():
        print(f"  {row['label']:>20s}: Δ={row['delta']:+.4f}  ({row['n_sigma_EH']:.2f}σ_EH)  [{row['status']}]")

    # ── Validation 2: τ by secondary structure ────────────────────────────
    print(f"[4/5] Secondary structure τ validation...")
    ss_df, ss_flags = validate_ss_tau(cell_stats, cell_stats_by_class)
    ss_path = os.path.join(args.out, 'validation_ss_tau.csv')
    ss_df.to_csv(ss_path, index=False)
    print(f"  Saved {ss_path}")
    for _, row in ss_df.iterrows():
        tau_str = f"{row['tau_pdb']:.3f}" if not np.isnan(row.get('tau_pdb', np.nan)) else "—"
        delta_str = f"{row['delta']:+.4f}" if not np.isnan(row.get('delta', np.nan)) else "—"
        print(f"  {row['label']:>18s}: τ_PDB={tau_str}  Δ={delta_str}  [{row['status']}]")

    # ── Validation 3: Bootstrap ───────────────────────────────────────────
    boot_flags = {}
    boot_df = pd.DataFrame()

    if args.skip_bootstrap:
        print(f"[5/5] Bootstrap skipped (--skip_bootstrap)")
    else:
        print(f"[5/5] Bootstrap confidence intervals "
              f"(n_boot={args.n_boot}, max_cells={args.max_boot_cells})...")
        # Only run on geo_cols that have coupling columns
        boot_geo = [c for c in geo_cols if f'{c}_coupling' in cell_stats.columns]
        if boot_geo:
            boot_df, boot_flags = validate_bootstrap(
                df, cell_stats, boot_geo,
                bin_size=args.bin_size,
                n_boot=args.n_boot,
                max_cells=args.max_boot_cells,
            )
            boot_path = os.path.join(args.out, 'validation_bootstrap.csv')
            boot_df.to_csv(boot_path, index=False)
            print(f"  Saved {boot_path}")
            for col, info in boot_flags.items():
                print(f"  {col:>20s}: {info['n_reliable']}/{info['n_tested']} cells reliable "
                      f"({100*info['frac_reliable']:.0f}%)")
        else:
            print(f"  No coupling columns found — skipping bootstrap")

    # ── Assemble all flags ────────────────────────────────────────────────
    all_flags = {}
    for col, status in eh_flags.items():
        all_flags[f'EH:{col}'] = status
    for region, status in ss_flags.items():
        all_flags[f'SS:{region}'] = status
    for col, info in boot_flags.items():
        fr = info.get('frac_reliable')
        if fr is None:
            all_flags[f'BOOT:{col}'] = 'MISSING'
        elif fr >= 0.60:
            all_flags[f'BOOT:{col}'] = 'PASS'
        elif fr >= 0.40:
            all_flags[f'BOOT:{col}'] = 'WARN'
        else:
            all_flags[f'BOOT:{col}'] = 'FAIL'

    flags_path = os.path.join(args.out, 'validation_flags.json')
    with open(flags_path, 'w') as f:
        json.dump(all_flags, f, indent=2)
    print(f"  Saved {flags_path}")

    # ── Report ────────────────────────────────────────────────────────────
    report = build_validation_report(eh_df, ss_df, boot_flags, all_flags)
    report_path = os.path.join(args.out, 'validation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n{report}")

    print(f"\n{'='*60}")
    print(f"  Validation complete in {time.time()-t0:.0f}s")
    print(f"  Outputs in {args.out}/")
    print(f"    validation_eh_comparison.csv")
    print(f"    validation_ss_tau.csv")
    if not args.skip_bootstrap and len(boot_df) > 0:
        print(f"    validation_bootstrap.csv")
    print(f"    validation_flags.json")
    print(f"    validation_report.txt")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()