#!/usr/bin/env python3
"""
paper1_figure7_rebuild.py — Corrected Figure 7 + Table S1
=============================================================
Rebuilds the AMBER vs. empirical k_phi comparison from scratch using:

  1. The REAL, verified ff14SB backbone torsion Fourier coefficients
     (extracted directly from protein.ff14SB.xml, the authoritative
     OpenMM/AmberTools distribution) -- NOT either of the two previously
     hardcoded arrays, both of which were confirmed wrong (one had the
     wrong sign; NEITHER matched the real published magnitudes/phases).
  2. The correct second-derivative sign convention:
       d2V/dtheta2 = -sum (Vn/2) n^2 cos(n*theta - gamma)
  3. The empirical secant k_eff = -tau_phi_correct/delta_phi (Methods 4.7),
     honestly reported AS a secant (not relabeled as a true local
     curvature), with bin occupancy and bootstrap 95% CI (resampled by
     structure).

Produces:
  - table_S1_stiffness_comparison.csv  (every populated bin: n, n_structures,
    empirical k_eff + CI, AMBER k_phi [verified], delta = empirical - AMBER)
  - figure7_corrected.png  (Panel A: delta_k heatmap; Panel C: alpha-helical
    corridor phi=-65 vs phi=-55, well-supported bins only, n>=100)

Usage:
  python paper1_figure7_rebuild.py --csv features_lj_FINAL_CLEAN.csv
"""

import argparse
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

PHI_REF, PSI_REF = -63.0, -43.0
MIN_DTHETA_DEG = 5.0
BIN_SIZE = 10
MIN_N_WELL_SUPPORTED = 100

# REAL, VERIFIED ff14SB backbone phi torsion (protein.ff14SB.xml)
# format: (periodicity n, Vn/2 kcal/mol, phase gamma in radians)
PHI_REAL = [(3, 1.75728, 0.0), (2, 1.12968, 0.0)]


def wrap180(x):
    return (x + 180) % 360 - 180


def amber_k_phi(phi_deg):
    """Correct-sign, verified AMBER ff14SB curvature at a given phi.
    Note: ff14SB's phi torsion term has NO psi-dependence by construction
    (the whole point of the separability comparison) -- so this is a
    function of phi alone, same value for every psi at that phi."""
    t = np.radians(phi_deg)
    return -sum(vn2 * n**2 * np.cos(n * t - g) for n, vn2, g in PHI_REAL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--out_prefix', default='.')
    args = ap.parse_args()

    print("Loading data...")
    df = pd.read_csv(args.csv, low_memory=False)
    df = df.dropna(subset=['phi_deg', 'psi_deg', 'tau_phi_correct'])
    print(f"  {len(df):,} residues, {df['pdb_id'].nunique()} structures")

    dphi_deg = wrap180(df['phi_deg'].values - PHI_REF)
    dphi_rad = np.radians(dphi_deg)
    valid = np.abs(dphi_deg) > MIN_DTHETA_DEG
    df = df[valid].copy()
    dphi_rad = dphi_rad[valid]
    df['k_eff_phi'] = -df['tau_phi_correct'].values / dphi_rad
    print(f"  {len(df):,} residues after |delta_phi| > {MIN_DTHETA_DEG} deg filter")

    phi_bins = np.arange(-180, 190, BIN_SIZE)
    psi_bins = np.arange(-180, 190, BIN_SIZE)
    df['pb'] = pd.cut(df['phi_deg'], phi_bins, labels=False, right=False)
    df['qb'] = pd.cut(df['psi_deg'], psi_bins, labels=False, right=False)
    df = df.dropna(subset=['pb', 'qb'])
    df['pb'] = df['pb'].astype(int)
    df['qb'] = df['qb'].astype(int)
    phi_centers = phi_bins[:-1] + BIN_SIZE / 2
    psi_centers = psi_bins[:-1] + BIN_SIZE / 2

    print(f"\nComputing per-bin occupancy + bootstrap CI + verified AMBER "
          f"comparison ({args.n_boot} iterations, resampled by structure)...")
    rng = np.random.default_rng(0)
    results = []
    for (pb, qb), grp in df.groupby(['pb', 'qb']):
        n = len(grp)
        phi_c = phi_centers[pb]
        psi_c = psi_centers[qb]
        mean_k = grp['k_eff_phi'].mean()

        structs = grp['pdb_id'].unique()
        struct_groups = grp.groupby('pdb_id')['k_eff_phi'].apply(list).to_dict()
        n_struct = len(structs)
        boot_means = np.empty(args.n_boot)
        for b in range(args.n_boot):
            sampled = rng.choice(structs, size=n_struct, replace=True)
            vals = np.concatenate([struct_groups[s] for s in sampled])
            boot_means[b] = vals.mean()
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

        k_amber = amber_k_phi(phi_c)  # VERIFIED, correct-sign AMBER value

        results.append({
            'phi_center': phi_c, 'psi_center': psi_c, 'n': n,
            'n_structures': n_struct,
            'k_empirical_secant': mean_k, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'ci_width': ci_hi - ci_lo,
            'k_amber_ff14SB_verified': k_amber,
            'delta_k': mean_k - k_amber,
            'well_supported': n >= MIN_N_WELL_SUPPORTED,
            'ci_excludes_amber': not (ci_lo <= k_amber <= ci_hi),
        })

    res_df = pd.DataFrame(results)
    out_csv = f"{args.out_prefix}/table_S1_stiffness_comparison.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(res_df)} populated bins)")

    well = res_df[res_df['well_supported']]
    print(f"\n{'='*70}")
    print(f"  Well-supported bins (n >= {MIN_N_WELL_SUPPORTED}): {len(well)} / {len(res_df)}")
    print(f"  Of these, fraction where CI excludes the AMBER value "
          f"(genuine, defensible discrepancy): "
          f"{100*well['ci_excludes_amber'].mean():.1f}%")
    print(f"{'='*70}")

    # ── Figure 7, rebuilt ────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: delta_k heatmap, well-supported bins only
    ax = axes[0]
    pivot = well.pivot_table(index='psi_center', columns='phi_center', values='delta_k')
    im = ax.imshow(pivot.values, origin='lower', aspect='auto',
                   extent=[pivot.columns.min(), pivot.columns.max(),
                           pivot.index.min(), pivot.index.max()],
                   cmap='RdBu_r', vmin=-10, vmax=10)
    plt.colorbar(im, ax=ax, label='Δk = empirical secant − AMBER (verified)')
    ax.set_xlabel('φ (°)'); ax.set_ylabel('ψ (°)')
    ax.set_title(f'A. Stiffness discrepancy\n(n≥{MIN_N_WELL_SUPPORTED} bins only)')

    # Panel C: alpha-helical corridor, phi=-65 vs phi=-55, well-supported only
    ax = axes[1]
    for phi_target, color in [(-65, '#D85A30'), (-55, '#378ADD')]:
        corridor = well[well['phi_center'] == phi_target].sort_values('psi_center')
        if len(corridor) == 0:
            continue
        ax.plot(corridor['psi_center'], corridor['k_empirical_secant'],
                'o-', color=color, label=f'Empirical, φ={phi_target}°')
        ax.fill_between(corridor['psi_center'], corridor['ci_lo'], corridor['ci_hi'],
                         color=color, alpha=0.2)
    # AMBER reference (psi-independent by construction)
    psi_range = np.linspace(-180, 180, 100)
    for phi_target, color in [(-65, '#D85A30'), (-55, '#378ADD')]:
        ax.axhline(amber_k_phi(phi_target), color=color, linestyle='--', alpha=0.6,
                   label=f'AMBER (verified), φ={phi_target}° (ψ-independent)')
    ax.set_xlabel('ψ (°)'); ax.set_ylabel('k_φ (kcal/mol/rad²)')
    ax.set_title(f'B. α-helical corridor\n(n≥{MIN_N_WELL_SUPPORTED} bins only)')
    ax.legend(fontsize=7)

    plt.tight_layout()
    out_png = f"{args.out_prefix}/figure7_corrected.png"
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved {out_png}")

    print(f"""
{'='*70}
IMPORTANT CAVEAT for the manuscript text:
  k_empirical is a SECANT (-tau_env/delta_phi relative to the alphaR
  reference), not a true local second derivative d2V/dphi2. This is the
  same quantity as Fig 5, honestly labeled as such (see response to
  Major Comment #4). AMBER's k_phi IS a true analytic second derivative
  (verified against the real ff14SB Fourier coefficients, correct sign).
  The comparison is therefore secant-vs-curvature, not curvature-vs-
  curvature -- report it as such rather than implying otherwise.
{'='*70}
""")


if __name__ == '__main__':
    main()
