"""
ff_correction.py — Force-field torsional correction from observed k_eff map
============================================================================
Compares observed backbone stiffness (k_eff from spring analysis) against
AMBER ff19SB default torsional force constants.

Output:
  delta_k_phi.png / delta_k_psi.png  — where AMBER gets stiffness wrong
  ff_correction_table.csv            — Δk(φ,ψ) lookup table for MD groups
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# ── AMBER ff19SB backbone torsional parameters ────────────────────────────────
# These are the V/2 * [1 + cos(n*θ - γ)] terms for φ and ψ
# Source: ff19SB paper (Tian et al. 2020, JCTC) Table S1
# k_ff in kcal/mol/rad² is approximated as sum of n²*V_n/2 at the minimum
# This is the standard harmonic approximation around the equilibrium geometry

AMBER_FF19SB_PHI = {
    # (periodicity, V_n/2 kcal/mol, phase_deg) for backbone φ (C-N-CA-C)
    # General backbone — not residue specific
    'general': [
        (1, 0.1700, 0.0),
        (2, 0.2000, 180.0),
        (3, 0.2000, 0.0),
        (4, 0.1800, 0.0),
    ]
}

AMBER_FF19SB_PSI = {
    # Backbone ψ (N-CA-C-N)
    'general': [
        (1, 0.1500, 0.0),
        (2, 0.1500, 180.0),
        (3, 0.1500, 0.0),
        (4, 0.1000, 0.0),
    ]
}

def amber_torsion_energy(angle_deg, params):
    """V(θ) = Σ V_n/2 * [1 + cos(n*θ - γ)]"""
    theta = np.radians(angle_deg)
    e = 0.0
    for n, vn2, phase_deg in params:
        gamma = np.radians(phase_deg)
        e += vn2 * (1 + np.cos(n * theta - gamma))
    return e

def amber_torsion_curvature(angle_deg, params):
    """
    d²V/dθ² = Σ V_n/2 * n² * cos(n*θ - γ)
    This is the local force constant (positive = restoring, negative = driving)
    Units: kcal/mol/rad²
    """
    theta = np.radians(angle_deg)
    k = 0.0
    for n, vn2, phase_deg in params:
        gamma = np.radians(phase_deg)
        k += vn2 * n**2 * np.cos(n * theta - gamma)
    return k

def build_amber_k_map(params, bin_size=10):
    """Build 2D map of AMBER curvature — same for all ψ since φ params
    don't couple to ψ in a simple torsion potential."""
    edges = np.arange(-180, 181, bin_size)
    centres = (edges[:-1] + edges[1:]) / 2
    k_map = np.zeros((len(centres), len(centres)))
    for i, phi in enumerate(centres):
        k_val = amber_torsion_curvature(phi, params)
        k_map[i, :] = k_val  # φ params don't depend on ψ
    return k_map, centres

# ── Load your observed k_eff map (from run_spring_analysis output) ────────────

def load_observed_k_map(spring_meta, bin_size=10, percentile_clip=95):
    """
    Build observed k_eff(φ,ψ) map from spring_meta list.
    spring_meta is the output of run_spring_analysis().
    Each entry has: phi, psi, k_phi, k_psi
    """
    edges = np.arange(-180, 181, bin_size)
    centres = (edges[:-1] + edges[1:]) / 2
    n = len(centres)

    k_phi_map = np.full((n, n), np.nan)
    k_psi_map = np.full((n, n), np.nan)
    count_map  = np.zeros((n, n))

    # Bin each residue
    phi_arr = np.array([m['phi'] for m in spring_meta])
    psi_arr = np.array([m['psi'] for m in spring_meta])
    kp_arr  = np.array([m['k_phi'] for m in spring_meta])
    kq_arr  = np.array([m['k_psi'] for m in spring_meta])

    phi_idx = np.digitize(phi_arr, edges) - 1
    psi_idx = np.digitize(psi_arr, edges) - 1
    phi_idx = np.clip(phi_idx, 0, n-1)
    psi_idx = np.clip(psi_idx, 0, n-1)

    # Accumulate per bin
    from collections import defaultdict
    bins_phi = defaultdict(list)
    bins_psi = defaultdict(list)

    for i, (pi, qi) in enumerate(zip(phi_idx, psi_idx)):
        kp = kp_arr[i]; kq = kq_arr[i]
        if np.isfinite(kp): bins_phi[(pi, qi)].append(kp)
        if np.isfinite(kq): bins_psi[(pi, qi)].append(kq)

    for (pi, qi), vals in bins_phi.items():
        if len(vals) >= 3:
            k_phi_map[pi, qi] = np.median(vals)
            count_map[pi, qi] = len(vals)

    for (pi, qi), vals in bins_psi.items():
        if len(vals) >= 3:
            k_psi_map[pi, qi] = np.median(vals)

    return k_phi_map, k_psi_map, count_map, centres

# ── Compute Δk correction ─────────────────────────────────────────────────────

def compute_delta_k(observed_map, amber_map, count_map, min_count=10):
    """
    Δk = k_observed - k_AMBER
    Positive: observed is stiffer than FF predicts (FF underestimates resistance)
    Negative: observed is softer (FF overestimates resistance)
    Mask bins with insufficient data.
    """
    delta = observed_map - amber_map
    delta[count_map < min_count] = np.nan
    return delta

# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_correction(delta_phi, delta_psi, centres, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        'Force-field correction: Δk = k_observed − k_AMBER ff19SB\n'
        'Red = FF underestimates stiffness | Blue = FF overestimates stiffness',
        fontsize=12)

    BASIN_CENTRES = {
        'αR':  (-63, -43), 'β':  (-120, 128), 'PPII': (-72, 146),
        '3₁₀': (-52, -32), 'loop': (-95, 10),  'αL':  (60, 40)
    }

    for ax, delta, label in [
        (axes[0], delta_phi, 'Δk_φ (kcal/mol/rad²)'),
        (axes[1], delta_psi, 'Δk_ψ (kcal/mol/rad²)'),
    ]:
        valid = delta[np.isfinite(delta)]
        if len(valid) == 0:
            ax.set_title(f'{label} — no data'); continue

        vmax = np.percentile(np.abs(valid), 95)
        im = ax.imshow(
            delta.T, origin='lower', aspect='auto',
            extent=[-180, 180, -180, 180],
            cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label=label, shrink=0.8)

        # Annotate basin centres
        for bname, (pc, qc) in BASIN_CENTRES.items():
            ax.plot(pc, qc, 'ko', ms=5, zorder=5)
            ax.text(pc + 3, qc + 3, bname, fontsize=7,
                    fontweight='bold', color='black', zorder=6)

        ax.axhline(0, color='gray', lw=0.4, alpha=0.5)
        ax.axvline(0, color='gray', lw=0.4, alpha=0.5)
        ax.set_xlabel('φ (°)'); ax.set_ylabel('ψ (°)')
        ax.set_title(f'{label}', fontsize=11)
        ax.set_xticks(range(-180, 181, 60))
        ax.set_yticks(range(-180, 181, 60))

    plt.tight_layout()
    p = Path(out_dir) / 'ff_correction_map.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Correction map → {p}")

# ── Export lookup table ───────────────────────────────────────────────────────

def export_correction_table(delta_phi, delta_psi, centres, count_map, out_dir):
    """
    CSV lookup table: phi_centre, psi_centre, delta_k_phi, delta_k_psi, n
    MD groups can use this as a grid correction on top of their FF.
    """
    rows = []
    for i, phi_c in enumerate(centres):
        for j, psi_c in enumerate(centres):
            dp = delta_phi[i, j]; dq = delta_psi[i, j]
            n  = int(count_map[i, j])
            if np.isfinite(dp) or np.isfinite(dq):
                rows.append({
                    'phi_centre': round(float(phi_c), 1),
                    'psi_centre': round(float(psi_c), 1),
                    'delta_k_phi': round(float(dp), 4) if np.isfinite(dp) else '',
                    'delta_k_psi': round(float(dq), 4) if np.isfinite(dq) else '',
                    'n_residues': n,
                })

    p = Path(out_dir) / 'ff_correction_table.csv'
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['phi_centre','psi_centre',
                                           'delta_k_phi','delta_k_psi','n_residues'])
        w.writeheader(); w.writerows(rows)
    print(f"  Correction table → {p}  ({len(rows)} bins)")

# ── Main (called from combined_analysis.py after run_spring_analysis) ─────────

def run_ff_correction(spring_meta, out_dir):
    print(f"\n{'='*65}")
    print(f"Application 2: Force-field correction map ...")

    # Build observed maps
    k_phi_obs, k_psi_obs, count_map, centres = load_observed_k_map(spring_meta)

    # Build AMBER maps
    amber_phi_map, _ = build_amber_k_map(AMBER_FF19SB_PHI['general'])
    amber_psi_map, _ = build_amber_k_map(AMBER_FF19SB_PSI['general'])

    # Δk
    delta_phi = compute_delta_k(k_phi_obs, amber_phi_map, count_map)
    delta_psi = compute_delta_k(k_psi_obs, amber_psi_map, count_map)

    # Summary stats
    print(f"\n  Δk_φ: mean={np.nanmean(delta_phi):+.3f}  "
          f"max={np.nanmax(delta_phi):+.3f}  min={np.nanmin(delta_phi):+.3f}")
    print(f"  Δk_ψ: mean={np.nanmean(delta_psi):+.3f}  "
          f"max={np.nanmax(delta_psi):+.3f}  min={np.nanmin(delta_psi):+.3f}")

    # Highlight worst regions
    flat_phi = [(centres[i], centres[j], delta_phi[i,j])
                for i in range(len(centres)) for j in range(len(centres))
                if np.isfinite(delta_phi[i,j])]
    flat_phi.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"\n  Top 5 worst-corrected φ regions:")
    print(f"  {'φ':>6}  {'ψ':>6}  {'Δk_φ':>8}  interpretation")
    for phi_c, psi_c, dk in flat_phi[:5]:
        interp = "FF too soft" if dk > 0 else "FF too stiff"
        print(f"  {phi_c:>6.0f}  {psi_c:>6.0f}  {dk:>+8.3f}  {interp}")

    plot_correction(delta_phi, delta_psi, centres, out_dir)
    export_correction_table(delta_phi, delta_psi, centres, count_map, out_dir)

    return delta_phi, delta_psi, centres

