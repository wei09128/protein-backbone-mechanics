"""
ff_curvature_split_panels.py
=============================
Splits the 4-panel force-field correction figure into 4 standalone PNG files.

Outputs (default to ./ff_panels/):
  panel_A_dk_phi_heatmap.png    — δk_φ Ramachandran heatmap (your strongest panel)
  panel_B_psi_scan.png          — ψ-scan at φ = -65° (PDB vs AMBER vs corrected)
  panel_C_alpha_corridor.png    — α-helix corridor: φ stiffness vs ψ
  panel_D_improvement_hist.png  — per-bin improvement distribution

NOTES ON PANEL VALIDITY:
  - Panels A and C plot the curvature correction directly. These are honest.
  - Panels B and D reconstruct energies from curvatures via cumulative
    integration, which introduces gauge errors. Use these with caution
    or replace with a proper fit-based comparison.

USAGE:
  python ff_curvature_split_panels.py --csv ff_correction_table.csv
  python ff_curvature_split_panels.py --csv ff_correction_table.csv --out_dir ./panels
  python ff_curvature_split_panels.py --csv ff_correction_table.csv --skip_bd
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import CubicSpline

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# ── Color scheme ──────────────────────────────────────────────────────────────
QM_BLUE    = "#4EADCF"   # PDB reference
AMBER_RED  = "#E05555"   # AMBER ff14SB
THIS_GREEN = "#5EC88B"   # this work / corrected

# ── AMBER ff14SB torsional terms ──────────────────────────────────────────────
AMBER_PHI = [(0.80, 1, 0.0), (2.00, 2, np.pi), (0.20, 3, 0.0), (0.10, 4, np.pi)]
AMBER_PSI = [(0.85, 1, 0.0), (0.80, 2, np.pi), (0.55, 3, np.pi), (0.15, 4, 0.0)]


def amber_E(theta_deg, terms):
    t = np.deg2rad(np.asarray(theta_deg, dtype=float))
    return sum(v * (1 + np.cos(n * t - g)) for v, n, g in terms)


def amber_k(theta_deg, terms):
    t = np.deg2rad(np.asarray(theta_deg, dtype=float))
    return sum(-n**2 * v * np.cos(n * t - g) for v, n, g in terms)


def load_data(csv_path):
    """Load correction table and compute derived quantities."""
    df = pd.read_csv(csv_path)
    total = df["n_residues"].sum()

    # PMF from populations (empirical reference)
    P = (df["n_residues"].values + 0.5) / (total + 0.5 * len(df))
    pmf = -0.593 * np.log(P)
    df["pmf"] = pmf - pmf.min()

    # AMBER separable energy surface
    df["V_amber_phi"] = amber_E(df["phi_centre"].values, AMBER_PHI)
    df["V_amber_psi"] = amber_E(df["psi_centre"].values, AMBER_PSI)
    df["V_amber"] = df["V_amber_phi"] + df["V_amber_psi"]
    df["V_amber"] -= df["V_amber"].min()

    # AMBER curvature (separable, ψ-independent for φ and vice versa)
    df["k_amber_phi"] = amber_k(df["phi_centre"].values, AMBER_PHI)
    df["k_amber_psi"] = amber_k(df["psi_centre"].values, AMBER_PSI)

    # Corrected curvature
    df["k_corr_phi"] = df["k_amber_phi"] + df["delta_k_phi"]
    df["k_corr_psi"] = df["k_amber_psi"] + df["delta_k_psi"]

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PANEL A: δk_φ heatmap
# ══════════════════════════════════════════════════════════════════════════════
def make_panel_A(df, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    piv = df.pivot_table(index="psi_centre", columns="phi_centre",
                          values="delta_k_phi", aggfunc="mean")
    vmax = np.nanmax(np.abs(piv.values))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.pcolormesh(piv.columns, piv.index, piv.values,
                        cmap="RdBu_r", norm=norm, shading="nearest")
    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("δk_φ  (kcal mol⁻¹ rad⁻²)", fontsize=10)
    cb.ax.tick_params(labelsize=8)

    ax.set_xlabel("φ (°)", fontsize=11)
    ax.set_ylabel("ψ (°)", fontsize=11)
    ax.set_title("Stiffness correction δk_φ across Ramachandran space",
                  fontsize=11, fontweight="bold", pad=10)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=9)

    # Mark canonical basins
    ax.plot(-63, -43, '*', color="white", ms=15, markeredgecolor="k",
            markeredgewidth=0.8, zorder=5)
    ax.plot(-135, 135, 's', color="white", ms=9, markeredgecolor="k",
            markeredgewidth=0.8, zorder=5)
    ax.annotate("α-helix", (-63, -43), (-25, -10), fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="k"))
    ax.annotate("β-sheet", (-135, 135), (-95, 165), fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="k"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL B: ψ scan at φ = -65°
# ══════════════════════════════════════════════════════════════════════════════
def make_panel_B(df, out_path, phi_slice=-65):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    sub = df[df["phi_centre"] == phi_slice].sort_values("psi_centre").copy()

    if len(sub) < 5:
        print(f"  Panel B: not enough data at φ = {phi_slice}°, skipping")
        plt.close(fig)
        return

    psi = sub["psi_centre"].values
    pmf = sub["pmf"].values.copy()
    v_amb = sub["V_amber"].values.copy()

    # Re-zero
    pmf -= pmf.min()
    v_amb -= v_amb.min()

    # Integrate δk_ψ twice to get δV (gauge-fixed by removing linear trend)
    # WARNING: this is an approximation; gauge errors may dominate the signal
    dk_psi = sub["delta_k_psi"].values
    step_rad = np.deg2rad(10.0)
    dk_clean = np.nan_to_num(dk_psi, nan=0.0)
    dv_prime = np.cumsum(dk_clean) * step_rad
    dv = np.cumsum(dv_prime) * step_rad
    x = np.arange(len(dv))
    if len(x) > 1:
        slope = (dv[-1] - dv[0]) / (x[-1] - x[0])
        dv -= slope * x + dv[0]

    v_corr = v_amb + dv
    v_corr -= v_corr.min()

    # Smooth via cubic spline
    psi_fine = np.linspace(psi.min(), psi.max(), 300)
    cs_pmf  = CubicSpline(psi, pmf)
    cs_amb  = CubicSpline(psi, v_amb)
    cs_corr = CubicSpline(psi, v_corr)

    ax.plot(psi_fine, cs_pmf(psi_fine), color=QM_BLUE, lw=2.5,
            label="PDB reference", zorder=3)
    ax.plot(psi_fine, cs_amb(psi_fine), color=AMBER_RED, lw=2.0, ls="--",
            label="AMBER ff14SB", zorder=2)
    ax.plot(psi_fine, cs_corr(psi_fine), color=THIS_GREEN, lw=2.2,
            label="This work", zorder=3)

    # RMSE annotations
    common = np.linspace(psi.min(), psi.max(), 200)
    rmse_amb  = float(np.sqrt(np.mean((cs_amb(common)  - cs_pmf(common))**2)))
    rmse_corr = float(np.sqrt(np.mean((cs_corr(common) - cs_pmf(common))**2)))

    ax.text(0.97, 0.97,
            f"RMSE vs reference\n"
            f"AMBER:     {rmse_amb:.2f} kcal/mol\n"
            f"This work: {rmse_corr:.2f} kcal/mol",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9))

    ax.set_xlabel("ψ (°)", fontsize=11)
    ax.set_ylabel("Relative energy (kcal mol⁻¹)", fontsize=11)
    ax.set_title(f"Ala dipeptide — ψ scan at φ = {phi_slice}°",
                  fontsize=11, fontweight="bold", pad=10)
    ax.legend(fontsize=9, loc="upper left", frameon=True,
              framealpha=0.9, edgecolor="#ccc")
    ax.tick_params(labelsize=9)
    ax.set_ylim(-0.3, None)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL C: α-helix corridor (φ stiffness depends on ψ)
# ══════════════════════════════════════════════════════════════════════════════
def make_panel_C(df, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for phi_val, ls, alpha_val in [(-65, "-", 0.9), (-55, "--", 0.6)]:
        sub = df[df["phi_centre"] == phi_val].sort_values("psi_centre")
        if len(sub) < 5:
            continue
        psi = sub["psi_centre"].values

        # AMBER: constant (flat line)
        k_amb = amber_k(phi_val, AMBER_PHI)
        ax.axhline(k_amb, color=AMBER_RED, lw=1.5, ls=ls, alpha=alpha_val * 0.7)

        # Corrected: varies with ψ
        ax.plot(psi, sub["k_corr_phi"].values, color=THIS_GREEN, lw=2.2, ls=ls,
                alpha=alpha_val, label=f"Corrected (φ={phi_val}°)")

        if phi_val == -65:
            ax.fill_between(psi, k_amb, sub["k_corr_phi"].values,
                             color=THIS_GREEN, alpha=0.10)

    ax.axhline(0, color="#bbb", lw=0.5)

    # Annotate AMBER flat line
    k_amb_65 = amber_k(-65, AMBER_PHI)
    ax.annotate("AMBER ff14SB\n(no ψ dependence)",
                xy=(80, k_amb_65), xytext=(100, k_amb_65 + 4),
                fontsize=9, color=AMBER_RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=AMBER_RED, lw=0.9))

    ax.set_xlabel("ψ (°)", fontsize=11)
    ax.set_ylabel("k_φ (kcal mol⁻¹ rad⁻²)", fontsize=11)
    ax.set_title("α-helix corridor: φ stiffness depends on ψ context",
                  fontsize=11, fontweight="bold", pad=10)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9, edgecolor="#ccc")
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL D: per-bin improvement histogram
# ══════════════════════════════════════════════════════════════════════════════
def make_panel_D(df, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    df_e = df.copy()
    df_e["V_amber"] -= df_e["V_amber"].min()
    df_e["pmf"] -= df_e["pmf"].min()

    res_amber = np.abs(df_e["V_amber"] - df_e["pmf"]).values

    # Build corrected V via per-φ-slice double integration of δk_ψ
    # WARNING: gauge errors from this integration can dominate the signal
    v_corr_all = np.full(len(df_e), np.nan)
    for phi_val in df_e["phi_centre"].unique():
        mask = df_e["phi_centre"] == phi_val
        sub = df_e[mask].sort_values("psi_centre")
        idx = sub.index

        if len(sub) < 3:
            v_corr_all[idx] = sub["V_amber"].values
            continue

        dk = sub["delta_k_psi"].values
        v_a = sub["V_amber"].values
        step_rad = np.deg2rad(10.0)

        dk_clean = np.nan_to_num(dk, nan=0.0)
        dv_p = np.cumsum(dk_clean) * step_rad
        dv = np.cumsum(dv_p) * step_rad
        x = np.arange(len(dv))
        if len(x) > 1:
            slope = (dv[-1] - dv[0]) / (x[-1] - x[0])
            dv -= slope * x + dv[0]

        v_corr_all[idx] = v_a + dv

    df_e["V_corr"] = v_corr_all
    df_e["V_corr"] -= np.nanmin(df_e["V_corr"])
    res_corr = np.abs(df_e["V_corr"] - df_e["pmf"]).values

    improvement = res_amber - res_corr
    n_better  = int(np.sum(improvement > 0.05))
    n_worse   = int(np.sum(improvement < -0.05))
    n_neutral = int(np.sum(np.abs(improvement) <= 0.05))

    bins = np.linspace(-3, 3, 61)
    ax.hist(improvement, bins=bins, color=THIS_GREEN, alpha=0.6,
            edgecolor="white", linewidth=0.3, label="Per-bin improvement")
    ax.axvline(0, color="k", lw=1.0, ls="-")
    ax.axvline(np.median(improvement), color=THIS_GREEN, lw=1.5, ls="--",
               label=f"Median = {np.median(improvement):.2f}")

    ax.axvspan(0.05, 3, alpha=0.05, color=THIS_GREEN)
    ax.axvspan(-3, -0.05, alpha=0.05, color=AMBER_RED)
    yt = ax.get_ylim()[1]
    ax.text(1.5, yt * 0.90, "Correction\nbetter", fontsize=9,
            ha="center", color=THIS_GREEN, fontweight="bold")
    ax.text(-1.5, yt * 0.90, "AMBER\nbetter", fontsize=9,
            ha="center", color=AMBER_RED, fontweight="bold")

    ax.set_xlabel("|AMBER residual| − |corrected residual|  (kcal mol⁻¹)",
                  fontsize=11)
    ax.set_ylabel("Number of (φ,ψ) bins", fontsize=11)
    ax.set_title("Per-bin improvement distribution",
                  fontsize=11, fontweight="bold", pad=10)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="#ccc", loc="upper left")
    ax.tick_params(labelsize=9)

    pct_better = n_better / len(improvement) * 100
    ax.text(0.97, 0.70,
            f"Improved: {n_better} ({pct_better:.0f}%)\n"
            f"Neutral:  {n_neutral}\n"
            f"Worse:    {n_worse}",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9))

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to ff_correction_table.csv")
    ap.add_argument("--out_dir", default="./ff_panels",
                    help="Output directory for the four PNGs")
    ap.add_argument("--skip_bd", action="store_true",
                    help="Skip panels B and D (recommended — see code notes)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.csv} ...")
    df = load_data(args.csv)
    print(f"  {len(df):,} bins, {df['n_residues'].sum():,} residues\n")

    print("Generating panels:")

    # Panel A — always make this one (it's solid)
    make_panel_A(df, out_dir / "panel_A_dk_phi_heatmap.png")

    # Panel C — always make this one (it's solid)
    make_panel_C(df, out_dir / "panel_C_alpha_corridor.png")

    if not args.skip_bd:
        make_panel_B(df, out_dir / "panel_B_psi_scan.png")
        make_panel_D(df, out_dir / "panel_D_improvement_hist.png")
        print("\nNote: Panels B and D depend on cumulative integration of δk")
        print("      to reconstruct energies, which introduces gauge errors.")
        print("      Use --skip_bd to omit them, or replace with a fit-based")
        print("      comparison for a more rigorous test.")
    else:
        print("\nPanels B and D skipped (--skip_bd).")
        print("Recommended: use only Panels A and C in the manuscript.")

    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()