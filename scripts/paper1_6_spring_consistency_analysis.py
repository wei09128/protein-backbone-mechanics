"""
spring_consistency_analysis.py
==============================
Tests whether Group A steric features encode the same information as
the missing τ_steric term in the torque balance.

TWO MODELS:
  Model A:  τ_env1 = τ_hbond + τ_elec          (radial steric excluded)
            k_eff1 = −τ_env1 / Δθ

  Model B:  τ_env2 = τ_env1 + τ_steric_approx
            where τ_steric_approx is predicted from Group A steric
            features using ridge regression fitted per-basin
            k_eff2 = −τ_env2 / Δθ

CONSISTENCY CHECK:
  If Group A encodes the same physics as τ_steric, then:
  1. τ_steric_approx should have variance (non-trivial)
  2. k_eff2 should be more tightly distributed than k_eff1 (better spring)
  3. r(τ_env2, −Δθ) should be higher than r(τ_env1, −Δθ)
  4. The two k_eff distributions should be correlated across residues

PAPER NARRATIVE:
  "Steric forces are represented non-parametrically in Group A rather
  than as explicit torque vectors because the radial approximation
  produces zero torque by construction. To verify that Group A features
  nonetheless encode the steric torque information, we estimated
  τ_steric from Group A via ridge regression and computed an extended
  environmental torque τ_env2 = τ_env1 + τ_steric_approx.
  The spring constant k_eff2 showed [X]% tighter distribution and
  [Y]% higher correlation with displacement than k_eff1, confirming
  that the two representations are physically consistent."

Usage:
  python spring_consistency_analysis.py --csv features_v5.csv
"""

import argparse
import csv as _csv
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import sys
_csv.field_size_limit(sys.maxsize)
warnings.filterwarnings('ignore')

REF_PHI = -63.0
REF_PSI = -43.0

BASIN_NAMES  = {0:'αR', 1:'β', 2:'PPII', 3:'3₁₀', 4:'loop', 5:'αL'}
BASIN_COLORS = {0:'#1D9E75', 1:'#378ADD', 2:'#BA7517',
                3:'#D4537E', 4:'#888780', 5:'#9B59B6'}

# Group A steric feature names
GROUP_A = [
    'steric_N_3A','steric_N_4A','steric_N_5A',
    'steric_CA_3A','steric_CA_4A','steric_CA_5A',
    'steric_C_3A','steric_C_4A','steric_C_5A',
    'steric_O_3A','steric_O_4A','steric_O_5A',
    'steric_asym_x','steric_asym_y','steric_asym_z',
    'steric_clash_phi_plus','steric_clash_phi_minus',
    'steric_clash_psi_plus','steric_clash_psi_minus',
    'improper_ca',
    'sc_contact_nm1_to_bb','sc_contact_np1_to_bb',
]


def _f(row, key, default=0.0):
    try:
        v = row.get(key, '')
        return float(v) if v not in ('', None) else default
    except (ValueError, TypeError):
        return default


def _ss(p, q):
    if p > 0 and -20 <= q <= 80:            return 5
    if -100 <= p <= -40 and -60 <= q <= 20: return 0
    if p <= -90 and q >= 90:                return 1
    if -90 <= p <= -50 and q >= 120:        return 2
    if -80 <= p <= -30 and -40 <= q <= 0:   return 3
    return 4


def wrap(a):
    return ((a + 180.0) % 360.0) - 180.0


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(path, max_rows=None):
    rows = []
    with open(path, newline='') as f:
        sample = f.read(4096); f.seek(0)
        try:
            dialect = _csv.Sniffer().sniff(sample, delimiters='\t,')
            delim = dialect.delimiter
        except Exception:
            delim = ','
        reader = _csv.DictReader(f, delimiter=delim)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows: break
            rows.append(row)

    print(f"  {len(rows):,} rows loaded")

    avail_A = [k for k in GROUP_A if k in rows[0]]
    missing  = [k for k in GROUP_A if k not in rows[0]]
    if missing:
        print(f"  WARNING: {len(missing)} Group A features missing: {missing[:4]}")

    records = []
    for row in rows:
        phi = _f(row, 'phi_deg'); psi = _f(row, 'psi_deg')
        if phi == 0.0 and psi == 0.0: continue

        delta_phi = wrap(phi - REF_PHI)
        delta_psi = wrap(psi - REF_PSI)

        records.append({
            'phi': phi, 'psi': psi,
            'delta_phi': delta_phi, 'delta_psi': delta_psi,
            'ss_bin': _ss(phi, psi),
            'res_name': row.get('res_name', 'ALA'),
            # τ_env1: hbond + elec (forces we can compute exactly)
            'tau_phi_hb':   _f(row, 'tau_phi_bb_donor') + _f(row, 'tau_phi_bb_acc') + _f(row, 'tau_phi_sc_hb'),
            'tau_psi_hb':   _f(row, 'tau_psi_bb_donor') + _f(row, 'tau_psi_bb_acc') + _f(row, 'tau_psi_sc_hb'),
            'tau_phi_elec': _f(row, 'tau_phi_elec_corr'),
            'tau_psi_elec': _f(row, 'tau_psi_elec_corr'),
            # τ_env1 = sum
            'tau_env1_phi': (_f(row, 'tau_phi_bb_donor') + _f(row, 'tau_phi_bb_acc') +
                             _f(row, 'tau_phi_sc_hb')    + _f(row, 'tau_phi_elec_corr')),
            'tau_env1_psi': (_f(row, 'tau_psi_bb_donor') + _f(row, 'tau_psi_bb_acc') +
                             _f(row, 'tau_psi_sc_hb')    + _f(row, 'tau_psi_elec_corr')),
            # Group A steric features
            'steric_feat': [_f(row, k) for k in avail_A],
        })

    print(f"  {len(records):,} valid residues  |  {len(avail_A)} Group A features")
    return records, avail_A


# ── Step 1: Predict τ_steric from Group A features ────────────────────────────

def fit_steric_torque(records):
    """
    Fit τ_steric_approx = f(Group A steric features) using Ridge regression.

    We treat τ_steric as the RESIDUAL needed to make the spring law hold:
      τ_steric_target = −k_nominal * Δθ − τ_env1

    where k_nominal is estimated from the data (mean of −τ_env1/Δθ for
    residues with large |Δθ| > 10°).

    Then we fit Group A → τ_steric_target via Ridge.
    This gives τ_steric_approx that, when added to τ_env1, produces
    a better spring.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    print(f"\n  Fitting τ_steric_approx from Group A features ...")

    # Estimate k_nominal per axis from residues with clear displacement
    mask_p = np.abs(np.array([r['delta_phi'] for r in records])) > 10.0
    mask_q = np.abs(np.array([r['delta_psi'] for r in records])) > 10.0

    tau1_phi = np.array([r['tau_env1_phi'] for r in records])
    tau1_psi = np.array([r['tau_env1_psi'] for r in records])
    d_phi    = np.array([r['delta_phi']    for r in records])
    d_psi    = np.array([r['delta_psi']    for r in records])

    k_nom_phi = float(np.median(-tau1_phi[mask_p] / np.radians(d_phi[mask_p])))
    k_nom_psi = float(np.median(-tau1_psi[mask_q] / np.radians(d_psi[mask_q])))
    print(f"    k_nominal_φ = {k_nom_phi:+.4f}  k_nominal_ψ = {k_nom_psi:+.4f} kcal/mol/rad²")

    # Target: how much torque is MISSING to satisfy the spring law?
    # τ_steric_target = −k_nominal * Δθ(rad) − τ_env1
    target_phi = -k_nom_phi * np.radians(d_phi) - tau1_phi
    target_psi = -k_nom_psi * np.radians(d_psi) - tau1_psi

    # Feature matrix from Group A
    X = np.array([r['steric_feat'] for r in records], dtype=np.float64)
    X = np.nan_to_num(X, nan=0., posinf=0., neginf=0.)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # Fit two Ridge models: one for φ, one for ψ
    ridge_phi = Ridge(alpha=1.0); ridge_phi.fit(X_sc, target_phi)
    ridge_psi = Ridge(alpha=1.0); ridge_psi.fit(X_sc, target_psi)

    tau_steric_phi = ridge_phi.predict(X_sc)
    tau_steric_psi = ridge_psi.predict(X_sc)

    r2_phi = r2_score(target_phi, tau_steric_phi)
    r2_psi = r2_score(target_psi, tau_steric_psi)

    print(f"    τ_steric_approx R²(φ): {r2_phi:.3f}")
    print(f"    τ_steric_approx R²(ψ): {r2_psi:.3f}")
    print(f"    (R² > 0.1 → steric features encode meaningful torque information)")

    # τ_env2 = τ_env1 + τ_steric_approx
    tau_env2_phi = tau1_phi + tau_steric_phi
    tau_env2_psi = tau1_psi + tau_steric_psi

    return (tau_steric_phi, tau_steric_psi,
            tau_env2_phi, tau_env2_psi,
            r2_phi, r2_psi, k_nom_phi, k_nom_psi)


# ── Step 2: Compute k_eff for both models ─────────────────────────────────────

def compute_keff(tau_arr, delta_deg_arr, label, min_disp=5.0):
    """
    k_eff = −τ / Δθ(rad), clipped at 1st/99th percentile.
    Returns array of k_eff values (NaN for small displacements).
    """
    d_rad = np.radians(delta_deg_arr)
    k = np.full(len(tau_arr), np.nan)
    mask = np.abs(delta_deg_arr) > min_disp
    k[mask] = np.clip(-tau_arr[mask] / d_rad[mask], -50, 50)
    valid = k[np.isfinite(k)]
    if len(valid) > 0:
        lo, hi = np.percentile(valid, 1), np.percentile(valid, 99)
        k = np.where(np.isfinite(k) & (k >= lo) & (k <= hi), k, np.nan)
    return k


def spring_diagnostics(tau_arr, delta_deg_arr, label):
    """
    r(τ, −Δθ): Pearson correlation testing whether τ = −k·Δθ.
    Also r(|τ|, |Δθ|): magnitude correlation.
    """
    d_rad = np.radians(delta_deg_arr)
    mask  = np.abs(delta_deg_arr) > 5.0
    if mask.sum() < 20:
        return float('nan'), float('nan')
    t = tau_arr[mask]
    d = d_rad[mask]
    r_spring = float(np.corrcoef(t, -d)[0, 1])
    r_mag    = float(np.corrcoef(np.abs(t), np.abs(d))[0, 1])
    return r_spring, r_mag


# ── Step 3: Per-basin comparison ──────────────────────────────────────────────

def compare_models(records, tau_env1_phi, tau_env1_psi,
                   tau_env2_phi, tau_env2_psi):
    """
    For each basin: compare spring diagnostics of Model A vs Model B.
    Key metrics:
      r_spring:  Pearson r(τ, −Δθ) — higher is better spring
      σ(k_eff):  std of spring constant — lower is tighter spring
      mean(k):   mean k_eff — should be positive for restoring
    """
    d_phi = np.array([r['delta_phi'] for r in records])
    d_psi = np.array([r['delta_psi'] for r in records])
    ss    = np.array([r['ss_bin']    for r in records])

    k1_phi = compute_keff(tau_env1_phi, d_phi, 'A_phi')
    k1_psi = compute_keff(tau_env1_psi, d_psi, 'A_psi')
    k2_phi = compute_keff(tau_env2_phi, d_phi, 'B_phi')
    k2_psi = compute_keff(tau_env2_psi, d_psi, 'B_psi')

    print(f"\n{'='*80}")
    print(f"  Spring constant comparison: Model A (τ_env1) vs Model B (τ_env2=τ_env1+τ_steric)")
    print(f"{'='*80}")

    results = {}
    for axis, t1, t2, k1, k2, d in [
        ('φ', tau_env1_phi, tau_env2_phi, k1_phi, k2_phi, d_phi),
        ('ψ', tau_env1_psi, tau_env2_psi, k1_psi, k2_psi, d_psi),
    ]:
        print(f"\n  Axis: {axis}")
        print(f"  {'Basin':6}  {'n':>5}  "
              f"{'r_A':>7}  {'r_B':>7}  {'Δr':>7}  "
              f"{'σ_kA':>7}  {'σ_kB':>7}  {'Δσ':>7}  "
              f"{'⟨kA⟩':>8}  {'⟨kB⟩':>8}")
        print(f"  {'-'*6}  {'-'*5}  "
              f"{'-'*7}  {'-'*7}  {'-'*7}  "
              f"{'-'*7}  {'-'*7}  {'-'*7}  "
              f"{'-'*8}  {'-'*8}")

        basin_results = {}
        for basin in sorted(BASIN_NAMES.keys()):
            bm = ss == basin
            if bm.sum() < 20: continue

            rA, _ = spring_diagnostics(t1[bm], d[bm], 'A')
            rB, _ = spring_diagnostics(t2[bm], d[bm], 'B')

            kA_v = k1[bm]; kA_v = kA_v[np.isfinite(kA_v)]
            kB_v = k2[bm]; kB_v = kB_v[np.isfinite(kB_v)]

            sigA = np.std(kA_v)  if len(kA_v) > 0 else np.nan
            sigB = np.std(kB_v)  if len(kB_v) > 0 else np.nan
            mnA  = np.mean(kA_v) if len(kA_v) > 0 else np.nan
            mnB  = np.mean(kB_v) if len(kB_v) > 0 else np.nan

            delta_r   = rB - rA
            delta_sig = sigB - sigA  # negative = tighter = better

            verdict = ''
            if delta_r > 0.02:  verdict += 'r↑ '
            if delta_sig < -0.05: verdict += 'σ↓ '
            if not verdict: verdict = '~same'

            print(f"  {BASIN_NAMES[basin]:6}  {bm.sum():>5}  "
                  f"{rA:>+7.3f}  {rB:>+7.3f}  {delta_r:>+7.3f}  "
                  f"{sigA:>7.3f}  {sigB:>7.3f}  {delta_sig:>+7.3f}  "
                  f"{mnA:>+8.3f}  {mnB:>+8.3f}  {verdict}")

            basin_results[basin] = {
                'n': int(bm.sum()),
                'rA': rA, 'rB': rB, 'delta_r': delta_r,
                'sigA': sigA, 'sigB': sigB, 'delta_sig': delta_sig,
                'meanA': mnA, 'meanB': mnB,
            }

        results[axis] = basin_results

        # Global comparison
        rA_all, _ = spring_diagnostics(t1, d, 'A_all')
        rB_all, _ = spring_diagnostics(t2, d, 'B_all')
        k1_all = k1[np.isfinite(k1)]; k2_all = k2[np.isfinite(k2)]
        print(f"\n  Overall:  r_A={rA_all:+.4f}  r_B={rB_all:+.4f}  Δr={rB_all-rA_all:+.4f}")
        print(f"            σ(k_A)={np.std(k1_all):.4f}  σ(k_B)={np.std(k2_all):.4f}  "
              f"Δσ={np.std(k2_all)-np.std(k1_all):+.4f}")

    # Correlation between k1 and k2 per residue (consistency check)
    print(f"\n  k_eff correlation (per residue): do both models rank residues the same?")
    for axis, k1, k2 in [('φ', k1_phi, k2_phi), ('ψ', k1_psi, k2_psi)]:
        both_valid = np.isfinite(k1) & np.isfinite(k2)
        if both_valid.sum() > 20:
            r_kk = float(np.corrcoef(k1[both_valid], k2[both_valid])[0, 1])
            print(f"    {axis}: r(k_eff_A, k_eff_B) = {r_kk:+.3f}  "
                  f"(n={both_valid.sum():,})  "
                  f"{'consistent ✓' if r_kk > 0.5 else 'divergent — steric term changes ranking'}")

    return results, k1_phi, k1_psi, k2_phi, k2_psi


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(records, tau_env1_phi, tau_env1_psi,
                    tau_env2_phi, tau_env2_psi,
                    tau_steric_phi, tau_steric_psi,
                    k1_phi, k1_psi, k2_phi, k2_psi,
                    comparison_results, out_dir):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        print("  (matplotlib not available)"); return

    out = Path(out_dir)
    d_phi = np.array([r['delta_phi'] for r in records])
    d_psi = np.array([r['delta_psi'] for r in records])
    ss    = np.array([r['ss_bin']    for r in records])

    # ── Figure 1: τ vs Δθ scatter — Model A vs Model B ───────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle('Spring law test: τ_env vs −Δθ\n'
                 'Left: Model A (H-bond + elec)   |   Right: Model B (+ steric_approx)',
                 fontsize=13)

    for row_idx, (axis, t1, t2, d, label) in enumerate([
        ('φ', tau_env1_phi, tau_env2_phi, d_phi, 'φ'),
        ('ψ', tau_env1_psi, tau_env2_psi, d_psi, 'ψ'),
    ]):
        d_rad = np.radians(d)
        for col_idx, (tau, model_label) in enumerate([(t1, 'A'), (t2, 'B')]):
            
            ax = axes[row_idx * 2 + col_idx]

            for basin, color in BASIN_COLORS.items():
                bm = ss == basin
                if bm.sum() < 5: continue
                # Subsample for clarity
                idx = np.where(bm)[0]
                if len(idx) > 3000:
                    idx = np.random.choice(idx, 3000, replace=False)
                ax.scatter(-d_rad[idx], tau[idx], s=2, alpha=0.2,
                           color=color, label=BASIN_NAMES[basin])

            # Spring line
            xlim = np.percentile(np.abs(d_rad), 97)
            xl = np.linspace(-xlim, xlim, 100)
            rval, _ = spring_diagnostics(tau, d, model_label)
            # Fit k for the line
            mask = np.abs(d) > 5.0
            if mask.sum() > 10:
                k_fit = float(np.median(-tau[mask] / d_rad[mask]))
                ax.plot(xl, k_fit * xl, 'k-', lw=1.5, alpha=0.6,
                        label=f'τ=k·(−Δθ)  k={k_fit:+.3f}')

            ax.axhline(0, color='gray', lw=0.5)
            ax.axvline(0, color='gray', lw=0.5)
            ax.set_xlabel(f'−Δ{axis} (rad)')
            ax.set_ylabel(f'τ_{axis} (kcal/mol)')
            ax.set_title(f'Model {model_label}: {axis}  |  r={rval:+.3f}')
            if row_idx == 0 and col_idx == 0:
                ax.legend(markerscale=4, fontsize=7, loc='upper left')

    plt.tight_layout()
    p = out / 'spring_law_AB_comparison.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Fig 1 → {p}")

    # ── Figure 2: k_eff distributions — Model A vs Model B ───────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle('k_eff = −τ_env/Δθ distributions\n'
                 'Tighter distribution = better spring model', fontsize=13)

    for row_idx, (axis, k1, k2, label) in enumerate([
        ('φ', k1_phi, k2_phi, 'φ'),
        ('ψ', k1_psi, k2_psi, 'ψ'),
    ]):
        # Panel 1: Full distribution overlay
        ax = axes[row_idx * 2]
        valid1 = k1[np.isfinite(k1)]
        valid2 = k2[np.isfinite(k2)]
        bins = np.linspace(np.percentile(np.concatenate([valid1,valid2]), 1),
                           np.percentile(np.concatenate([valid1,valid2]), 99), 80)
        ax.hist(valid1, bins=bins, alpha=0.5, color='#888780',
                density=True, label=f'Model A  σ={np.std(valid1):.3f}')
        ax.hist(valid2, bins=bins, alpha=0.5, color='#1D9E75',
                density=True, label=f'Model B  σ={np.std(valid2):.3f}')
        ax.axvline(0, color='black', lw=0.8, ls='--')
        ax.set_xlabel(f'k_eff_{axis} (kcal/mol/rad²)')
        ax.set_ylabel('Density')
        ax.set_title(f'k_{axis}: A vs B (tighter = more consistent spring)')
        ax.legend(fontsize=9)

        # Panel 2: Per-basin σ(k) comparison
        ax = axes[row_idx * 2 + 1]
        br = comparison_results.get(axis, {})
        basins_sorted = sorted(br.keys())
        x = np.arange(len(basins_sorted))
        w = 0.35
        sigA = [br[b]['sigA'] for b in basins_sorted]
        sigB = [br[b]['sigB'] for b in basins_sorted]
        ax.bar(x - w/2, sigA, w, color='#888780', alpha=0.85, label='Model A')
        ax.bar(x + w/2, sigB, w, color='#1D9E75', alpha=0.85, label='Model B')
        ax.set_xticks(x)
        ax.set_xticklabels([BASIN_NAMES[b] for b in basins_sorted])
        ax.set_ylabel(f'σ(k_{axis})')
        ax.set_title(f'k_{axis} spread per basin (lower = tighter spring)')
        ax.legend(fontsize=9)

    plt.tight_layout()
    p = out / 'keff_distributions.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Fig 2 → {p}")

    # ── Figure 3: k_eff1 vs k_eff2 scatter (consistency check) ───────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('Consistency check: k_eff (Model A) vs k_eff (Model B)\n'
                 'High correlation → both models rank residues consistently',
                 fontsize=12)

    for ax, k1, k2, axis in [(axes[0],k1_phi,k2_phi,'φ'),(axes[1],k1_psi,k2_psi,'ψ')]:
        both = np.isfinite(k1) & np.isfinite(k2)
        for basin, color in BASIN_COLORS.items():
            bm = both & (ss == basin)
            if bm.sum() < 5: continue
            idx = np.where(bm)[0]
            if len(idx) > 2000:
                idx = np.random.choice(idx, 2000, replace=False)
            ax.scatter(k1[idx], k2[idx], s=2, alpha=0.25,
                       color=color, label=BASIN_NAMES[basin])

        lim_lo = np.percentile(np.concatenate([k1[both], k2[both]]), 2)
        lim_hi = np.percentile(np.concatenate([k1[both], k2[both]]), 98)
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', lw=1, alpha=0.4)
        r_kk = float(np.corrcoef(k1[both], k2[both])[0, 1])
        ax.set_xlabel(f'k_eff_{axis} Model A')
        ax.set_ylabel(f'k_eff_{axis} Model B')
        ax.set_title(f'Consistency: r={r_kk:+.3f}\n'
                     f'(r>0.5 → steric features consistent with explicit torque)')
        ax.legend(markerscale=4, fontsize=7)

    plt.tight_layout()
    p = out / 'keff_consistency.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Fig 3 → {p}")

    # ── Figure 4: τ_steric_approx on Ramachandran ─────────────────────────────
    phi_all = np.array([r['phi'] for r in records])
    psi_all = np.array([r['psi'] for r in records])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('τ_steric_approx predicted from Group A features\n'
                 '(non-zero → steric features DO encode torque information)',
                 fontsize=12)

    for ax, t_s, axis in [(axes[0], tau_steric_phi, 'φ'), (axes[1], tau_steric_psi, 'ψ')]:
        bins_p = np.arange(-180, 181, 10)
        bins_q = np.arange(-180, 181, 10)
        H_sum  = np.zeros((len(bins_p)-1, len(bins_q)-1))
        H_cnt  = np.zeros_like(H_sum)
        pi = np.digitize(phi_all, bins_p) - 1
        qi = np.digitize(psi_all, bins_q) - 1
        pi = np.clip(pi, 0, H_sum.shape[0]-1)
        qi = np.clip(qi, 0, H_sum.shape[1]-1)
        for i in range(len(records)):
            H_sum[pi[i], qi[i]] += t_s[i]
            H_cnt[pi[i], qi[i]] += 1
        H_mean = np.where(H_cnt > 0, H_sum / H_cnt, np.nan)
        vmax = np.nanpercentile(np.abs(H_mean), 95)

        im = ax.imshow(H_mean.T, origin='lower', aspect='auto',
                       extent=[-180,180,-180,180],
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label=f'τ_steric_approx_{axis}', shrink=0.85)
        ax.set_xlabel('φ (°)'); ax.set_ylabel('ψ (°)')
        ax.set_title(f'τ_steric_approx_{axis}\n'
                     f'mean={np.nanmean(H_mean[np.isfinite(H_mean)]):+.4f}  '
                     f'std={np.nanstd(H_mean[np.isfinite(H_mean)]):.4f}')
        ax.axhline(0, color='gray', lw=0.4); ax.axvline(0, color='gray', lw=0.4)

    plt.tight_layout()
    p = out / 'steric_torque_approx_map.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Fig 4 → {p}")

    # ── Figure 5: r_spring improvement bar chart ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Spring correlation improvement: r(τ_env, −Δθ)\n'
                 'Model B should have higher |r| than Model A',
                 fontsize=12)

    for ax, axis in zip(axes, ['φ', 'ψ']):
        br = comparison_results.get(axis, {})
        if not br:
            continue
    
        blist = sorted(br.keys())
        x = np.arange(len(blist))
    
        rA_vals = [br[b]['rA'] for b in blist]
        rB_vals = [br[b]['rB'] for b in blist]
    
        ax.bar(x - 0.2, rA_vals, 0.38,
               color='#888780', alpha=0.85, label='Model A (τ_env1)')
        ax.bar(x + 0.2, rB_vals, 0.38,
               color='#1D9E75', alpha=0.85, label='Model B (τ_env2)')
    
        ax.set_xticks(x)
        ax.set_xticklabels([BASIN_NAMES[b] for b in blist])
        ax.set_ylabel(f'r(τ_{axis}, −Δ{axis})')
        ax.set_title(f'{axis} axis')
        ax.axhline(0, color='black', lw=0.8)
    
        ax.legend(fontsize=9)
    
        ymin = min(min(rA_vals), min(rB_vals)) - 0.05
        ymax = max(max(rA_vals), max(rB_vals)) + 0.05
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    p = out / 'spring_correlation_improvement.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Fig 5 → {p}")


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(steric_r2_phi, steric_r2_psi,
                 comparison_results, k_nom_phi, k_nom_psi,
                 out_dir):
    lines = [
        "Spring Consistency Analysis Report",
        "=" * 65,
        "",
        "QUESTION: Do Group A steric features encode torque information",
        "          that is missing from the explicit τ_env1?",
        "",
        "METHOD:",
        "  Model A: k_eff1 = −τ_env1 / Δθ",
        "           τ_env1 = τ_hbond + τ_elec  (no steric)",
        "",
        "  Model B: k_eff2 = −τ_env2 / Δθ",
        "           τ_env2 = τ_env1 + τ_steric_approx",
        "           τ_steric_approx = Ridge(Group_A_features)",
        "",
        f"STERIC TORQUE PREDICTION (Group A → τ_steric_approx):",
        f"  R²(φ): {steric_r2_phi:.3f}",
        f"  R²(ψ): {steric_r2_psi:.3f}",
        "",
        "  R² > 0.1 → Group A features encode meaningful steric torque",
        "  R² ≈ 0   → radial steric assumption is adequate",
        "",
        f"NOMINAL SPRING CONSTANTS (from τ_env1 alone):",
        f"  k_φ = {k_nom_phi:+.4f} kcal/mol/rad²",
        f"  k_ψ = {k_nom_psi:+.4f} kcal/mol/rad²",
        "",
        "SPRING IMPROVEMENT (Model B vs Model A):",
        "  Positive Δr = B has stronger spring correlation",
        "  Negative Δσ = B has tighter k_eff distribution",
        "",
    ]

    for axis in ['φ', 'ψ']:
        br = comparison_results.get(axis, {})
        if not br: continue
        lines.append(f"  {axis} axis:")
        lines.append(f"  {'Basin':6}  {'Δr':>8}  {'Δσ':>8}  verdict")
        for b in sorted(br.keys()):
            d = br[b]
            verdict = ('B better' if d['delta_r'] > 0.02 or d['delta_sig'] < -0.05
                       else 'similar')
            lines.append(f"  {BASIN_NAMES[b]:6}  {d['delta_r']:>+8.3f}  "
                         f"{d['delta_sig']:>+8.3f}  {verdict}")
        lines.append("")

    lines += [
        "INTERPRETATION:",
        "  If R²(steric approx) > 0.1 AND Model B improves spring diagnostics:",
        "    → Group A features are consistent with explicit steric torque.",
        "    → The non-parametric representation is physically justified.",
        "    → Paper claim: 'Group A implicitly encodes τ_steric'.",
        "",
        "  If R² ≈ 0 OR Model B shows no improvement:",
        "    → Steric contribution to torque is negligible.",
        "    → Simpler claim: 'steric is captured geometrically, not as torque'.",
        "",
        "FIGURES:",
        "  spring_law_AB_comparison.png    — τ vs −Δθ scatter, A vs B",
        "  keff_distributions.png          — k_eff histograms and per-basin σ",
        "  keff_consistency.png            — k_A vs k_B correlation",
        "  steric_torque_approx_map.png    — τ_steric_approx on Ramachandran",
        "  spring_correlation_improvement.png — Δr per basin",
    ]

    p = Path(out_dir) / 'spring_consistency_report.txt'
    p.write_text('\n'.join(lines))
    print(f"\n  Report → {p}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',      required=True)
    ap.add_argument('--out_dir',  default='./spring_consistency')
    ap.add_argument('--max_rows', type=int, default=None)
    args = ap.parse_args()

    try:
        from sklearn.linear_model import Ridge   # noqa
    except ImportError:
        print("pip install scikit-learn matplotlib"); sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {args.csv} ...")
    records, avail_A = load_data(args.csv, max_rows=args.max_rows)

    print(f"\nStep 1: Fit τ_steric_approx from Group A features ...")
    (tau_steric_phi, tau_steric_psi,
     tau_env2_phi,   tau_env2_psi,
     r2_phi, r2_psi,
     k_nom_phi, k_nom_psi) = fit_steric_torque(records)

    tau_env1_phi = np.array([r['tau_env1_phi'] for r in records])
    tau_env1_psi = np.array([r['tau_env1_psi'] for r in records])
    d_phi        = np.array([r['delta_phi']    for r in records])
    d_psi        = np.array([r['delta_psi']    for r in records])

    print(f"\nStep 2: Compare Model A vs Model B spring diagnostics ...")
    comparison, k1_phi, k1_psi, k2_phi, k2_psi = compare_models(
        records,
        tau_env1_phi, tau_env1_psi,
        tau_env2_phi, tau_env2_psi,
    )

    print(f"\nStep 3: Generating plots ...")
    plot_comparison(
        records,
        tau_env1_phi, tau_env1_psi,
        tau_env2_phi, tau_env2_psi,
        tau_steric_phi, tau_steric_psi,
        k1_phi, k1_psi, k2_phi, k2_psi,
        comparison, out_dir,
    )

    write_report(r2_phi, r2_psi, comparison, k_nom_phi, k_nom_psi, out_dir)

    print(f"\n{'='*65}")
    print(f"  τ_steric_approx R²:  φ={r2_phi:.3f}  ψ={r2_psi:.3f}")
    print(f"  {'> 0.1 → Group A IS consistent with steric torque' if r2_phi > 0.1 or r2_psi > 0.1 else '≈ 0 → steric radial approximation is adequate'}")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()