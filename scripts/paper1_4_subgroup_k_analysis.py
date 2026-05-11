"""
combined_analysis.py — Unified Ramachandran coverage + force-basin analysis
============================================================================
Combines: analyse_backbone_coverage.py, force_basin_analysis.py,
          nonlinear_analysis.py into one pipeline.

Outputs (all plots + reports):
  Fig 1: rama_coverage.png         — 6-state vs +8 new states (heatmap)
  Fig 2: torque_vectors.png        — Net torque arrows per basin (3 panels)
           Panel A: raw net torque (near-zero diagnostic)
           Panel B: restoring projection (toward/away from αR)
           Panel C: per-source decomposition (which forces dominate)
  Fig 3: r2_comparison.png         — Bar chart: linear vs RF vs MLP
  Fig 4: basin_accuracy.png        — Per-basin accuracy all models
  Fig 5: feature_importance.png    — RF importances for φ and ψ
  Fig 6: residuals.png             — Error distribution per basin
  Fig 7: corrected_rama_coverage.png — Actual vs force-predicted (φ,ψ)
  report.txt                       — Full text summary

Usage:
  python combined_analysis.py --csv features_v3.csv
  python combined_analysis.py --csv features_v3.csv --out_dir ./results --target 0.95
"""

import argparse
import csv as _csv_mod
import sys
import warnings
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
_csv_mod.field_size_limit(sys.maxsize)
warnings.filterwarnings('ignore')

# ── References ────────────────────────────────────────────────────────────────
REF_PHI = -63.0
REF_PSI = -43.0

# 6 basins (v3: includes αL as basin 5)
BASIN_NAMES   = {0:'αR', 1:'β', 2:'PPII', 3:'3₁₀', 4:'loop', 5:'αL'}
BASIN_CENTRES = {0:(-63,-43), 1:(-120,128), 2:(-72,146),
                 3:(-52,-32), 4:(-95,10),   5:(60,40)}
BASIN_COLORS  = {0:'#1D9E75', 1:'#378ADD', 2:'#BA7517',
                 3:'#D4537E', 4:'#888780', 5:'#9B59B6'}

# Current 6-state grid from quantum_hamiltonian.py
CURRENT_STATES = [
    ('αR',   -63,  -43,  25, 22),
    ('β',   -120,  130,  28, 25),
    ('3₁₀',  -49,  -26,  20, 18),
    ('PPII', -75,  145,  22, 20),
    ('π',    -57,  -70,  18, 16),
    ('αL',    60,   40,  22, 20),
]
COVERAGE_SIGMA = 1.5


def wrap(a):
    return ((a + 180.0) % 360.0) - 180.0


def _ss_bin(phi_deg, psi_deg):
    """Assign basin — now includes αL as basin 5."""
    p, q = phi_deg, psi_deg
    if p > 0 and -20 <= q <= 80:          return 5  # αL (GLY-dominated)
    if -100 <= p <= -40 and -60 <= q <= 20: return 0  # αR
    if p <= -90 and q >= 90:                return 1  # β
    if -90 <= p <= -50 and q >= 120:        return 2  # PPII
    if -80 <= p <= -30 and -40 <= q <= 0:   return 3  # 3₁₀
    return 4                                           # loop


# ══════════════════════════════════════════════════════════════════════════════
# PART 0: Data loading
# ══════════════════════════════════════════════════════════════════════════════

def _f(row, key, default=0.0):
    try:
        v = row.get(key, '')
        return float(v) if v not in ('', None) else default
    except (ValueError, TypeError):
        return default


# Feature keys — matches collect_backbone_features_v5.py output
# GROUP A: Steric hindrance field
# GROUP B: Five forces (3 active + 2 always-zero)
# GROUP C: Minimal context — NO leaky features
FEAT_KEYS = [
    # GROUP A: Steric field (22)
    'steric_N_3A','steric_N_4A','steric_N_5A',
    'steric_CA_3A','steric_CA_4A','steric_CA_5A',
    'steric_C_3A','steric_C_4A','steric_C_5A',
    'steric_O_3A','steric_O_4A','steric_O_5A',
    'steric_asym_x','steric_asym_y','steric_asym_z',
    'steric_clash_phi_plus','steric_clash_phi_minus',
    'steric_clash_psi_plus','steric_clash_psi_minus',
    'improper_ca',
    'sc_contact_nm1_to_bb','sc_contact_np1_to_bb',
    # GROUP B: Forces (14)
    'tau_phi_correct','tau_psi_correct',
    'tau_phi_bb_donor','tau_psi_bb_donor',
    'tau_phi_bb_acc','tau_psi_bb_acc',
    'tau_phi_sc_hb','tau_psi_sc_hb',
    'tau_phi_steric','tau_psi_steric',
    'tau_phi_elec_corr','tau_psi_elec_corr',
    'chi1_rad','has_chi1',
    # GROUP C: Context (20)
    'chi2_rad','has_chi2',
    'sc_mass','sc_n_heavy','sc_n_rotatable','sc_rigidity',
    'sc_is_branched','sc_is_aromatic','sc_lever_arm',
    'hb_n_strong','hb_best_e',
    'bfactor_ca','is_pro_np1',
    'angle_NCaC','angle_CaCN','angle_CNCa',
    'dist_ca_m2','dist_ca_p2',
    'sc_mass_nm1','sc_mass_np1',
]

# Indices of just the 2 net-torque features within FEAT_KEYS
_TAU_NET_INDICES = [0, 1]  # tau_phi_correct, tau_psi_correct


def load_data(csv_path, max_rows=None):
    """Load CSV, return records list and feature matrices."""
    rows = []
    with open(csv_path, newline='') as f:
        sample = f.read(4096); f.seek(0)
        try:
            dialect = _csv_mod.Sniffer().sniff(sample, delimiters='\t,')
            delim = dialect.delimiter
        except Exception:
            delim = ','
        reader = _csv_mod.DictReader(f, delimiter=delim)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows: break
            rows.append(row)

    print(f"  {len(rows):,} CSV rows loaded")

    # Auto-detect which features are available in the CSV
    available_keys = [k for k in FEAT_KEYS if k in rows[0]]
    missing_keys   = [k for k in FEAT_KEYS if k not in rows[0]]
    if missing_keys:
        print(f"  NOTE: {len(missing_keys)} features not in CSV (v2 data?): "
              f"{missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        print(f"  Using {len(available_keys)} available features")

    records = []
    for row in rows:
        phi = _f(row, 'phi_deg'); psi = _f(row, 'psi_deg')
        if abs(phi) < 0.5 or abs(psi) < 0.5: continue
        omega = _f(row, 'omega_deg', 180.0)
        phi_rad = np.radians(phi); psi_rad = np.radians(psi); om_rad = np.radians(omega)
        feat = [_f(row, k) for k in available_keys]
        records.append({
            'phi': phi, 'psi': psi, 'omega': omega,
            'delta_phi': wrap(phi - REF_PHI),
            'delta_psi': wrap(psi - REF_PSI),
            'sin_phi': np.sin(phi_rad), 'cos_phi': np.cos(phi_rad),
            'sin_psi': np.sin(psi_rad), 'cos_psi': np.cos(psi_rad),
            'sin_omega': np.sin(om_rad), 'cos_omega': np.cos(om_rad),
            'ss_bin':    _ss_bin(phi, psi),
            'res_name':  row.get('res_name', 'ALA'),
            'tau_phi':   _f(row, 'tau_phi_correct'),
            'tau_psi':   _f(row, 'tau_psi_correct'),
            'feat':      feat,
        })

    X  = np.array([r['feat'] for r in records], dtype=np.float64)
    X  = np.nan_to_num(X, nan=0., posinf=0., neginf=0.)
    yp = np.array([r['delta_phi'] for r in records])
    yq = np.array([r['delta_psi'] for r in records])
    ss = np.array([r['ss_bin']    for r in records], dtype=np.int32)

    # Sin/cos targets (reference-free, includes ω)
    y_sincos = {
        'sin_phi': np.array([r['sin_phi'] for r in records]),
        'cos_phi': np.array([r['cos_phi'] for r in records]),
        'sin_psi': np.array([r['sin_psi'] for r in records]),
        'cos_psi': np.array([r['cos_psi'] for r in records]),
        'sin_omega': np.array([r['sin_omega'] for r in records]),
        'cos_omega': np.array([r['cos_omega'] for r in records]),
    }

    print(f"  {len(records):,} residues parsed, feature matrix: {X.shape}")

    # Also build the 2-feature matrix (just net torques)
    X_tau = np.column_stack([
        np.array([r['tau_phi'] for r in records]),
        np.array([r['tau_psi'] for r in records]),
    ])
    X_tau = np.nan_to_num(X_tau, nan=0., posinf=0., neginf=0.)

    return records, X, X_tau, yp, yq, y_sincos, ss, available_keys


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Ramachandran coverage (6-state → +8 new states)
# ══════════════════════════════════════════════════════════════════════════════

def _is_covered(phi, psi, states, threshold=COVERAGE_SIGMA):
    for name, phi_c, psi_c, s_phi, s_psi in states:
        dp = wrap(phi - phi_c) / s_phi
        dq = wrap(psi - psi_c) / s_psi
        if np.sqrt(dp*dp + dq*dq) <= threshold:
            return True
    return False


def find_new_states(records, target=0.95, max_new=8, bin_size=5, min_pop=0.0002):
    """Greedy: find new backbone states that maximally improve coverage."""
    from scipy.ndimage import uniform_filter

    total = len(records)
    phi_arr = np.array([r['phi'] for r in records])
    psi_arr = np.array([r['psi'] for r in records])

    def residue_coverage(states):
        return sum(1 for p, q in zip(phi_arr, psi_arr)
                   if _is_covered(p, q, states)) / total

    phi_edges = np.arange(-180, 181, bin_size)
    psi_edges = np.arange(-180, 181, bin_size)
    H, _, _ = np.histogram2d(phi_arr, psi_arr, bins=[phi_edges, psi_edges])
    H_smooth = uniform_filter(H.astype(float), size=3)
    phi_c = (phi_edges[:-1] + phi_edges[1:]) / 2
    psi_c = (psi_edges[:-1] + psi_edges[1:]) / 2

    all_states  = list(CURRENT_STATES)
    initial_cov = residue_coverage(all_states)
    print(f"  Initial coverage: {initial_cov:.1%} ({int(initial_cov*total):,}/{total:,})")

    # Bin-level covered mask
    covered_mask = np.zeros_like(H, dtype=bool)
    for i, p in enumerate(phi_c):
        for j, q in enumerate(psi_c):
            if _is_covered(p, q, all_states):
                covered_mask[i, j] = True

    min_count  = max(1, min_pop * total)
    new_states = []

    for iteration in range(max_new):
        if residue_coverage(all_states) >= target:
            print(f"  Target {target:.0%} reached after {iteration} new states")
            break

        search = H_smooth.copy()
        search[covered_mask]  = 0
        search[H < min_count] = 0

        if search.max() == 0:
            search = H_smooth.copy()
            search[covered_mask] = 0
            if search.max() == 0:
                print(f"  No uncovered bins remaining"); break

        bi, bj = np.unravel_index(search.argmax(), search.shape)
        best_phi = float(phi_c[bi])
        best_psi = float(psi_c[bj])
        best_pop = float(H[bi, bj]) / total

        # Estimate sigma from HWHM
        peak_val = H_smooth[bi, bj]
        half = peak_val * 0.5

        def _hw(arr, ci):
            for d in range(1, len(arr)):
                lo = max(ci - d, 0); hi = min(ci + d, len(arr) - 1)
                if arr[lo] < half or arr[hi] < half:
                    return d * bin_size
            return 3 * bin_size

        sig_phi = float(np.clip(max(_hw(H_smooth[:, bj], bi), bin_size*1.5), 10, 40))
        sig_psi = float(np.clip(max(_hw(H_smooth[bi, :], bj), bin_size*1.5), 10, 40))

        # Name the region
        name = _name_region(best_phi, best_psi, len(new_states))
        state = (name, best_phi, best_psi, sig_phi, sig_psi)
        new_states.append(state)
        all_states.append(state)

        # Update mask
        for i, p in enumerate(phi_c):
            for j, q in enumerate(psi_c):
                if not covered_mask[i, j] and _is_covered(p, q, [state]):
                    covered_mask[i, j] = True

        cov = residue_coverage(all_states)
        print(f"  +{name:12s}  φ={best_phi:>7.1f}°  ψ={best_psi:>7.1f}°  "
              f"pop={best_pop:.2%}  coverage → {cov:.1%}")

    final_cov = residue_coverage(all_states)
    return new_states, initial_cov, final_cov


def _name_region(phi, psi, idx):
    if phi > 0:             return 'αL_ext'
    if -90<=phi<=-30 and -60<=psi<=10:    return 'bridge'
    if -160<=phi<=-90 and 50<=psi<=100:   return 'turnI'
    if -90<=phi<=-40 and 90<=psi<=130:    return 'turnII'
    if phi<-140 and -60<=psi<=40:         return 'βL'
    if -80<=phi<=-40 and 130<=psi<=180:   return 'γ_inv'
    return f'loop{idx+1}'


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Torque vector analysis + R² (net torque vs full model)
# ══════════════════════════════════════════════════════════════════════════════

def compute_torque_r2(X_tau, X_full, yp, yq, y_sincos=None):
    """
    Compute R² for multiple target representations:
      1. Δφ/Δψ from αR reference (original)
      2. sin/cos encoding (reference-free)

    For each: net torque only vs full feature set, Ridge with 5-fold CV.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import r2_score

    results = {}

    # ── A. Delta targets (original, reference-dependent) ──────────────────────
    print(f"\n  Target: Δφ/Δψ from αR reference (−63°, −43°)")
    for label, X_raw in [('τ_net only (2 feat)', X_tau),
                          ('Full model (Δ ref)', X_full)]:
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_raw)

        # Stage 1: φ
        m1 = Ridge(alpha=1.0)
        cv1 = cross_val_score(m1, X_sc, yp, cv=5, scoring='r2')
        m1.fit(X_sc, yp)
        yp_pred = m1.predict(X_sc)
        r2_phi = r2_score(yp, yp_pred)

        # Stage 2: ψ | φ
        X2 = np.hstack([X_sc, yp_pred.reshape(-1, 1), yp.reshape(-1, 1)])
        sc2 = StandardScaler()
        X2_sc = sc2.fit_transform(X2)
        m2 = Ridge(alpha=1.0)
        cv2 = cross_val_score(m2, X2_sc, yq, cv=5, scoring='r2')
        m2.fit(X2_sc, yq)
        yq_pred = m2.predict(X2_sc)
        r2_psi = r2_score(yq, yq_pred)

        results[label] = {
            'r2_phi': r2_phi, 'r2_psi': r2_psi,
            'cv_phi': cv1, 'cv_psi': cv2,
            'yp_pred': yp_pred, 'yq_pred': yq_pred,
        }
        print(f"  {label:30s}  φ R²={r2_phi:.3f} (CV {cv1.mean():.3f}±{cv1.std():.3f})  "
              f"ψ R²={r2_psi:.3f} (CV {cv2.mean():.3f}±{cv2.std():.3f})")

    # ── B. Sin/cos targets (reference-free) ───────────────────────────────────
    if y_sincos is not None:
        print(f"\n  Target: sin/cos encoding (reference-free)")
        for label, X_raw in [('τ_net sincos (2 feat)', X_tau),
                              ('Full model (sincos)', X_full)]:
            scaler = StandardScaler()
            X_sc = scaler.fit_transform(X_raw)

            # Predict all 4 targets: sin_φ, cos_φ, sin_ψ, cos_ψ
            sc_r2 = {}
            sc_cv = {}
            sc_pred = {}
            for target_name in ['sin_phi', 'cos_phi', 'sin_psi', 'cos_psi']:
                y_t = y_sincos[target_name]
                m = Ridge(alpha=1.0)
                cv = cross_val_score(m, X_sc, y_t, cv=5, scoring='r2')
                m.fit(X_sc, y_t)
                pred = m.predict(X_sc)
                sc_r2[target_name] = r2_score(y_t, pred)
                sc_cv[target_name] = cv
                sc_pred[target_name] = pred

            # Reconstruct angles from sin/cos predictions
            pred_phi = np.degrees(np.arctan2(sc_pred['sin_phi'], sc_pred['cos_phi']))
            pred_psi = np.degrees(np.arctan2(sc_pred['sin_psi'], sc_pred['cos_psi']))
            actual_phi = np.degrees(np.arctan2(y_sincos['sin_phi'], y_sincos['cos_phi']))
            actual_psi = np.degrees(np.arctan2(y_sincos['sin_psi'], y_sincos['cos_psi']))

            # Circular R²: computed on the reconstructed angles
            # Use wrapped residuals for fair comparison
            resid_phi = np.array([wrap(p - a) for p, a in zip(pred_phi, actual_phi)])
            resid_psi = np.array([wrap(p - a) for p, a in zip(pred_psi, actual_psi)])
            var_phi = np.var(np.array([wrap(a - np.mean(actual_phi)) for a in actual_phi]))
            var_psi = np.var(np.array([wrap(a - np.mean(actual_psi)) for a in actual_psi]))
            circ_r2_phi = 1.0 - np.var(resid_phi) / var_phi if var_phi > 0 else 0.0
            circ_r2_psi = 1.0 - np.var(resid_psi) / var_psi if var_psi > 0 else 0.0

            results[label] = {
                'r2_phi': circ_r2_phi, 'r2_psi': circ_r2_psi,
                'cv_phi': np.mean([sc_cv['sin_phi'], sc_cv['cos_phi']], axis=0),
                'cv_psi': np.mean([sc_cv['sin_psi'], sc_cv['cos_psi']], axis=0),
                'yp_pred': np.array([wrap(p - REF_PHI) for p in pred_phi]),
                'yq_pred': np.array([wrap(p - REF_PSI) for p in pred_psi]),
                'sincos_r2': sc_r2,
            }

            print(f"  {label:30s}  "
                  f"sin_φ R²={sc_r2['sin_phi']:.3f}  cos_φ R²={sc_r2['cos_phi']:.3f}  "
                  f"sin_ψ R²={sc_r2['sin_psi']:.3f}  cos_ψ R²={sc_r2['cos_psi']:.3f}")
            print(f"  {'':30s}  "
                  f"→ circular φ R²={circ_r2_phi:.3f}  ψ R²={circ_r2_psi:.3f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: Non-linear model comparison
# ══════════════════════════════════════════════════════════════════════════════

def run_nonlinear_comparison(X, yp, yq, ss, feat_names):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import r2_score

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    models = {
        'Linear Ridge': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=12,
            min_samples_leaf=5, n_jobs=-1, random_state=42),
        'Gradient Boost': GradientBoostingRegressor(
            n_estimators=200, max_depth=5,
            learning_rate=0.05, random_state=42),
        'MLP (2-layer)': MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation='relu', max_iter=500,
            random_state=42, early_stopping=True),
    }

    results = {}
    print(f"\n  {'Model':20}  {'R²_φ':>8}  {'CV_φ':>10}  "
          f"{'R²_ψ':>8}  {'CV_ψ':>10}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}")

    for name, model in models.items():
        model_phi = model.__class__(**model.get_params())
        cv_phi = cross_val_score(model_phi, X_sc, yp, cv=5,
                                  scoring='r2', n_jobs=-1)
        model_phi.fit(X_sc, yp)
        r2_phi = r2_score(yp, model_phi.predict(X_sc))

        yp_pred = model_phi.predict(X_sc)
        X_aug = np.hstack([X_sc, yp_pred.reshape(-1, 1)])
        model_psi = model.__class__(**model.get_params())
        cv_psi = cross_val_score(model_psi, X_aug, yq, cv=5,
                                  scoring='r2', n_jobs=-1)
        model_psi.fit(X_aug, yq)
        r2_psi = r2_score(yq, model_psi.predict(X_aug))

        results[name] = {
            'model_phi': model_phi, 'model_psi': model_psi,
            'scaler': scaler,
            'r2_phi': r2_phi, 'r2_psi': r2_psi,
            'cv_phi': cv_phi, 'cv_psi': cv_psi,
            'yp_pred': yp_pred,
            'yq_pred': model_psi.predict(X_aug),
        }

        print(f"  {name:20}  {r2_phi:>8.3f}  "
              f"{cv_phi.mean():>6.3f}±{cv_phi.std():.3f}  "
              f"{r2_psi:>8.3f}  "
              f"{cv_psi.mean():>6.3f}±{cv_psi.std():.3f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: Basin coverage from force-predicted angles
# ══════════════════════════════════════════════════════════════════════════════

def compute_basin_coverage(records, model_results):
    actual_phi = np.array([r['phi'] for r in records])
    actual_psi = np.array([r['psi'] for r in records])
    actual_ss  = np.array([r['ss_bin'] for r in records])

    baseline = sum(1 for p, q in zip(actual_phi, actual_psi)
                   if _is_covered(p, q, CURRENT_STATES)) / len(records)
    print(f"\n  Baseline 6-state grid coverage: {baseline:.1%}")

    results = {}
    for model_name, mr in model_results.items():
        # CORRECT reconstruction: ref + predicted delta
        pred_phi = np.array([wrap(REF_PHI + d) for d in mr['yp_pred']])
        pred_psi = np.array([wrap(REF_PSI + d) for d in mr['yq_pred']])
        pred_ss  = np.array([_ss_bin(p, q) for p, q in zip(pred_phi, pred_psi)])

        basin_acc = float(np.mean(pred_ss == actual_ss))

        per_basin = {}
        for basin in range(6):
            mask = actual_ss == basin
            if mask.sum() < 10: continue
            per_basin[basin] = float(np.mean(pred_ss[mask] == actual_ss[mask]))

        results[model_name] = {
            'basin_accuracy': basin_acc,
            'per_basin': per_basin,
            'pred_phi': pred_phi, 'pred_psi': pred_psi,
            'pred_ss': pred_ss,
        }

        print(f"  {model_name:20s}: basin accuracy = {basin_acc:.1%}")
        for b, acc in sorted(per_basin.items()):
            print(f"    {BASIN_NAMES[b]:6s}: {acc:.1%}")

    return results, baseline


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(records, new_states, initial_cov, final_cov,
             torque_r2, model_comparison, coverage_results, baseline,
             feat_names, out_dir):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LogNorm

    out = Path(out_dir)
    phi_all = np.array([r['phi'] for r in records])
    psi_all = np.array([r['psi'] for r in records])
    ss_all  = np.array([r['ss_bin'] for r in records])

    # ══════════════════════════════════════════════════════════════════════════
    # Fig 1: Ramachandran coverage — 6-state vs +8 new states
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    all_states = list(CURRENT_STATES) + list(new_states)

    for ax, title, states, c_new in [
        (axes[0], f'Current 6 states  (coverage = {initial_cov:.0%})',
         CURRENT_STATES, []),
        (axes[1], f'+ {len(new_states)} new states  (coverage = {final_cov:.0%})',
         all_states, new_states),
    ]:
        H, xe, ye = np.histogram2d(phi_all, psi_all, bins=72,
                                    range=[[-180,180],[-180,180]])
        h_plot = np.where(H.T > 0, H.T, np.nan)
        ax.imshow(h_plot, origin='lower', aspect='auto',
                  extent=[-180,180,-180,180],
                  cmap='Blues', norm=LogNorm(vmin=1))

        for name, phi_c, psi_c, s_phi, s_psi in CURRENT_STATES:
            ax.plot(phi_c, psi_c, 'o', color='#1D9E75', ms=8, zorder=5)
            ell = mpatches.Ellipse((phi_c, psi_c),
                                    2*s_phi*COVERAGE_SIGMA,
                                    2*s_psi*COVERAGE_SIGMA,
                                    fill=False, edgecolor='#1D9E75',
                                    linewidth=1.2, zorder=4)
            ax.add_patch(ell)
            ax.text(phi_c+3, psi_c+3, name, fontsize=7, color='#085041', zorder=6)

        for name, phi_c, psi_c, s_phi, s_psi in c_new:
            ax.plot(phi_c, psi_c, 's', color='#D85A30', ms=9, zorder=5)
            ell = mpatches.Ellipse((phi_c, psi_c),
                                    2*s_phi*COVERAGE_SIGMA,
                                    2*s_psi*COVERAGE_SIGMA,
                                    fill=False, edgecolor='#D85A30',
                                    linewidth=1.5, linestyle='--', zorder=4)
            ax.add_patch(ell)
            ax.text(phi_c+3, psi_c+3, name, fontsize=7, color='#712B13', zorder=6)

        ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
        ax.set_xlabel('φ (degrees)'); ax.set_ylabel('ψ (degrees)')
        ax.set_title(title, fontsize=11)
        ax.axhline(0, color='gray', lw=0.4, alpha=0.5)
        ax.axvline(0, color='gray', lw=0.4, alpha=0.5)
        ax.set_xticks(range(-180,181,60)); ax.set_yticks(range(-180,181,60))

    handles = [
        mpatches.Patch(color='#1D9E75', label='Current states (6)'),
        mpatches.Patch(color='#D85A30', label=f'New states ({len(new_states)})'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    plt.tight_layout()
    p = out / 'rama_coverage.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Fig 1 → {p}")

    # ══════════════════════════════════════════════════════════════════════════
    # Fig 2: Torque vectors — 3 panels
    #   Left:   Raw net torque per basin
    #   Center: Restoring projection (toward αR = green, away = red)
    #   Right:  Per-source torque decomposition
    # ══════════════════════════════════════════════════════════════════════════

    # ── Gather per-basin torques (net + per-source) ───────────────────────────
    source_keys = {
        'bb_donor':  ('tau_phi_bb_donor',  'tau_psi_bb_donor'),
        'bb_acc':    ('tau_phi_bb_acc',    'tau_psi_bb_acc'),
        'sc_hb':     ('tau_phi_sc_hb',     'tau_psi_sc_hb'),
        'steric':    ('tau_phi_steric',    'tau_psi_steric'),
        'elec':      ('tau_phi_elec_corr', 'tau_psi_elec_corr'),
    }

    basin_data = defaultdict(lambda: defaultdict(list))
    for r in records:
        b = r['ss_bin']
        basin_data[b]['tau_phi'].append(r['tau_phi'])
        basin_data[b]['tau_psi'].append(r['tau_psi'])
        for src, (pk, qk) in source_keys.items():
            try:
                pi = feat_names.index(pk)
                qi = feat_names.index(qk)
                basin_data[b][f'tau_phi_{src}'].append(r['feat'][pi])
                basin_data[b][f'tau_psi_{src}'].append(r['feat'][qi])
            except (ValueError, IndexError):
                basin_data[b][f'tau_phi_{src}'].append(0.0)
                basin_data[b][f'tau_psi_{src}'].append(0.0)

    # ── Print near-zero diagnostic ────────────────────────────────────────────
    print(f"\n  Net torque per basin (non-zero = unmodeled backbone spring + solvent):")
    print(f"  {'Basin':6}  {'n':>6}  {'⟨τ_φ⟩':>10}  {'⟨τ_ψ⟩':>10}  "
          f"{'σ(τ_φ)':>9}  {'σ(τ_ψ)':>9}")
    for basin in sorted(basin_data.keys()):
        d = basin_data[basin]
        tp = np.array(d['tau_phi']); tq = np.array(d['tau_psi'])
        print(f"  {BASIN_NAMES[basin]:6}  {len(tp):>6}  "
              f"{np.mean(tp):>+10.4f}  {np.mean(tq):>+10.4f}  "
              f"{np.std(tp):>9.4f}  {np.std(tq):>9.4f}")

    # ── Compute restoring projection ──────────────────────────────────────────
    # Direction: basin_center → αR reference (-63, -43)
    # Positive projection = torque pushes toward αR (restoring)
    # Negative projection = torque pushes away from αR (driving into basin)
    print(f"\n  Restoring projection (+ = toward αR, − = drives into basin):")
    print(f"  {'Basin':6}  {'direction':>20}  {'⟨proj⟩':>10}  {'interp':>12}")

    restoring_data = {}
    for basin in sorted(basin_data.keys()):
        d = basin_data[basin]
        phi_c, psi_c = BASIN_CENTRES[basin]
        dv = np.array([wrap(REF_PHI - phi_c), wrap(REF_PSI - psi_c)])
        dv_norm = np.linalg.norm(dv)

        if dv_norm < 1.0:  # αR itself
            restoring_hat = np.array([0.0, 0.0])
        else:
            restoring_hat = dv / dv_norm

        tp = np.array(d['tau_phi']); tq = np.array(d['tau_psi'])
        projections = tp * restoring_hat[0] + tq * restoring_hat[1]
        mean_proj = float(np.mean(projections))

        interp = "RESTORING" if mean_proj > 0 else "DRIVING"
        if basin == 0: interp = "(reference)"

        restoring_data[basin] = {
            'hat': restoring_hat,
            'mean_proj': mean_proj,
            'mean_tau': (float(np.mean(tp)), float(np.mean(tq))),
        }

        print(f"  {BASIN_NAMES[basin]:6}  "
              f"({restoring_hat[0]:>+.3f}, {restoring_hat[1]:>+.3f})  "
              f"{mean_proj:>+10.4f}  {interp:>12}")

    # ── Draw 3-panel figure ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(26, 8))

    def _draw_bg(ax):
        H, xe, ye = np.histogram2d(phi_all, psi_all, bins=72,
                                    range=[[-180,180],[-180,180]])
        ax.imshow(H.T, origin='lower', aspect='auto',
                  extent=[-180,180,-180,180],
                  cmap='Greys', norm=LogNorm(vmin=0.5), alpha=0.55)
        ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
        ax.axhline(0, color='gray', lw=0.4, alpha=0.4)
        ax.axvline(0, color='gray', lw=0.4, alpha=0.4)
        ax.set_xlabel('φ (°)'); ax.set_ylabel('ψ (°)')
        ax.scatter([REF_PHI], [REF_PSI], s=130, marker='*',
                   color='black', zorder=6)

    # ── Panel A: Raw net torque arrows ────────────────────────────────────────
    ax = axes[0]
    _draw_bg(ax)
    ax.set_title('(A) Environmental torque (= −τ_resistance)\n'
                 'Non-zero because backbone spring\n'
                 '+ solvent balance these at equilibrium', fontsize=10)

    # Compute global max across all basins for consistent arrow scaling
    all_net_mags = []
    for basin in BASIN_COLORS:
        d = basin_data.get(basin, {})
        if 'tau_phi' not in d or len(d['tau_phi']) < 10: continue
        rd = restoring_data[basin]
        mt, mq = rd['mean_tau']
        all_net_mags.append(max(abs(mt), abs(mq)))
    global_net_max = max(all_net_mags) if all_net_mags else 1.0
    net_scale = 30.0 / (global_net_max + 1e-8)

    for basin, color in BASIN_COLORS.items():
        d = basin_data.get(basin, {})
        if 'tau_phi' not in d or len(d['tau_phi']) < 10: continue
        rd = restoring_data[basin]
        mean_tp, mean_tq = rd['mean_tau']
        phi_c, psi_c = BASIN_CENTRES[basin]

        ax.annotate('',
            xy=(phi_c + mean_tp*net_scale, psi_c + mean_tq*net_scale),
            xytext=(phi_c, psi_c),
            arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
        ax.scatter([phi_c], [psi_c], s=80, color=color, zorder=5,
                   label=f"{BASIN_NAMES[basin]} n={len(d['tau_phi']):,}")

    ax.legend(fontsize=7, loc='lower right')

    # ── Panel B: Restoring projection ─────────────────────────────────────────
    ax = axes[1]
    _draw_bg(ax)
    ax.set_title('(B) Restoring projection\n'
                 'Green → toward αR (restoring)\n'
                 'Red → away from αR (driving into basin)', fontsize=10)

    # Draw dashed lines from each basin toward αR for reference
    for basin, color in BASIN_COLORS.items():
        if basin == 0: continue
        phi_c, psi_c = BASIN_CENTRES[basin]
        ax.plot([phi_c, REF_PHI], [psi_c, REF_PSI],
                '--', color=color, lw=0.8, alpha=0.3, zorder=2)

    for basin, color in BASIN_COLORS.items():
        d = basin_data.get(basin, {})
        if 'tau_phi' not in d or len(d['tau_phi']) < 10: continue
        rd = restoring_data[basin]
        phi_c, psi_c = BASIN_CENTRES[basin]

        if basin == 0:
            ax.scatter([phi_c], [psi_c], s=100, color=color, zorder=5,
                       label=f"{BASIN_NAMES[basin]} (ref)")
            continue

        hat = rd['hat']
        proj = rd['mean_proj']

        # Arrow along restoring direction, scaled by projection magnitude
        # Projections range ~0.04–0.19, so scale up significantly for visibility
        arrow_scale = 120.0
        dx = proj * hat[0] * arrow_scale
        dy = proj * hat[1] * arrow_scale

        # Clamp for visibility (min 12, max 40 degrees on plot)
        arrow_len = np.sqrt(dx**2 + dy**2)
        if arrow_len > 0.01:
            target_len = np.clip(arrow_len, 12.0, 40.0)
            dx *= target_len / arrow_len
            dy *= target_len / arrow_len

        arrow_color = '#1D9E75' if proj > 0 else '#E24B4A'

        ax.annotate('',
            xy=(phi_c + dx, psi_c + dy),
            xytext=(phi_c, psi_c),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=3.0))
        ax.scatter([phi_c], [psi_c], s=80, color=color, zorder=5)
        ax.text(phi_c + 5, psi_c - 10,
                f"{BASIN_NAMES[basin]}\nproj={proj:+.3f}",
                fontsize=7, color=color, fontweight='bold', zorder=7)

    ax.legend(fontsize=7, loc='lower right')

    # ── Panel C: Per-source torque decomposition ──────────────────────────────
    ax = axes[2]
    _draw_bg(ax)
    ax.set_title('(C) Per-source torque decomposition\n'
                 'Which forces dominate in each basin?', fontsize=10)

    source_colors = {
        'bb_donor': '#1D9E75',
        'bb_acc':   '#378ADD',
        'sc_hb':    '#BA7517',
        'steric':   '#D4537E',
        'elec':     '#9B59B6',
    }
    source_labels = {
        'bb_donor': 'bb N-H donor',
        'bb_acc':   'bb C=O acceptor',
        'sc_hb':    'SC H-bond',
        'steric':   'SC steric (Cγ)',
        'elec':     'electrostatic',
    }

    # Global scale for consistent arrow lengths
    all_mags = []
    for basin in BASIN_COLORS:
        d = basin_data.get(basin, {})
        for src in source_colors:
            k1 = f'tau_phi_{src}'; k2 = f'tau_psi_{src}'
            if k1 in d and len(d[k1]) > 0:
                all_mags.append(max(abs(np.mean(d[k1])), abs(np.mean(d[k2]))))
    global_max = max(all_mags) if all_mags else 1.0
    src_scale = 22.0 / (global_max + 1e-8)

    legend_added = set()
    for basin in BASIN_COLORS:
        d = basin_data.get(basin, {})
        if 'tau_phi_bb_donor' not in d or len(d['tau_phi_bb_donor']) < 10:
            continue
        phi_c, psi_c = BASIN_CENTRES[basin]
        ax.scatter([phi_c], [psi_c], s=60, color=BASIN_COLORS[basin],
                   zorder=5, edgecolors='black', linewidth=0.5)
        ax.text(phi_c + 4, psi_c - 12, BASIN_NAMES[basin],
                fontsize=7, color=BASIN_COLORS[basin], fontweight='bold', zorder=7)

        # Draw each source as a separate arrow from basin center
        # Slight offset per source so they don't stack on top of each other
        offsets = [(-4, 4), (4, 4), (-4, -4), (4, -4), (0, 7)]
        for si, (src, scolor) in enumerate(source_colors.items()):
            k1 = f'tau_phi_{src}'; k2 = f'tau_psi_{src}'
            mt = float(np.mean(d[k1])); mq = float(np.mean(d[k2]))
            if abs(mt) < 1e-6 and abs(mq) < 1e-6: continue

            ox, oy = offsets[si % len(offsets)]
            ax.annotate('',
                xy=(phi_c + ox + mt*src_scale, psi_c + oy + mq*src_scale),
                xytext=(phi_c + ox, psi_c + oy),
                arrowprops=dict(arrowstyle='->', color=scolor, lw=1.8, alpha=0.85))

            if src not in legend_added:
                ax.plot([], [], color=scolor, lw=2, label=source_labels[src])
                legend_added.add(src)

    ax.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    p = out / 'torque_vectors.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Fig 2 → {p}")

    # ══════════════════════════════════════════════════════════════════════════
    # Fig 3: R² comparison bar chart (all models)
    # ══════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 5))

    # Combine torque_r2 and model_comparison for one unified chart
    all_models = {}
    for label, mr in torque_r2.items():
        all_models[label] = mr
    for label, mr in model_comparison.items():
        all_models[label] = mr

    names = list(all_models.keys())
    x = np.arange(len(names))
    r2p = [all_models[n]['r2_phi'] for n in names]
    r2q = [all_models[n]['r2_psi'] for n in names]
    cvp = [all_models[n]['cv_phi'].mean() for n in names]
    cvq = [all_models[n]['cv_psi'].mean() for n in names]

    ax.bar(x - 0.2, r2p, 0.38, label='R² φ (train)', color='#1D9E75', alpha=0.9)
    ax.bar(x + 0.2, r2q, 0.38, label='R² ψ (train)', color='#378ADD', alpha=0.9)
    ax.scatter(x - 0.2, cvp, color='#085041', s=40, zorder=5, label='CV R² φ')
    ax.scatter(x + 0.2, cvq, color='#0C447C', s=40, zorder=5, label='CV R² ψ')

    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel('R²')
    ax.set_title('Model comparison: net torque → linear → non-linear')
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    p = out / 'r2_comparison.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Fig 3 → {p}")

    # ══════════════════════════════════════════════════════════════════════════
    # Fig 4: Basin accuracy per basin, all models
    # ══════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 5))
    basins = sorted(set(b for cr in coverage_results.values()
                        for b in cr['per_basin']))
    n_models = len(coverage_results)
    width = 0.8 / max(n_models, 1)
    model_colors = ['#1D9E75','#378ADD','#BA7517','#D4537E','#9B59B6','#888780']

    for mi, (mname, cr_m) in enumerate(coverage_results.items()):
        xpos = np.arange(len(basins)) + (mi - n_models/2 + 0.5) * width
        vals = [cr_m['per_basin'].get(b, 0) for b in basins]
        ax.bar(xpos, vals, width*0.9, label=mname,
               color=model_colors[mi % len(model_colors)], alpha=0.85)

    ax.set_xticks(np.arange(len(basins)))
    ax.set_xticklabels([BASIN_NAMES[b] for b in basins])
    ax.set_ylim(0, 1.05); ax.set_ylabel('Basin accuracy')
    ax.set_title('Per-basin accuracy: does force model predict the correct Ramachandran region?')
    ax.legend(fontsize=8)
    ax.axhline(1.0/len(basins), color='gray', lw=0.8, ls=':', alpha=0.5)
    plt.tight_layout()
    p = out / 'basin_accuracy.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Fig 4 → {p}")

    # ══════════════════════════════════════════════════════════════════════════
    # Fig 5: Feature importance (RF) — 3 colors: steric / force / context
    # ══════════════════════════════════════════════════════════════════════════
    if 'Random Forest' in model_comparison:
        rf_phi = model_comparison['Random Forest']['model_phi']
        rf_psi = model_comparison['Random Forest']['model_psi']

        # Color map: which group does each feature belong to?
        GROUP_A = {'steric_N_3A','steric_N_4A','steric_N_5A',
                   'steric_CA_3A','steric_CA_4A','steric_CA_5A',
                   'steric_C_3A','steric_C_4A','steric_C_5A',
                   'steric_O_3A','steric_O_4A','steric_O_5A',
                   'steric_asym_x','steric_asym_y','steric_asym_z',
                   'steric_clash_phi_plus','steric_clash_phi_minus',
                   'steric_clash_psi_plus','steric_clash_psi_minus',
                   'improper_ca','sc_contact_nm1_to_bb','sc_contact_np1_to_bb'}
        GROUP_B = {'tau_phi_correct','tau_psi_correct',
                   'tau_phi_bb_donor','tau_psi_bb_donor',
                   'tau_phi_bb_acc','tau_psi_bb_acc',
                   'tau_phi_sc_hb','tau_psi_sc_hb',
                   'tau_phi_steric','tau_psi_steric',
                   'tau_phi_elec_corr','tau_psi_elec_corr',
                   'chi1_rad','has_chi1'}
        # GROUP_C = everything else

        def _feat_color(name):
            if name in GROUP_A: return '#1D9E75'   # green = steric
            if name in GROUP_B: return '#378ADD'    # blue = force
            return '#BA7517'                         # amber = context

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        for ax, model_rf, title, extra_names in [
            (axes[0], rf_phi, 'RF importance for φ', feat_names),
            (axes[1], rf_psi, 'RF importance for ψ (given φ)', feat_names + ['predicted_φ']),
        ]:
            if not hasattr(model_rf, 'feature_importances_'): continue
            imp = sorted(zip(extra_names, model_rf.feature_importances_),
                         key=lambda x: x[1], reverse=True)[:15]
            names_i = [x[0] for x in imp]
            vals_i  = [x[1] for x in imp]
            colors_i = [_feat_color(n) for n in names_i]
            ax.barh(range(len(names_i)), vals_i, color=colors_i, alpha=0.85)
            ax.set_yticks(range(len(names_i)))
            ax.set_yticklabels(names_i, fontsize=8)
            ax.set_xlabel('Importance')
            ax.set_title(title, fontsize=11)
            ax.invert_yaxis()

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1D9E75', alpha=0.85, label='Group A: Steric'),
            Patch(facecolor='#378ADD', alpha=0.85, label='Group B: Forces'),
            Patch(facecolor='#BA7517', alpha=0.85, label='Group C: Context'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3,
                   bbox_to_anchor=(0.5, -0.02), fontsize=10)
        plt.tight_layout()
        p = out / 'feature_importance.png'
        plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  Fig 5 → {p}")

    # ══════════════════════════════════════════════════════════════════════════
    # Fig 6: Residual analysis (best model)
    # ══════════════════════════════════════════════════════════════════════════
    best_name = max(model_comparison,
                    key=lambda n: model_comparison[n]['cv_psi'].mean())
    best_mr = model_comparison[best_name]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    yp_arr = np.array([r['delta_phi'] for r in records])
    yq_arr = np.array([r['delta_psi'] for r in records])

    for ax, residuals, label in [
        (axes[0], best_mr['yp_pred'] - yp_arr, 'φ'),
        (axes[1], best_mr['yq_pred'] - yq_arr, 'ψ'),
    ]:
        for basin, color in BASIN_COLORS.items():
            mask = ss_all == basin
            if mask.sum() < 5: continue
            ax.hist(residuals[mask], bins=60, alpha=0.5,
                    color=color, label=BASIN_NAMES.get(basin), density=True)
        ax.axvline(0, color='black', lw=1)
        rmse = float(np.sqrt(np.mean(residuals**2)))
        ax.set_xlabel(f'Residual Δ{label} (°)')
        ax.set_ylabel('Density')
        ax.set_title(f'{label} residuals — RMSE={rmse:.1f}°  ({best_name})')
        ax.legend(fontsize=7)
    plt.tight_layout()
    p = out / 'residuals.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Fig 6 → {p}")

    # ══════════════════════════════════════════════════════════════════════════
    # Fig 7: Actual vs force-predicted Ramachandran (best model)
    # ══════════════════════════════════════════════════════════════════════════
    cr = coverage_results[best_name]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, p_arr, q_arr, title in [
        (axes[0], phi_all, psi_all,
         f'Actual (φ,ψ) — grid coverage={baseline:.1%}'),
        (axes[1], cr['pred_phi'], cr['pred_psi'],
         f'Force-predicted (φ,ψ) — {best_name}\n'
         f'basin accuracy={cr["basin_accuracy"]:.1%}'),
    ]:
        H, xe, ye = np.histogram2d(p_arr, q_arr, bins=72,
                                    range=[[-180,180],[-180,180]])
        ax.imshow(H.T, origin='lower', aspect='auto',
                  extent=[-180,180,-180,180],
                  cmap='Blues', norm=LogNorm(vmin=0.5))
        for name, phi_c, psi_c, s_phi, s_psi in CURRENT_STATES:
            ell = mpatches.Ellipse((phi_c, psi_c),
                                    2*s_phi*COVERAGE_SIGMA,
                                    2*s_psi*COVERAGE_SIGMA,
                                    fill=False, edgecolor='#D85A30',
                                    linewidth=1.5, linestyle='--', zorder=4)
            ax.add_patch(ell)
        ax.set_xlim(-180,180); ax.set_ylim(-180,180)
        ax.axhline(0,color='gray',lw=0.4,alpha=0.4)
        ax.axvline(0,color='gray',lw=0.4,alpha=0.4)
        ax.set_xlabel('φ (°)'); ax.set_ylabel('ψ (°)')
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    p = out / 'corrected_rama_coverage.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Fig 7 → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

def write_report(records, new_states, initial_cov, final_cov,
                 torque_r2, model_comparison, coverage_results, baseline,
                 feat_names, out_dir):
    best_name = max(model_comparison,
                    key=lambda n: model_comparison[n]['cv_psi'].mean())
    cr = coverage_results[best_name]
    tau_r = torque_r2['τ_net only (2 feat)']
    full_r = torque_r2['Full model (Δ ref)']

    lines = [
        "Combined Ramachandran + Force-Basin Analysis Report",
        "=" * 65,
        "",
        f"Total residues: {len(records):,}",
        "",
        "─── PART 1: RAMACHANDRAN COVERAGE ───",
        f"  6-state grid:           {initial_cov:.1%}",
        f"  + {len(new_states)} new states:       {final_cov:.1%}  "
        f"(+{final_cov - initial_cov:.1%})",
        "",
        "  New states:",
    ]
    for name, phi, psi, sp, sq in new_states:
        lines.append(f"    {name:12s}  φ={phi:>7.1f}°  ψ={psi:>7.1f}°  "
                     f"σ=({sp:.0f}°,{sq:.0f}°)")

    lines += [
        "",
        "─── PART 2: TORQUE → ANGLE PREDICTION ───",
        f"  Net torque only (τ_φ, τ_ψ → Δφ, Δψ):",
        f"    φ R²={tau_r['r2_phi']:.3f}  CV={tau_r['cv_phi'].mean():.3f}",
        f"    ψ R²={tau_r['r2_psi']:.3f}  CV={tau_r['cv_psi'].mean():.3f}",
        f"",
        f"  Full model ({len(feat_names)} features → Δφ, Δψ):",
        f"    φ R²={full_r['r2_phi']:.3f}  CV={full_r['cv_phi'].mean():.3f}",
        f"    ψ R²={full_r['r2_psi']:.3f}  CV={full_r['cv_psi'].mean():.3f}",
        "",
        "  This shows that decomposed per-source torques + geometry",
        "  explain more variance than net torque alone.",
        "",
        "─── PART 3: NON-LINEAR MODEL COMPARISON ───",
        f"  {'Model':20}  {'R²_φ':>7}  {'CV_φ':>8}  {'R²_ψ':>7}  {'CV_ψ':>8}",
        f"  {'-'*20}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*8}",
    ]
    for name, mr in model_comparison.items():
        lines.append(
            f"  {name:20}  {mr['r2_phi']:>7.3f}  "
            f"{mr['cv_phi'].mean():>6.3f}±{mr['cv_phi'].std():.2f}  "
            f"{mr['r2_psi']:>7.3f}  "
            f"{mr['cv_psi'].mean():>6.3f}±{mr['cv_psi'].std():.2f}")

    lines += [
        "",
        "─── PART 4: FORCE-PREDICTED BASIN COVERAGE ───",
        f"  6-state grid baseline: {baseline:.1%}",
        f"  Best model: {best_name}",
        f"  Basin accuracy: {cr['basin_accuracy']:.1%}",
        "",
        "  Per-basin:",
    ]
    for b, acc in sorted(cr['per_basin'].items()):
        lines.append(f"    {BASIN_NAMES[b]:6s}: {acc:.1%}")

    lines += [
        "",
        "─── FIGURE INDEX ───",
        "  Fig 1: rama_coverage.png              — 6-state vs +N new states",
        "  Fig 2: torque_vectors.png             — 3-panel: raw / restoring / decomposition",
        "  Fig 3: r2_comparison.png              — R² bar chart all models",
        "  Fig 4: basin_accuracy.png             — Per-basin accuracy",
        "  Fig 5: feature_importance.png         — RF feature importance",
        "  Fig 6: residuals.png                  — Prediction error by basin",
        "  Fig 7: corrected_rama_coverage.png    — Actual vs predicted Ramachandran",
    ]

    p = Path(out_dir) / 'report.txt'
    p.write_text('\n'.join(lines))
    print(f"\n  Report → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 5: Layer cascade — steric only → +forces → +context
# ══════════════════════════════════════════════════════════════════════════════

# Feature group membership for cascade
_GROUP_A = {'steric_N_3A','steric_N_4A','steric_N_5A',
            'steric_CA_3A','steric_CA_4A','steric_CA_5A',
            'steric_C_3A','steric_C_4A','steric_C_5A',
            'steric_O_3A','steric_O_4A','steric_O_5A',
            'steric_asym_x','steric_asym_y','steric_asym_z',
            'steric_clash_phi_plus','steric_clash_phi_minus',
            'steric_clash_psi_plus','steric_clash_psi_minus',
            'improper_ca','sc_contact_nm1_to_bb','sc_contact_np1_to_bb'}
_GROUP_B = {'tau_phi_correct','tau_psi_correct',
            'tau_phi_bb_donor','tau_psi_bb_donor',
            'tau_phi_bb_acc','tau_psi_bb_acc',
            'tau_phi_sc_hb','tau_psi_sc_hb',
            'tau_phi_steric','tau_psi_steric',
            'tau_phi_elec_corr','tau_psi_elec_corr',
            'chi1_rad','has_chi1'}


def run_cascade(X, yp, yq, y_sincos, feat_names):
    """
    Run the R² cascade: steric only → +forces → +context.
    Uses sin/cos targets for ψ (reference-free) and Δ-reference for φ.
    Both Ridge and RF with 5-fold CV.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import r2_score

    # Split features into groups by index
    idx_A = [i for i, k in enumerate(feat_names) if k in _GROUP_A]
    idx_B = [i for i, k in enumerate(feat_names) if k in _GROUP_B]
    idx_C = [i for i, k in enumerate(feat_names)
             if k not in _GROUP_A and k not in _GROUP_B]

    print(f"  Group A (steric):  {len(idx_A)} features")
    print(f"  Group B (forces):  {len(idx_B)} features")
    print(f"  Group C (context): {len(idx_C)} features")

    cascade_configs = [
        ('Steric only (A)',      idx_A),
        ('+ Forces (A+B)',       idx_A + idx_B),
        ('+ Context (A+B+C)',    idx_A + idx_B + idx_C),
        ('Forces only (B)',      idx_B),
        ('Context only (C)',     idx_C),
    ]

    # Use sin/cos targets for better ψ
    y_sp = y_sincos['sin_phi']; y_cp = y_sincos['cos_phi']
    y_sq = y_sincos['sin_psi']; y_cq = y_sincos['cos_psi']

    print(f"\n  {'Config':25}  {'Ridge φ':>8}  {'Ridge ψ':>8}  "
          f"{'RF φ (CV)':>10}  {'RF ψ (CV)':>10}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")

    results = {}
    for label, indices in cascade_configs:
        if not indices:
            print(f"  {label:25}  (no features)")
            continue

        X_sub = X[:, indices]
        sc = StandardScaler()
        X_sc = sc.fit_transform(X_sub)

        # Ridge on sin/cos: predict 4 targets, reconstruct circular R²
        ridge_r2 = {}
        for tname, y_t in [('sin_phi', y_sp), ('cos_phi', y_cp),
                            ('sin_psi', y_sq), ('cos_psi', y_cq)]:
            m = Ridge(alpha=1.0)
            m.fit(X_sc, y_t)
            ridge_r2[tname] = r2_score(y_t, m.predict(X_sc))

        # Circular R² from reconstructed angles
        pred_phi = np.degrees(np.arctan2(
            Ridge(alpha=1.0).fit(X_sc, y_sp).predict(X_sc),
            Ridge(alpha=1.0).fit(X_sc, y_cp).predict(X_sc)))
        pred_psi = np.degrees(np.arctan2(
            Ridge(alpha=1.0).fit(X_sc, y_sq).predict(X_sc),
            Ridge(alpha=1.0).fit(X_sc, y_cq).predict(X_sc)))
        actual_phi = np.degrees(np.arctan2(y_sp, y_cp))
        actual_psi = np.degrees(np.arctan2(y_sq, y_cq))
        resid_p = np.array([wrap(p-a) for p,a in zip(pred_phi, actual_phi)])
        resid_q = np.array([wrap(p-a) for p,a in zip(pred_psi, actual_psi)])
        var_p = np.var([wrap(a - np.mean(actual_phi)) for a in actual_phi])
        var_q = np.var([wrap(a - np.mean(actual_psi)) for a in actual_psi])
        ridge_circ_phi = 1 - np.var(resid_p)/var_p if var_p > 0 else 0
        ridge_circ_psi = 1 - np.var(resid_q)/var_q if var_q > 0 else 0

        # RF on Δ-reference (two-stage, 5-fold CV)
        rf = RandomForestRegressor(n_estimators=200, max_depth=12,
                                    min_samples_leaf=5, n_jobs=-1, random_state=42)
        cv_phi = cross_val_score(rf, X_sc, yp, cv=5, scoring='r2', n_jobs=-1)
        rf.fit(X_sc, yp)
        yp_pred = rf.predict(X_sc)
        X_aug = np.hstack([X_sc, yp_pred.reshape(-1,1)])
        rf2 = RandomForestRegressor(n_estimators=200, max_depth=12,
                                     min_samples_leaf=5, n_jobs=-1, random_state=42)
        cv_psi = cross_val_score(rf2, X_aug, yq, cv=5, scoring='r2', n_jobs=-1)

        results[label] = {
            'ridge_phi': ridge_circ_phi, 'ridge_psi': ridge_circ_psi,
            'rf_cv_phi': cv_phi.mean(), 'rf_cv_psi': cv_psi.mean(),
            'rf_cv_phi_std': cv_phi.std(), 'rf_cv_psi_std': cv_psi.std(),
        }

        print(f"  {label:25}  {ridge_circ_phi:>8.3f}  {ridge_circ_psi:>8.3f}  "
              f"{cv_phi.mean():>6.3f}±{cv_phi.std():.3f}  "
              f"{cv_psi.mean():>6.3f}±{cv_psi.std():.3f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PART 6: Two-spring + one-joint model — estimate k_backbone, k_sidechain
# ══════════════════════════════════════════════════════════════════════════════

_SC_N_ROT = {
    'GLY':0,'ALA':0,'VAL':1,'LEU':2,'ILE':2,'PRO':0,
    'PHE':2,'TYR':2,'TRP':2,'SER':1,'THR':1,'CYS':1,
    'MET':3,'ASP':2,'ASN':2,'GLU':3,'GLN':3,'LYS':4,'ARG':4,'HIS':2,
}

def run_spring_analysis(records, out_dir):
    """
    Estimate k_eff = −τ_env / Δθ per residue.
    Test whether k is constant (rigid body) or varies (2-spring model).
    """
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out_dir)

    print(f"\n  Estimating k_eff = −τ_env / Δθ per residue ...")

    meta = []
    for r in records:
        dphi_rad = np.radians(wrap(r['phi'] - REF_PHI))
        dpsi_rad = np.radians(wrap(r['psi'] - REF_PSI))
        kp = np.clip(-r['tau_phi']/dphi_rad, -50, 50) if abs(dphi_rad) > np.radians(5) else np.nan
        kq = np.clip(-r['tau_psi']/dpsi_rad, -50, 50) if abs(dpsi_rad) > np.radians(5) else np.nan
        meta.append({
            'basin': r['ss_bin'], 'res_name': r['res_name'],
            'n_rot': _SC_N_ROT.get(r['res_name'], 0),
            'k_phi': kp, 'k_psi': kq,
            'phi': r['phi'], 'psi': r['psi'],
        })

    k_phi = np.array([m['k_phi'] for m in meta])
    k_psi = np.array([m['k_psi'] for m in meta])

    # ── Test 1: k by basin ────────────────────────────────────────────────────
    print(f"\n  k_eff by basin:")
    print(f"  {'Basin':6}  {'n':>5}  {'⟨k_φ⟩':>8}  {'σ_φ':>7}  {'⟨k_ψ⟩':>8}  {'σ_ψ':>7}")
    for b in sorted(BASIN_NAMES.keys()):
        mask = np.array([m['basin']==b for m in meta])
        kp = k_phi[mask]; kp = kp[np.isfinite(kp)]
        kq = k_psi[mask]; kq = kq[np.isfinite(kq)]
        if len(kp) < 10: continue
        print(f"  {BASIN_NAMES[b]:6}  {len(kp):>5}  {np.mean(kp):>+8.3f}  "
              f"{np.std(kp):>7.3f}  {np.mean(kq):>+8.3f}  {np.std(kq):>7.3f}")

    # ── Test 2: k by n_rotatable (2-spring prediction) ────────────────────────
    print(f"\n  k_eff by rotatable bonds (2-spring test):")
    print(f"  {'n_rot':>5}  {'n':>5}  {'⟨|k_φ|⟩':>8}  {'⟨|k_ψ|⟩':>8}  {'examples':>20}")
    ex = {0:'GLY/ALA/PRO', 1:'VAL/SER/THR', 2:'LEU/PHE/ASP', 3:'MET/GLU/GLN', 4:'LYS/ARG'}
    for nr in range(5):
        mask = np.array([m['n_rot']==nr for m in meta])
        kp = k_phi[mask]; kp = kp[np.isfinite(kp)]
        kq = k_psi[mask]; kq = kq[np.isfinite(kq)]
        if len(kp) < 10: continue
        print(f"  {nr:>5}  {len(kp):>5}  {np.mean(np.abs(kp)):>8.3f}  "
              f"{np.mean(np.abs(kq)):>8.3f}  {ex.get(nr,''):>20}")

    # ── Test 3: k by amino acid ───────────────────────────────────────────────
    print(f"\n  k_eff by amino acid:")
    print(f"  {'AA':>3}  {'n':>5}  {'⟨k_φ⟩':>8}  {'⟨k_ψ⟩':>8}  {'n_rot':>5}")
    for aa in ['GLY','ALA','PRO','VAL','LEU','ILE','PHE','TRP','LYS','ARG','GLU']:
        mask = np.array([m['res_name']==aa for m in meta])
        kp = k_phi[mask]; kp = kp[np.isfinite(kp)]
        kq = k_psi[mask]; kq = kq[np.isfinite(kq)]
        if len(kp) < 5: continue
        print(f"  {aa:>3}  {len(kp):>5}  {np.mean(kp):>+8.3f}  "
              f"{np.mean(kq):>+8.3f}  {_SC_N_ROT.get(aa,0):>5}")

    # ── Test 4: k by basin × amino acid (subgroup analysis) ──────────────────
    print(f"\n  k_eff by basin × amino acid (does same AA have different k in different basins?):")
    test_aas = ['GLY','ALA','LEU','GLU','LYS','ARG','PRO']
    for aa in test_aas:
        aa_mask = np.array([m['res_name']==aa for m in meta])
        n_total = np.sum(aa_mask & np.isfinite(k_phi))
        if n_total < 20: continue
        print(f"\n  {aa} (n_rot={_SC_N_ROT.get(aa,0)}):")
        print(f"    {'Basin':6}  {'n':>5}  {'⟨k_φ⟩':>8}  {'σ_φ':>7}  {'⟨k_ψ⟩':>8}  {'σ_ψ':>7}")
        for b in sorted(BASIN_NAMES.keys()):
            mask = np.array([m['res_name']==aa and m['basin']==b for m in meta])
            kp = k_phi[mask]; kp = kp[np.isfinite(kp)]
            kq = k_psi[mask]; kq = kq[np.isfinite(kq)]
            if len(kp) < 5: continue
            print(f"    {BASIN_NAMES[b]:6}  {len(kp):>5}  {np.mean(kp):>+8.3f}  "
                  f"{np.std(kp):>7.3f}  {np.mean(kq):>+8.3f}  {np.std(kq):>7.3f}")

    # ── Test 5: Chemistry group analysis ──────────────────────────────────────
    chem_groups = {
        'nonpolar':  ['ALA','VAL','LEU','ILE','MET','PHE','TRP','PRO'],
        'polar':     ['SER','THR','ASN','GLN','CYS','TYR'],
        'charged+':  ['LYS','ARG','HIS'],
        'charged-':  ['ASP','GLU'],
        'special':   ['GLY'],
    }
    print(f"\n  k_eff by chemistry group:")
    print(f"  {'Group':10}  {'n':>6}  {'⟨k_φ⟩':>8}  {'⟨k_ψ⟩':>8}  {'⟨|k_φ|⟩':>8}  {'⟨|k_ψ|⟩':>8}")
    for gname, aas in chem_groups.items():
        mask = np.array([m['res_name'] in aas for m in meta])
        kp = k_phi[mask]; kp = kp[np.isfinite(kp)]
        kq = k_psi[mask]; kq = kq[np.isfinite(kq)]
        if len(kp) < 10: continue
        print(f"  {gname:10}  {len(kp):>6}  {np.mean(kp):>+8.3f}  {np.mean(kq):>+8.3f}  "
              f"{np.mean(np.abs(kp)):>8.3f}  {np.mean(np.abs(kq)):>8.3f}")

    # ── Plot 1: k_eff heatmap on Ramachandran ─────────────────────────────────
    phi_all = np.array([m['phi'] for m in meta])
    psi_all = np.array([m['psi'] for m in meta])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Resistance map: k_eff = −τ_env / Δθ\n'
                 'Red = stiff (resists displacement) | Blue = compliant', fontsize=12)

    for ax, k_arr, label in [(axes[0], k_phi, 'k_φ'), (axes[1], k_psi, 'k_ψ')]:
        bs = 10
        pe = np.arange(-180, 181, bs); qe = np.arange(-180, 181, bs)
        k_map = np.full((len(pe)-1, len(qe)-1), np.nan)
        for i in range(len(pe)-1):
            for j in range(len(qe)-1):
                mask = ((phi_all>=pe[i])&(phi_all<pe[i+1])&
                        (psi_all>=qe[j])&(psi_all<qe[j+1]))
                kv = k_arr[mask]; kv = kv[np.isfinite(kv)]
                if len(kv) >= 3: k_map[i,j] = np.median(kv)
        valid = k_map[np.isfinite(k_map)]
        if len(valid) == 0: continue
        vmax = np.percentile(np.abs(valid), 95)
        im = ax.imshow(k_map.T, origin='lower', aspect='auto',
                       extent=[-180,180,-180,180], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label=f'{label} (kcal/mol/rad²)', shrink=0.8)
        ax.set_xlabel('φ (°)'); ax.set_ylabel('ψ (°)'); ax.set_title(f'{label} resistance map')
        for b,(pc,qc) in BASIN_CENTRES.items():
            ax.plot(pc,qc,'ko',ms=6,zorder=5)
            ax.text(pc+3,qc+3,BASIN_NAMES[b],fontsize=7,fontweight='bold',zorder=6)
    plt.tight_layout()
    p = out / 'spring_constant_map.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Fig 8: Spring constant map → {p}")

    # ── Plot 2: |k| vs n_rotatable ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('2-spring test: does sidechain flexibility reduce backbone stiffness?\n'
                 'If yes → bars decrease left to right', fontsize=11)
    for ax, k_arr, label in [(axes[0], k_phi, 'k_φ'), (axes[1], k_psi, 'k_ψ')]:
        means=[]; stds=[]; ns=[]
        for nr in range(5):
            mask = np.array([m['n_rot']==nr for m in meta])
            k = k_arr[mask]; k = k[np.isfinite(k)]
            if len(k)<10: continue
            means.append(np.mean(np.abs(k))); stds.append(np.std(np.abs(k))/np.sqrt(len(k))); ns.append(nr)
        ax.bar(ns, means, yerr=stds, color='#378ADD', alpha=0.8, capsize=4)
        ax.set_xlabel('Rotatable bonds (χ joints)')
        ax.set_ylabel(f'⟨|{label}|⟩')
        ax.set_title(f'{label}')
        ax.set_xticks(range(5))
        ax.set_xticklabels(['0\nGLY/ALA','1\nVAL/SER','2\nLEU/PHE','3\nMET/GLU','4\nLYS/ARG'],fontsize=8)
    plt.tight_layout()
    p = out / 'spring_vs_flexibility.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Fig 9: Spring vs flexibility → {p}")

    return meta


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Combined Ramachandran coverage + force-basin analysis')
    ap.add_argument('--csv',      required=True, help='features CSV (v2 or v3)')
    ap.add_argument('--out_dir',  default='./combined_results')
    ap.add_argument('--target',   type=float, default=0.95,
                    help='Target coverage for new states (default 0.95)')
    ap.add_argument('--max_new',  type=int, default=8,
                    help='Max new backbone states (default 8)')
    ap.add_argument('--max_rows', type=int, default=None,
                    help='Limit rows for fast testing')
    args = ap.parse_args()

    try:
        from scipy.ndimage import uniform_filter  # noqa
        from sklearn.linear_model import Ridge     # noqa
    except ImportError:
        print("pip install scipy scikit-learn matplotlib"); sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Loading {args.csv} ...")
    records, X_full, X_tau, yp, yq, y_sincos, ss, feat_names = load_data(
        args.csv, max_rows=args.max_rows)

    # ── Part 1: Coverage ──────────────────────────────────────────────────────
    #print(f"\n{'='*65}")
    #print(f"Part 1: Ramachandran coverage (target {args.target:.0%}) ...")
    #new_states, initial_cov, final_cov = find_new_states(
    #    records, target=args.target, max_new=args.max_new)

    # ── Part 2: Torque R² (Δ reference vs sin/cos) ───────────────────────────
    #print(f"\n{'='*65}")
    #print(f"Part 2: Torque → angle R² (Δ-reference vs sin/cos encoding) ...")
    #torque_r2 = compute_torque_r2(X_tau, X_full, yp, yq, y_sincos=y_sincos)

    # ── Part 2.5: Layer cascade ───────────────────────────────────────────────
    #print(f"\n{'='*65}")
    #print(f"Part 2.5: Layer cascade (steric → +forces → +context) ...")
    #cascade = run_cascade(X_full, yp, yq, y_sincos, feat_names)

    # ── Part 3: Non-linear comparison ─────────────────────────────────────────
    #print(f"\n{'='*65}")
    #print(f"Part 3: Non-linear model comparison ...")
    #model_comparison = run_nonlinear_comparison(X_full, yp, yq, ss, feat_names)

    # # ── Part 4: Basin coverage ────────────────────────────────────────────────
    # print(f"\n{'='*65}")
    # print(f"Part 4: Force-predicted basin coverage ...")
    # coverage_results, baseline = compute_basin_coverage(records, model_comparison)

    # # ── Plots ─────────────────────────────────────────────────────────────────
    # print(f"\n{'='*65}")
    # print(f"Generating plots ...")
    # plot_all(records, new_states, initial_cov, final_cov,
    #          torque_r2, model_comparison, coverage_results, baseline,
    #          feat_names, out_dir)

    # ── Part 6: Spring constant analysis ──────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Part 6: Two-spring + joint model — k_eff estimation ...")
    spring_meta = run_spring_analysis(records, out_dir)

    # # ── Report ────────────────────────────────────────────────────────────────
    # write_report(records, new_states, initial_cov, final_cov,
    #              torque_r2, model_comparison, coverage_results, baseline,
    #              feat_names, out_dir)

    # # ── Summary ───────────────────────────────────────────────────────────────
    # best = max(model_comparison, key=lambda n: model_comparison[n]['cv_psi'].mean())
    # best_cr = coverage_results[best]
    # tau_r = torque_r2['τ_net only (2 feat)']
    # full_r = torque_r2['Full model (Δ ref)']

    # print(f"\n{'='*65}")
    # print(f"  COVERAGE:  {initial_cov:.1%} → {final_cov:.1%}  "
    #       f"(+{len(new_states)} states)")
    # print(f"")
    # print(f"  NET TORQUE ONLY:   φ R²={tau_r['r2_phi']:.3f}  "
    #       f"ψ R²={tau_r['r2_psi']:.3f}")
    # print(f"  FULL LINEAR:       φ R²={full_r['r2_phi']:.3f}  "
    #       f"ψ R²={full_r['r2_psi']:.3f}")
    # print(f"  BEST NON-LINEAR:   φ R²={model_comparison[best]['r2_phi']:.3f}  "
    #       f"ψ R²={model_comparison[best]['r2_psi']:.3f}  ({best})")
    # print(f"  BASIN ACCURACY:    {best_cr['basin_accuracy']:.1%}")
    # print(f"{'='*65}\n")

    from ff_correction import run_ff_correction

    # After Part 6:
    delta_phi, delta_psi, centres = run_ff_correction(spring_meta, out_dir)

if __name__ == '__main__':
    main()