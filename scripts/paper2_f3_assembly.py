"""
paper2_F3_assembly.py — Figure 3: the non-local channel hierarchy
===================================================================

CHANGES from original:
  1. effect_size() now also returns a bootstrap standard error, so
     Panel B bars can show error bars (previously no uncertainty was
     computed at all for these quartile-effect-size values).
  2. Panel B bars now plotted with xerr = bootstrap SE.
  3. fig.suptitle(...) removed.
  4. Row gap (hspace) narrowed between row 0 (A,B) and row 1 (C,D),
     and `top` raised slightly to reclaim the space freed by dropping
     the suptitle.
  5. Local feature importances (model.feature_importances_) are now
     extracted and printed per angle, so claims like "psi dominant
     (importance 0.79)" in the manuscript text are backed by a real,
     reproducible number from this script rather than an unverified
     figure. Not currently plotted as a panel -- console/CSV output
     only, since Fig. 3/4 as designed has no local-importance panel.

Four panels showing that backbone bond angles couple to the non-local
environment to different degrees:

  Panel A: Local R² bar chart — which angles are local-dominated?
  Panel B: Top non-local feature effect sizes (2x2 mini-grid), now with
           bootstrap SE error bars
  Panel C: τ residual vs hb_n_strong (residual-tracker) — the non-local
           channel visualized
  Panel D: ω_dev residual vs steric_CA_5A (residual-tracker) — the
           "ω is environment-driven" story

Re-fits the GBR models from paper2_07 to get indexed residuals. Adds the
trans-only + omega_dev transform for ω (same as paper2_07 patched).

Usage
-----
    python paper2_F3_assembly_v2.py --csv features.csv
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr


_LOCAL_FEATURES = [
    'phi_deg', 'psi_deg', 'res_type',
    'chi1_rad', 'chi2_rad',
    'sc_mass', 'sc_n_heavy',
    'sc_is_branched', 'sc_is_aromatic',
    'has_chi1', 'has_chi2',
]

_NONLOCAL_FEATURES = [
    'steric_CA_5A', 'steric_N_5A', 'steric_C_5A', 'steric_O_5A',
    'steric_asym_x', 'steric_asym_y', 'steric_asym_z',
    'hb_n_bonds', 'hb_n_strong', 'hb_best_e',
    'hb_mean_d_HO', 'hb_best_d_HO',
    'sc_contact_nm1_to_bb', 'sc_contact_np1_to_bb',
    'bfactor_ca',
]

_N_BOOTSTRAP = 200  # resamples for effect-size SE


def fit_and_residuals(df, target, features, random_state=42):
    """Fit GBR, return model, R², and residuals Series indexed to df."""
    d = df.dropna(subset=features + [target]).copy()
    X = d[features].values
    y = d[target].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=random_state)

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=random_state)
    model.fit(X_tr, y_tr)

    pred_te = model.predict(X_te)
    ss_res = float(np.sum((y_te - pred_te) ** 2))
    ss_tot = float(np.sum((y_te - np.mean(y_te)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    pred_full = model.predict(X)
    residuals = pd.Series(y - pred_full, index=d.index, name='residual')
    return model, r2, residuals, d


def _quartile_effect(x, y, residual_std):
    """Point estimate: quartile-mean spread of y, in units of residual_std."""
    q = pd.qcut(x, 4, labels=False, duplicates='drop')
    grp_means = pd.Series(y).groupby(q).mean()
    return float(grp_means.max() - grp_means.min()) / residual_std


def effect_size(x, y, residual_std, n_boot=_N_BOOTSTRAP, seed=42):
    """
    Quartile spread of mean y across quartiles of x, in units of residual_std,
    plus a bootstrap standard error obtained by resampling (x, y) pairs with
    replacement and recomputing the quartile-mean spread each time.

    Returns (effect, se). se is np.nan if the point estimate itself fails.
    """
    try:
        point = _quartile_effect(x, y, residual_std)
    except Exception:
        return float('nan'), float('nan')

    rng = np.random.default_rng(seed)
    n = len(x)
    boot_vals = np.empty(n_boot)
    ok = 0
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boot_vals[ok] = _quartile_effect(x[idx], y[idx], residual_std)
            ok += 1
        except Exception:
            continue
    if ok < max(20, n_boot // 4):
        return point, float('nan')
    se = float(np.std(boot_vals[:ok], ddof=1))
    return point, se


def compute_nonlocal_effects(d, residuals, feats):
    """Return a DataFrame with r, effect, and effect_se for each non-local feature."""
    res_std = float(residuals.std())
    rows = []
    for feat in feats:
        if feat not in d.columns:
            continue
        sub = d[feat].dropna()
        common = residuals.index.intersection(sub.index)
        if len(common) < 1000:
            continue
        x = sub.loc[common].values
        y = residuals.loc[common].values
        r, p = pearsonr(x, y)
        eff, eff_se = effect_size(x, y, res_std)
        rows.append(dict(feature=feat, r=r, p=p, effect=eff, effect_se=eff_se,
                          n=len(common)))
    return pd.DataFrame(rows)


def tracker_plot(ax, x_values, residuals, feat_name, n_bins=10,
                  title='', color='#2c3e50'):
    """Bin x_values into quantiles and plot mean residual ± 95% CI per bin."""
    mask = np.isfinite(x_values) & np.isfinite(residuals)
    x = np.asarray(x_values)[mask]
    y = np.asarray(residuals)[mask]

    try:
        q_edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        q_edges = np.unique(q_edges)
    except Exception:
        q_edges = np.linspace(x.min(), x.max(), n_bins + 1)

    centers, means, sems, counts = [], [], [], []
    for i in range(len(q_edges) - 1):
        lo, hi = q_edges[i], q_edges[i + 1]
        if i < len(q_edges) - 2:
            m = (x >= lo) & (x < hi)
        else:
            m = (x >= lo) & (x <= hi)
        if m.sum() < 30:
            continue
        centers.append(0.5 * (lo + hi))
        means.append(y[m].mean())
        sems.append(y[m].std() / np.sqrt(m.sum()))
        counts.append(int(m.sum()))

    centers = np.array(centers); means = np.array(means); sems = np.array(sems)
    ci95 = 1.96 * sems

    ax.fill_between(centers, means - ci95, means + ci95,
                     color=color, alpha=0.2, linewidth=0)
    ax.plot(centers, means, 'o-', color=color, lw=1.8, ms=6,
             markeredgecolor='white', markeredgewidth=0.8)
    ax.axhline(0, color='k', lw=0.4, alpha=0.5)
    ax.set_xlabel(feat_name, fontsize=10)
    ax.set_ylabel('Mean residual (deg)', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.bar(centers, counts,
             width=(centers[-1] - centers[0]) / len(centers) * 0.6
                   if len(centers) > 1 else 1,
             alpha=0.15, color='#888', zorder=0)
    ax2.set_ylabel('bin count', fontsize=8, color='#888')
    ax2.tick_params(axis='y', labelsize=7, colors='#888')
    ax2.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--n_sample', type=int, default=100_000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='paper2_F3_nonlocal_channel_v2.png')
    ap.add_argument('--dpi', type=int, default=220)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found"); sys.exit(1)

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    if len(df) > args.n_sample:
        df = df.sample(n=args.n_sample, random_state=args.seed)
        print(f"  sampled to {len(df):,} rows")

    available = [f for f in _LOCAL_FEATURES if f in df.columns]

    angles_spec = [
        ('tau',  'τ',        'tau_deg',            None),
        ('omega','ω',        'omega_measured_deg', 'trans_dev'),
        ('n_cb', '∠N-Cα-Cβ', 'angle_N_CA_CB',      None),
        ('c_cb', '∠C-Cα-Cβ', 'angle_C_CA_CB',      None),
    ]

    results = {}
    for key, label, target, transform in angles_spec:
        print(f"\nFitting {label} ({target}) ...")
        df_a = df.copy()
        if transform == 'trans_dev':
            df_a = df_a[np.abs(df_a[target]) > 150].copy()
            df_a['omega_dev'] = 180.0 - np.abs(df_a[target])
            eff_target = 'omega_dev'
            print(f"  trans-filtered: {len(df_a):,} residues")
        else:
            eff_target = target
        model, r2, resid, d = fit_and_residuals(df_a, eff_target, available)
        print(f"  computing non-local effects + bootstrap SE "
              f"({_N_BOOTSTRAP} resamples per feature)...")
        nl = compute_nonlocal_effects(d, resid, _NONLOCAL_FEATURES)

        # Local feature importances -- verifies/replaces claims like
        # "psi dominant (importance 0.79)" with a real number from this fit.
        local_imp = pd.Series(model.feature_importances_, index=available)
        local_imp = local_imp.sort_values(ascending=False)

        results[key] = dict(label=label, r2=r2, resid=resid, d=d,
                             nl=nl, eff_target=eff_target,
                             local_importance=local_imp)
        print(f"  R² = {r2:.4f},   max |effect| = "
              f"{nl['effect'].abs().max():.3f} SD")
        print(f"  top local features: "
              + ", ".join(f"{feat}={imp:.3f}"
                           for feat, imp in local_imp.head(3).items()))

    print(f"\nAssembling figure ...")

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(
        2, 2,
        hspace=0.20, wspace=0.28,   # narrowed from 0.42
        left=0.07, right=0.96, top=0.97, bottom=0.07,  # top raised from 0.92 (no suptitle)
    )

    # ── Panel A: Local R² bar chart (top-left) ───────────────────────────────
    axA = fig.add_subplot(gs[0, 0])
    keys = ['tau', 'omega', 'n_cb', 'c_cb']
    labels = [results[k]['label'] for k in keys]
    r2s = [results[k]['r2'] for k in keys]
    colors_r2 = ['#2a5d9f' if r > 0.3 else ('#d9a05b' if r > 0.15 else '#999')
                  for r in r2s]
    bars = axA.bar(labels, r2s, color=colors_r2, edgecolor='white', linewidth=1.2)
    axA.axhline(0.3, color='#888', lw=0.6, ls='--', alpha=0.7)
    axA.text(axA.get_xlim()[1] * 0.98, 0.305, 'R² = 0.3',
              color='#888', fontsize=9, va='bottom', ha='right')
    axA.set_ylim(0, max(0.85, max(r2s) * 1.2))
    axA.set_ylabel('Local-model R²', fontsize=11)
    for bar, r in zip(bars, r2s):
        axA.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.018,
                  f'{r:.3f}', ha='center', fontsize=10.5, fontweight='bold')
    axA.grid(True, axis='y', alpha=0.25)
    axA.tick_params(axis='x', labelsize=11)

    # ── Panel B: 2x2 top-non-local features block (top-right) ────────────────
    gs_B = gs[0, 1].subgridspec(2, 2, hspace=0.75, wspace=0.55)
    axB_parent = fig.add_subplot(gs[0, 1])
    axB_parent.set_xticks([]); axB_parent.set_yticks([])
    for spine in axB_parent.spines.values():
        spine.set_visible(False)
    axB_parent.patch.set_alpha(0)

    for idx, key in enumerate(keys):
        ax = fig.add_subplot(gs_B[idx // 2, idx % 2])
        nl = results[key]['nl'].copy()
        nl['abs_eff'] = nl['effect'].abs()
        top = nl.sort_values('abs_eff', ascending=False).head(5)
        y = np.arange(len(top))
        colors_b = ['#c0392b' if abs(e) > 0.08 else '#aaa'
                     for e in top['effect']]
        ax.barh(y, top['effect'], xerr=top['effect_se'],
                 color=colors_b, edgecolor='white', linewidth=0.8,
                 error_kw=dict(ecolor='#333', elinewidth=0.9, capsize=2.5))
        ax.set_yticks(y)
        ax.set_yticklabels(top['feature'], fontsize=8)
        ax.axvline(0, color='k', lw=0.5)
        ax.axvline(+0.08, color='#ccc', lw=0.5, ls=':')
        ax.axvline(-0.08, color='#ccc', lw=0.5, ls=':')
        ax.set_title(results[key]['label'], fontsize=10, pad=4)
        ax.invert_yaxis()
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(True, axis='x', alpha=0.2)
        if idx >= 2:
            ax.set_xlabel('effect (SD)', fontsize=9)

    # ── Panel C: τ residual vs hb_n_strong (bottom-left) ─────────────────────
    axC = fig.add_subplot(gs[1, 0])
    tau_d = results['tau']['d']
    tau_r = results['tau']['resid']
    common = tau_d.index.intersection(tau_r.index)
    tracker_plot(axC,
                  tau_d.loc[common, 'hb_n_strong'].values,
                  tau_r.loc[common].values,
                  'hb_n_strong  (number of strong backbone H-bonds)',
                  n_bins=8,
                  title='',
                  color='#2a5d9f')

    # ── Panel D: ω_dev residual vs steric_CA_5A (bottom-right) ───────────────
    axD = fig.add_subplot(gs[1, 1])
    om_d = results['omega']['d']
    om_r = results['omega']['resid']
    common = om_d.index.intersection(om_r.index)
    tracker_plot(axD,
                  om_d.loc[common, 'steric_CA_5A'].values,
                  om_r.loc[common].values,
                  'steric_CA_5A  (atoms within 5Å of Cα)',
                  n_bins=10,
                  title='',
                  color='#c0392b')

    # NOTE: fig.suptitle(...) intentionally removed per request.

    plt.savefig(args.out, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved: {args.out}  ({args.dpi} dpi)")

    print("\n" + "=" * 78)
    print("FIGURE 3 SUMMARY")
    print("=" * 78)
    for key in keys:
        r = results[key]
        top_nl = r['nl'].copy()
        top_nl['abs_eff'] = top_nl['effect'].abs()
        top_row = top_nl.sort_values('abs_eff', ascending=False).iloc[0]
        se_str = f"± {top_row['effect_se']:.3f}" if np.isfinite(top_row['effect_se']) else "(SE n/a)"
        top_local_feat = r['local_importance'].index[0]
        top_local_imp = r['local_importance'].iloc[0]
        print(f"  {r['label']:<10s}  R² = {r['r2']:.3f}  "
              f"top local: {top_local_feat}={top_local_imp:.3f}  "
              f"top non-local: {top_row['feature']:<22s} "
              f"effect = {top_row['effect']:+.3f} SD {se_str}")

    print("\n" + "-" * 78)
    print("Full non-local effects table per angle (for manuscript verification)")
    print("Bonferroni threshold = 0.01 / n_features_tested per angle")
    print("-" * 78)
    for key in keys:
        r = results[key]
        nl = r['nl'].copy()
        nl['abs_eff'] = nl['effect'].abs()
        nl = nl.sort_values('abs_eff', ascending=False)
        n_tested = len(nl)
        bonf_thresh = 0.01 / n_tested if n_tested > 0 else float('nan')
        print(f"\n  {r['label']}  (n_features_tested={n_tested}, "
              f"Bonferroni p-threshold={bonf_thresh:.2e}):")
        for _, row in nl.iterrows():
            se_str = f"{row['effect_se']:.3f}" if np.isfinite(row['effect_se']) else "n/a"
            sig = "sig." if row['p'] < bonf_thresh else "n.s."
            print(f"    {row['feature']:<22s} effect={row['effect']:+.3f} SD "
                  f"(SE={se_str})  r={row['r']:+.3f}  p={row['p']:.3g}  [{sig}]")

    print("\n" + "-" * 78)
    print("Full local feature importances per angle (for manuscript verification)")
    print("-" * 78)
    for key in keys:
        r = results[key]
        print(f"\n  {r['label']}:")
        for feat, imp in r['local_importance'].items():
            print(f"    {feat:<18s} {imp:.4f}")


if __name__ == '__main__':
    main()