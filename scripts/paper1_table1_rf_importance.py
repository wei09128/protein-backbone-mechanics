"""
paper1_table1_rf_importance.py — RF feature importance (Fig 3 data)
================================================================================
RF's CV run took only 7 min, so this refits once (not 5-fold CV, just a
single full-data fit per stage) purely to extract feature importances for
Fig 3 -- consistent with the corrected 58-feature set (tau_phi_lj/psi_lj
excluded) and the corrected clash/mindist features.

Usage:
  python paper1_table1_rf_importance.py --cache table1_cache.npz
"""

import argparse
import numpy as np

try:
    import cupy as cp
    from cuml.ensemble import RandomForestRegressor as cuRF
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    from sklearn.ensemble import RandomForestRegressor as skRF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='table1_cache.npz')
    args = ap.parse_args()

    print(f"GPU backend: {'ENABLED (cuML)' if GPU_AVAILABLE else 'CPU sklearn fallback'}")
    d = np.load(args.cache)
    X_full_sc = d['X_full_sc'].astype(np.float32)
    y_phi = d['y_phi'].astype(np.float32)
    y_psi = d['y_psi'].astype(np.float32)
    feat_names = list(d['feat_names'])
    print(f"  {len(y_phi):,} residues, {len(feat_names)} features")

    def make_rf(seed):
        if GPU_AVAILABLE:
            return cuRF(n_estimators=200, max_depth=12, min_samples_leaf=5,
                        n_bins=128, random_state=seed)
        return skRF(n_estimators=200, max_depth=12, min_samples_leaf=5,
                    random_state=seed, n_jobs=-1)

    print("Fitting RF for phi...")
    rf_phi = make_rf(0)
    if GPU_AVAILABLE:
        rf_phi.fit(cp.asarray(X_full_sc), cp.asarray(y_phi))
        phi_pred_train = cp.asnumpy(rf_phi.predict(cp.asarray(X_full_sc)))
    else:
        rf_phi.fit(X_full_sc, y_phi)
        phi_pred_train = rf_phi.predict(X_full_sc)

    print("Fitting RF for psi (with phi prediction appended)...")
    X_aug = np.column_stack([X_full_sc, phi_pred_train]).astype(np.float32)
    rf_psi = make_rf(0)
    if GPU_AVAILABLE:
        rf_psi.fit(cp.asarray(X_aug), cp.asarray(y_psi))
    else:
        rf_psi.fit(X_aug, y_psi)

    imp_phi = np.array(rf_phi.feature_importances_)
    imp_psi = np.array(rf_psi.feature_importances_)[:-1]

    import pandas as pd
    phi_series = pd.Series(imp_phi, index=feat_names).sort_values(ascending=False)
    psi_series = pd.Series(imp_psi, index=feat_names).sort_values(ascending=False)

    print("\n=== Top 10 features for phi ===")
    print(phi_series.head(10).to_string())
    print("\n=== Top 10 features for psi ===")
    print(psi_series.head(10).to_string())

    phi_series.to_csv('rf_importance_phi.csv', header=['importance'])
    psi_series.to_csv('rf_importance_psi.csv', header=['importance'])
    print("\nSaved rf_importance_phi.csv and rf_importance_psi.csv")


if __name__ == '__main__':
    main()
