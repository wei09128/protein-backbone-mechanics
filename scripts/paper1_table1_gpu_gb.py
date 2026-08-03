"""
paper1_table1_gpu_gb.py — Gradient Boosting on GPU (XGBoost), shared cache
================================================================================
cuML has no native Gradient Boosting regressor; XGBoost's GPU support
(tree_method='hist', device='cuda') is the standard GPU-accelerated GBM
in a RAPIDS environment and matches sklearn's GradientBoostingRegressor
closely in behavior. Falls back to sklearn CPU if xgboost is unavailable
or no GPU is detected.

Usage:
  python paper1_table1_gpu_gb.py --cache table1_cache.npz
"""

import argparse
import time
import numpy as np
from sklearn.model_selection import GroupKFold

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    from sklearn.ensemble import GradientBoostingRegressor


def make_model(use_gpu):
    if XGB_AVAILABLE:
        params = dict(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=0)
        if use_gpu:
            params.update(tree_method='hist', device='cuda')
        return xgb.XGBRegressor(**params)
    return GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=0)


def gpu_probe():
    if not XGB_AVAILABLE:
        return False
    # XGBoost does NOT raise when device='cuda' is requested but no GPU is
    # visible -- it emits a UserWarning ("No visible GPU is found...") and
    # silently falls back to CPU internally. A bare try/except therefore
    # always "succeeds" and reports GPU=True even on a CPU-only machine.
    # We must inspect the warnings to detect the silent fallback.
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            test = xgb.XGBRegressor(n_estimators=2, tree_method='hist', device='cuda')
            test.fit(np.zeros((10, 2), dtype=np.float32), np.zeros(10, dtype=np.float32))
        except Exception:
            return False
    for w in caught:
        if 'no visible gpu' in str(w.message).lower() or 'changed from gpu to cpu' in str(w.message).lower():
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='table1_cache.npz')
    ap.add_argument('--n_splits', type=int, default=5)
    args = ap.parse_args()

    use_gpu = gpu_probe()
    backend = ('XGBoost GPU' if use_gpu else
               'XGBoost CPU' if XGB_AVAILABLE else
               'sklearn CPU fallback')
    print(f"Backend: {backend}")

    print(f"Loading {args.cache}...")
    d = np.load(args.cache)
    X_full_sc = d['X_full_sc'].astype(np.float32)
    y_phi = d['y_phi'].astype(np.float32)
    y_psi = d['y_psi'].astype(np.float32)
    groups = d['groups']
    print(f"  {len(y_phi):,} residues, {len(np.unique(groups)):,} groups, "
          f"{X_full_sc.shape[1]} features")

    cv = GroupKFold(n_splits=args.n_splits)
    splits = list(cv.split(X_full_sc, y_phi, groups=groups))

    def cv_predict(X, y):
        oof = np.zeros(len(y), dtype=np.float32)
        for train_idx, test_idx in splits:
            model = make_model(use_gpu)
            model.fit(X[train_idx], y[train_idx])
            oof[test_idx] = model.predict(X[test_idx])
        return oof

    t0 = time.time()
    print("Stage 1: predicting phi...")
    phi_oof = cv_predict(X_full_sc, y_phi)
    cv_phi = 1.0 - (np.mean((y_phi - phi_oof) ** 2) / np.var(y_phi))
    print(f"  CV_phi = {cv_phi:.3f}  ({(time.time()-t0)/60:.1f} min elapsed)")

    print("Stage 2: predicting psi...")
    X_aug = np.column_stack([X_full_sc, phi_oof]).astype(np.float32)
    psi_oof = cv_predict(X_aug, y_psi)
    cv_psi = 1.0 - (np.mean((y_psi - psi_oof) ** 2) / np.var(y_psi))

    elapsed = time.time() - t0
    line = f"Gradient Boosting (full feat, D-ref)\t{cv_phi:.3f}\t{cv_psi:.3f}"
    with open('table1_result_gb.txt', 'w') as f:
        f.write(line + '\n')
    print(f"\nDone in {elapsed/60:.1f} min. {line}")
    print("Saved table1_result_gb.txt")


if __name__ == '__main__':
    main()
