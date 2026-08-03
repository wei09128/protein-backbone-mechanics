"""
paper1_table1_gpu_rf.py — Random Forest on GPU (cuML), reads the shared cache
================================================================================
Run after paper1_table1_preprocess.py has produced table1_cache.npz.
Safe to run alongside the GB and MLP GPU scripts in separate terminals --
each is a separate process and cuML/XGBoost/PyTorch will share the GPU via
its own scheduler, though if you hit memory pressure running all 3 at once,
stagger them instead.

Usage:
  python paper1_table1_gpu_rf.py --cache table1_cache.npz
"""

import argparse
import time
import numpy as np
from sklearn.model_selection import GroupKFold

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
    ap.add_argument('--n_splits', type=int, default=5)
    args = ap.parse_args()

    print(f"GPU backend: {'ENABLED (cuML)' if GPU_AVAILABLE else 'NOT FOUND -- CPU sklearn fallback'}")
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

    def make_rf(seed):
        if GPU_AVAILABLE:
            return cuRF(n_estimators=100, max_depth=12, n_bins=128, random_state=seed)
        return skRF(n_estimators=100, max_depth=12, random_state=seed, n_jobs=-1)

    def cv_predict(X, y, seed):
        oof = np.zeros(len(y), dtype=np.float32)
        for train_idx, test_idx in splits:
            model = make_rf(seed)
            if GPU_AVAILABLE:
                Xtr = cp.asarray(X[train_idx]); Xte = cp.asarray(X[test_idx])
                ytr = cp.asarray(y[train_idx])
                model.fit(Xtr, ytr)
                pred = model.predict(Xte)
                oof[test_idx] = cp.asnumpy(pred)
            else:
                model.fit(X[train_idx], y[train_idx])
                oof[test_idx] = model.predict(X[test_idx])
        return oof

    t0 = time.time()
    print("Stage 1: predicting phi...")
    phi_oof = cv_predict(X_full_sc, y_phi, seed=0)
    cv_phi = 1.0 - (np.mean((y_phi - phi_oof) ** 2) / np.var(y_phi))
    print(f"  CV_phi = {cv_phi:.3f}  ({(time.time()-t0)/60:.1f} min elapsed)")

    print("Stage 2: predicting psi (phi_oof appended as a feature)...")
    X_aug = np.column_stack([X_full_sc, phi_oof]).astype(np.float32)
    psi_oof = cv_predict(X_aug, y_psi, seed=0)
    cv_psi = 1.0 - (np.mean((y_psi - psi_oof) ** 2) / np.var(y_psi))

    elapsed = time.time() - t0
    line = f"Random Forest (full feat, D-ref)\t{cv_phi:.3f}\t{cv_psi:.3f}"
    with open('table1_result_rf.txt', 'w') as f:
        f.write(line + '\n')
    print(f"\nDone in {elapsed/60:.1f} min. {line}")
    print("Saved table1_result_rf.txt")


if __name__ == '__main__':
    main()
