"""
paper1_table1_ridge.py — Ridge models only (env torque, D-ref, sin/cos)
================================================================================
Fast (no GPU needed) -- run in a 4th terminal alongside RF/GB/MLP.

Usage:
  python paper1_table1_ridge.py --cache table1_cache.npz
"""

import argparse
import time
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.linear_model import Ridge


def circular_r2(y_true_deg, y_pred_deg):
    rad_diff = np.radians(y_true_deg - y_pred_deg)
    res_var = np.mean(1.0 - np.cos(rad_diff))
    true_rad = np.radians(y_true_deg)
    mean_sin = np.mean(np.sin(true_rad)); mean_cos = np.mean(np.cos(true_rad))
    mean_angle = np.arctan2(mean_sin, mean_cos)
    tot_var = np.mean(1.0 - np.cos(true_rad - mean_angle))
    return 1.0 - (res_var / tot_var) if tot_var > 1e-9 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='table1_cache.npz')
    ap.add_argument('--n_splits', type=int, default=5)
    args = ap.parse_args()

    print(f"Loading {args.cache}...")
    d = np.load(args.cache)
    X_full_sc = d['X_full_sc']; X_tau_sc = d['X_tau_sc']
    y_phi = d['y_phi']; y_psi = d['y_psi']
    y_sin_phi = d['y_sin_phi']; y_cos_phi = d['y_cos_phi']
    y_sin_psi = d['y_sin_psi']; y_cos_psi = d['y_cos_psi']
    phi = d['phi']; psi = d['psi']; groups = d['groups']
    print(f"  {len(y_phi):,} residues, {len(np.unique(groups)):,} groups, "
          f"{X_full_sc.shape[1]} features")

    cv = GroupKFold(n_splits=args.n_splits)
    ridge = Ridge(alpha=1.0)
    t0 = time.time()
    lines = []

    print("Environmental torque only...")
    phi_oof_tau = cross_val_predict(ridge, X_tau_sc, y_phi, cv=cv, groups=groups)
    cv_phi_tau = 1.0 - (np.mean((y_phi - phi_oof_tau)**2) / np.var(y_phi))
    X_tau_aug = np.column_stack([X_tau_sc, phi_oof_tau])
    psi_oof_tau = cross_val_predict(ridge, X_tau_aug, y_psi, cv=cv, groups=groups)
    cv_psi_tau = 1.0 - (np.mean((y_psi - psi_oof_tau)**2) / np.var(y_psi))
    lines.append(f"Environmental torque only (2 feat)\t{cv_phi_tau:.3f}\t{cv_psi_tau:.3f}")
    print(f"  CV_phi={cv_phi_tau:.3f}  CV_psi={cv_psi_tau:.3f}")

    print("Linear Ridge (full feat, D-ref)...")
    phi_oof_full = cross_val_predict(ridge, X_full_sc, y_phi, cv=cv, groups=groups)
    cv_phi_full = 1.0 - (np.mean((y_phi - phi_oof_full)**2) / np.var(y_phi))
    X_full_aug = np.column_stack([X_full_sc, phi_oof_full])
    psi_oof_full = cross_val_predict(ridge, X_full_aug, y_psi, cv=cv, groups=groups)
    cv_psi_full = 1.0 - (np.mean((y_psi - psi_oof_full)**2) / np.var(y_psi))
    lines.append(f"Linear Ridge (full feat, D-ref)\t{cv_phi_full:.3f}\t{cv_psi_full:.3f}")
    print(f"  CV_phi={cv_phi_full:.3f}  CV_psi={cv_psi_full:.3f}")

    print("Linear Ridge (full feat, sin/cos)...")
    oof_sin_phi = cross_val_predict(ridge, X_full_sc, y_sin_phi, cv=cv, groups=groups)
    oof_cos_phi = cross_val_predict(ridge, X_full_sc, y_cos_phi, cv=cv, groups=groups)
    phi_pred_deg = np.degrees(np.arctan2(oof_sin_phi, oof_cos_phi))
    cv_phi_sincos = circular_r2(phi, phi_pred_deg)
    X_sincos_aug = np.column_stack([X_full_sc, oof_sin_phi, oof_cos_phi])
    oof_sin_psi = cross_val_predict(ridge, X_sincos_aug, y_sin_psi, cv=cv, groups=groups)
    oof_cos_psi = cross_val_predict(ridge, X_sincos_aug, y_cos_psi, cv=cv, groups=groups)
    psi_pred_deg = np.degrees(np.arctan2(oof_sin_psi, oof_cos_psi))
    cv_psi_sincos = circular_r2(psi, psi_pred_deg)
    lines.append(f"Linear Ridge (full feat, sin/cos)\t{cv_phi_sincos:.3f}\t{cv_psi_sincos:.3f}")
    print(f"  CV_phi={cv_phi_sincos:.3f}  CV_psi={cv_psi_sincos:.3f}")

    elapsed = time.time() - t0
    with open('table1_result_ridge.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nDone in {elapsed/60:.1f} min. Saved table1_result_ridge.txt")


if __name__ == '__main__':
    main()
