#!/usr/bin/env python3
"""
paper1_table2_correct.py — Structure-level bootstrap CI for the REAL Table 2
================================================================================
Table 2 is about restoring projection (tau_proj), per Methods 4.7:
  "the restoring direction is the unit vector toward the alphaR reference...
   mean net torque is projected onto this direction; positive = restoring
   (toward alphaR), negative = driving (away from alphaR). For alphaR itself,
   the convention is inverted: positive = forces oppose displacement from
   center (restoring within well)."

This is a DIFFERENT quantity from k_phi/k_psi (Fig 5/7's secant stiffness,
computed axis-by-axis) -- tau_proj is the net (tau_phi, tau_psi) vector
projected onto a single restoring direction, per basin.

For non-alphaR basins: restoring direction is FIXED per basin (unit vector
from that basin's center to the alphaR reference).
For alphaR itself: restoring direction is PER-RESIDUE (unit vector from
that residue's own (phi,psi) back to the alphaR center) -- i.e. opposing
its own displacement from center, using wrapped angular differences to
handle the +/-180 boundary correctly.

Usage:
  python paper1_table2_correct.py --csv features_lj_v3_FINAL_CLEAN.csv
"""

import argparse
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

BASIN_CENTERS = {
    'alphaR': (-63.0, -43.0),
    'beta':   (-120.0, 128.0),
    'PPII':   (-72.0, 146.0),
    'loop':   (-95.0, 10.0),
    'alphaL': (60.0, 40.0),
}
REF_PHI, REF_PSI = -63.0, -43.0


def wrap180(x):
    return (x + 180) % 360 - 180


def assign_basin(phi, psi):
    if -100 <= phi < -30 and -70 <= psi < -15:
        return 'alphaR'
    if -100 <= phi < -30 and 45 <= psi < 180:
        return 'PPII'
    if -180 <= phi < -30 and (psi >= 45 or psi < -170):
        return 'beta'
    if 0 <= phi < 180 and -20 <= psi < 100:
        return 'alphaL'
    return 'loop'


def bootstrap_by_structure(values, pdb_ids, n_boot=2000, statistic=np.mean, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'v': values, 'pdb_id': pdb_ids})
    groups = df.groupby('pdb_id')['v'].apply(list).to_dict()
    structs = np.array(list(groups.keys()))
    n_struct = len(structs)
    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(structs, size=n_struct, replace=True)
        vals = np.concatenate([groups[s] for s in sampled])
        boot_stats[b] = statistic(vals)
    return statistic(values), np.percentile(boot_stats, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--n_boot', type=int, default=2000)
    args = ap.parse_args()

    print("Loading data...")
    df = pd.read_csv(args.csv, low_memory=False)
    df = df.dropna(subset=['phi_deg', 'psi_deg', 'tau_phi_correct', 'tau_psi_correct'])
    print(f"  {len(df):,} residues, {df['pdb_id'].nunique()} structures")

    df['basin'] = [assign_basin(p, s) for p, s in zip(df['phi_deg'], df['psi_deg'])]

    tau_proj = np.full(len(df), np.nan)
    phi = df['phi_deg'].values
    psi = df['psi_deg'].values
    tau_phi = df['tau_phi_correct'].values
    tau_psi = df['tau_psi_correct'].values
    basin_arr = df['basin'].values

    for basin, (bc_phi, bc_psi) in BASIN_CENTERS.items():
        mask = basin_arr == basin
        if basin == 'alphaR':
            d_phi = wrap180(REF_PHI - phi[mask])
            d_psi = wrap180(REF_PSI - psi[mask])
        else:
            d_phi = np.full(mask.sum(), wrap180(REF_PHI - bc_phi))
            d_psi = np.full(mask.sum(), wrap180(REF_PSI - bc_psi))
        norm = np.sqrt(d_phi**2 + d_psi**2)
        norm[norm < 1e-9] = 1e-9
        dir_phi = d_phi / norm
        dir_psi = d_psi / norm
        tau_proj[mask] = tau_phi[mask] * dir_phi + tau_psi[mask] * dir_psi

    df['tau_proj'] = tau_proj

    print(f"\n{'='*80}")
    print("  TABLE 2 (CORRECTED): restoring projection (tau_proj), bootstrap 95% CI")
    print(f"  ({args.n_boot} resamples, resampled by structure)")
    print(f"{'='*80}")
    print(f"{'Basin':8s} {'n':>10s} {'<tau_proj>':>12s} {'95% CI':>22s} {'% Rest.':>9s} {'% Driv.':>9s}")
    for basin in ['alphaR', 'beta', 'PPII', 'loop', 'alphaL']:
        sub = df[df['basin'] == basin].dropna(subset=['tau_proj'])
        vals = sub['tau_proj'].values
        ids = sub['pdb_id'].values
        if len(vals) < 10:
            continue
        mean_v, (ci_lo, ci_hi) = bootstrap_by_structure(vals, ids, args.n_boot, np.mean)
        pct_rest = 100 * (vals > 0).mean()
        pct_driv = 100 * (vals < 0).mean()
        print(f"{basin:8s} {len(vals):>10,} {mean_v:>+12.3f} "
              f"[{ci_lo:>+8.3f}, {ci_hi:>+8.3f}] {pct_rest:>8.1f}% {pct_driv:>8.1f}%")

    print(f"\n{'='*80}")
    print("Compare against the ORIGINAL Table 2 (uncorrected, no CI):")
    print("  alphaR  n=759,198  tau_proj=-0.067  43.4% rest  49.3% driv")
    print("  beta    n=453,980  tau_proj=+1.232  84.5% rest   8.1% driv")
    print("  PPII    n=202,351  tau_proj=+0.194  44.4% rest  33.4% driv")
    print("  loop    n=180,975  tau_proj=+0.639  67.4% rest  17.8% driv")
    print("  alphaL  n=66,322   tau_proj=+0.069  37.8% rest  46.1% driv")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
