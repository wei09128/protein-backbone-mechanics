#!/usr/bin/env python3
"""
residual_analysis.py — Is there real signal beyond regression-to-default?

Context: mm_local_relax.py's d_tau (tau_post - tau_pre) turned out to be
dominated by a basin-blind artifact -- residues relax toward AMBER
ff14SB's single fixed tau equilibrium (~111.1 deg) regardless of which
Ramachandran basin they're in. Confirmed by:
  - d_tau correlates with (111.1 - tau_pre) at r~0.94-0.95, slope~0.58,
    identical whether or not --mechlib is applied.
This means alpha/alphaL's "confirmed" sign agreement is largely coincidence
(their crystal tau is also usually below 111, so the pull is in the same
direction as the real basin effect), while beta/PPII's mismatch is because
the true compression is below 111 and the fixed-equilibrium pull opposes it.

This script:
  1. Fits d_tau = slope*(TAU0 - tau_pre) + intercept as a GLOBAL baseline
     (not per-basin -- the whole point is this artifact is basin-blind).
  2. Computes residual = d_tau_observed - d_tau_predicted for every residue.
  3. Reports per (klass, basin) mean +/- SEM of the RAW d_tau (what you had
     before) side by side with the RESIDUAL (what's left after removing the
     fixed-equilibrium pull) -- and re-runs the sign-agreement test against
     the crystallographic expected sign on the residual instead of raw d_tau.

If residuals in alpha/alphaL are still reliably positive and residuals in
beta/PPII are near zero or negative, that's real evidence for "other terms
partially reinforce alpha but can't overcome the angle penalty in beta"
-- a defensible number for the manuscript, unlike raw d_tau.

Usage:
    python residual_analysis.py relax_results.csv
    python residual_analysis.py relax_results.csv --tau0 111.1
    python residual_analysis.py relax_results.csv --xtal_dtau xtal_dtau.csv
"""
import argparse
import csv
import numpy as np


# Crystallographic Delta-tau (vs residue-type median), from the full
# 1.5M-residue paper2_01b run -- used only to determine the EXPECTED SIGN
# per (klass, basin) for the agreement test. Override with --xtal_dtau if
# you have a CSV (columns: klass,basin,dtau) instead, e.g. if you rerun
# paper2_01b on an updated feature CSV.
_DEFAULT_XTAL_DTAU = {
 ('ALA','alpha'):0.310, ('ALA','beta'):-1.777, ('ALA','PPII'):-0.709, ('ALA','alphaL'):0.983,
 ('ARG','alpha'):0.386, ('ARG','beta'):-1.833, ('ARG','PPII'):-0.925, ('ARG','alphaL'):1.056,
 ('ASN','alpha'):0.331, ('ASN','beta'):-2.393, ('ASN','PPII'):-1.370, ('ASN','alphaL'):0.963,
 ('ASP','alpha'):0.382, ('ASP','beta'):-2.505, ('ASP','PPII'):-1.246, ('ASP','alphaL'):0.984,
 ('CYS','alpha'):0.658, ('CYS','beta'):-1.365, ('CYS','PPII'):-0.444, ('CYS','alphaL'):1.412,
 ('GLN','alpha'):0.312, ('GLN','beta'):-1.817, ('GLN','PPII'):-0.972, ('GLN','alphaL'):0.890,
 ('GLU','alpha'):0.286, ('GLU','beta'):-1.994, ('GLU','PPII'):-1.118, ('GLU','alphaL'):0.996,
 ('HIS','alpha'):0.503, ('HIS','beta'):-1.779, ('HIS','PPII'):-0.999, ('HIS','alphaL'):1.228,
 ('ILE','alpha'):0.914, ('ILE','beta'):-1.450, ('ILE','PPII'):-0.616, ('ILE','alphaL'):1.447,
 ('LEU','alpha'):0.447, ('LEU','beta'):-1.884, ('LEU','PPII'):-0.823, ('LEU','alphaL'):1.093,
 ('LYS','alpha'):0.372, ('LYS','beta'):-1.853, ('LYS','PPII'):-1.049, ('LYS','alphaL'):0.869,
 ('MET','alpha'):0.434, ('MET','beta'):-1.653, ('MET','PPII'):-0.725, ('MET','alphaL'):0.943,
 ('PHE','alpha'):0.563, ('PHE','beta'):-1.354, ('PHE','PPII'):-0.642, ('PHE','alphaL'):1.615,
 ('SER','alpha'):0.552, ('SER','beta'):-1.539, ('SER','PPII'):-0.578, ('SER','alphaL'):1.179,
 ('THR','alpha'):0.553, ('THR','beta'):-1.414, ('THR','PPII'):-0.717, ('THR','alphaL'):1.588,
 ('TRP','alpha'):0.599, ('TRP','beta'):-1.499, ('TRP','PPII'):-0.720, ('TRP','alphaL'):1.396,
 ('TYR','alpha'):0.506, ('TYR','beta'):-1.304, ('TYR','PPII'):-0.665, ('TYR','alphaL'):1.396,
 ('VAL','alpha'):1.027, ('VAL','beta'):-1.171, ('VAL','PPII'):-0.442, ('VAL','alphaL'):1.053,
}


def load_xtal_dtau(path):
    d = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            d[(row['klass'], row['basin'])] = float(row['dtau'])
    return d


def mean_sem(vals):
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    if n == 0:
        return float('nan'), float('nan'), 0
    if n == 1:
        return float(vals[0]), float('nan'), 1
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(n)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results_csv', help='output of mm_local_relax.py '
                     '(needs columns: klass, basin, tau_pre, d_tau)')
    ap.add_argument('--tau0', type=float, default=None,
                     help='fixed reference equilibrium in degrees (default: '
                          'fit it jointly with the slope, rather than '
                          'assuming 111.1 -- see note below)')
    ap.add_argument('--xtal_dtau', default=None,
                     help='CSV with columns klass,basin,dtau -- overrides '
                          'the built-in crystallographic reference table')
    ap.add_argument('--min_n', type=int, default=5,
                     help='minimum residues per (klass,basin) cell to report '
                          '(default 5 -- lower than usual since this is '
                          'meant to work on small pilot batches too)')
    args = ap.parse_args()

    xtal_dtau = load_xtal_dtau(args.xtal_dtau) if args.xtal_dtau else _DEFAULT_XTAL_DTAU

    rows = []
    with open(args.results_csv) as f:
        for row in csv.DictReader(f):
            try:
                rows.append(dict(
                    klass=row['klass'], basin=row['basin'],
                    tau_pre=float(row['tau_pre']), d_tau=float(row['d_tau'])))
            except (KeyError, ValueError):
                continue
    print(f"Loaded {len(rows)} usable rows from {args.results_csv}")

    tau_pre = np.array([r['tau_pre'] for r in rows])
    d_tau = np.array([r['d_tau'] for r in rows])

    # ── Fit the global regression-to-fixed-equilibrium baseline ─────────────
    if args.tau0 is not None:
        # Fix tau0, fit slope+intercept: d_tau = slope*(tau0-tau_pre)+intercept
        x = args.tau0 - tau_pre
        slope, intercept = np.polyfit(x, d_tau, 1)
        tau0 = args.tau0
    else:
        # Fit d_tau = m*tau_pre + c directly, then re-express as slope/tau0:
        # choose tau0 as this line's root (d_tau=0), so intercept is 0 by
        # construction when expressed in terms of x=(tau0-tau_pre) -- refit
        # explicitly below anyway so both branches use the same code path.
        m, c = np.polyfit(tau_pre, d_tau, 1)
        slope_guess = -m
        tau0 = -c / m if m != 0 else float('nan')
        x = tau0 - tau_pre
        slope, intercept = np.polyfit(x, d_tau, 1)

    d_tau_pred = slope * (tau0 - tau_pre) + intercept
    residual = d_tau - d_tau_pred
    r = np.corrcoef(tau0 - tau_pre, d_tau)[0, 1]

    print(f"\nFitted baseline: d_tau_predicted = {slope:.3f} * ({tau0:.2f} - tau_pre) "
          f"+ ({intercept:+.3f})")
    print(f"  (target tau0 = {tau0:.2f} deg, slope = {slope:.3f}, "
          f"intercept = {intercept:+.3f}, r = {r:.3f})")
    print(f"  Interpretation: {100*abs(slope):.0f}% of the gap between each "
          f"residue's crystal tau and {tau0:.1f} deg closes during relaxation, "
          f"regardless of basin -- that's the artifact this analysis removes.")

    for i, r_ in enumerate(rows):
        r_['residual'] = float(residual[i])
        r_['d_tau_pred'] = float(d_tau_pred[i])

    # ── Per (klass, basin) table: raw d_tau vs residual, both vs xtal sign ──
    cells = {}
    for r_ in rows:
        cells.setdefault((r_['klass'], r_['basin']), []).append(r_)

    print("\n" + "=" * 100)
    print(f"{'klass':<6}{'basin':<8}{'n':>4}  "
          f"{'raw d_tau':>16}  {'residual':>16}  {'xtal Δτ':>9}  "
          f"{'raw sign':>9}  {'resid sign':>11}")
    print("-" * 100)

    raw_agree = raw_total = 0
    resid_agree = resid_total = 0

    for (klass, basin), items in sorted(cells.items()):
        if len(items) < args.min_n:
            continue
        raw_vals = [it['d_tau'] for it in items]
        resid_vals = [it['residual'] for it in items]
        raw_m, raw_sem, n = mean_sem(raw_vals)
        res_m, res_sem, _ = mean_sem(resid_vals)
        xtal = xtal_dtau.get((klass, basin), float('nan'))

        raw_sign_ok = ''
        resid_sign_ok = ''
        if not np.isnan(xtal):
            raw_total += 1
            resid_total += 1
            if np.sign(raw_m) == np.sign(xtal):
                raw_agree += 1
                raw_sign_ok = 'match'
            else:
                raw_sign_ok = 'MISS'
            if np.sign(res_m) == np.sign(xtal):
                resid_agree += 1
                resid_sign_ok = 'match'
            else:
                resid_sign_ok = 'MISS'

        print(f"{klass:<6}{basin:<8}{n:>4}  "
              f"{raw_m:+7.3f}±{raw_sem:5.3f}  "
              f"{res_m:+7.3f}±{res_sem:5.3f}  "
              f"{xtal:+8.3f}  {raw_sign_ok:>9}  {resid_sign_ok:>11}")

    print("-" * 100)
    if raw_total:
        print(f"\nSign agreement vs crystallographic direction:")
        print(f"  Using RAW d_tau:  {raw_agree}/{raw_total} "
              f"({100*raw_agree/raw_total:.0f}%)")
        print(f"  Using RESIDUAL:   {resid_agree}/{resid_total} "
              f"({100*resid_agree/resid_total:.0f}%)")
        print()
        if resid_agree > raw_agree:
            print("  -> Removing the fixed-equilibrium regression artifact IMPROVED "
                  "agreement.\n     There is real basin-dependent signal underneath it.")
        elif resid_agree < raw_agree:
            print("  -> Agreement got WORSE after removing the artifact.\n"
                  "     The raw agreement in alpha/alphaL was likely coincidental "
                  "(same-direction\n     overlap with the artifact), not real signal -- "
                  "re-examine before claiming\n     MM validation in ANY basin.")
        else:
            print("  -> No change. Inconclusive at this sample size.")


if __name__ == '__main__':
    main()
