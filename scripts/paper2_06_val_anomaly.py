"""
paper2_06_val_anomaly.py — Why does VAL's χ1 rotamer effect flip sign?
========================================================================

In paper2_05 we saw that VAL is the ONLY residue whose ∠N-Cα-Cβ (and τ)
rotamer shift goes the opposite direction from all 16 others:

    ∠N-Cα-Cβ α:  HIS +0.45  ILE +1.88  LEU +0.37  THR +1.54  ...  VAL −1.49
    τ α:         HIS +0.92  ILE +1.66  LEU +0.58  THR +0.81  ...  VAL −0.37

Hypothesis: this is a χ1 NOMENCLATURE ARTIFACT, not a mechanical reversal.
VAL has two chemically identical methyl groups at Cβ (CG1, CG2). The PDB
convention labels one of them as "CG1" (and hence defines χ1 as
N-Cα-Cβ-CG1). If that label is assigned systematically such that what we
call "VAL g⁻" is geometrically equivalent to other residues' "g⁺", then
VAL's sign flip is entirely about the labeling, not the mechanism.

Test: compute BOTH possible χ1 definitions for VAL:
    χ1(A) = N - Cα - Cβ - CG1   (standard definition)
    χ1(B) = N - Cα - Cβ - CG2   (alternative using the other methyl)
χ1(B) will differ from χ1(A) by ~120° (the angle between the two methyls
around the Cβ tetrahedron). If we redo the rotamer analysis using χ1(B),
the VAL sign should flip back to agreeing with the others.

If this is confirmed:
  → VAL's apparent anomaly is an artifact of CG1/CG2 label assignment
  → The universal lever story holds for all 17 residues with sidechains
  → Add a footnote to paper 2 about VAL's symmetric methyl group

If NOT confirmed (VAL still flips under χ1(B)):
  → There's a real VAL-specific mechanical effect, needs investigation

We also check ILE for consistency: ILE has CG1 (leading to CD1) vs CG2 (methyl),
which are NOT equivalent. ILE's χ1 is unambiguous, and ILE agrees with the
pack — which is consistent with "sign flip requires identical substituents".

Usage
-----
    python paper2_06_val_anomaly.py --csv features.csv \\
        --pdb_dir /mnt/f/Protein_Folding/pdb_cache/

    (or just --pdb_dir if features.csv is already in that path)

This script reads PDB files directly to extract CG1 and CG2 positions
for VAL residues, because features.csv only stores the standard χ1.
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


# Import pdb_loader to walk real PDB files
sys.path.insert(0, str(Path(__file__).parent))
from pdb_loader import load_structure
from molcore import dihedral_angle


def _rot_label(chi1_rad):
    """Classify χ1 value into gauche-/trans/gauche+ wells. NaN-safe."""
    if np.isnan(chi1_rad):
        return None
    if abs(chi1_rad) >= np.pi * 5 / 6:
        return 'trans'
    if -np.pi * 5 / 6 < chi1_rad < -np.pi / 6:
        return 'g-'
    if np.pi / 6 < chi1_rad < np.pi * 5 / 6:
        return 'g+'
    return None


def _wrap_deg(x):
    """Wrap to [-180, 180]."""
    return (x + 180) % 360 - 180


def compute_val_alt_chi1(pdb_dir, max_structures=None, verbose=False):
    """
    Walk PDB files and extract for each VAL residue:
      - chi1_A: dihedral N-Cα-Cβ-CG1  (standard)
      - chi1_B: dihedral N-Cα-Cβ-CG2  (alternative)
      - phi, psi, tau, ∠N-Cα-Cβ, ∠C-Cα-Cβ

    Returns a DataFrame with one row per VAL residue.
    """
    rows = []
    pdb_files = sorted(Path(pdb_dir).glob('*.pdb'))
    if max_structures:
        pdb_files = pdb_files[:max_structures]
    print(f"  scanning {len(pdb_files):,} PDB files for VAL residues ...")

    for idx, pdb_path in enumerate(pdb_files):
        if verbose and idx % 500 == 0:
            print(f"    {idx:>6,}/{len(pdb_files):,}  ({pdb_path.name})")
        try:
            s = load_structure(str(pdb_path))
        except Exception:
            continue

        # Skip if CG1 or CG2 absent from this structure
        if 'CG1' not in s.coords or 'CG2' not in s.coords:
            continue

        val_idxs = [i for i, r in enumerate(s.sequence) if r == 'VAL']
        for i in val_idxs:
            if i == 0 or i == s.n_res - 1:
                continue
            # Check chain breaks
            if (i - 1) in s.chain_breaks or i in s.chain_breaks:
                continue

            N   = s.coords['N'][i]
            CA  = s.coords['CA'][i]
            C   = s.coords['C'][i]
            CB  = s.coords['CB'][i]
            CG1 = s.coords['CG1'][i]
            CG2 = s.coords['CG2'][i]

            if any(np.any(np.isnan(p)) for p in (N, CA, C, CB, CG1, CG2)):
                continue

            # Compute two χ1 definitions
            try:
                chi1_A = dihedral_angle(N, CA, CB, CG1)
                chi1_B = dihedral_angle(N, CA, CB, CG2)
            except Exception:
                continue

            # Compute phi, psi of this residue
            C_prev = s.coords['C'][i - 1]
            N_next = s.coords['N'][i + 1]
            if any(np.any(np.isnan(p)) for p in (C_prev, N_next)):
                continue

            try:
                phi = dihedral_angle(C_prev, N, CA, C)
                psi = dihedral_angle(N, CA, C, N_next)
            except Exception:
                continue

            # Bond angles
            def _ang(a, b, c):
                v1 = a - b; v2 = c - b
                n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
                if n1 < 1e-10 or n2 < 1e-10:
                    return float('nan')
                cos_a = float(np.dot(v1, v2) / (n1 * n2))
                return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos_a)))))

            rows.append({
                'pdb_id':   s.pdb_id,
                'res_idx':  i,
                'chi1_A_deg': float(np.degrees(chi1_A)),
                'chi1_B_deg': float(np.degrees(chi1_B)),
                'phi_deg':  float(np.degrees(phi)),
                'psi_deg':  float(np.degrees(psi)),
                'tau':      _ang(N, CA, C),
                'angle_N_CA_CB': _ang(N, CA, CB),
                'angle_C_CA_CB': _ang(C, CA, CB),
            })

    return pd.DataFrame(rows)


def analyse(df, label_col='chi1_A_deg', angle_col='tau',
            basin='α', per_residue_median=111.0):
    """
    Given a DataFrame with chi1 column `label_col` and angle `angle_col`,
    compute g− vs g+ mean delta for the given basin.
    """
    phi_range = {'α': (-80, -40), 'β': (-150, -90), 'PPII': (-90, -60)}[basin]
    psi_range = {'α': (-60, -20), 'β': (100, 160), 'PPII': (120, 160)}[basin]

    m = (df['phi_deg'].between(*phi_range)
         & df['psi_deg'].between(*psi_range))
    sub = df[m].copy()
    sub['chi1_rad'] = np.radians(sub[label_col])
    sub['rot'] = sub['chi1_rad'].apply(_rot_label)
    sub['delta'] = sub[angle_col] - per_residue_median

    gm = sub.loc[sub['rot'] == 'g-', 'delta'].values
    gp = sub.loc[sub['rot'] == 'g+', 'delta'].values

    if len(gm) < 30 or len(gp) < 30:
        return None
    shift = float(gp.mean() - gm.mean())
    sem = float(np.sqrt(gm.var() / len(gm) + gp.var() / len(gp)))
    return dict(gm_mean=float(gm.mean()), gm_n=len(gm),
                gp_mean=float(gp.mean()), gp_n=len(gp),
                shift=shift, shift_sem=sem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdb_dir', required=True,
                    help='Directory of PDB files to scan')
    ap.add_argument('--max_pdbs', type=int, default=None,
                    help='Limit number of PDBs (for quick test)')
    ap.add_argument('--out', default='paper2_06_val_anomaly.png')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    pdb_dir = Path(args.pdb_dir)
    if not pdb_dir.is_dir():
        print(f"ERROR: {pdb_dir} is not a directory"); sys.exit(1)

    print("Extracting VAL residues with both CG1 and CG2 ...")
    df = compute_val_alt_chi1(pdb_dir, max_structures=args.max_pdbs,
                                verbose=args.verbose)
    print(f"  extracted {len(df):,} VAL residues")

    if len(df) < 100:
        print("  ERROR: too few VALs — increase --max_pdbs or check path")
        sys.exit(1)

    # ── Sanity: chi1_B should equal chi1_A ± 120° ────────────────────────────
    diffs = _wrap_deg(df['chi1_B_deg'] - df['chi1_A_deg'])
    print(f"\nSanity check on χ1(B) − χ1(A) (should be ±120° for VAL):")
    print(f"  median: {float(np.median(diffs)):+.2f}°")
    print(f"  mean:   {float(np.mean(diffs)):+.2f}°")
    print(f"  std:    {float(np.std(diffs)):.2f}°")
    near_pm120 = np.abs(np.abs(diffs) - 120) < 15
    print(f"  fraction within ±15° of ±120°: "
          f"{int(near_pm120.sum())}/{len(df)} "
          f"({100*near_pm120.mean():.1f}%)")

    # ── Run rotamer analysis with both χ1 definitions ────────────────────────
    # Compute VAL's median τ and Cβ angles from the full data for normalization
    med_tau = float(df['tau'].median())
    med_ncacb = float(df['angle_N_CA_CB'].median())
    med_ccacb = float(df['angle_C_CA_CB'].median())
    print(f"\nVAL medians: τ={med_tau:.3f}°  "
          f"∠N-Cα-Cβ={med_ncacb:.3f}°  "
          f"∠C-Cα-Cβ={med_ccacb:.3f}°")

    angles = [
        ('tau',            'τ',           med_tau),
        ('angle_N_CA_CB',  '∠N-Cα-Cβ',    med_ncacb),
        ('angle_C_CA_CB',  '∠C-Cα-Cβ',    med_ccacb),
    ]
    basins = ['α', 'β', 'PPII']

    print("\n" + "=" * 82)
    print("VAL: Δ(g⁺) − Δ(g⁻) using χ1(A) vs χ1(B)")
    print("=" * 82)
    print(f"  {'angle':<12} {'basin':<6}  "
          f"{'shift(A)':>15}  {'shift(B)':>15}   expected from pack")
    print("  " + "-" * 80)

    # From paper2_05 summary: expected sign across 16 non-VAL residues
    expected_sign = {
        ('τ', 'α'):    +1,  ('τ', 'β'):    +1,  ('τ', 'PPII'):    +1,
        ('∠N-Cα-Cβ', 'α'): +1, ('∠N-Cα-Cβ', 'β'): +1, ('∠N-Cα-Cβ', 'PPII'): +1,
        ('∠C-Cα-Cβ', 'α'): +1, ('∠C-Cα-Cβ', 'β'): +1, ('∠C-Cα-Cβ', 'PPII'): +1,
    }

    results = {}
    for ang_col, ang_label, med in angles:
        for basin in basins:
            rA = analyse(df, 'chi1_A_deg', ang_col, basin, per_residue_median=med)
            rB = analyse(df, 'chi1_B_deg', ang_col, basin, per_residue_median=med)
            exp = expected_sign[(ang_label, basin)]
            sA = f"{rA['shift']:+7.3f}±{rA['shift_sem']:.3f}" if rA else '  n<30  '
            sB = f"{rB['shift']:+7.3f}±{rB['shift_sem']:.3f}" if rB else '  n<30  '
            exp_str = '+' if exp > 0 else '−'
            flag_A = '' if (rA and rA['shift'] * exp > 0) else '  ← mismatch'
            flag_B = '' if (rB and rB['shift'] * exp > 0) else '  ← mismatch'
            print(f"  {ang_label:<12} {basin:<6}  "
                  f"{sA:>15}  {sB:>15}      {exp_str}   "
                  f"{flag_A or flag_B}")
            results[(ang_col, basin)] = (rA, rB)

    # ── Verdict ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 82)
    print("VERDICT")
    print("=" * 82)
    A_mismatches = sum(
        1 for (rA, _) in results.values()
        if rA and rA['shift'] * expected_sign[(
            {'tau':'τ','angle_N_CA_CB':'∠N-Cα-Cβ','angle_C_CA_CB':'∠C-Cα-Cβ'}
            [rA['gm_n']>0 and list(results.keys())[
                list(results.values()).index((rA, _))][0] or ''],
            list(results.keys())[
                list(results.values()).index((rA, _))][1])] < 0)
    # Simplified: just count directly
    mismatch_A = mismatch_B = 0
    total = 0
    for (ang_col, basin), (rA, rB) in results.items():
        ang_label = {'tau':'τ','angle_N_CA_CB':'∠N-Cα-Cβ','angle_C_CA_CB':'∠C-Cα-Cβ'}[ang_col]
        exp = expected_sign[(ang_label, basin)]
        if rA:
            total += 1
            if rA['shift'] * exp < 0: mismatch_A += 1
        if rB and rB['shift'] * exp < 0: mismatch_B += 1

    print(f"  Using χ1(A) = N-Cα-Cβ-CG1:  {mismatch_A}/{total} "
          f"mismatches vs pack expectation")
    print(f"  Using χ1(B) = N-Cα-Cβ-CG2:  {mismatch_B}/{total} "
          f"mismatches vs pack expectation")
    print()
    if mismatch_A >= 5 and mismatch_B <= 2:
        print("  → CONFIRMED: VAL's anomaly is a CG1/CG2 labeling artifact.")
        print("    Using CG2 as the χ1 reference atom makes VAL agree with")
        print("    all other residues. The lever story is universal.")
    elif mismatch_A <= 2 and mismatch_B >= 5:
        print("  → χ1(B) is WORSE — labeling is NOT the explanation.")
        print("    There's a real VAL-specific mechanical effect to investigate.")
    elif mismatch_A <= 2 and mismatch_B <= 2:
        print("  → Both definitions agree with the pack. Original paper2_05")
        print("    result is not reproducing here; double-check data.")
    else:
        print("  → Neither definition cleanly resolves the anomaly. VAL has")
        print("    a real and complex behaviour that needs deeper analysis.")

    # ── Figure ───────────────────────────────────────────────────────────────
    print(f"\nGenerating {args.out} ...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for k, (ang_col, ang_label, med) in enumerate(angles):
        ax = axes[k]
        x = np.arange(len(basins))
        a_vals, a_errs, b_vals, b_errs = [], [], [], []
        for basin in basins:
            rA, rB = results[(ang_col, basin)]
            a_vals.append(rA['shift'] if rA else 0)
            a_errs.append(rA['shift_sem'] if rA else 0)
            b_vals.append(rB['shift'] if rB else 0)
            b_errs.append(rB['shift_sem'] if rB else 0)
        w = 0.36
        ax.bar(x - w/2, a_vals, w, yerr=a_errs, label='χ1 = N-Cα-Cβ-CG1  (std)',
                color='#5aa369', capsize=3)
        ax.bar(x + w/2, b_vals, w, yerr=b_errs, label='χ1 = N-Cα-Cβ-CG2  (alt)',
                color='#c0392b', capsize=3)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels(basins)
        ax.set_title(f'{ang_label}   VAL only', fontsize=11)
        ax.set_ylabel('Δ(g⁺) − Δ(g⁻) (deg)')
        ax.grid(True, axis='y', alpha=0.25)
        if k == 0:
            ax.legend(fontsize=8, loc='best')
    plt.suptitle('VAL: testing χ1 definition as source of rotamer anomaly',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches='tight')
    print(f"Figure saved: {args.out}")


if __name__ == '__main__':
    main()