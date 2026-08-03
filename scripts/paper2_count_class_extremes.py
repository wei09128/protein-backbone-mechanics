"""
count_class_extremes.py — tally which sidechain class hits max/min |Δ|
========================================================================

Reuses compute_class_data() from paper2_F2_assembly.py (must be importable,
i.e. run from the same directory, or add its path to sys.path below).

For each of the 16 (angle x basin) columns, ranks the 5 classes (G, A, U, B, Ar)
by |Δ| and records which class is largest (|max|) and which is smallest (|min|).
G is included in the ranking (not excluded as a trivial baseline).
B and Ar are kept as SEPARATE classes throughout (never combined).

Usage (from the directory containing paper2_F2_assembly.py and your features csv):
    python count_class_extremes.py --csv features_lj_v3_FINAL_CLEAN.csv
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# import the real computation logic from your existing figure script
sys.path.insert(0, str(Path(__file__).parent))
from paper2_f2_assembly import compute_class_data, _CLASS_ORDER, _REGION_ORDER, _ANGLES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    args = ap.parse_args()

    print(f"Loading {args.csv} ...")
    df_raw = pd.read_csv(args.csv)
    print(f"  {len(df_raw):,} total rows")

    class_data = compute_class_data(df_raw)  # {angle_label: {(basin,class): (mean,sem,n)}}

    max_count = {c: 0 for c in _CLASS_ORDER}
    min_count = {c: 0 for c in _CLASS_ORDER}
    n_cells_used = 0

    print("\n" + "=" * 100)
    print(f"{'angle':<16}{'basin':<6}" + ''.join(f"{c:>12}" for c in _CLASS_ORDER)
          + f"{'  |max|->':>14}{'  |min|->':>10}")
    print("=" * 100)

    for _, _, label in _ANGLES:
        cells = class_data[label]
        for reg in _REGION_ORDER:
            row_abs = {}
            row_str = []
            valid = True
            for cls in _CLASS_ORDER:
                m, sem, n = cells[(reg, cls)]
                if n < 30 or not np.isfinite(m):
                    valid = False
                    row_str.append(f"{'n/a':>12}")
                    continue
                row_abs[cls] = abs(m)
                row_str.append(f"{m:+11.3f}")
            if not valid or len(row_abs) < len(_CLASS_ORDER):
                print(f"{label:<16}{reg:<6}" + ''.join(row_str) + f"{'--skip--':>14}")
                continue

            n_cells_used += 1
            max_cls = max(row_abs, key=row_abs.get)
            min_cls = min(row_abs, key=row_abs.get)
            max_count[max_cls] += 1
            min_count[min_cls] += 1

            print(f"{label:<16}{reg:<6}" + ''.join(row_str)
                  + f"{max_cls:>14}{min_cls:>10}")

    print("=" * 100)
    print(f"\nCells used (all 5 classes present, n>=30): {n_cells_used} / 16\n")

    print(f"{'class':<8}{'|max| count':>14}{'|min| count':>14}")
    for c in _CLASS_ORDER:
        print(f"{c:<8}{max_count[c]:>14}{min_count[c]:>14}")

    print("\nInterpretation notes:")
    print(" - B and Ar are kept separate (never combined) throughout.")
    print(" - G is included in the ranking, not excluded as a size baseline.")
    print(" - A cell is 'skipped' only if any class has n<30 in that basin"
          " (e.g. sparse alphaL bins).")


if __name__ == '__main__':
    main()
