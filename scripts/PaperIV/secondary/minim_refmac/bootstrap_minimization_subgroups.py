#!/usr/bin/env python3
"""Read-only structure bootstrap of the existing Paper IV minimization split."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


def interval(values):
    ordered = sorted(values)
    n = len(ordered)
    return [ordered[int(0.025 * (n - 1))], ordered[int(0.975 * (n - 1))]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=Path, required=True,
                        help='MINIMIZATION_REFMAC_ROWS.tsv from the existing read-only audit')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260925)
    args = parser.parse_args()
    if args.replicates < 1000:
        parser.error('At least 1000 resamples required')
    with args.rows.open(newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    if len(rows) != 300 or len({r['pdb_id'] for r in rows}) != 300:
        parser.error('Expected exactly 300 unique frozen structure rows')
    counts = {name: sum(r['program'] == name for r in rows)
              for name in ('REFMAC', 'PHENIX')}
    if counts != {'REFMAC': 87, 'PHENIX': 201}:
        parser.error(f'Refinement groups differ from frozen diagnostic: {counts}')
    rng = random.Random(args.seed)
    output = {'status': 'POST_HOC_STRUCTURE_BOOTSTRAP',
              'estimand': '100 * (1 - sum(MechLib RMSD) / sum(fixed AMBER RMSD))',
              'source_sha256': hashlib.sha256(args.rows.read_bytes()).hexdigest(),
              'replicates': args.replicates, 'seed': args.seed, 'groups': {}}
    for name in ('ALL', 'REFMAC', 'PHENIX'):
        group = rows if name == 'ALL' else [r for r in rows if r['program'] == name]
        pairs = [(float(r['amber_rmsd']), float(r['mechlib_rmsd'])) for r in group]
        if not all(0 < a < 100 and 0 < m < 100 for a, m in pairs):
            parser.error(f'Nonpositive or implausible RMSD in {name}')
        a_total = sum(a for a, _ in pairs)
        m_total = sum(m for _, m in pairs)
        estimate = 100 * (1 - m_total / a_total)
        if name == 'ALL' and not 33.05 <= estimate <= 33.07:
            parser.error(f'Full cohort failed frozen 33.059% cross-check: {estimate}')
        draws = []
        n = len(pairs)
        for _ in range(args.replicates):
            sample = rng.choices(pairs, k=n)
            draws.append(100 * (1 - sum(m for _, m in sample) / sum(a for a, _ in sample)))
        output['groups'][name] = {'n_structures': n, 'reduction_percent': estimate,
                                  'bootstrap_ci95_percent': interval(draws)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output['groups'], indent=2))


if __name__ == '__main__':
    main()
