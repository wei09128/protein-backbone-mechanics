#!/usr/bin/env python3
"""Post hoc signed-offset sensitivity of frozen Q6 geometry, never a new gate."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def read(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--detail', type=Path, required=True)
    ap.add_argument('--endpoint', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    endpoint = json.loads(args.endpoint.read_text())
    if endpoint.get('status') != 'FAIL' or endpoint.get('qc_pass_cases') != 80:
        raise ValueError('Expected the original 80/80 Q6 registered FAIL; stop.')
    rows = read(args.detail)
    if len({r['case_id'] for r in rows}) != 80 or len(rows) < 800:
        raise ValueError('Unexpected detail size; stop.')
    groups = defaultdict(list)
    for row in rows:
        groups[row['term']].append(row)
    term_result = []
    case_errors = defaultdict(list)
    for term, group in sorted(groups.items()):
        if len({r['residue'] for r in group}) < 2:
            raise ValueError('Insufficient distinct residues for leave-one-residue-out offset')
        signed = {model: defaultdict(list) for model in ('amber', 'mechlib')}
        for row in group:
            for model in signed:
                signed[model][row['residue']].append(float(row[f'{model}_value']) - float(row['qm_value']))
        residuals = {model: [] for model in signed}
        for row in group:
            residue = row['residue']
            scale = math.sqrt(float(row['force_constant'])) * (math.pi / 180 if row['kind'] == 'angle' else 1)
            for model in signed:
                train = [v for other, vals in signed[model].items() if other != residue for v in vals]
                corrected = float(row[f'{model}_value']) - float(row['qm_value']) - sum(train)/len(train)
                weighted = scale * abs(corrected)
                residuals[model].append(weighted)
                case_errors[row['case_id'], model].append(weighted)
        amber = sum(residuals['amber']) / len(group)
        mechlib = sum(residuals['mechlib']) / len(group)
        term_result.append({'term': term, 'n': len(group), 'amber_weighted_abs': f'{amber:.8g}',
                            'mechlib_weighted_abs': f'{mechlib:.8g}',
                            'delta_positive_favors_mechlib': f'{amber-mechlib:.8g}'})
    case_delta = []
    for case in sorted({r['case_id'] for r in rows}):
        a, m = (sum(case_errors[case, model])/len(case_errors[case, model])
                for model in ('amber', 'mechlib'))
        case_delta.append(a-m)
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / 'posthoc_terms.tsv'
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(term_result[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(term_result)
    summary = {
        'label': 'post_hoc_exploratory_leave_one_residue_out_signed_offset',
        'registered_Q6_status_unchanged': endpoint['status'],
        'registered_Q6_primary_mean_delta_unchanged': endpoint['primary_mean_delta'],
        'case_mean_offset_centered_delta_positive_favors_mechlib': sum(case_delta)/len(case_delta),
        'n_cases': len(case_delta), 'n_terms': len(term_result),
        'detail_sha256': hashlib.sha256(args.detail.read_bytes()).hexdigest(),
        'endpoint_sha256': hashlib.sha256(args.endpoint.read_bytes()).hexdigest(),
        'interpretation': 'Leave-one-residue-out offsets are derived from the same 80-case benchmark; exploratory sensitivity, not new independent QM or a replacement endpoint.',
    }
    (args.out / 'posthoc_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
