#!/usr/bin/env python3
"""Read-only, post hoc refinement-software split of frozen minimization rows."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def program(path):
    if not path.is_file():
        return 'UNKNOWN', 'PDB_MISSING'
    labels = []
    with path.open(errors='replace') as f:
        for line in f:
            if line.startswith('REMARK   3   PROGRAM'):
                labels.append(line[19:].strip())
            if line.startswith('CRYST1'):
                break
    raw = ' '.join(labels).upper()
    if 'PHENIX' in raw:
        return 'PHENIX', raw
    if 'REFMAC' in raw:
        return 'REFMAC', raw
    if raw:
        return 'OTHER_KNOWN', raw
    return 'UNKNOWN', 'NO_REMARK_3_PROGRAM'


def find_tables(root):
    result = []
    for path in sorted(root.rglob('*.csv')):
        if len(path.relative_to(root).parts) > 4 or path.stat().st_size > 100_000_000:
            continue
        try:
            with path.open(newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames or []
                samples = [row for _, row in zip(range(3), reader)]
        except (OSError, UnicodeError, csv.Error):
            continue
        result.append({'path': str(path), 'bytes': path.stat().st_size,
                       'columns': cols, 'sample_rows': samples})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', type=Path, required=True)
    p.add_argument('--pdb-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--table', type=Path)
    p.add_argument('--id-column')
    p.add_argument('--amber-column')
    p.add_argument('--mechlib-column')
    p.add_argument('--shuffle-column')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if not a.results_dir.is_dir() or not a.pdb_dir.is_dir():
        p.error('Results and PDB cache must exist; no calculations are rerun.')
    if a.preflight:
        inventory = find_tables(a.results_dir)
        print(json.dumps({'status': 'INVENTORY_ONLY', 'tables': inventory}, indent=2))
        return
    if not a.table or not all((a.id_column, a.amber_column, a.mechlib_column)):
        p.error('Execute requires explicit table, ID, AMBER, and MechLib columns from preflight.')
    table = a.table.resolve()
    if not table.is_file() or not table.is_relative_to(a.results_dir.resolve()):
        p.error('Table must be an existing CSV inside results-dir.')
    with table.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        names = [a.id_column, a.amber_column, a.mechlib_column]
        if a.shuffle_column:
            names.append(a.shuffle_column)
        if not set(names).issubset(reader.fieldnames or []):
            p.error('Column mapping does not match the table header.')
        rows = list(reader)
    if len(rows) != 300:
        p.error(f'Expected 300 frozen structure rows; found {len(rows)}. Inspect preflight before proceeding.')
    output = []
    for row in rows:
        pdb_id = row[a.id_column].strip().lower().removesuffix('.pdb')
        if not (len(pdb_id) == 4 and pdb_id.isalnum()):
            p.error(f'Cannot resolve PDB identifier: {row[a.id_column]!r}')
        pdb_path = a.pdb_dir / (pdb_id + '.pdb')
        if not pdb_path.exists():
            pdb_path = a.pdb_dir / (pdb_id.upper() + '.pdb')
        category, raw = program(pdb_path)
        try:
            amber = float(row[a.amber_column]); mechlib = float(row[a.mechlib_column])
            shuffled = float(row[a.shuffle_column]) if a.shuffle_column else None
        except ValueError as exc:
            p.error(f'Non-numeric RMSD for {pdb_id}: {exc}')
        if not (0 < amber < 100 and 0 < mechlib < 100):
            p.error(f'Implausible RMSD for {pdb_id}; check column units and mapping.')
        output.append({'pdb_id': pdb_id, 'program': category, 'program_raw': raw,
                       'amber_rmsd': amber, 'mechlib_rmsd': mechlib,
                       'shuffled_rmsd': shuffled, 'delta_rmsd': amber - mechlib,
                       'reduction_percent': 100 * (amber - mechlib) / amber})
    ids = [r['pdb_id'] for r in output]
    if len(set(ids)) != len(ids):
        p.error('Repeated structure ID: the table may be per-arm or per-replicate; do not aggregate.')
    groups = {}
    for group in ('ALL', 'REFMAC', 'PHENIX', 'OTHER_KNOWN', 'UNKNOWN'):
        selected = [r for r in output if group == 'ALL' or r['program'] == group]
        if not selected:
            continue
        groups[group] = {'n_structures': len(selected),
                         'mean_amber_rmsd': mean(r['amber_rmsd'] for r in selected),
                         'mean_mechlib_rmsd': mean(r['mechlib_rmsd'] for r in selected),
                         'mean_paired_delta_rmsd': mean(r['delta_rmsd'] for r in selected),
                         'ratio_of_group_mean_reduction_percent': 100 * sum(r['delta_rmsd'] for r in selected) / sum(r['amber_rmsd'] for r in selected),
                         'mean_individual_reduction_percent': mean(r['reduction_percent'] for r in selected),
                         'amber_better_count': sum(r['delta_rmsd'] < 0 for r in selected),
                         'ties_count': sum(r['delta_rmsd'] == 0 for r in selected)}
    if not (1 <= groups.get('REFMAC', {}).get('n_structures', 0) < 300):
        p.error('REFMAC subset missing or all 300; review PDB paths and REMARK 3 metadata.')
    a.out.mkdir(parents=True, exist_ok=True)
    report = {'status': 'POST_HOC_DIAGNOSTIC', 'original_frozen_endpoint_unchanged': True,
              'source_csv': str(table), 'source_sha256': digest(table),
              'classification_rule': 'PDB REMARK 3 PROGRAM; proxy for refinement software, not restraint usage',
              'groups': groups,
              'caveats': ['This is a post hoc subgroup analysis of the same 300 structures.',
                          'A refinement program name does not prove whether CDL restraints were enabled.',
                          'The original 33.1% endpoint may use a different reduction formula; do not overwrite it.']}
    (a.out / 'MINIMIZATION_REFMAC_DIAGNOSTIC.json').write_text(json.dumps(report, indent=2) + '\n')
    with (a.out / 'MINIMIZATION_REFMAC_ROWS.tsv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output[0].keys(), delimiter='\t')
        writer.writeheader(); writer.writerows(output)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
