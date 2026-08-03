#!/usr/bin/env python3
"""
fetch_release_dates.py — Batch fetch PDB initial release dates via RCSB GraphQL
====================================================================================
Used to build a temporally blind train/test split: build the MechLib library
on structures released before a cutoff date, evaluate strain reduction on
structures released after -- structures the library genuinely never saw.

Usage:
  python fetch_release_dates.py --pdb_ids_file all_pdb_ids.txt --out release_dates.csv
"""

import argparse
import json
import time
import urllib.request
import urllib.error


def fetch_batch(pdb_ids):
    ids_str = ','.join(f'"{p}"' for p in pdb_ids)
    gql = {'query': (
        '{ entries(entry_ids: [' + ids_str + ']) { '
        'rcsb_id rcsb_accession_info { initial_release_date } } }'
    )}
    req = urllib.request.Request(
        'https://data.rcsb.org/graphql',
        data=json.dumps(gql).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            result = json.loads(r.read())
    except Exception as e:
        print(f"  batch error: {e}")
        return {}

    out = {}
    for entry in result.get('data', {}).get('entries', []) or []:
        if entry is None:
            continue
        pdb_id = entry.get('rcsb_id')
        info = entry.get('rcsb_accession_info') or {}
        date = info.get('initial_release_date')
        if date:
            out[pdb_id] = date[:10]  # keep just YYYY-MM-DD
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdb_ids_file', required=True,
                    help='Text file, one PDB ID per line')
    ap.add_argument('--out', default='release_dates.csv')
    ap.add_argument('--batch_size', type=int, default=50)
    args = ap.parse_args()

    with open(args.pdb_ids_file) as f:
        pdb_ids = [l.strip().upper() for l in f if l.strip()]
    print(f"{len(pdb_ids)} PDB IDs to fetch")

    date_map = {}
    t0 = time.time()
    for i in range(0, len(pdb_ids), args.batch_size):
        batch = pdb_ids[i:i + args.batch_size]
        result = fetch_batch(batch)
        date_map.update(result)
        if (i // args.batch_size) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+len(batch):5d}/{len(pdb_ids)}]  {elapsed:.0f}s elapsed")

    print(f"\nFetched dates for {len(date_map)}/{len(pdb_ids)} entries")

    with open(args.out, 'w') as f:
        f.write("pdb_id,release_date\n")
        for pdb_id in pdb_ids:
            date = date_map.get(pdb_id, '')
            f.write(f"{pdb_id},{date}\n")
    print(f"Written {args.out}")


if __name__ == '__main__':
    main()
