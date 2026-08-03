#!/usr/bin/env python3
"""
fetch_resolution.py — Batch fetch PDB resolution values
============================================================
Used to test whether MechLib's strain reduction depends on structure
quality (resolution) -- if the effect is independent of resolution, that
argues against "we are just fitting low-resolution coordinate noise."

Usage:
  python fetch_resolution.py --pdb_ids_file all_protein_ids.txt --out resolutions.csv
"""

import argparse
import json
import time
import urllib.request


def fetch_batch(pdb_ids):
    ids_str = ','.join(f'"{p}"' for p in pdb_ids)
    gql = {'query': (
        '{ entries(entry_ids: [' + ids_str + ']) { '
        'rcsb_id rcsb_entry_info { resolution_combined } } }'
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
        info = entry.get('rcsb_entry_info') or {}
        res_list = info.get('resolution_combined')
        if res_list:
            out[pdb_id] = res_list[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdb_ids_file', required=True)
    ap.add_argument('--out', default='resolutions.csv')
    ap.add_argument('--batch_size', type=int, default=50)
    args = ap.parse_args()

    with open(args.pdb_ids_file) as f:
        pdb_ids = [l.strip().upper() for l in f if l.strip()]
    print(f"{len(pdb_ids)} PDB IDs to fetch")

    res_map = {}
    t0 = time.time()
    for i in range(0, len(pdb_ids), args.batch_size):
        batch = pdb_ids[i:i + args.batch_size]
        result = fetch_batch(batch)
        res_map.update(result)
        if (i // args.batch_size) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+len(batch):5d}/{len(pdb_ids)}]  {elapsed:.0f}s elapsed")

    print(f"\nFetched resolution for {len(res_map)}/{len(pdb_ids)} entries")

    with open(args.out, 'w') as f:
        f.write("pdb_id,resolution\n")
        for pdb_id in pdb_ids:
            res = res_map.get(pdb_id, '')
            f.write(f"{pdb_id},{res}\n")
    print(f"Written {args.out}")


if __name__ == '__main__':
    main()
