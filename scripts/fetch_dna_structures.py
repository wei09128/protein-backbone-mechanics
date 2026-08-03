#!/usr/bin/env python3
"""
fetch_dna_structures.py — Download DNA structures via RCSB Search API v2
============================================================================
BGSU's nrlist service is RNA-only (confirmed 404 on /nrlist/dna/...), so
this uses RCSB's search API directly, filtered to nucleic acid entries
containing DNA.

Filters:
  - entity_poly.rcsb_entity_polymer_type includes DNA
    (covers pure DNA and DNA/RNA hybrid structures — protein-DNA
    complexes are also included, which is useful since you likely want
    backbone geometry from both free DNA duplexes AND protein-bound DNA)
  - resolution <= 3.0 Å  (DNA structures generally resolve worse than
    protein; 2.0 Å would exclude the majority of the DNA-only PDB)
  - X-ray only (matches your protein pipeline convention)

Usage:
  python fetch_dna_structures.py --out_dir ./dna_pdb_cache
  python fetch_dna_structures.py --out_dir ./dna_pdb_cache --max_resolution 2.5
  python fetch_dna_structures.py --out_dir ./dna_pdb_cache --pure_dna_only
"""

import argparse
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def parse_args():
    ap = argparse.ArgumentParser(description='Download DNA-containing PDB structures')
    ap.add_argument('--out_dir', default='./dna_pdb_cache')
    ap.add_argument('--max_resolution', type=float, default=3.0,
                    help='Max resolution in Å (default: 3.0, DNA needs looser cutoff than protein)')
    ap.add_argument('--pure_dna_only', action='store_true',
                    help='Exclude protein-DNA complexes; only pure DNA/DNA-RNA structures')
    ap.add_argument('--max_pdbs', type=int, default=None)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--pdb_ids_file', default=None,
                    help='Download exactly these IDs instead of re-querying RCSB '
                         '(e.g. the deduplicated list from check_dna_redundancy.py)')
    return ap.parse_args()


def build_query(max_resolution, pure_dna_only):
    nodes = [
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal",
                "value": max_resolution
            }
        },
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "exptl.method",
                "operator": "exact_match",
                "value": "X-RAY DIFFRACTION"
            }
        },
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "entity_poly.rcsb_entity_polymer_type",
                "operator": "exact_match" if pure_dna_only else "in",
                "value": "DNA" if pure_dna_only else ["DNA", "NA-hybrid"]
            }
        }
    ]
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": nodes
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 10000},
            "results_verbosity": "compact"
        }
    }


def fetch_pdb_ids(max_resolution, pure_dna_only, verbose=False):
    query = build_query(max_resolution, pure_dna_only)
    data = json.dumps(query).encode()
    req = urllib.request.Request(
        RCSB_SEARCH_URL, data=data,
        headers={'Content-Type': 'application/json',
                 'User-Agent': 'Mozilla/5.0'})

    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            result = json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors='ignore')
        print(f"  HTTP {e.code}: {body[:500]}")
        return []
    except Exception as e:
        print(f"  ERROR: {e}")
        return []

    result_set = result.get('result_set', [])
    if result_set and isinstance(result_set[0], dict):
        ids = [h['identifier'] for h in result_set]
    else:
        ids = list(result_set)  # compact mode returns plain ID strings
    total = result.get('total_count', len(ids))
    if verbose:
        print(f"  API reports total_count={total}, returned {len(ids)}")
    return ids


def download_one(pdb_id, out_dir, verbose=False):
    out_path = Path(out_dir) / f"{pdb_id.lower()}.pdb"
    if out_path.exists() and out_path.stat().st_size > 500:
        return pdb_id, 'skip', ''

    url = RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id.lower())
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            if len(data) < 500:
                return pdb_id, 'fail', 'too small'
            out_path.write_bytes(data)
            return pdb_id, 'ok', f"{len(data)/1024:.1f}KB"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return pdb_id, 'fail', '404'
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return pdb_id, 'fail', 'timeout/error'


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("fetch_dna_structures.py — RCSB Search API v2")
    print("=" * 70)
    print(f"  Resolution  : <= {args.max_resolution} Å")
    print(f"  Method      : X-ray only")
    print(f"  Polymer     : {'pure DNA only' if args.pure_dna_only else 'DNA + DNA/RNA hybrid (incl. protein-DNA complexes)'}")
    print(f"  Output      : {out_dir}")

    print("\n[1] Getting PDB ID list...")
    if args.pdb_ids_file:
        print(f"  Using fixed ID list: {args.pdb_ids_file}")
        pdb_ids = [l.strip() for l in open(args.pdb_ids_file) if l.strip()]
    else:
        print("  Querying RCSB for DNA-containing structures...")
        pdb_ids = fetch_pdb_ids(args.max_resolution, args.pure_dna_only, verbose=True)

    if not pdb_ids:
        print("\nERROR: no structures returned. Query may need adjustment.")
        print("Try checking valid polymer_type values manually at:")
        print("  https://search.rcsb.org/redoc/index.html")
        return

    if args.max_pdbs:
        pdb_ids = pdb_ids[:args.max_pdbs]

    print(f"  {len(pdb_ids)} PDB IDs to download")

    print(f"\n[2] Downloading to {out_dir} ...")
    n_ok = n_skip = n_fail = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_one, pid, out_dir, args.verbose): pid
                   for pid in pdb_ids}
        for i, fut in enumerate(as_completed(futures), 1):
            pdb_id, status, msg = fut.result()
            if status == 'ok': n_ok += 1
            elif status == 'skip': n_skip += 1
            else: n_fail += 1
            if args.verbose and status == 'fail':
                print(f"  [FAIL] {pdb_id}: {msg}")
            if i % 200 == 0 or i == len(pdb_ids):
                elapsed = time.time() - t0
                print(f"  [{i:5d}/{len(pdb_ids)}]  ok={n_ok}  skip={n_skip}  "
                      f"fail={n_fail}  {i/max(elapsed,1):.1f} PDB/s")

    print(f"\n{'='*70}")
    print(f"  Done. ok={n_ok}  skip={n_skip}  fail={n_fail}")
    print(f"  Total files: {len(list(out_dir.glob('*.pdb')))}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
