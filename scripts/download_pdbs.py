#!/usr/bin/env python3
"""
download_pdbs.py — Download PDB dataset via PISCES culled list
==============================================================
Fetches a non-redundant, quality-filtered PDB chain list from PISCES,
then downloads each PDB file from RCSB.

PISCES filters applied:
  - Resolution    <= 2.0 Å
  - R-free        <= 0.25
  - Sequence identity <= 25%  (non-redundant)
  - Chain length  >= 40 residues

Usage:
  python download_pdbs.py --out_dir ./pdb_cache
  python download_pdbs.py --out_dir ./pdb_cache --max_pdbs 100   # test run
  python download_pdbs.py --out_dir ./pdb_cache --workers 8      # parallel

Author: Wei (Cvek Lab, LSUHSC)
"""

import argparse
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── PISCES query URL ──────────────────────────────────────────────────────────
# These parameters match the original dataset filters.
PISCES_URL = (
    "https://dunbrack.fccc.edu/pisces/download/cullpdb_pc25_res2.0_R0.25"
    "_d230101_chains18235.gz"
)

# Fallback: PISCES CGI query (generates fresh list)
PISCES_CGI = (
    "https://dunbrack.fccc.edu/pisces/download/pisces.cgi"
    "?resolution=2.0&Rfactor=0.25&seqId=25&chains=1"
    "&MinLength=40&MaxLength=10000&CA=0&Submit=Submit"
)

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description='Download PISCES-filtered PDB dataset')
    ap.add_argument('--out_dir', default='./pdb_cache',
                    help='Directory to save PDB files (default: ./pdb_cache)')
    ap.add_argument('--pisces_file', default=None,
                    help='Use a pre-downloaded PISCES list file instead of fetching')
    ap.add_argument('--max_pdbs', type=int, default=None,
                    help='Limit number of PDBs (for testing)')
    ap.add_argument('--workers', type=int, default=4,
                    help='Parallel download workers (default: 4)')
    ap.add_argument('--skip_existing', action='store_true', default=True,
                    help='Skip already-downloaded PDBs (default: True)')
    ap.add_argument('--verbose', action='store_true',
                    help='Print per-PDB status')
    return ap.parse_args()

# ── PISCES list fetching ──────────────────────────────────────────────────────

def fetch_pisces_list(pisces_file=None):
    """
    Returns list of (pdb_id, chain_id) tuples from PISCES.
    Tries local file first, then RCSB-hosted culled list, then CGI.
    """
    if pisces_file and Path(pisces_file).exists():
        print(f"  Using local PISCES file: {pisces_file}")
        return _parse_pisces_file(Path(pisces_file).read_text())

    # Try to fetch a known static culled list from PISCES
    # The URL format changes with date; try several known recent ones
    candidates = [
        "https://dunbrack.fccc.edu/pisces/download/cullpdb_pc25_res2.0_R0.25_d240101_chains18793.gz",
        "https://dunbrack.fccc.edu/pisces/download/cullpdb_pc25_res2.0_R0.25_d230101_chains18235.gz",
        "https://dunbrack.fccc.edu/pisces/download/cullpdb_pc25_res2.0_R0.25_d220101_chains17612.gz",
    ]

    for url in candidates:
        try:
            print(f"  Trying PISCES URL: {url}")
            import gzip, io
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            text = gzip.decompress(data).decode('utf-8', errors='ignore')
            chains = _parse_pisces_file(text)
            if chains:
                print(f"  Got {len(chains)} chains from PISCES")
                return chains
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    # Fallback: use RCSB search API with quality filters
    print("  PISCES unavailable — using RCSB search API fallback")
    return _fetch_rcsb_list()


def _parse_pisces_file(text):
    """Parse PISCES culled list text → list of (pdb_id, chain_id)."""
    chains = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or line.lower().startswith('pdbid'):
            continue
        parts = line.split()
        if len(parts) < 1:
            continue
        entry = parts[0]  # e.g. '1UBQA' or '1UBQ_A'
        if '_' in entry:
            pdb_id, chain = entry.split('_', 1)
        elif len(entry) == 5:
            pdb_id, chain = entry[:4], entry[4]
        elif len(entry) == 4:
            pdb_id, chain = entry, 'A'
        else:
            continue
        chains.append((pdb_id.lower(), chain))
    return chains


def _fetch_rcsb_list():
    """
    Fallback: query RCSB REST API for structures with:
      resolution <= 2.0, R_free <= 0.25, polymer type = protein
    Returns list of (pdb_id, chain) — chain defaults to 'A'.
    """
    import json
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": 2.0
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "refine.ls_rfactor_rfree",
                        "operator": "less_or_equal",
                        "value": 0.25
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "Protein"
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 50000},
            "results_verbosity": "compact"
        }
    }
    url = "https://search.rcsb.org/rcsbsearch/v1/query"
    data = json.dumps(query).encode()
    req = urllib.request.Request(url, data=data,
                                  headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            result = json.loads(r.read())
        ids = [h['identifier'] for h in result.get('result_set', [])]
        print(f"  RCSB returned {len(ids)} structures")
        return [(pdb_id.lower(), 'A') for pdb_id in ids]
    except Exception as e:
        print(f"  RCSB API also failed: {e}")
        return []

# ── PDB downloading ───────────────────────────────────────────────────────────

def download_one(pdb_id, out_dir, skip_existing=True, verbose=False):
    """Download a single PDB file. Returns (pdb_id, status, message)."""
    out_path = Path(out_dir) / f"{pdb_id}.pdb"
    if skip_existing and out_path.exists() and out_path.stat().st_size > 1000:
        return pdb_id, 'skip', ''

    url = RCSB_URL.format(pdb_id=pdb_id)
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            if len(data) < 500:
                return pdb_id, 'fail', 'file too small'
            out_path.write_bytes(data)
            if verbose:
                print(f"  [OK] {pdb_id}  ({len(data)/1024:.1f} KB)")
            return pdb_id, 'ok', f"{len(data)/1024:.1f}KB"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return pdb_id, 'fail', '404'
            time.sleep(2 ** attempt)
        except Exception as e:
            time.sleep(2 ** attempt)
    return pdb_id, 'fail', 'timeout/error'


def download_all(pdb_ids, out_dir, workers=4, skip_existing=True, verbose=False):
    """Download all PDB files in parallel."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = n_fail = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(download_one, pdb_id, out_dir, skip_existing, verbose): pdb_id
            for pdb_id in pdb_ids
        }
        for i, fut in enumerate(as_completed(futures), 1):
            pdb_id, status, msg = fut.result()
            if status == 'ok':    n_ok   += 1
            elif status == 'skip': n_skip += 1
            else:                  n_fail += 1

            if i % 500 == 0 or i == len(pdb_ids):
                elapsed = time.time() - t0
                rate = (n_ok + n_skip) / max(elapsed, 1)
                print(f"  [{i:5d}/{len(pdb_ids)}]  "
                      f"ok={n_ok}  skip={n_skip}  fail={n_fail}  "
                      f"{rate:.1f} PDB/s  {elapsed/60:.1f} min")

    return n_ok, n_skip, n_fail

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 70)
    print("download_pdbs.py — PISCES-filtered PDB dataset")
    print("=" * 70)
    print(f"  Output dir : {args.out_dir}")
    print(f"  Workers    : {args.workers}")
    print(f"  Filters    : resolution ≤ 2.0 Å, R-free ≤ 0.25, "
          f"seqid ≤ 25%, len ≥ 40")

    print("\n[1] Fetching PISCES chain list...")
    chains = fetch_pisces_list(args.pisces_file)
    if not chains:
        print("ERROR: could not obtain PDB list. "
              "Download a PISCES list manually from:\n"
              "  https://dunbrack.fccc.edu/pisces/\n"
              "and pass it with --pisces_file <path>")
        sys.exit(1)

    # Deduplicate by PDB ID (we download full PDB, not per-chain)
    seen = set()
    pdb_ids = []
    for pdb_id, chain in chains:
        if pdb_id not in seen:
            seen.add(pdb_id)
            pdb_ids.append(pdb_id)

    if args.max_pdbs:
        pdb_ids = pdb_ids[:args.max_pdbs]
        print(f"  Limiting to {args.max_pdbs} PDBs (--max_pdbs)")

    print(f"  {len(chains)} chains → {len(pdb_ids)} unique PDB files to fetch")

    print(f"\n[2] Downloading to {args.out_dir} ...")
    n_ok, n_skip, n_fail = download_all(
        pdb_ids, args.out_dir,
        workers=args.workers,
        skip_existing=args.skip_existing,
        verbose=args.verbose)

    print(f"\n{'='*70}")
    print(f"  Done.")
    print(f"  Downloaded : {n_ok}")
    print(f"  Skipped    : {n_skip}  (already existed)")
    print(f"  Failed     : {n_fail}")
    print(f"  Total PDBs : {len(list(Path(args.out_dir).glob('*.pdb')))}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
