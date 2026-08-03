#!/usr/bin/env python3
"""
check_dna_redundancy.py — Sequence redundancy check for DNA PDB dataset
===========================================================================
Fetches the canonical DNA sequence for every entry in a PDB ID list via
RCSB's GraphQL API and reports how redundant the set is (exact-sequence
duplicates). This is a first-pass redundancy check — NOT a proper
sequence-identity clustering like PISCES does for proteins (that would
need pairwise alignment). Exact-match duplication is still very
informative for DNA, since crystallographers reuse the same handful of
well-behaved duplex sequences (Dickerson dodecamer variants, etc.)
across hundreds of different crystal forms / ligand-bound structures.

Usage:
  python check_dna_redundancy.py --pdb_ids_file dna_ids.txt
  python check_dna_redundancy.py --query   # re-runs the RCSB search itself
"""

import argparse
import json
import time
import urllib.request
import urllib.error
from collections import Counter, defaultdict

GRAPHQL_URL = "https://data.rcsb.org/graphql"
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"


def fetch_pure_dna_ids(max_resolution=3.0):
    """Re-run the same verified query used in fetch_dna_structures.py."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "less_or_equal", "value": max_resolution}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "exptl.method",
                    "operator": "exact_match", "value": "X-RAY DIFFRACTION"}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "entity_poly.rcsb_entity_polymer_type",
                    "operator": "exact_match", "value": "DNA"}}
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 10000},
            "results_verbosity": "compact"
        }
    }
    data = json.dumps(query).encode()
    req = urllib.request.Request(SEARCH_URL, data=data,
                                  headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as r:
        result = json.loads(r.read())
    result_set = result.get('result_set', [])
    if result_set and isinstance(result_set[0], dict):
        return [h['identifier'] for h in result_set]
    return list(result_set)


def fetch_sequences_batch(pdb_ids_batch):
    """Fetch canonical DNA sequences for a batch of entry IDs via GraphQL."""
    ids_str = ','.join(f'"{pid}"' for pid in pdb_ids_batch)
    query = {
        "query": (
            "{ entries(entry_ids: [" + ids_str + "]) { "
            "rcsb_id polymer_entities { entity_poly { "
            "pdbx_seq_one_letter_code_can type } } } }"
        )
    }
    data = json.dumps(query).encode()
    req = urllib.request.Request(GRAPHQL_URL, data=data,
                                  headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            result = json.loads(r.read())
    except Exception as e:
        print(f"  batch error: {e}")
        return {}

    out = {}
    for entry in result.get('data', {}).get('entries', []) or []:
        pdb_id = entry.get('rcsb_id')
        seqs = []
        for pe in entry.get('polymer_entities') or []:
            ep = pe.get('entity_poly') or {}
            if ep.get('type') == 'polydeoxyribonucleotide':
                seq = ep.get('pdbx_seq_one_letter_code_can')
                if seq:
                    seqs.append(seq.upper().replace(' ', ''))
        if seqs:
            # Combine multi-chain DNA entities (e.g. duplex) into one
            # canonical key: sorted tuple of chain sequences
            out[pdb_id] = tuple(sorted(seqs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdb_ids_file', default=None,
                    help='Text file, one PDB ID per line')
    ap.add_argument('--query', action='store_true',
                    help='Re-run the RCSB search to get the ID list fresh')
    ap.add_argument('--max_resolution', type=float, default=3.0)
    ap.add_argument('--batch_size', type=int, default=50)
    ap.add_argument('--out', default='dna_sequences.tsv')
    args = ap.parse_args()

    if args.pdb_ids_file:
        pdb_ids = [l.strip() for l in open(args.pdb_ids_file) if l.strip()]
    elif args.query:
        print("Querying RCSB for pure-DNA entry list...")
        pdb_ids = fetch_pure_dna_ids(args.max_resolution)
    else:
        print("ERROR: provide --pdb_ids_file or --query")
        return

    print(f"{len(pdb_ids)} PDB IDs to check")

    seq_map = {}
    t0 = time.time()
    for i in range(0, len(pdb_ids), args.batch_size):
        batch = pdb_ids[i:i + args.batch_size]
        result = fetch_sequences_batch(batch)
        seq_map.update(result)
        if (i // args.batch_size) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+len(batch):5d}/{len(pdb_ids)}]  "
                  f"{elapsed:.0f}s elapsed")

    print(f"\nFetched sequences for {len(seq_map)}/{len(pdb_ids)} entries "
          f"({len(pdb_ids)-len(seq_map)} had no DNA polymer_entity — "
          f"may be RNA-DNA hybrids misclassified, or API mismatch)")

    # ── Redundancy analysis ──────────────────────────────────────────────
    seq_counts = Counter(seq_map.values())
    n_unique = len(seq_counts)
    n_total = len(seq_map)

    print(f"\n{'='*60}")
    print(f"  REDUNDANCY SUMMARY")
    print(f"{'='*60}")
    print(f"  Total structures with sequence data : {n_total}")
    print(f"  Unique exact DNA sequences           : {n_unique}")
    print(f"  Redundancy ratio                     : {n_total/n_unique:.2f}x")
    print(f"  (1.0x = no redundancy, higher = more repeated sequences)")

    print(f"\n  Top 15 most-repeated sequences:")
    for seq, count in seq_counts.most_common(15):
        seq_display = seq[0][:40] + ('...' if len(seq[0]) > 40 else '') if seq else ''
        print(f"    {count:4d}x   {seq_display}")

    # ── Write full mapping for downstream dedup ─────────────────────────
    with open(args.out, 'w') as f:
        f.write("pdb_id\tsequence\tduplicate_count\n")
        for pdb_id, seq in seq_map.items():
            f.write(f"{pdb_id}\t{'|'.join(seq)}\t{seq_counts[seq]}\n")
    print(f"\nWrote {args.out}")

    # ── Suggest a deduplicated ID list (one representative per sequence) ──
    seen_seqs = set()
    representatives = []
    for pdb_id, seq in seq_map.items():
        if seq not in seen_seqs:
            seen_seqs.add(seq)
            representatives.append(pdb_id)

    dedup_file = args.out.replace('.tsv', '_nonredundant_ids.txt')
    with open(dedup_file, 'w') as f:
        f.write('\n'.join(sorted(representatives)))
    print(f"Wrote {len(representatives)} non-redundant representative IDs "
          f"to {dedup_file}")


if __name__ == '__main__':
    main()
