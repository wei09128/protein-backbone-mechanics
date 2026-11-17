"""Reserve split for the environment-resolved geometry study (fixed 17 Nov 2026, before any analysis of that study).
Rule: a sequence group is RESERVED iff int(SHA-256(sequence_group as UTF-8), 16) mod 10 in {0, 1, 2}; otherwise DISCOVERY.
Reads only structure identifiers and group labels (no geometry). The rule, not this list, is the definition: any structure
added later is assigned by applying the rule to its sequence group.
Usage: python -I make_reserve_split.py structure_sequence_groups.tsv RESERVE_SPLIT.tsv"""
import csv, hashlib, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
with open(sys.argv[2], 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t'); w.writerow(['pdb_id', 'sequence_group', 'split'])
    for r in rows:
        g = r['sequence_group']; res = int(hashlib.sha256(g.encode()).hexdigest(), 16) % 10 in (0, 1, 2)
        w.writerow([r['pdb_id'], g, 'RESERVED' if res else 'DISCOVERY'])
groups = {r['sequence_group'] for r in rows}
rg = {g for g in groups if int(hashlib.sha256(g.encode()).hexdigest(), 16) % 10 in (0, 1, 2)}
rs = sum(1 for r in rows if r['sequence_group'] in rg)
print(f"groups {len(groups)}: reserved {len(rg)} ({len(rg)/len(groups):.1%}); structures {len(rows)}: reserved {rs} ({rs/len(rows):.1%})")
