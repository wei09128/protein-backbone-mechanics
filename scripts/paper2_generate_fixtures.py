"""
generate_fixtures.py — Prepare validation ground truth
=======================================================
One-time script. Run once from the v6_refactor root to set up test fixtures.

What it does:
  1. Ensures a list of reference PDBs exist in pdb_cache (downloads missing ones)
  2. Runs mkdssp on the small reference structures (1UBQ, 1CRN)
  3. Parses DSSP output into a canonical H-bond list
  4. Saves fixtures/<pdb_id>_dssp.pkl for use by later validation scripts

Reference set:
  1UBQ   — ubiquitin, 76 res, α + β, classic test case
  1CRN   — crambin, 46 res, two short helices, very clean
  2QKE   — KaiB ground state (fold-switcher)
  5JYT   — KaiB fold-switched
  2OUG   — RfaH autoinhibited (CTD α-helical)
  6C6S   — RfaH active (CTD β-barrel)
  1J8I   — XCL1 canonical
  2JP1   — XCL1 alternative fold

After this script runs you should see:
  pdb_cache/1ubq.pdb, 1crn.pdb, 2qke.pdb, ...
  fixtures/1ubq_dssp.pkl, 1crn_dssp.pkl

Usage:
    python generate_fixtures.py --pdb_cache F:/Protein_Folding/pdb_cache \
                                --fixtures  F:/Protein_Folding/v6_refactor/tests/fixtures
"""

import argparse
import pickle
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


# ── Reference structures ─────────────────────────────────────────────────────
REFERENCE_PDBS = {
    '1ubq': 'ubiquitin, 76 res (classic α+β)',
    '1crn': 'crambin, 46 res (clean, small)',
    '2qke': 'KaiB ground state (fold-switcher)',
    '5jyt': 'KaiB fold-switched',
    '2oug': 'RfaH autoinhibited (CTD α-helical)',
    '6c6s': 'RfaH active (CTD β-barrel)',
    '1j8i': 'XCL1 canonical chemokine',
    '2jp1': 'XCL1 alternative dimer fold',
}

# Which ones to run mkdssp on. We do them all so Layer 4 of the
# hbond_finder self-test can diff against each one. Extra cost: ~seconds.
DSSP_TARGETS = list(REFERENCE_PDBS.keys())

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


# ══════════════════════════════════════════════════════════════════════════════
# PDB download
# ══════════════════════════════════════════════════════════════════════════════

def download_pdb(pdb_id: str, cache_dir: Path) -> bool:
    """Download a PDB file from RCSB if not already present. Returns True if ok."""
    # Check for common case variations (pdb_cache may use UPPER or lower)
    candidates = [
        cache_dir / f"{pdb_id.lower()}.pdb",
        cache_dir / f"{pdb_id.upper()}.pdb",
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            print(f"  [exists] {c.name}")
            return True

    # Download
    target = cache_dir / f"{pdb_id.lower()}.pdb"
    url = RCSB_URL.format(pdb_id=pdb_id.lower())
    try:
        print(f"  [fetch ] {url}", end=' ', flush=True)
        urllib.request.urlretrieve(url, target)
        size_kb = target.stat().st_size / 1024
        print(f"OK ({size_kb:.1f} KB)")
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Chain extraction — DSSP and our H-bond finder must see the same atoms
# ══════════════════════════════════════════════════════════════════════════════

# Standard 20 amino acids. Anything else (nucleotides, ligands, waters) is
# rejected when picking the first protein chain.
_PROTEIN_RESIDUES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    # Common non-standard protein residues worth keeping
    'MSE',  # selenomethionine — treated as MET
    'SEC',  # selenocysteine
    'PYL',  # pyrrolysine
    'HID', 'HIE', 'HIP',  # His protonation variants
    'CYX', 'CYM',  # Cys variants
}


def _chain_composition(pdb_path: Path) -> list:
    """
    Return a list of (chain_id, n_protein_residues, first_resname) for every
    chain that contains at least one ATOM record, in order of first appearance.

    For NMR ensembles, only the first MODEL is considered, matching
    extract_first_chain() and mkdssp's default behavior.
    """
    seen = {}        # chain_id → dict(n_protein=int, first_res=str, order=int)
    seen_residues = set()  # (chain_id, resseq+icode) → don't double-count
    order = 0
    in_model = None  # None=outside, 1=in first model, 'done'=past it

    with open(pdb_path) as f:
        for line in f:
            if line.startswith('MODEL'):
                if in_model is None:
                    in_model = 1
                else:
                    in_model = 'done'
                continue
            if line.startswith('ENDMDL'):
                if in_model == 1:
                    in_model = 'done'
                continue
            if in_model == 'done':
                continue
            if not line.startswith('ATOM'):
                continue
            if len(line) < 27:
                continue
            chain_id = line[21]
            resname = line[17:20].strip()
            resseq = line[22:27]  # includes insertion code

            if chain_id not in seen:
                seen[chain_id] = {'n_protein': 0, 'first_res': resname,
                                  'order': order}
                order += 1

            # Count this residue once (regardless of how many atoms it has)
            key = (chain_id, resseq)
            if key in seen_residues:
                continue
            seen_residues.add(key)

            if resname in _PROTEIN_RESIDUES:
                seen[chain_id]['n_protein'] += 1

    # Return in original order of appearance
    result = sorted(seen.items(), key=lambda kv: kv[1]['order'])
    return [(ch, info['n_protein'], info['first_res']) for ch, info in result]


def extract_first_chain(pdb_path: Path, out_path: Path,
                        min_protein_residues: int = 10) -> str:
    """
    Write a new PDB containing only the first *protein* chain.

    "First protein chain" = the earliest-appearing chain whose ATOM records
    contain at least `min_protein_residues` standard amino acids. Chains
    that are pure DNA, RNA, ligand, or tiny peptide fragments are skipped.

    Returns the chain ID that was kept (e.g. 'A'), or '' if no protein
    chain was found.

    Why: mkdssp processes all chains, but our H-bond finder runs on
    NeRFBuilder which parses only the first chain. To diff the two bond
    lists they must see the same atoms. And in modern cryo-EM depositions
    (6C6S, ribosomes, polymerase complexes, ...) chain A is often DNA
    or RNA, so picking chain A naively gives zero protein bonds.
    """
    # Find the first chain with enough protein content
    comp = _chain_composition(pdb_path)
    chain_to_keep = None
    for chain_id, n_prot, first_res in comp:
        if n_prot >= min_protein_residues:
            chain_to_keep = chain_id
            break

    if chain_to_keep is None:
        return ''

    # Copy headers + ATOM records for that chain only.
    # NMR structures contain multiple MODEL/ENDMDL blocks of the same
    # residues at slightly different coordinates. mkdssp processes only
    # model 1 by default, so we must do the same or our residue indices
    # will desync from the DSSP fixture. We stop at the first ENDMDL.
    header_lines = []
    atom_lines = []
    in_model = None   # None=outside, 1=in first model, 'done'=past first model
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(('HEADER', 'TITLE ', 'COMPND', 'SOURCE',
                                'CRYST1', 'SCALE', 'ORIGX')):
                header_lines.append(line)
                continue
            if line.startswith('MODEL'):
                if in_model is None:
                    in_model = 1
                else:
                    in_model = 'done'  # second MODEL seen → stop
                continue
            if line.startswith('ENDMDL'):
                if in_model == 1:
                    in_model = 'done'
                continue
            if in_model == 'done':
                continue
            if not line.startswith('ATOM'):
                continue
            if line[21] != chain_to_keep:
                continue
            atom_lines.append(line)

    with open(out_path, 'w') as f:
        if not any(h.startswith('HEADER') for h in header_lines):
            f.write(f"HEADER    EXTRACTED CHAIN {chain_to_keep}"
                    f"                                                  \n")
        f.writelines(header_lines)
        f.writelines(atom_lines)
        f.write(f"TER\nEND\n")

    return chain_to_keep


# ══════════════════════════════════════════════════════════════════════════════
# mkdssp driver
# ══════════════════════════════════════════════════════════════════════════════

def check_mkdssp() -> bool:
    """Check that mkdssp is available on PATH."""
    if shutil.which('mkdssp') is None:
        print("  ERROR: mkdssp not found on PATH")
        return False
    try:
        r = subprocess.run(['mkdssp', '--version'],
                           capture_output=True, text=True, timeout=10)
        print(f"  mkdssp: {r.stdout.strip() or r.stderr.strip()}")
        return True
    except Exception as e:
        print(f"  ERROR running mkdssp: {e}")
        return False


def run_mkdssp(pdb_path: Path, out_path: Path) -> bool:
    """
    Run mkdssp pdb_path → out_path (classic text DSSP format).

    mkdssp 4.x defaults to mmCIF output; we force the classic format with
    --output-format dssp, which matches the original Kabsch & Sander layout.
    """
    try:
        r = subprocess.run(
            ['mkdssp', '--output-format', 'dssp', str(pdb_path), str(out_path)],
            capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            print(f"    mkdssp failed (rc={r.returncode}): {r.stderr.strip()[:200]}")
            return False
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception as e:
        print(f"    mkdssp exception: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# DSSP parser — extract H-bonds from mkdssp text output
# ══════════════════════════════════════════════════════════════════════════════
#
# DSSP output format (classic + mkdssp 4.x):
#
#   #  RESIDUE AA STRUCTURE BP1 BP2  ACC  N-H-->O    O-->H-N   N-H-->O    O-->H-N ...
#
# After the '  #  RESIDUE' header line, each residue occupies one line.
# The four H-bond columns are paired offset/energy values:
#     N-H-->O   (current residue as DONOR,    two bonds: best and 2nd best)
#     O-->H-N   (current residue as ACCEPTOR, two bonds: best and 2nd best)
#
# Each pair is formatted like " -12,-2.8" meaning:
#   partner is current residue index + (-12), with energy -2.8 kcal/mol
#
# We extract all four (donor×2, acceptor×2) and emit bonds as (donor_idx, acceptor_idx, energy).

def parse_dssp_hbonds(dssp_path: Path):
    """
    Parse a DSSP file and return a list of bond dicts:
      {'donor': int, 'acceptor': int, 'energy': float, 'dssp_idx_donor': int, 'dssp_idx_acceptor': int}

    Indices here are DSSP's internal residue numbering (1-based, sequential,
    matching the leftmost '#' column in DSSP). The caller should map these
    to whatever indexing the rest of the pipeline uses.
    """
    lines = dssp_path.read_text(errors='replace').splitlines()

    # Find the residue section: starts with a line containing "#  RESIDUE AA"
    start = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith('#') and 'RESIDUE' in line and ' AA ' in line:
            start = i + 1
            break
    if start is None:
        raise ValueError(f"{dssp_path}: no '#  RESIDUE AA' header found")

    # Each residue record: DSSP index is columns 0-5 ('  #' field)
    # H-bond columns begin at a fixed offset. We'll find them by searching
    # the header line for positions of 'N-H-->O' and 'O-->H-N'.
    header = lines[start - 1]
    # There are two 'N-H-->O' and two 'O-->H-N' columns.
    # Their character positions in the header are stable, but we'll locate them robustly.
    nho_positions = []
    ohn_positions = []
    i = 0
    while i < len(header):
        if header[i:i + 7] == 'N-H-->O':
            nho_positions.append(i)
            i += 7
        elif header[i:i + 7] == 'O-->H-N':
            ohn_positions.append(i)
            i += 7
        else:
            i += 1
    if len(nho_positions) < 2 or len(ohn_positions) < 2:
        raise ValueError(f"{dssp_path}: expected 2 N-H-->O and 2 O-->H-N columns, "
                         f"got {len(nho_positions)} and {len(ohn_positions)}")

    # Each field is actually 10 characters wide and *starts one column
    # before* the header marker. DSSP right-aligns the offset in a 3-char
    # slot ending at the comma: positive offsets are ' 62,' and negative
    # offsets are '-62,', both occupying the same visual width. So the
    # minus sign sits in the column immediately before the header marker.
    # Width 10 is enough for '-62,-3.2' plus trailing spaces.
    FIELD_START_OFFSET = -1
    FIELD_WIDTH = 10

    def read_field(line: str, header_col: int):
        """Read one ' offset, energy' field. Returns (offset, energy) or None."""
        start = header_col + FIELD_START_OFFSET
        frag = line[start:start + FIELD_WIDTH]
        if ',' not in frag:
            return None
        try:
            off_str, e_str = frag.split(',', 1)
            off = int(off_str.strip())
            e = float(e_str.strip())
            if off == 0:
                return None  # null bond
            return off, e
        except (ValueError, IndexError):
            return None

    bonds = []
    seen = set()  # dedupe: donor bonds and acceptor bonds report the same pair

    for line in lines[start:]:
        if len(line) < 30:
            continue
        # DSSP index column: leftmost integer
        try:
            dssp_idx = int(line[0:5].strip())
        except ValueError:
            continue  # chain-break marker

        # Skip chain-break markers (residue field is '!')
        if len(line) > 13 and line[13] == '!':
            continue

        # Two donor bonds (this residue's N-H → O of partner)
        for col in nho_positions:
            result = read_field(line, col)
            if result is None:
                continue
            off, e = result
            partner = dssp_idx + off
            key = ('nho', dssp_idx, partner)
            if key in seen:
                continue
            seen.add(key)
            bonds.append({
                'donor':    dssp_idx,
                'acceptor': partner,
                'energy':   e,
                'source':   'N-H-->O',
            })

        # Two acceptor bonds (this residue's C=O ← H-N of partner)
        for col in ohn_positions:
            result = read_field(line, col)
            if result is None:
                continue
            off, e = result
            partner = dssp_idx + off
            key = ('ohn', partner, dssp_idx)  # canonical: donor first
            if key in seen:
                continue
            seen.add(key)
            bonds.append({
                'donor':    partner,
                'acceptor': dssp_idx,
                'energy':   e,
                'source':   'O-->H-N',
            })

    # Dedupe: the same physical bond appears from both sides.
    # Key by (donor, acceptor); keep the one with the more negative energy
    # (DSSP sometimes prints slightly different values depending on side).
    merged = {}
    for b in bonds:
        key = (b['donor'], b['acceptor'])
        if key not in merged or b['energy'] < merged[key]['energy']:
            merged[key] = b

    result = sorted(merged.values(), key=lambda b: b['energy'])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Self-test — validates the DSSP parser without needing network or big PDBs
# ══════════════════════════════════════════════════════════════════════════════

SYNTHETIC_DSSP = """\
==== Secondary Structure Definition by the program DSSP, test fixture              ==== DATE=2026-01-01        .
REFERENCE TEST                                                                                                                 .
HEADER    SYNTHETIC                                                                                                            .
    5  1  0  0  0 TOTAL NUMBER OF RESIDUES                                                                                     .
  #  RESIDUE AA STRUCTURE BP1 BP2  ACC     N-H-->O    O-->H-N    N-H-->O    O-->H-N    TCO  KAPPA ALPHA  PHI   PSI    X-CA   Y-CA   Z-CA
    1    1 A A              0   0  100      0, 0.0     2,-1.5     0, 0.0     0, 0.0   0.000 360.0 360.0 360.0 -40.0    0.0    0.0    0.0
    2    2 A A              0   0  100      0, 0.0     0, 0.0     0, 0.0     0, 0.0   0.900 100.0 100.0 -60.0 -40.0    4.0    0.0    0.0
    3    3 A A              0   0  100     -2,-1.5     0, 0.0     0, 0.0     0, 0.0   0.900 100.0 100.0 -60.0 -40.0    8.0    0.0    0.0
    4    4 A A              0   0  100    -62,-3.2     0, 0.0     0, 0.0     0, 0.0   0.000 360.0 360.0 360.0 -40.0   12.0    0.0    0.0
    5    5 A A              0   0  100      0, 0.0   104,-2.7     0, 0.0     0, 0.0   0.000 360.0 360.0 360.0 -40.0   16.0    0.0    0.0
"""


def _self_test():
    """Validate the DSSP parser on synthetic DSSP text with known bonds.

    Test cases:
      - Residue 1/3: short positive/negative offset, one shared bond (dedup test)
      - Residue 4:   large negative offset (-62, originally misread as +62)
      - Residue 5:   large positive offset (+104), tests wide field handling
    """
    import tempfile
    print("  [parser self-test]")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dssp', delete=False) as f:
        f.write(SYNTHETIC_DSSP)
        tmp = Path(f.name)
    try:
        bonds = parse_dssp_hbonds(tmp)
    finally:
        tmp.unlink()

    # Build a quick lookup by (donor, acceptor)
    by_pair = {(b['donor'], b['acceptor']): b for b in bonds}

    ok = True

    # 1. The (3 → 1, -1.5) bond should be there — it is reported from both
    #    sides (residue 1's O-->H-N and residue 3's N-H-->O) and should dedupe.
    b = by_pair.get((3, 1))
    if not b or abs(b['energy'] - (-1.5)) > 1e-6:
        print(f"    [FAIL] missing (3→1, -1.5); got {b}")
        ok = False

    # 2. Residue 4's N-H-->O field is '-62,-3.2' → donor=4, partner=4+(-62)=-58.
    #    In a real file partner -58 is invalid, but the parser should still
    #    read the offset correctly. We check the SIGN of the offset is
    #    preserved (this is the bug that v2.0 had).
    b = by_pair.get((4, -58))
    if not b or abs(b['energy'] - (-3.2)) > 1e-6:
        print(f"    [FAIL] missing (4→-58, -3.2) — negative offset broken; got {b}")
        ok = False

    # 3. Residue 5's O-->H-N field is '104,-2.7' → donor=5+104=109, acceptor=5.
    b = by_pair.get((109, 5))
    if not b or abs(b['energy'] - (-2.7)) > 1e-6:
        print(f"    [FAIL] missing (109→5, -2.7) — large positive offset broken; got {b}")
        ok = False

    # Expected: exactly 3 unique bonds (the three above, no extras)
    if len(bonds) != 3:
        print(f"    [FAIL] expected 3 bonds, got {len(bonds)}: {bonds}")
        ok = False

    if ok:
        print(f"    [PASS] 3 bonds: (3→1, -1.5), (4→-58, -3.2), (109→5, -2.7)")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdb_cache',
                    help='Directory containing (or to receive) PDB files')
    ap.add_argument('--fixtures',
                    help='Output directory for fixture files')
    ap.add_argument('--skip_download', action='store_true',
                    help='Do not download missing PDBs (assume they exist)')
    ap.add_argument('--self_test', action='store_true',
                    help='Run parser self-test on synthetic input, then exit')
    args = ap.parse_args()

    if args.self_test:
        print("=" * 70)
        print("generate_fixtures.py  —  parser self-test")
        print("=" * 70)
        ok = _self_test()
        sys.exit(0 if ok else 1)

    if not args.pdb_cache or not args.fixtures:
        ap.error("--pdb_cache and --fixtures are required (or use --self_test)")

    cache = Path(args.pdb_cache)
    fix = Path(args.fixtures)
    cache.mkdir(parents=True, exist_ok=True)
    fix.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("generate_fixtures.py")
    print("=" * 70)
    print(f"PDB cache:   {cache}")
    print(f"Fixtures:    {fix}")

    # ── Step 1: Ensure PDBs exist ────────────────────────────────────────────
    print("\n[1] Reference PDBs")
    ok_pdbs = []
    for pdb_id, desc in REFERENCE_PDBS.items():
        print(f"  {pdb_id.upper()}  ({desc})")
        if args.skip_download:
            found = any((cache / f"{pdb_id}.pdb").exists() or
                        (cache / f"{pdb_id.upper()}.pdb").exists()
                        for pdb_id in [pdb_id])
            if found:
                ok_pdbs.append(pdb_id)
        else:
            if download_pdb(pdb_id, cache):
                ok_pdbs.append(pdb_id)

    print(f"\n  → {len(ok_pdbs)}/{len(REFERENCE_PDBS)} PDBs available")

    # ── Step 2: Check mkdssp ─────────────────────────────────────────────────
    print("\n[2] mkdssp")
    if not check_mkdssp():
        print("\n  Cannot proceed without mkdssp. Install it and re-run.")
        sys.exit(1)

    # ── Step 3: Run mkdssp + parse H-bonds for targets ───────────────────────
    print("\n[3] Running mkdssp on reference targets (chain A only)")
    for pdb_id in DSSP_TARGETS:
        if pdb_id not in ok_pdbs:
            print(f"  [skip ] {pdb_id.upper()} (PDB missing)")
            continue
        pdb_path = cache / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            pdb_path = cache / f"{pdb_id.upper()}.pdb"
        chain_pdb = fix / f"{pdb_id}_chainA.pdb"
        dssp_txt = fix / f"{pdb_id}_dssp.txt"
        dssp_pkl = fix / f"{pdb_id}_dssp.pkl"

        print(f"  {pdb_id.upper()}")
        print(f"    input:  {pdb_path}")

        # Show all chains in the original file so the choice is transparent
        comp = _chain_composition(pdb_path)
        if len(comp) > 1:
            summary = ', '.join(
                f"{ch}={n}{'*' if r not in _PROTEIN_RESIDUES else ''}"
                for ch, n, r in comp[:12])
            if len(comp) > 12:
                summary += f", ...(+{len(comp)-12} more)"
            print(f"    chains: {summary}   (*=non-protein first residue)")

        # Extract the first chain with ≥10 protein residues
        chain = extract_first_chain(pdb_path, chain_pdb)
        if not chain:
            print(f"    [FAIL] no protein chain found")
            continue
        print(f"    chain:  {chain} → {chain_pdb.name}")

        if not run_mkdssp(chain_pdb, dssp_txt):
            print(f"    [FAIL] mkdssp did not produce output")
            continue

        try:
            bonds = parse_dssp_hbonds(dssp_txt)
        except Exception as e:
            print(f"    [FAIL] parser: {e}")
            continue

        # Quick statistics
        strong = sum(1 for b in bonds if b['energy'] < -0.5)
        weak = len(bonds) - strong
        if bonds:
            e_min = min(b['energy'] for b in bonds)
            e_max = max(b['energy'] for b in bonds)
            print(f"    parsed: {len(bonds)} bonds "
                  f"({strong} strong, {weak} weak)  "
                  f"E range [{e_min:.2f}, {e_max:.2f}]")
        else:
            print(f"    parsed: 0 bonds  [WARNING]")

        # Save parsed fixture.  We also stash the absolute path of the
        # chain-A PDB so downstream validation can load it without having
        # to re-extract.
        with open(dssp_pkl, 'wb') as f:
            pickle.dump({
                'pdb_id':    pdb_id,
                'source':    'mkdssp',
                'chain':     chain,
                'chain_pdb': str(chain_pdb),
                'bonds':     bonds,
                'n_bonds':   len(bonds),
                'n_strong':  strong,
                'n_weak':    weak,
            }, f)
        print(f"    saved:  {dssp_pkl.name}")

    print("\n" + "=" * 70)
    print("  Fixtures ready.")
    print("=" * 70)


if __name__ == '__main__':
    main()