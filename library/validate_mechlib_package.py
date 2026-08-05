#!/usr/bin/env python3
"""
validate_mechlib_package.py
============================
Standalone sanity checks for the MechLib release bundle:
  constants_library.json, constants_chi1.json,
  MechLib_protein_library.csv, MechLib_protein_chi1_library.csv,
  MechLib_DNA_library.csv,
  backbone_geometry_library.py, apply_library_corrections.py, README.md

Note: the AMBER-native frcmod delivery path was deliberately dropped
(standard frcmod ANGLE/BOND terms are keyed by atom type, not residue
identity, so this library's per-residue phi/psi-dependent corrections
can't be expressed that way without custom atom-typing). Section 4
below checks that the removal was done cleanly (no orphaned file, no
dangling README promise) rather than re-validating frcmod content.

Run from the directory containing all of the above.
Exits non-zero if any CRITICAL check fails.
"""
import json
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
FAIL = []
WARN = []

def critical(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(name)

def warn(name, cond, detail=""):
    status = "OK" if cond else "WARN"
    print(f"[{status}]   {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        WARN.append(name)


# ---------------------------------------------------------------- load ----
lib = json.load(open(HERE / "constants_library.json"))
chi1 = json.load(open(HERE / "constants_chi1.json"))
prot_csv = pd.read_csv(HERE / "MechLib_protein_library.csv")
chi1_csv = pd.read_csv(HERE / "MechLib_protein_chi1_library.csv")
dna_csv = pd.read_csv(HERE / "MechLib_DNA_library.csv")
readme = (HERE / "README.md").read_text()
mod_src = (HERE / "backbone_geometry_library.py").read_text()
stub_src = (HERE / "apply_library_corrections.py").read_text()

sys.path.insert(0, str(HERE))
from backbone_geometry_library import GeometryLibrary  # noqa: E402  # PATCHED

print("=" * 70)
print("1. SCHEMA / FIELD COVERAGE")
print("=" * 70)

all_keys = set()
n_cells = 0
for res, phis in lib.items():
    for phi, psis in phis.items():
        for psi, cell in psis.items():
            n_cells += 1
            all_keys.update(cell.keys())

critical("constants_library.json parses and has residue+ALL top level",
          set(lib.keys()) >= {"ALA", "ALL"})
warn("README table promises *_std / *_k / *_coupling fields",
     any(k.endswith(("_std", "_k", "_coupling")) for k in all_keys),
     f"none found; actual fields = {sorted(all_keys)}")
warn("README/API doc-string promises omega_deg_eq (lib.get_omega example)",
     any("omega" in k for k in all_keys),
     "0 of {} cells contain an omega field -> get_omega() always returns "
     "the hardcoded AMBER default (180.0), never a data-driven value".format(n_cells))

print()
print("=" * 70)
print("2. CHI1 SUB-LIBRARY BIN-SIZE CONSISTENCY (the big one)")
print("=" * 70)
g = GeometryLibrary(str(HERE / "constants_library.json"), str(HERE / "constants_chi1.json"))
main_bin = g._bin_size
# infer chi1's actual bin size from its own keys
sample_res = next(iter(chi1))
chi1_phi_keys = sorted(int(k) for k in chi1[sample_res].keys())
chi1_bin = chi1_phi_keys[1] - chi1_phi_keys[0] if len(chi1_phi_keys) > 1 else None
# NOTE: the two bin sizes are legitimately allowed to differ (the chi1
# sub-library is commonly exported coarser than the main table) -- the
# real invariant is that GeometryLibrary must resolve chi1 lookups using
# chi1's OWN bin size, not silently reuse the main library's. Check that
# directly against the library's internal state instead of requiring
# equality.
critical("GeometryLibrary resolves chi1 lookups using constants_chi1.json's "
         "own bin size (not the main library's)",
         getattr(g, "_chi1_bin_size", None) == chi1_bin,
         f"main lib bin={main_bin} deg, chi1 lib bin={chi1_bin} deg "
         f"({chi1_phi_keys[:5]}...); GeometryLibrary._chi1_bin_size="
         f"{getattr(g, '_chi1_bin_size', 'MISSING')} -- if this doesn't "
         f"match chi1_bin, chi1 lookups will silently miss")

# brute-force confirm
hits = sum(
    1 for phi in np.arange(-180, 180, 5) for psi in np.arange(-180, 180, 5)
    for rot in ("g-", "t", "g+")
    if g.get_chi1_correction(phi, psi, "ARG", rot) is not None
)
critical("get_chi1_correction() returns a non-None value at least once "
         "on a dense (phi,psi,rotamer) grid for ARG",
         hits > 0, f"{hits} hits out of a dense grid probe -> fully dead code path")

print()
print("=" * 70)
print("3. CSV <-> JSON <-> README COUNT CONSISTENCY")
print("=" * 70)
warn("README protein cell count claim matches shipped protein CSV row count",
     len(prot_csv) == 5303, f"CSV actually has {len(prot_csv)} rows")
warn("README chi1 entry count claim matches shipped chi1 CSV row count",
     len(chi1_csv) == 2987, f"CSV actually has {len(chi1_csv)} rows")
# NOTE: reference counts updated 5303/2987 to match the independently
# re-verified rebuild (see README's "Update" section) -- the previous
# 745/2425 checked here were the pre-rebuild release's counts and no
# longer describe what this bundle ships.

print()
print("=" * 70)
print("4. AMBER-NATIVE PATH — CONFIRM CLEAN REMOVAL")
print("=" * 70)
critical("library_correction.frcmod is not shipped in the release directory",
         not (HERE / "library_correction.frcmod").exists(),
         "the frcmod file is still present; the AMBER-native path was "
         "supposed to be dropped (undeliverable in standard frcmod syntax "
         "-- ANGLE/BOND terms are keyed by atom type, not residue identity)")
critical("README no longer promises a 'library_correction.frcmod' deliverable",
         "library_correction.frcmod" not in readme,
         "README still references library_correction.frcmod somewhere -- "
         "dangling promise for a file that is no longer shipped")
critical("README no longer advertises an 'AMBER native' user path in the "
         "Who Uses What table",
         "(AMBER native)" not in readme,
         "README's 'Who Uses What' table still lists an AMBER-native row "
         "pointing at a file that no longer exists")
warn("README explains WHY no frcmod is shipped, not just that it's absent",
     "atom type" in readme.lower(),
     "removing the row silently (no rationale) will just prompt the same "
     "question again from the next AMBER user who reads the README")

print()
print("=" * 70)
print("5. apply_corrections() OBSERVABLE COVERAGE")
print("=" * 70)
angle_calls = re.findall(r"add_angle\(atoms\['(\w+)'\](?:,\s*|.*?)atoms\['(\w+)'\]"
                          r"(?:,\s*|.*?)(?:atoms|next_atoms)\['(\w+)'\]", mod_src)
bond_calls = re.findall(r"add_bond\(atoms\['(\w+)'\](?:,\s*|.*?)(?:atoms|next_atoms)\['(\w+)'\]", mod_src)
has_custom_bond_force = "CustomBondForce" in mod_src
print(f"Angle triples referenced via add_angle() in apply_corrections(): {angle_calls}")
print(f"Bond pairs referenced via add_bond() in apply_corrections(): {bond_calls}")
n_angle_kinds = len(set(angle_calls))  # order matters for angles -- the middle
                                        # atom is the vertex, so ('CA','C','N') and
                                        # ('C','N','CA') are genuinely different angles
n_bond_kinds = len({tuple(sorted(t)) for t in bond_calls})  # order doesn't matter for bonds
warn("apply_corrections() corrects all 11 library observables "
     "(README: 'One line to correct all backbone angle equilibria')",
     n_angle_kinds >= 6 and n_bond_kinds >= 5 and has_custom_bond_force,
     f"found {n_angle_kinds} distinct angle kinds and {n_bond_kinds} distinct "
     f"bond kinds referenced (need 6 angles + 5 bonds), "
     f"CustomBondForce present={has_custom_bond_force}")

print()
print("=" * 70)
print("6. apply_library_corrections.py — SHIM DELEGATES TO REAL IMPL")
print("=" * 70)
critical("apply_library_corrections.py no longer contains its own hardcoded "
         "phi/psi logic (the old alpha-helix-bin-for-every-residue bug)",
         "phi_key, psi_key = \"-65\", \"-45\"" not in stub_src,
         "the hardcoded '-65'/'-45' fallback is still present in this file's "
         "own source -- it should now just re-export "
         "backbone_geometry_library.apply_corrections instead of "
         "reimplementing it")
sys.path.insert(0, str(HERE))
import importlib
import apply_library_corrections as shim
import backbone_geometry_library as real_mod
importlib.reload(shim)
critical("apply_library_corrections.apply_corrections is the SAME function "
         "object as backbone_geometry_library.apply_corrections (not a "
         "second, divergent implementation)",
         shim.apply_corrections is real_mod.apply_corrections,
         "these are two different function objects -- there are still two "
         "implementations to keep in sync, which is exactly how the "
         "original bug happened")

print()
print("=" * 70)
print("7. CLI DOCUMENTED IN README")
print("=" * 70)
critical("backbone_geometry_library.py has a __main__/argparse CLI "
         "(README documents `python backbone_geometry_library.py --phi ...`)",
         "__main__" in mod_src,
         "no `if __name__ == '__main__':` block exists in the file at all — "
         "the documented CLI usage cannot run")

print()
print("=" * 70)
print("8. TRUTHY-ZERO FALLBACK BUG (edge case)")
print("=" * 70)
phi_test, psi_test = 0.0, 0.0
fallback_triggers = ("phi = phi or -63.0" in mod_src)
critical("phi/psi fallback uses an explicit `is None` check, not truthiness",
         "phi is None" in mod_src and "phi = phi if phi is not None else" in mod_src,
         "`phi = phi or -63.0` / `psi = psi or -43.0` treat a legitimately "
         "computed dihedral of exactly 0.0 deg as falsy and silently replace it "
         "with the hardcoded default — rare in practice but a real bug")

print()
print("=" * 70)
print("9. DNA LIBRARY WIRING")
print("=" * 70)
critical("Some code path loads/consumes MechLib_DNA_library.csv or a DNA JSON",
         "dna" in mod_src.lower() or "DNA" in mod_src,
         "no class or function anywhere in backbone_geometry_library.py or "
         "apply_library_corrections.py references DNA at all — the DNA table "
         "is an orphaned CSV with zero supporting code")

print()
print("=" * 70)
print(f"SUMMARY: {len(FAIL)} CRITICAL failures, {len(WARN)} warnings")
print("=" * 70)
for f in FAIL:
    print("  CRITICAL:", f)
for w in WARN:
    print("  WARN:", w)

sys.exit(1 if FAIL else 0)
