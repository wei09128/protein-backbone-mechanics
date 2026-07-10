# MechLib (Rebuilt & Independently Verified)

This folder contains a **rebuilt** version of the conformation-dependent
geometry library described in `../library/`, produced after institutional
relocation (LSUS → LSUHSC), on an independently re-curated PISCES dataset
(8,292 structures, 1,698,562 residues; vs. the original 10,979 structures,
1,770,786 residues), with **every reference force-field parameter
independently re-verified against its authoritative primary distribution**
(AMBER ff14SB, OPLS-AA/M, CHARMM36, AMBER parm10/bsc1) rather than
propagated from earlier tabulations.

## Why the numbers differ from `../library/`

The original library reported a 31.5% strain reduction against AMBER
ff14SB. This rebuild reports **38.7%** (bootstrap 95% CI 38.1–39.3%,
resampled by structure). The difference is not a change in method or
dataset curation — it results from correcting four AMBER ff14SB
equilibrium-value discrepancies found by checking every parameter
directly against the official `protein.ff14SB.xml` distribution rather
than relying on previously tabulated values (see manuscript Methods 2.2
for the full diagnostic). Three of the four corrected values are
individually small (0.4–1.0°, or 0.009 Å) — comparable in magnitude to
the systematic conformation-dependent shifts this library is designed to
detect — which is exactly why they are easy to propagate silently through
a computational pipeline without dedicated verification against a primary
source. We verified algebraically and empirically that force-constant
errors cancel exactly in the strain-reduction percentage (the same
constant is applied to both the fixed and corrected terms), so only the
equilibrium-value corrections affect the reported numbers.

## What's new here vs. the original library

- **Independently verified parameters** for AMBER ff14SB, OPLS-AA/M,
  CHARMM36 CMAP, and AMBER parm10/bsc1 (DNA), each checked directly
  against its official distribution.
- **Bootstrap 95% confidence intervals** (resampled by structure, not
  residue, since residues within one PDB entry are not independent).
- **Temporally blind train/test validation**: library built only on
  structures released before 2020, evaluated only on structures released
  2020–2026. Held-out reduction (39.7%) slightly exceeds in-sample
  (37.8%) — evidence against overfitting.
- **Extension to the DNA sugar-phosphate backbone** (3,804 non-redundant
  structures, 157,562 nucleotides): 85.4% median strain reduction
  against AMBER parm10/bsc1 (bootstrap 95% CI 84.9–85.9%), consistent
  across all four bases and BI/BII backbone states, plus a formal
  ANOVA-style test finding genuine (non-separable) coupling between
  sugar pucker and glycosidic torsion for the sugar ring-closure angle
  (η²coupling = 0.073).
- **Documented data-quality exclusions**: 91 protein residues (43
  structures, 0.52%) confirmed as genuine local crystallographic
  modeling defects via direct atomic-distance inspection
  (`EXCLUSIONS_COMPLETE.csv`), plus correction of a chain-connectivity
  artifact in multi-stranded DNA crystal structures.

## Files

| File | Description |
|---|---|
| `MechLib_protein_library.csv` | 5,303 (residue, φ-bin, ψ-bin) entries, 11 backbone bond/angle observables |
| `MechLib_DNA_library.csv` | 74 (base, BI/BII, sugar-pucker-class) entries, 26 backbone bond/angle observables |
| `mechlib.py` | Reference Python loader with automatic fallback (specific bin → pooled bin → fixed equilibrium) |
| `EXCLUSIONS_COMPLETE.csv` | 91 protein residues excluded as documented crystallographic defects (PDB ID + residue index) |

## Usage

```python
from mechlib import MechLibProtein, MechLibDNA

protein_lib = MechLibProtein("MechLib_protein_library.csv")
tau_corrected = protein_lib.get("ALA", phi=-60.0, psi=-45.0, observable="tau_deg")

dna_lib = MechLibDNA("MechLib_DNA_library.csv")
bond_corrected = dna_lib.get("DA", bi_bii="BI", pucker_class="C2-endo",
                             observable="bond_P_O5")
```

Both classes fall back to a pooled average, then to the fixed
AMBER equilibrium, if a specific combination isn't in the table
(insufficient data in the original PDB-derived dataset) — a lookup
always returns a usable number.

## Relationship to the original library (`../library/`)

The original `constants_library.json`/`constants_chi1.json` (chi1-rotamer
resolved, MAE-based validation) remains a valid, separately-published
resource and is unchanged. A follow-up check on this rebuilt dataset
found that adding χ1 as a third binning dimension yields a further modest
improvement (40.2% vs. 38.9% reduction, for residues possessing a χ1
angle) beyond `(residue, φ, ψ)` binning alone — consistent with, and a
quantitative confirmation of, the original library's χ1-dependent
correction layer, though not re-implemented as a standalone table in this
rebuild.
