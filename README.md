# Protein Backbone Mechanics: A Four-Paper Series

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Data: 1.7M residues](https://img.shields.io/badge/data-1.7M%20residues-green.svg)](#dataset)

**A complete framework for understanding and correcting protein and DNA backbone geometry, from first-principles mechanics to a practical, cross-force-field correction library.**

Wei Chen, Xiaohong Lu
Louisiana State University Health Sciences Center, Shreveport

---

## Overview

This repository contains the data, code, and geometry library for a four-paper series that reinterprets protein backbone geometry as a mechanical system and delivers MechLib, a practical, conformation-dependent correction library for molecular mechanics force fields, extended to DNA.

| Paper | Title | Key Finding |
|:---:|---|---|
| **I** | Protein Backbone Mechanics I: The Ramachandran Plot as a Mechanical Force Field | The Ramachandran plot behaves as a mechanical force field, with two orthogonal steric discontinuities at φ ≈ −60° and ψ ≈ −40° |
| **II** | Protein Backbone Mechanics II: Systematic Bond-Angle Deformation Across Conformational Space | τ (N–Cα–C) varies 8.4° peak-to-peak across the Ramachandran plane; deformation is not fixed but conformation-dependent |
| **III** | Protein Backbone Mechanics III: Geometric Coupling and Bonded Strain | φ×ψ coupling adds ΔR² = 0.3–2.7% beyond additive models; quantum-mechanical corrections (n→π*, hyperconjugation) are geometrically redundant with it |
| **IV** | MechLib: A Conformation-Dependent Geometry Library for Protein and Nucleic Acid Backbone Mechanics | Drop-in correction library for AMBER/OPLS-AA/M/CHARMM36; reduces protein phantom strain by 38.7%, extends to the DNA sugar-phosphate backbone |

**The central deliverable is MechLib** — a conformation-dependent geometry correction library that replaces the fixed backbone bond-length and bond-angle equilibria used in AMBER, CHARMM, and OPLS with values derived directly from experimental structures, for both protein and DNA backbones.

---

## Quick Start

### Installation

```bash
git clone https://github.com/wei09128/protein-backbone-mechanics.git
cd protein-backbone-mechanics/library
pip install -r requirements.txt   # numpy only, for using the library itself
```

### Basic Usage

```python
from backbone_geometry_library import GeometryLibrary

lib = GeometryLibrary()

# Look up geometry for alanine in alpha-helix
geom = lib.get(phi=-63, psi=-43, residue='ALA')
print(f"tau = {geom['tau']:.1f} deg")
print(f"N-Ca bond = {geom['bond_NCA']:.3f} Ang")
```

### DNA Backbone Geometry

```python
from backbone_geometry_library import DNAGeometryLibrary

dna = DNAGeometryLibrary()
geom = dna.get(base='DA', bi_bii='BI', pucker_class='C2-endo')
# Returns corrected bond lengths/angles for this (base, BI/BII, pucker)
# combination, or None if unpopulated -- no pooled fallback for DNA.
```

### Command-Line Query

```bash
python backbone_geometry_library.py --phi -63 --psi -43 --residue GLY --compare
python backbone_geometry_library.py --phi -120 --psi 130 --all_residues
```

### OpenMM Force-Field Integration

```python
from openmm.app import PDBFile, ForceField
from backbone_geometry_library import apply_corrections

pdb = PDBFile('protein.pdb')
ff = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
system = ff.createSystem(pdb.topology)

system = apply_corrections(system, pdb.topology, pdb.positions,
                            force_field='amber')  # or 'charmm' or 'opls'
```

Note: `apply_corrections()` computes phi/psi and looks up each residue's
correction **once, from the structure passed in** — the resulting target
geometry is fixed for the rest of that simulation, not re-queried every
timestep as the structure evolves.

---

## Repository Structure

```
protein-backbone-mechanics/
├── README.md
├── features_lj_v3_FINAL_CLEAN.csv     # Raw per-residue feature table (protein), 1,698,562 rows
│
├── library/                           # The MechLib deliverable (Paper IV)
│   ├── constants_library.json         # Main library: 5,303 (residue, phi, psi) cells x 21 classes
│   ├── constants_chi1.json            # chi1-dependent Cbeta angle corrections (2,987 entries)
│   ├── MechLib_protein_library.csv    # Flat, human-readable version of constants_library.json
│   ├── MechLib_protein_chi1_library.csv
│   ├── MechLib_DNA_library.csv        # DNA sugar-phosphate backbone corrections (74 entries)
│   ├── backbone_geometry_library.py   # GeometryLibrary, DNAGeometryLibrary, apply_corrections()
│   ├── apply_library_corrections.py   # Thin re-export shim (see module docstring)
│   ├── validate_mechlib_package.py    # Automated package validation suite (see below)
│   ├── requirements.txt               # numpy (only dependency to USE the library)
│   ├── requirements-dev.txt           # + pandas, for running validate_mechlib_package.py
│   └── README.md                      # Full library API documentation
│
├── manuscript/
│   ├── paper1_rdm_v3_clear.docx
│   ├── paper2_gd_v3_clear.docx
│   ├── paper3_v3_JCIM_MAIN.docx
│   ├── paper3_v3_JCIM_SI.docx
│   ├── paper4_v3_JCIM_MAIN.docx
│   └── paper4_v3_JCIM_SI.docx
│
├── figures_and_supplementary_material/
│   ├── paper1_figure1.png ... paper1_figure7.png, paper1_table_S1_stiffness_comparison.csv
│   ├── paper2_TOC_graphic.png, paper2_figure1.png ... paper2_figure5.png,
│   │   paper2_figure_s3_sidechain_scatter.png, paper2_table_s1.csv, paper2_s2.csv,
│   │   paper2_table_s4_relaxing.xlsx, paper2_log.txt
│   ├── paper3_TOC.png, paper3_figure1.png, paper3_figure1_new.png,
│   │   paper3_figure2A.png, paper3_figure2B.png, paper3_figure3.png, paper3_figure4.png
│   └── paper4_TOC.png, paper4_figure1_protein_amber_mechlib.png ... paper4_figure9_resolution_check.png
│
└── scripts/                           # Analysis pipeline, prefixed by paper (paper1_*, paper2_*,
                                        # paper3_*, paper4_*) plus shared utilities (geom_utils.py,
                                        # molcore.py, nerf_builder.py, pdb_loader.py,
                                        # features_collector.py, download_pdbs.py, etc.).
                                        # See scripts/ for the complete, current listing --
                                        # this section intentionally isn't duplicated here to
                                        # avoid drifting out of sync with it again.
```

.github/workflows/validate-mechlib.yml runs `library/validate_mechlib_package.py`
automatically on every push or pull request touching `library/`.

---

## Key Results

Verified directly against `library/` and the current manuscripts (Table S2 in Paper IV's Supporting Information):

| Comparison | Result | Detail | n |
|---|---|---|---|
| Protein vs. AMBER ff14SB | **38.7%** phantom strain reduction | 95% CI [38.1%, 39.3%] | 1,698,562 residues |
| Protein, temporally blind holdout (2020-2026 structures) | **39.7%** | cf. 37.8% in-sample | 428,750 residues |
| Protein vs. OPLS-AA/M | ≈ same as AMBER | 10 of 11 bonded terms numerically identical | 1,698,562 residues |
| Protein vs. CHARMM36 CMAP | r = -0.25, p < 0.001 | independent geometric vs. energetic correction | 740 bins |
| DNA vs. AMBER parm10 (median, overall) | **85.4%** | 95% CI [84.9%, 85.9%] | 157,562 nucleotides |
| DNA, BI / BII backbone state | 85.4% / 85.8% | both states after chain-connectivity fix | 118,254 / 20,508 nt |
| DNA delta-chi coupling (C1'-O4'-C4' angle) | eta-squared = 0.073 | vs. 0.026 for a geometrically unrelated control angle | 150,022 nt |

The temporally blind result is the strongest single piece of evidence here: a correction
library built only from structures deposited before 2020 performs, if anything, slightly
*better* on structures deposited after 2020 than on its own training data — the pattern
expected of a real, generalizable geometric regularity rather than overfitting.

---

## Paper Summaries

### Paper I — The Ramachandran Plot as a Mechanical Force Field
Reinterprets the Ramachandran plot in explicitly mechanical terms: local (phi,psi) conformations
represent dynamic equilibria among steric, electrostatic, and hydrogen-bonding forces, with two
orthogonal steric discontinuities at phi ~ -60 deg and psi ~ -40 deg.

### Paper II — Systematic Bond-Angle Deformation Across Conformational Space
Shows backbone bond angles are not fixed: tau (N-Ca-C) varies 8.4 deg peak-to-peak across the
Ramachandran plane, deforming systematically with conformation and residue identity.

### Paper III — Geometric Coupling and Bonded Strain
Tests whether backbone bonded geometry separates into independent phi- and psi-dependent
contributions. GAM decomposition shows non-additive coupling (DeltaR^2 = 0.3-2.7%), strongest
in coil regions, with a second, independent coupling channel introduced by sidechain chi1
rotamer state. Candidate electronic-structure explanations (n->pi*, hyperconjugation) are
geometrically redundant with the observed coupling.

### Paper IV — MechLib
Builds, validates, and delivers the correction library itself: consistent phantom-strain
reduction across four independently verified force fields, a temporally blind holdout test
ruling out overfitting, and extension of the entire framework to the DNA sugar-phosphate
backbone, including a formally quantified delta-chi coupling.

---

## Dataset

Backbone geometry extracted from crystal structures via the PISCES server (sequence
identity and resolution filters as described in each paper's Methods), yielding
**1,698,562** quality-filtered protein residues. The raw per-residue feature table,
`features_lj_v3_FINAL_CLEAN.csv`, is included at the repository root.

DNA backbone geometry was extracted analogously from a non-redundant DNA structure set;
see Paper IV Methods 2.6 for the full curation pipeline.

---

## Validating the Library

```bash
cd library
pip install -r requirements-dev.txt
python validate_mechlib_package.py
```

This runs the full automated check (schema coverage, chi1 bin-size resolution, CSV/JSON/README
count consistency, correct removal of the AMBER-native frcmod path, `apply_corrections()`
observable coverage, shim delegation, CLI presence, the phi/psi truthy-zero edge case, and DNA
library wiring). A GitHub Actions workflow runs this automatically on every push or pull request
touching `library/`.

---

## Citation

If you use MechLib or any part of this work, please cite the relevant paper(s) in this series
(Papers I-IV, Chen & Lu). Full citation details will be added once each paper's DOI is assigned;
see `manuscript/` for the current preprint/submission text of each.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

The geometry library (`constants_library.json`, `constants_chi1.json`, `MechLib_DNA_library.csv`)
is released under CC-BY-4.0 for maximum reuse.

---

## Contact

**Wei Chen** and **Xiaohong Lu** — Louisiana State University Health Sciences Center, Shreveport

Issues and pull requests welcome.
