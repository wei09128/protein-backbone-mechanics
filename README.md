\# Mechanical Origins of Backbone Conformation

\### A force-decomposition framework for Ramachandran statistics



\[!\[License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

\[!\[Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

\[!\[Dataset: PDB](https://img.shields.io/badge/dataset-PDB-orange.svg)](https://www.rcsb.org/)



\---



\## The Question



Why does a residue adopt φ = −63°, ψ = −43° rather than any other point on the 

Ramachandran map? The standard answer — steric exclusion — explains what is 

\*forbidden\* but not what \*selects\* among the permitted regions.



This repository provides a mechanical answer: backbone conformation is determined 

by a hierarchy of three force layers, each contributing distinct physical information.



\---



\## The Short Answer



| Layer | Features | Predicts | R² (φ / ψ) |

|-------|----------|----------|------------|

| Steric field | Contact shells, clash counts | Which region is accessible | 0.622 / 0.507 |

| + Mechanical forces | H-bond, electrostatic torques | Which basin is occupied | 0.690 / 0.809 |

| + Molecular context | Sidechain mechanics, B-factors | Fine-scale positioning | 0.719 / 0.820 |



\*\*Key finding:\*\* φ is sterically dominated (top features: clash counts, 

sidechain–backbone contacts). ψ is electrostatic: a single feature — 

electrostatic torque about the φ axis (τ\_φ,elec) — accounts for 52% of 

Random Forest importance for ψ prediction. The same force is mechanically 

invisible at its origin dihedral but dominates the coupled one, a direct 

consequence of Cα sp³ geometry routing torsional stress toward the more 

compliant degree of freedom.



\---



\## Key Results



\*\*1. Three force layers, not one\*\*  

Steric exclusion sets the accessible landscape. Mechanical torques 

(H-bond donors/acceptors, electrostatics) resolve which sterically permitted 

basin is actually occupied. Molecular context parameterises how efficiently 

force is transmitted through the sidechain lever.



\*\*2. The net torque paradox\*\*  

The net environmental torque predicts almost nothing (R² = 0.004 / 0.012).  

At equilibrium, forces cancel. Predictive signal lies in the \*decomposition\* — 

which sources are active, their relative signs and magnitudes — not the sum.



\*\*3. Basin-specific mechanical signatures\*\*  

\- \*\*αR\*\*: universal mechanical attractor. Every residue, every chemistry, 

&#x20; experiences coherent driving force into αR (100% sign consistency, p ≈ 0).  

\- \*\*PPII\*\*: genuine local minimum with measurable restoring curvature 

&#x20; (k = −0.021, p = 4×10⁻⁶⁶). Pro shows the deepest well; Gly is neutral.  

\- \*\*αL\*\*: two mechanisms. Gly accesses αL through steric permissiveness 

&#x20; (no Cβ); charged/polar residues are actively driven in by local 

&#x20; electrostatics (66% driving sign consistency vs 51% for Gly).  

\- \*\*β\*\*: mechanically unselective at the residue level. Stability is 

&#x20; entirely cooperative — no single residue chemistry shows coherent 

&#x20; restoring force.  

\- \*\*Loop\*\*: a transitional catch-all, not a minimum. Displacement 

&#x20; increases driving \*away\* from the basin centre.



\*\*4. φ ≠ ψ in mechanical origin\*\*  

φ is a steric degree of freedom. ψ is an electrostatic one. This asymmetry 

is not an empirical observation but a geometric consequence: the tetrahedral 

Cα couples both dihedrals and routes torsional stress toward the axis of 

greater conformational freedom.



\---



\## Repository Structure

```

├── src/
│   ├── collect_backbone_features.py  # Feature extraction
│   └── subgroup_k_analysis.py        # Main analysis pipeline
├── figures/                          # All manuscript figures (Fig 1–3G)
├── data/                             # (Folder exists, but CSV is hosted externally)
├── environment.yml                   # Conda environment setup
└── README.md

```
Note on Data: The feature matrix (features_11k.csv, ~170MB) is too large for GitHub.


\---



\## Reproducing the Analysis

```bash

\# Install dependencies

pip install numpy scipy scikit-learn matplotlib



\# Run full pipeline on pre-extracted features

python combined\_analysis.py --csv data/features\_v3.csv --out\_dir ./results



\# Fast test on subset

python combined\_analysis.py --csv data/features\_v3.csv --max\_rows 50000



\# Skip Parts 1–4 if cached results exist (Parts 5–6 only)

\# The script auto-detects cached .pkl files in --out\_dir

python combined\_analysis.py --csv data/features\_v3.csv --out\_dir ./results

```



Parts 1–4 cache automatically to `results/part{1-4}.pkl` on first run.  

Subsequent runs load from cache and proceed directly to Parts 5–6.



\---



\## Feature Groups



| Group | Description | n features |

|-------|-------------|-----------|

| \*\*A: Steric field\*\* | Per-atom contact shells (3/4/5Å), clash counts, asymmetry vector | 22 |

| \*\*B: Forces\*\* | Torques from bb H-bond donors, bb acceptors, electrostatics, sidechain H-bonds | 14 |

| \*\*C: Context\*\* | Sidechain mass/rigidity, B-factors, bond angles, neighbour properties | 20 |



The steric torque (τ\_steric) is zero by construction: a radial force F ∝ −r̂ 

is parallel to its own lever arm, producing zero cross product. Steric effects 

are instead captured non-parametrically by Group A.



\---



\## Data



\- \*\*Source\*\*: RCSB Protein Data Bank, non-redundant set  

\- \*\*Size\*\*: \~600,000 residues across \~X structures  

\- \*\*Resolution filter\*\*: ≤ 2.5 Å  

\- \*\*Exclusions\*\*: chain termini, missing Cα, cis-Pro, alternate conformations  

\- \*\*Feature extraction\*\*: `collect\_backbone\_features\_v5.py`



Pre-extracted features available at: \[link to Zenodo / figshare]



\---



\## Citation

```bibtex

@article{yourname2026,

&#x20; title   = {Mechanical Origins of Backbone Conformation: 

&#x20;            A Force-Decomposition Framework for Ramachandran Statistics},

&#x20; author  = {Your Name},

&#x20; journal = {Journal Name},

&#x20; year    = {2026},

&#x20; doi     = {}

}

```



\---



\## License



MIT — see \[LICENSE](LICENSE) for details.

