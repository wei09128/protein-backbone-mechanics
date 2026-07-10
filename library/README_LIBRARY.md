# Backbone Geometry Library — Documentation

**A conformation-dependent replacement for fixed backbone constants in protein modelling.**

This library provides (φ,ψ,residue)-dependent equilibrium values for 11 backbone bond lengths and bond angles, derived from 1.77 million residues across 10,979 protein crystal structures. It replaces the single fixed values used by AMBER, CHARMM, OPLS, and NeRF reconstruction algorithms.

---

## Files

| File | Size | Description |
|---|---|---|
| `constants_library.json` | ~13 MB | Main library: 745 Ramachandran cells × 21 residue classes × 12 observables |
| `constants_chi1.json` | ~1 MB | χ₁-dependent corrections for ∠N–Cα–Cβ and ∠C–Cα–Cβ (2,425 entries) |
| `backbone_geometry_library.py` | ~20 KB | Python module with complete API (only dependency: numpy) |

Place all three files in the same directory. The Python module auto-discovers the JSON files.

---

## JSON Schema

### constants_library.json

Nested dictionary: **residue class → φ center → ψ center → observables**

```
{
  "ALA": {                          ← residue class (20 AAs + "ALL")
    "-65": {                        ← φ bin center in degrees (string)
      "-45": {                      ← ψ bin center in degrees (string)
        "n": 12847,                 ← number of observations in this cell
        "tau_deg_eq": 111.02,       ← equilibrium τ (∠N–Cα–C) [degrees]
        "tau_deg_std": 1.62,        ← standard deviation [degrees]
        "tau_deg_k": 823.4,         ← empirical spring constant [kcal/mol/rad²]
        "tau_deg_coupling": 0.12,   ← Paper 3 coupling correction [degrees]
        "tau_deg_delta_amber": -0.08, ← deviation from AMBER ff14SB
        "angle_N_CA_CB_eq": 110.31, ← ∠N–Cα–Cβ [degrees]
        "angle_N_CA_CB_std": 1.14,
        "angle_N_CA_CB_k": 892.1,
        ...                         ← (same pattern for all 12 observables)
        "bond_N_CA_eq": 1.4601,     ← N–Cα bond length [Å]
        "bond_N_CA_std": 0.0091,
        "bond_N_CA_k": 7204.3,
        ...
        "omega_deg_eq": 179.82,     ← ω (circular mean) [degrees]
        "omega_dev_eq": 3.21,       ← |180° − |ω|| deviation [degrees]
        "omega_frac_cis": 0.002,    ← fraction of cis peptides
        "bond_C_O_coupling": 0.0,               ← zeroed (unreliable)
        "bond_C_O_coupling_reliable": false,     ← reliability flag
        "bond_C_N_next_coupling": 0.0,           ← zeroed (unreliable)
        "bond_C_N_next_coupling_reliable": false
      },
      "-35": { ... },
      ...
    },
    "-55": { ... },
    ...
  },
  "GLY": { ... },
  "VAL": { ... },
  ...
  "ALL": { ... }                    ← pooled across all residue types
}
```

**Bin centers** range from −175 to +175 in steps of 10 (the grid is 10° × 10°). Keys are string representations of integers.

### constants_chi1.json

Nested dictionary: **residue → φ center → ψ center → χ₁ rotamer → observables**

```
{
  "VAL": {
    "-70": {
      "-50": {
        "g+": {                      ← gauche+ (χ₁ = 30°–150°)
          "n": 342,
          "angle_N_CA_CB_eq": 109.81,
          "angle_N_CA_CB_std": 1.08,
          "angle_C_CA_CB_eq": 110.52,
          "angle_C_CA_CB_std": 0.94
        },
        "t": { ... },               ← trans (|χ₁| ≥ 150°)
        "g-": { ... }               ← gauche− (χ₁ = −150° to −30°)
      }
    }
  },
  "LEU": { ... },
  ...
}
```

Uses 20° bins (coarser than the main library) with a minimum of 20 observations per cell.

---

## The 12 Observables

### Bond Angles (6)

| Key prefix | Angle | AMBER default | Typical PDB range |
|---|---|---|---|
| `tau_deg` | τ (∠N–Cα–C) | 111.1° | 109–114° |
| `angle_N_CA_CB` | ∠N–Cα–Cβ | 110.1° | 108–113° |
| `angle_C_CA_CB` | ∠C–Cα–Cβ | 110.1° | 108–112° |
| `angle_CaCN` | ∠Cα–C–N | 116.6° | 114–119° |
| `angle_CNCa` | ∠C–N–Cα | 121.9° | 119–124° |
| `angle_CA_C_O` | ∠Cα–C=O | 120.4° | 119–122° |

### Bond Lengths (5)

| Key prefix | Bond | AMBER default | Typical PDB range |
|---|---|---|---|
| `bond_N_CA` | N–Cα | 1.458 Å | 1.45–1.47 Å |
| `bond_CA_C` | Cα–C | 1.522 Å | 1.51–1.54 Å |
| `bond_C_O` | C=O | 1.229 Å | 1.22–1.24 Å |
| `bond_C_N_next` | C–N (peptide) | 1.335 Å | 1.32–1.34 Å |
| `bond_CA_CB` | Cα–Cβ | 1.526 Å | 1.52–1.55 Å |

### Peptide Planarity (1)

| Key prefix | Parameter | Default | Notes |
|---|---|---|---|
| `omega_deg` | ω | 180.0° | Computed via circular mean; `omega_dev_eq` gives deviation from planarity |

---

## The 21 Residue Classes

The library stores separate lookup tables for each of the 20 standard amino acids plus a pooled `ALL` class:

```
ALL, GLY, ALA, VAL, LEU, ILE, PRO,
PHE, TYR, TRP, SER, THR, CYS,
MET, ASP, ASN, GLU, GLN, LYS, ARG, HIS
```

Coverage varies by residue: GLY has 555 populated cells (broadest Ramachandran coverage), PRO has 114 (narrowest due to constrained φ).

---

## Lookup Logic and Fallback Chain

When you query `lib.get(phi, psi, residue)`, the library follows this priority:

```
1. Look up the specific amino acid (e.g., "VAL") at the (φ,ψ) bin
   → Found? Return it.

2. Fall back to "ALL" (pooled across all residues) at the same bin
   → Found? Return it.

3. Fall back to AMBER ff14SB fixed defaults
   → Always available. Never fails.
```

This means **every query returns a value**. You never need to handle missing data.

The `stats` property tracks how often each fallback level is used:

```python
lib = GeometryLibrary()
# ... do many lookups ...
print(lib.stats)
# {'hits': 1982, 'fallback_all': 312, 'fallback_default': 6, 'total': 2300}
```

---

## Per-Field Suffixes

Each observable has multiple fields:

| Suffix | Meaning | Example |
|---|---|---|
| `_eq` | Equilibrium value (cell mean) | `tau_deg_eq = 111.02` |
| `_std` | Within-cell standard deviation | `tau_deg_std = 1.62` |
| `_k` | Empirical spring constant = kT/σ² | `tau_deg_k = 823.4` |
| `_coupling` | Coupling correction from Paper 3 | `tau_deg_coupling = 0.12` |
| `_delta_amber` | Deviation from AMBER constant | `tau_deg_delta_amber = -0.08` |
| `_coupling_reliable` | Bootstrap reliability flag | `true` or `false` |

**Note on `_k` values:** These are *observed* spring constants from crystallographic variance, not force-field parameters. They include refinement restraint effects and are ~10× larger than AMBER's force-field k values. Use for diagnostic purposes only; for energy calculations, use AMBER/CHARMM/OPLS spring constants.

**Note on `_coupling`:** Set to 0.0 for bond C=O and bond C–N (statistically unreliable at 10° bin resolution). The `_coupling_reliable` flag indicates this.

---

## Python API

### Initialization

```python
from backbone_geometry_library import GeometryLibrary

# Auto-discover JSON files in the same directory
lib = GeometryLibrary()

# Or specify paths explicitly
lib = GeometryLibrary(
    library_path='path/to/constants_library.json',
    chi1_path='path/to/constants_chi1.json'
)
```

### Core Methods

```python
# Get all geometry for one residue
geom = lib.get(phi=-63.0, psi=-43.0, residue='ALA')
# Returns dict with 12 keys: tau, angle_NCaCB, ..., bond_NCA, ..., omega

# Individual lookups
tau = lib.get_tau(phi=-63.0, psi=-43.0, residue='GLY')       # float, degrees
omega = lib.get_omega(phi=-63.0, psi=-43.0, residue='ALA')   # float, degrees

# Grouped lookups
bonds = lib.get_bonds(phi=-120.0, psi=130.0, residue='VAL')
# {'NCA': 1.459, 'CAC': 1.524, 'CO': 1.232, 'CN': 1.333, 'CACB': 1.540}

angles = lib.get_angles(phi=-63.0, psi=-43.0, residue='ALA')
# {'N_CA_C': 111.0, 'N_CA_CB': 110.3, 'C_CA_CB': 110.1,
#  'CA_C_N': 116.7, 'C_N_CA': 121.4, 'CA_C_O': 120.4}

# χ₁ correction (returns None if no data for this cell)
corr = lib.get_chi1_correction(
    phi=-63.0, psi=-43.0, residue='VAL', chi1_rotamer='t'
)
# {'N_CA_CB': 109.8, 'C_CA_CB': 110.5}  or  None
```

### Utility

```python
# Available residue types
lib.available_residues
# ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
#  'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
#  'THR', 'TRP', 'TYR', 'VAL']

# Lookup statistics
lib.stats
# {'hits': 1982, 'fallback_all': 312, 'fallback_default': 6, 'total': 2300}

lib.reset_stats()
```

---

## Usage Examples

### Example 1: NeRF Reconstruction

Replace hardcoded constants in your NeRF builder:

```python
from backbone_geometry_library import GeometryLibrary

lib = GeometryLibrary()

def build_residue(phi, psi, residue_name, prev_atoms):
    """Place N, CA, C for one residue using library geometry."""
    geom = lib.get(phi, psi, residue_name)
    
    # These replace your hardcoded values:
    bond_cn  = geom['bond_CN']       # was: 1.335
    bond_nca = geom['bond_NCA']      # was: 1.458
    bond_cac = geom['bond_CAC']      # was: 1.522
    tau      = geom['tau']           # was: 111.1
    angle_cn = geom['angle_CNCa']   # was: 121.9
    angle_cc = geom['angle_CaCN']   # was: 116.6
    omega    = geom['omega']         # was: 180.0
    
    # ... your existing NeRF placement code ...
```

### Example 2: Structure Validation

Check whether a bond angle is an outlier given its conformation:

```python
lib = GeometryLibrary()

# A residue with τ = 109.5° — is this an outlier?
# With AMBER's fixed 111.1°: deviation = 1.6° (flagged)
# With library:
expected = lib.get_tau(phi=-120, psi=130, residue='VAL')  # 109.6°
deviation = 109.5 - expected  # only -0.1° — NOT an outlier!
```

### Example 3: Comparing Force Fields

```python
import json

with open('constants_library.json') as f:
    lib = json.load(f)

# Compare AMBER vs PDB for glycine in αR
cell = lib['GLY']['-65']['-45']
print(f"PDB τ for GLY at αR: {cell['tau_deg_eq']:.1f}°")  # ~113.1°
print(f"AMBER τ (all residues): 111.1°")
print(f"Deviation: {cell['tau_deg_eq'] - 111.1:.1f}°")     # +2.0°
```

### Example 4: Batch Processing

```python
import pandas as pd

lib = GeometryLibrary()

# Read your structure data
df = pd.read_csv('my_residues.csv')  # needs phi_deg, psi_deg, res_name columns

# Vectorized lookup
df['tau_predicted'] = [
    lib.get_tau(row.phi_deg, row.psi_deg, row.res_name)
    for _, row in df.iterrows()
]

# Compare to observed
df['tau_error'] = df['tau_observed'] - df['tau_predicted']
print(f"MAE: {df['tau_error'].abs().mean():.3f}°")
```

### Example 5: OpenMM Force-Field Correction

```python
from openmm.app import PDBFile, ForceField
from backbone_geometry_library import apply_corrections

pdb = PDBFile('protein.pdb')

# Works with any force field:
ff = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
system = ff.createSystem(pdb.topology)
system = apply_corrections(system, pdb.topology, pdb.positions,
                           force_field='amber')

# Or CHARMM:
# ff = ForceField('charmm36.xml', 'charmm36/water.xml')
# system = apply_corrections(..., force_field='charmm')

# Or OPLS:
# system = apply_corrections(..., force_field='opls')
```

---

## Validation Summary

### Engh & Huber (2001) Crystallographic Reference

All 11 observables pass (|Δ| < 0.5σ_EH). Largest deviation: ∠Cα–C–N at 0.27σ.

### Secondary Structure τ

| Region | Reference | Library | Δ | Status |
|---|---|---|---|---|
| αR helix | 111.60° (Lovell 2003) | 111.64° | +0.04° | PASS |
| β-strand | 110.40° (Lovell 2003) | 109.17° | −1.23° | WARN |
| PPII | 111.00° (Berkholz 2009) | 110.43° | −0.57° | PASS |
| GLY αR | 113.10° (Lovell 2003) | 112.85° | −0.25° | PASS |
| β < αR constraint | — | Δ = 2.47° | — | PASS |

The β-strand WARN reflects differences in region definition, not a data error.

### Bootstrap Reliability

| Observable | Cells Tested | Reliable (|correction| > 2σ) | Status |
|---|---|---|---|
| τ | 500 | 65% | PASS |
| ∠N–Cα–Cβ | 377 | 61% | PASS |
| ∠Cα–C–N | 500 | 69% | PASS |
| bond C=O | 500 | 22% | **FAIL → coupling zeroed** |
| bond C–N | 500 | 39% | **FAIL → coupling zeroed** |

---

## Known Limitations

1. **Coupling corrections for bond C=O and C–N are zeroed** — statistically unreliable at 10° bin resolution. The equilibrium values (`_eq`) are still valid; only the coupling corrections are affected.

2. **ω wrapping** — computed via circular mean to avoid the ±180° averaging artifact. The `omega_dev_eq` field (|180° − |ω||) is the recommended metric for peptide planarity.

3. **Crystallographic bias** — the library reflects refined crystal structures. Solution-state geometry may differ due to dynamics and solvent effects.

4. **No resolution stratification** — the current version does not separate by crystallographic resolution. A future version may provide resolution-dependent variants.

5. **Sparse cells** — some (φ,ψ) bins for rare amino acids have few observations. The `n` field indicates cell population; cells with n < 50 should be used cautiously.

6. **Static equilibria** — the library provides fixed equilibrium values per (φ,ψ) cell. During MD simulation, as φ and ψ change, the corrections are applied based on the *initial* coordinates. A fully dynamic implementation would update corrections each timestep.

---

## How to Cite

```bibtex
@article{chen2025library,
  title={A Conformation-Dependent Geometry Library for 
         Protein Backbone Reconstruction},
  author={Chen, Wei and Cvek, Ur{\v{s}}ka and Trutschl, Marjan},
  journal={[submitted]},
  year={2025}
}
```

---

## Questions?

Open an issue on GitHub or contact the authors directly.
