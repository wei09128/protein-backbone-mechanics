# MechLib QM Gate Q6 confirmatory analysis

This directory evaluates the frozen `MechLib-QM-Q1-v1` endpoint using the
80 audited Q5 constrained-QM geometries and the model predictions frozen
before confirmatory outcome access.

## Result

The registered primary endpoint is **FAIL** under the contract's conservative
zero-threshold interpretation of “materially worse.” All 80/80 registered
cases passed QC. The overall weighted error difference favored MechLib
(`mean delta = 0.00933099`; residue-clustered 95% bootstrap CI
`[0.00263134, 0.01665963]`), and 14/20 residue means were positive. However,
three registered bond terms were significantly worse for MechLib after Holm
correction: `bond_N_CA`, `bond_C_O`, and `bond_C_N_next`. Thus three of four
success conditions passed and the complete conjunction failed.

This is mixed confirmatory evidence. It supports an overall local-geometry
advantage across the registered term average but does not support the stronger
claim that MechLib uniformly improves the registered geometry set. It does not
test dynamics, populations, folding thermodynamics, or NMR/RDC observables.

## Reproduction

Run `python q6_analyze.py` from the directory containing `work_q5/` and this
`work_q6/` directory. The script uses deterministic seed `20260915` and 10,000
bootstrap replicates. `SHA256SUMS.txt` covers every primary result and figure.
