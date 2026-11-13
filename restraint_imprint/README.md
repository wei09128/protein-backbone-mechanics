# Restraint imprint (Chen & Lu, submitted to IUCrJ)

Code, frozen specifications, library files, registered predictions and result tables for
"The restraint imprint: how crystallographic refinement shapes deposited protein geometry".
The other folders of this repository belong to earlier papers in the series and are unchanged.

## MechLib in this paper
One library: MechLib version 2 (next-Pro split; `library/mechlib_v2_training_targets.npz`,
SHA-256 ef7b86b3...) with chi1 conditioning for N-CA-CB and C-CA-CB only
(`library/mechlib_chi1_model.npz`, SHA-256 6210897c...). All other terms use version 2.
Earlier MechLib files elsewhere in this repository (e.g. version 1) are not used in this paper.

## Contents
- `specs/` frozen specifications; each kit checks its specification's SHA-256 before reading data.
  - rotamer conditioning (CHI1_FAIR_COMPARISON_SPEC, v1.1)
  - home advantage (HOME_ADVANTAGE_SPEC, v1.1)
  - prospective swap (SWAP_PREDICTION_SPEC, v1.0)
  - training mix (TRAINING_MIX_SPEC, v1.0)
  - QM re-score (QM_RESCORE_SPEC, v1.0; reuses the registered contract MechLib-QM-Q1-v1)
- `kits/` analysis code as run.
- `registered_predictions/PREDICTIONS.tsv` swap forecast frozen before refinement (SHA-256 8b3f6277...).
- `results/` returned result tables and run locks for each analysis (Figure 4, Tables S6, S7, S10).
- `figures/scripts/` figure code; `figures/output/` vector figures as submitted.
- `manuscript/` manuscript and SI sources and the build script.
- `SHA256SUMS.txt` checksums of every file here.

Figures 1-3 are built from earlier run records in `run_records/`; raw refinement outputs are not
included because of size. Diffraction data and coordinates are from the Protein Data Bank.
