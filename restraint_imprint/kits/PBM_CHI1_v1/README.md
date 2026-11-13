# PBM chi1 kit (spec v1.1), 12 Nov 2026

Builds MechLib v2 + chi1 (chi1-conditioned surfaces for tau, N-CA-CB and C-CA-CB; all other terms stay v2) and tests it on held-out deposits refined with neither CDL nor MechLib (REFMAC). No refinement, MD or QM.

## Run (WSL, the numpy/pandas/matplotlib environment used for the clean benchmark)
```bash
cd /mnt/c/Wei/PBM
bash PBM_CHI1_v1/run.sh
```
Upload the printed `pbm_chi1.*.tar.gz`. Expected time: roughly 30-90 min (about 7,200 coordinate files are parsed; `WORKERS=8 bash ...` uses more cores).

Inputs found automatically under /mnt/c/Wei/PBM and hash-checked (the run stops on any mismatch): the training geometry (pbm_altloc_training_audit_results_v1), the frozen v2 targets (PBM_CLEAN_REBUILD_v1/results), the held-out junction table, metadata and strata (as in rescore.sh), pdb_cache, pbm_v2_frozen_coordinates and the PHENIX CDL file (path from rescore.sh; override with CDL=...).

## Built-in stops
1. Spec hash. 2. Rebuild gate: v2's tau, N-CA-CB and C-CA-CB grids rebuilt from the training geometry must equal the frozen arrays exactly. 3. Same-atom gate: training chi1 kept only if tau from the same atoms matches the frozen value within 0.01 deg; held-out values kept only if Ca-C-N and O-C-N match within 0.01 deg; the run stops before scoring if coverage is below 75%. 4. Controls (Ca-C-N, O-C-N) must be unchanged by chi1 conditioning.

## Outputs (results/)
REBUILD_GATE.json, CHI1_COVERAGE.json, CHI1_MODEL.npz, FALLBACK_FRACTIONS.tsv, STRUCTURE_MAE_CHI1.tsv, CONTRASTS.json, Chi1.png/pdf, RUN_LOCK.json (includes which terms MechLib reports with chi1, by the frozen rule).
