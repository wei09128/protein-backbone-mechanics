# PBM training-mix sensitivity (spec v1.0), 12 Nov 2026
Refits the Ca-C-N and O-C-N surfaces with v2's exact rules on different refinement-program mixes (natural PHENIX+REFMAC, balanced 50/50, REFMAC-only, PHENIX-only) and scores them on the frozen held-out REFMAC and PHENIX deposits. Sensitivity analysis only: the reported MechLib stays v2; nothing is promoted.

Run (WSL, the numpy/pandas/matplotlib environment used for the chi1 kit; a few minutes):
```bash
cd /mnt/c/Wei/PBM
bash PBM_TRAINING_MIX_v1/run.sh
```
Upload the printed pbm_training_mix.*.tar.gz. Stops if: spec or any input hash differs; the all-program refit does not reproduce the frozen v2 arrays exactly; program labels cover < 90% of training structures.
