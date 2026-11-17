# PBM de-imprint calculation (spec v1.0), 13 Nov 2026
Estimates, from the archive alone, the effective transmission of CDL's pattern into PHENIX deposits and of flat CCP4 targets into REFMAC deposits, and tests out of sample whether one transmission model with one data-supported pattern explains both. No library is built.

Run (WSL, the numpy/pandas/scipy/matplotlib environment used for the home-advantage kit; roughly 20-60 min):
```bash
cd /mnt/c/Wei/PBM
bash PBM_DEIMPRINT_v1/run.sh
```
Upload the printed pbm_deimprint.*.tar.gz. Stops if the spec, features, headers, groups or CDL hash differs, or if the self-test fails.
