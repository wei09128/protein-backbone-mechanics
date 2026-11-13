# PBM prospective swap test (CDL -> MechLib v2), spec v1.0, kit v1.0.2 (stage-2 number reader fix; refinement.main.nproc=1), 12 Nov 2026

Question: can the response measured in each of the 19 batch structures predict, BEFORE refinement, what swapping the carbonyl-angle targets from CDL to MechLib v2 does to the refined coordinates?

## Stage 1: predictions only (minutes, no PHENIX refinement)
```bash
cd /mnt/c/Wei/PBM
bash PBM_SWAP_v1/run_stage1.sh
```
It reads the frozen batch (pbm_transmission_batch.j1pKgh), builds each structure's site matrices from the isolated arms, computes the planned v2-minus-CDL edit at every site (with a gate that the CDL value at the computed lookup node equals the stock restraint), and writes `pbm_swap_stage1/PREDICTIONS.tsv`. It prints a SHA256.

**Send that SHA256 (and pbm_swap_stage1.tar.gz) before running stage 2.** It is recorded in the evidence ledger; that is what makes the test prospective.

## Stage 2: 38 one-cycle refinements + scoring (about 1-2 h with JOBS=4)
```bash
bash PBM_SWAP_v1/run_stage2.sh <the registered SHA256>
```
Runs a new stock arm and a swap arm for each structure (same model, MTZ, protocol.phil as the batch; nproc=1), audits them (identical restraints except the three swapped targets; realised edit = planned edit at every scored site; same reflections and free flags; live lookup trace), and scores the predictions. Upload the printed `pbm_swap_stage2.*.tar.gz` (MTZ files are excluded to keep it small).

## Reading rule (frozen)
Primary: mean over structures of MAE(target copy) - MAE(measured response), with a 95% interval from structure resampling within resolution bands; "predicted" if the interval is above 0. Everything else (mean matrix, angle sum, edit size, bands, R values) is reported without a rule.
