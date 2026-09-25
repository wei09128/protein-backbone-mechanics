# M2E scalar-coupling scoring contract — frozen v1

**State: scoring definition frozen before trajectory arm scores were computed.** This is an exploratory 1 ns pilot, not Gate M. Freeze identity is the separate `M2E_SCORE_LOCK_v1.json`; a scoring run must verify every lock hash and all 259 input manifest entries before reading DCD frames.

## Input set and exclusions

- Use `m2d_pilot_out_v101_GB3_corruption_rerun_v1`: six freshly regenerated GB3 lineages and six byte-matched copied ubiquitin lineages. Original damaged GB3 DCDs and forensic staged files are excluded. The approved engineering integrity report is `M2E_GB3_CORRUPTION_RERUN_VALIDATION_v1.json` (120 readable chunks, 1200 frames).
- Input hash inventory must contain all 120 DCDs, all 120 state reports, 12 audits, both prepared PDBs, both transcribed experimental tables, the two source PDFs and the integrity report. Reject mismatched hashes, altered row counts, missing frames, nonfinite angles or mapping errors; do not substitute a different source after scoring.
- GB3: Roche/Ying/Bax Table S1 transcription, 50 `primary_non_gly=YES` rows. Ubiquitin: Wang/Bax 1996 Table S3 HNHA column exclusively, 63 `primary_non_gly=YES` rows. Both have 1-based prepared PDB numbering. No glycine in the primary set, no interpolation, and no outlier removal after opening predictions. Verify every row's residue letter and connected peptide backbone. Report unmappable rows and block scoring if any of the 113 primary rows fail mapping.

## Locked analysis proposal

- All 100 frames per lineage: ten chunks in order, ten frames per chunk. No burn-in or chosen time window. Use identical frame treatment in both arms.
- For each frame and residue compute conventional backbone phi = dihedral C(previous)–N–CA–C in radians. Set theta = phi minus 60 degrees; predict 3J(HN,Halpha) in Hz as `6.7*cos(theta)**2 - 1.3*cos(theta) + 1.5`. Average **predicted J values over frames**, not dihedral angles. This is a phi-based approximation fitted in unrelated proteins; no fitting on GB3 or ubiquitin and no use of published coefficients tuned to either test protein.
- For each protein, arm, seed, compare its 50 or 63 frame-averaged predicted couplings to the same experimental observations using mean squared error (Hz²) and RMSE (Hz). Primary paired contrast for each protein and seed is `MSE(AMBER_FIXED) - MSE(AMBER_MECHLIB_V1)`; positive favors MechLib. Report six paired differences, each protein's mean of its three paired differences, and the equal-weight mean of the two protein means. Also give arm RMSEs and per-residue residuals. Do not average the two arm trajectories before scoring.
- Treat the six protein-by-seed pairs as descriptive paired observations. Report their signs, ranges and time-block drift. Do not calculate a residue- or frame-based significance test; no inferential CI or universal force-field improvement claim from three seeds and two proteins. A positive pooled contrast with both protein means positive is a *pilot-level concordant signal* only; any other outcome is mixed or unfavorable under these rules. There is no Gate M PASS threshold here.
- The 298 K simulations and experimental sample conditions differ; disclose recorded temperatures, buffers and pH. Apply no retrospective correction. No claims of converged ensemble agreement from 1 ns.

## Freeze identity and release

Input manifest SHA-256: `dea8327709f23b5a84037a033cb9ddef2aa56fea34f695c4d37765897d605561` (259 distinct paths). The independent forward-model expression is recorded in the original Ludvigsen, Andersen and Poulsen 1991 abstract (PMID 2005622, DOI 10.1016/0022-2836(91)90529-f), parameterized using CI-2 and barnase. Frozen execution environment: mdtraj 1.10.3 and NumPy 2.2.6. The preflight verifies every input SHA-256 and 113 mapped observations without scoring. Score and publish both arms and every registered residue regardless of sign. Keep Q6 separate: 80/80 QC PASS and positive aggregate effect, but strict registered endpoint FAIL because N–Cα, C–O and C–N(next) worsened. Paper IV and PBM active bundle remain unchanged until the MD/NMR endpoint has been frozen and adjudicated.
