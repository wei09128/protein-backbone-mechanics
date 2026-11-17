# QM noise-floor pilot: specification v1.0 (13 Nov 2026, Claude)

## Purpose
Measure how much constrained-QM backbone geometry moves when the method changes (basis set, functional, implicit water). The result sizes the panel of the final QM referee for MechLib v3 and decides which comparisons are resolvable. It is not a test of any library.

## What the pilot does not do
- It reads no library (MechLib v2/v3, CDL, AMBER targets). No prediction is scored.
- Its results cannot be used to build or tune v3. The v3 specification will be frozen without reference to the pilot outputs, other than the method-noise numbers (term-level SDs) used to size the final panel.

## Cases (fixed)
12 of the 80 registered Q6 cases: ALA, GLY, SER x alpha, beta, ppii, left (case IDs C-<RES>-<region>). The cases cover a residue with no side chain beyond CB, one with no CB, and one with a polar side chain. Registered phi/psi from cases_confirmatory.tsv (sha 160e3672...).

## Start geometry and constraints
Each arm starts from the frozen Q5 converged geometry of the case (work_q5/<case>/qm/qm_final_geometry.xyz). Before any calculation, the kit recomputes the 11 Q6 terms from that geometry and requires agreement with Q6_TERM_DETAIL.tsv (sha 3af8c2fe...) within 1e-4 deg and 1e-6 A. Constraints, optimizer and restart rule are those of the frozen Q5 runner: central phi and psi fixed at their registered values (geomeTRIC $set, --enforce 0.01 --conmethod 1), at most 2 automatic restarts from the last frame, constraint tolerance 0.5 deg, connectivity audit as in Q5.

## Arms (run in this order)
| Arm | Method | Basis | Medium | Role |
|---|---|---|---|---|
| R0 | wB97X-D | def2-SVP | gas | reproduction of the frozen reference (expected shift ~0) |
| S1 | wB97X-D | def2-SVP | implicit water | solvent effect |
| B1 | wB97X-D | def2-TZVP | gas | basis effect |
| F1 | wB97M-V | def2-TZVP | gas | functional effect (compare with B1) |
Psi4 settings as in Q5 (scf_type df, rhf, e/d convergence 1e-8, maxiter 200, memory 4 GB). Implicit water: ddX PCM (water) if the preflight shows working ddX gradients, otherwise PCMSolver IEFPCM if its gradients work; the choice is recorded at preflight and locked at the first run. If neither works, S1 is not run and this is reported.

## Outputs
For each arm, case and term (the 11 Q6 terms plus O-C-N): value and shift from the frozen reference. Per arm and term: n, mean shift (systematic), SD of shift across cases, mean |shift|, and shape SD (shift with the residue mean across the four regions removed). R0 must reproduce the reference: largest |shift| is reported, with a flag if any angle moves > 0.05 deg or bond > 0.0005 A.

## Use of the results
Method-to-method SDs define the resolution of the final QM referee: a library difference smaller than the relevant SD is called unresolved in advance. The final QM specification (method, medium, panel, endpoint, decision rule) will be frozen after this pilot and after v3 is frozen, before any v3 prediction is scored.

## Amendment 1 (13 Nov 2026, after the first preflight, before any peptide calculation)
The first preflight (water tests only) passed R0 and B1, but wB97M-V, ddX and PCMSolver all failed in the installed Psi4, so no arm was marked available and nothing ran. Changes: (1) each arm is available on its own (R0 required; S1, B1, F1 run if their water test passes); (2) F1 uses the first functional in the fixed order wB97M-V, M06-2X, PBE0 that passes the water test with def2-TZVP; (3) preflight prints the Psi4/geomeTRIC error lines of any failed test. No peptide geometry had been computed by any arm when this amendment was made.

## Amendment 2 (13 Nov 2026, after the second preflight, before any peptide calculation)
Second preflight: R0 and B1 passed; wB97M-V failed (no gradient returned); M06-2X passed and becomes F1 under Amendment 1. Because M06-2X is sensitive to the integration grid (the E6 calculation stalled at the default-grid noise floor), F1 uses a (99, 590) grid; R0, S1 and B1 keep the Q5 default grid so that R0 reproduces the frozen reference. Implicit water failed because the ddX Python module is not installed and PCMSolver returns no gradient. The frozen environment mechlib_qm_clean is not modified: S1 requires a cloned environment with pyddx added, and then all arms run in that clone (same Psi4 build; versions recorded at preflight).

## Amendment 3 (16 Nov 2026, from runtime only; no B1 or F1 result existed)
At low priority beside the MD, water (S1) cases took about 14 h each, and the def2-TZVP arms (B1, F1) would take several times longer: weeks for 24 cases. B1 and F1 are therefore restricted to the four ALA cases (alpha, beta, ppII, left). R0 and S1 keep all 12 cases. The basis rule of the QM referee draft (TZVP if the B1 - R0 shape SD exceeds 0.10 deg) is applied to these four cases; F1 is reported descriptively. The restriction was decided from runtime alone, before any B1 or F1 case finished.
