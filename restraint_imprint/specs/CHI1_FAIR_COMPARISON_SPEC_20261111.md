# Clean chi1 refit and provenance-neutral comparison: specification v1.1 (frozen before scoring)
11 Nov 2026, Claude; v1.1 amended 12 Nov 2026 before any chi1 value was extracted or any score computed. Run with chi1_test.py (PBM_CHI1_v1 kit), which checks this file's SHA256 before reading geometry. Nothing below may be changed after scores are seen; any change is an amendment with a new hash and a reason.

Amendment log (v1.0 was never run): (1) chi1 is not present in the corrected feature exports, so it is computed from the cached coordinates with a same-atom gate (below); (2) the v2 surfaces are rebuilt from the training geometry and must reproduce the frozen v2 arrays exactly before any chi1 surface is fitted; (3) chi1 cells are nested within v2's next-Pro split; (4) CDL is evaluated by periodic Catmull-Rom as in score.py (v1.0 wrongly said nearest node); (5) held-out observations for tau, N-Ca-Cb and C-Ca-Cb are computed from the frozen coordinates of each scored junction; (6) the rule that defines the single reported MechLib is fixed here; (7) Holm is applied to bootstrap p-values.

## Question
Does conditioning on the side-chain chi1 rotamer add predictive information about Ca-centred backbone angles beyond phi/psi, when scored on deposits refined with neither CDL nor MechLib?

Why this is the fair test: CDL has no chi1 index and REFMAC's targets are flat in phi/psi, so a chi1 gain on REFMAC deposits cannot be a direct copy of either library's pattern. Indirect leakage through coupled restraints is still possible; the result is evidence of added information, not proof of physical correctness.

## The single reported MechLib (fixed now)
MechLib as reported in the paper = clean v2 (next-Pro; target sha ef7b86b3...) plus chi1 conditioning for each primary term (tau, N-Ca-Cb, C-Ca-Cb) whose "chi1 adds information" reading rule passes; for a term that fails, MechLib uses the v2 value. All other terms are v2. Both the with- and without-chi1 results are reported. No other version is considered.

## chi1 source and same-atom gate
- Training: pdb_cache/<id>.pdb; held-out: pbm_v2_frozen_coordinates/<ID>.cif.gz. First model only; blank-altloc atoms only (atoms carrying an alternate-location label are ignored). chi1 = N-CA-CB-Xg with Xg = CG (default), SG (Cys), OG (Ser), OG1 (Thr), CG1 (Ile, Val). Gly, Ala and Pro have no chi1 term here.
- Training gate: a residue's chi1 is used only if tau recomputed from the same blank atoms matches the frozen training tau_deg within 0.01 deg and the match within its chain and residue name is unique.
- Held-out gate: a junction's computed values are used only if Ca-C-N and O-C-N recomputed from the same atoms match the frozen junction table (cacn_deg, ocn_deg) within 0.01 deg and the match is unique.
- Coverage stop: if fewer than 75% of eligible chi1-bearing rows pass the gate in either set, the run stops and reports coverage; no relaxation of the gate.

## Models
- M_v2: clean v2 as frozen. Its tau, N-Ca-Cb and C-Ca-Cb grids are rebuilt from the training geometry (pbm_altloc_training_audit_results_v1/blank_only_geometry.csv.gz, sha a2929836...; training chains as frozen; eligibility as the full-term fit) and must equal training_targets.npz (count grids exact; mean grids within 1e-12), else the run stops.
- M_v2chi: for residues with a chi1 class, a (residue, next-Pro, chi1 class, phi node, psi node) cell uses its training mean if it has >= 10 observations, otherwise the M_v2 value at that node. chi1 class: nearest of g- (-60), t (180), g+ (+60) by periodic distance; more than 40 deg from every well = eclipsed, which uses M_v2. Residues without chi1 or without a gated chi1 use M_v2. Same periodic Catmull-Rom at bin centres (-175..175). Fallback fractions are reported per term.
- CDL: cctbx CDL v1.2 (sha 79c197e5...), group by residue class and next-Pro, periodic Catmull-Rom at nodes (-180..170), as in score.py. Columns mNAC (tau), mNAB (N-Ca-Cb), mBAC (C-Ca-Cb), mACN, mOCN.

## Terms
Primary (chi1 applied): tau, N-Ca-Cb, C-Ca-Cb. Controls (no chi1 conditioning; M_v2chi must equal M_v2 within 1e-6 or the run is invalid): Ca-C-N, O-C-N.

## Test set and strata
Frozen held-out junction table, metadata and strata (score.py hashes); B30_primary subset; common coverage across the three predictors per term. Primary stratum: REFMAC. Sensitivity: REFMAC + SHELXL; PHENIX post-default (provenance effect, not a test of chi1). Others excluded. A structure enters a term with >= 5 scored junctions.

## Estimands and uncertainty
Per-structure MAE per term; averaged within sequence components; components equally weighted.
- Delta_chi = MAE(M_v2chi) - MAE(M_v2).
- Delta_CDL = MAE(M_v2chi) - MAE(CDL).
95% percentile intervals from 10,000 sequence-component bootstrap draws (multinomial), seed 20261111, shared weights across paired predictors. Two-sided bootstrap p; Holm across the three primary terms within each contrast family.

## Reading rules (fixed now)
- "chi1 adds information" for a term: Delta_chi < 0 and Holm-adjusted p < 0.05 in the primary stratum.
- "Lower error than CDL on neutral deposits" for a term: Delta_CDL < 0 and Holm-adjusted p < 0.05 in the primary stratum. Otherwise "not lower".
- Every outcome is reported, including nulls, reversals, fallback fractions and gate coverage. No term, stratum, threshold or class definition is added or dropped after scoring.

## Outputs
CHI1_MODEL.npz (+ sha), REBUILD_GATE.json, CHI1_COVERAGE.json, FALLBACK_FRACTIONS.tsv, STRUCTURE_MAE_CHI1.tsv, CONTRASTS.json, Chi1.png/pdf, RUN_LOCK.json (this file's hash and all input hashes).

## What it will and will not show
Will show whether chi1 carries predictive information about these angles beyond phi/psi on deposits refined with neither CDL nor MechLib, and how MechLib compares with CDL there. Will not show physical correctness, effect in refinement, or behaviour in simulation. REFMAC deposits are the home of CCP4's targets (flat for these angles); that does not favour either compared predictor.
