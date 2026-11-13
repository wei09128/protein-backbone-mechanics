# Prospective prediction of a library swap (CDL -> MechLib v2): specification v1.0 (frozen before any swap refinement)
12 Nov 2026, Claude. Run with PBM_SWAP_v1 in two stages. Stage 1 (no refinement) writes the predictions and their SHA256; Wei sends that hash to be recorded in the evidence ledger; stage 2 refuses to run unless given the same hash. Nothing below may change after stage 1; any change is an amendment with a new hash and a reason.

## Question
Can the coupled response measured in a structure from isolated one-degree edits predict, before refinement, what replacing the restraint library's carbonyl-angle targets (CDL -> MechLib v2) does to that structure's refined coordinates? This tests Rule 2 of the paper (forecast a library's imprint before deploying it). It is not a comparison of libraries.

## Structures
The 19 structures that completed the frozen transmission batch (pbm_transmission_batch.j1pKgh; freeze lock af428334...): 9QDV, 9AYN, 9KWV, 4JP6, 2WFI, 5AL6 (<=1.0 A); 4YSL, 2XRH, 5FAF, 9LTB, 3BMZ, 2OPC, 1Y8A (1.0-1.5 A); 3LWA, 3B4Q, 8KIH, 1MG7, 3UCE, 8CIH (1.5-2.0 A). Input model and common MTZ hashes as in BATCH_CONFIG. No structure is added, dropped or replaced after stage 1; a structure that fails a stage-2 audit is reported as failed.

## The swap
- In PHENIX's live CDL lookup (discrete nearest-node, as in the batch), for every junction whose lookup returns a CDL row (trans, standard residues), the three carbonyl-angle means mACN (Ca-C-N), mACO (Ca-C-O) and mOCN (O-C-N) are replaced by clean MechLib v2 (target sha ef7b86b3...) for the central residue and next-Pro class, evaluated by periodic Catmull-Rom at the lookup node (phi, psi). Raw v2 O-C-N is used (no closure). All sigmas and all other CDL fields are unchanged. Nothing else in the protocol changes.
- Protocol: the batch protocol.phil (sha c510fc19...; one macro-cycle, fixed wxc 1.0, weight optimisation off), same model and MTZ, nproc 1. Two new arms per structure, run identically: stock (unedited) and swap. The observed response is swap minus new stock.

## Prediction (stage 1, before any swap run)
- Sites: blank-altloc carbonyl triplets eligible in the batch matrix analysis (same code: identify / triplets on the stock initial restraints and the cacn, aco, ocn arm pairs).
- Site matrix T (3x3; rows and columns Ca-C-N, Ca-C-O, O-C-N): columns (q_plus - q_minus)/2 from the batch's isolated cacn, aco and ocn arms, as in the batch analysis.
- Planned edit e (per site) = v2 target - CDL target at the site's lookup node. The node is computed from the input model's phi/psi (rounded to 10 degrees as CDL does). Key gate: the CDL table value at that node must equal the stock initial restraint ideal for all three angles within 0.006 deg; sites failing the gate are excluded before prediction and counted; if more than 10% of a structure's sites fail, that structure is excluded at stage 1 (reported).
- Predictors: site matrix (T e; primary); same-structure mean matrix; target copy (e); zero.

## Stage-2 audits
Swap and new-stock initial restraints must have identical proxy sets, sigmas and weights; every changed ideal must be one of the three terms; at every scored site the realised ideal change must equal e within 0.006 deg. Observation and free-flag support must match between the two arms. A live lookup trace must be present. Sites failing the realised-edit audit are excluded and counted.

## Primary estimand and reading rule
Per structure: MAE over scored sites and the three angles of (observed - predicted), for each predictor. Primary contrast: D = MAE(copy) - MAE(site matrix), positive = matrix better. Mean of D over structures (each structure equal weight), 95% percentile interval from 10,000 structure resamples within resolution bands, seed 20261113.
- "The measured response predicted the library swap": mean D > 0 with the interval above 0. Otherwise "not shown". The number of structures with D > 0 is reported either way.
Secondary (reported, no reading rule): mean-matrix and zero predictors; observed versus targeted change in the three-angle sum; sites with max|e| >= 1 deg versus < 1 deg (because the half-amplitude comparison failed in 4YCA); per band; R-work/R-free of both arms.

## Outputs
Stage 1: PREDICTIONS.tsv, PREDICTION_LOCK.json (its SHA256 printed). Stage 2: AUDITS.json, SITE_RESULTS.tsv, STRUCTURE_RESULTS.tsv, SUMMARY.json, Swap.png/pdf, RUN_LOCK.json.

## What it will and will not show
Will show whether a response measured once per structure forecasts a real library change before it is made, including the knock-on change in Ca-C-O, at the size of the actual CDL -> v2 differences. Will not show which library is better, convergence, or the effect of stiffness changes (sigmas are not swapped). One macro-cycle, fixed weights, as in the matrix measurement.
