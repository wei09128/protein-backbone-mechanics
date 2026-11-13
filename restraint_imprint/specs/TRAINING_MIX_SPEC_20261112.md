# Training-mix sensitivity of a learned backbone library: specification v1.0 (frozen before fitting)
12 Nov 2026, Claude. Run with PBM_TRAINING_MIX_v1 (checks this file's SHA256 first). Sensitivity analysis only: the reported MechLib remains clean v2 whatever the outcome; no fit below is called a library version and none replaces v2.

## Question
How much does a learned library depend on the refinement-program composition of its training set, and does balancing PHENIX and REFMAC training change held-out error on deposits from each program?

## Fits (identical code, grid, next-Pro split, minimum count and fallback rules as clean v2)
Training rows: the frozen clean training cohort (5,926 chains; training geometry sha a2929836...), labelled by refinement program from the archive headers (family PHENIX or REFMAC; other programs and unlabelled structures excluded from F1-F4). Terms: Ca-C-N and O-C-N.
- F0 v2 (all programs, natural mix): must reproduce the frozen v2 arrays exactly (gate), else stop.
- F1 natural-PR: PHENIX + REFMAC structures, natural mix (every junction weight 1).
- F2 balanced-PR: same structures; junction weights scaled so PHENIX and REFMAC carry equal total weight (within each program, weights stay natural).
- F3 REFMAC-only. F4 PHENIX-only (all PHENIX builds).
Cell means are weighted means; the >= 10 rule and fallback use unweighted counts, as in v2. If fewer than 90% of training structures carry any program label in the archive headers, the run stops (label coverage, not PHENIX/REFMAC share, is gated; the PHENIX/REFMAC share is reported).

## Test set
Frozen held-out junction table, metadata and strata (score.py hashes); B30 primary subset; strata REFMAC and PHENIX_post_default; common coverage across all fits per term; structures with >= 5 junctions.

## Estimands
Per-structure MAE; averaged within sequence components; components equally weighted.
- Primary: Delta_bal(stratum) = MAE(F2) - MAE(F1), for each stratum and term; and the gap change G = [MAE(F2) - MAE(F1)](REFMAC) - [MAE(F2) - MAE(F1)](PHENIX_post).
- Secondary: F3 and F4 relative to F1 in each stratum (the program-specific extremes); surface shift = mean |F2 - F1| and |F3 - F4| over cells supported (>= 10) in both, by term.
95% percentile intervals, 10,000 component bootstrap draws, seed 20261114; Holm over the two terms within each family.

## Reading (descriptive; fixed now)
- "Balancing changed held-out error" for a stratum and term if Holm-adjusted p < 0.05 for Delta_bal, with its sign reported.
- A smaller program gap under balancing is expected by construction (balancing averages the two programs' imprints) and is NOT evidence of better geometry. The informative outcomes are the size of the surface shift and whether error on BOTH strata holds or improves.
- No fit is promoted. Every outcome is reported.

## Outputs
REBUILD_GATE.json, LABEL_COVERAGE.json, SURFACE_SHIFT.tsv, STRUCTURE_MAE.tsv, CONTRASTS.json, TrainingMix.png/pdf, RUN_LOCK.json.

## What it will and will not show
Will show how strongly the learned surfaces and their held-out agreement depend on the training set's program mix. Will not show which mix is physically correct (agreement with deposits is the endpoint throughout) or remove any imprint.
