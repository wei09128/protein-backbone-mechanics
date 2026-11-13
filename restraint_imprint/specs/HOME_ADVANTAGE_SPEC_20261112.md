# Home-advantage test of three backbone libraries: specification v1.1 (frozen before scoring)
12 Nov 2026, Claude. Run with home_advantage.py (PBM_HOME_ADVANTAGE_v1 kit), which checks this file's SHA256 before reading any geometry. Nothing below may change after scores are seen; any change is an amendment with a new hash and a reason.

Amendment log: v1.0 (hash 4a5e75c8...) was never run. v1.1, written while coding and before any data were read, (1) moves C1/C2 to the corrected archive feature file, because the held-out cohort is the post-training (recent) set and cannot supply pre-2015 PHENIX or pre-2018 REFMAC strata, and no learned model enters C1/C2, so training overlap is irrelevant there; (2) corrects the CDL lookup to periodic Catmull-Rom, which is what the existing benchmark (score.py) actually uses; (3) drops C-N-Ca (incoming-link mapping not verified in this kit) and the v2-vs-CCP4 descriptive (needs CCP4 predictions on held-out junctions); (4) states junction filters, eligibility minimums and the Holm rule exactly.

## Question
Does each restraint library agree best with deposits refined using that library, and how large is that home advantage? This applies Rule 1 of the paper (do not score a library on structures it helped refine) as a measurement. It is not a ranking of libraries, and no result will be reported as "best library".

## Prediction recorded now
Each library's error, relative to another library, is smaller on its home stratum than on an away stratum (DiD < 0 below). Sizes are not predicted.

## Predictors (none refitted)
- CDL: cctbx CDL v1.2 (cdl_database.py, sha 79c197e5...), group by residue class and next-Pro, periodic Catmull-Rom at the junction's phi/psi, as in score.py.
- CCP4-old / CCP4-new: MonomerLibrary/monomers commits a683db68 (before) and 2bc9cc94 (after); mon_lib_list.cif hashes f3b2b8ed... and 4fcb5cca.... Ca-C-N and O-C-N from the TRANS link, or PTRANS when the next residue is Pro (values as recovered in LINK_ANGLE_DIFF.tsv). tau (N-Ca-C) from the residue monomer. Ca-C-O: new = DEL-OXT override 120.614; old = residue monomer value (the old effective Ca-C-O used by depositors is uncertain; Ca-C-O results for CCP4-old are flagged accordingly). Flat in phi/psi.
- MechLib v2 (C3 only): clean next-Pro model, target sha ef7b86b3..., as scored in PBM_CLEAN_BENCHMARK_RESULTS_v1.

## Data
- C1, C2 and their secondaries: corrected archive features pbm_altloc_full_rdd_audit_v1/blank_only_geometry.csv.gz (sha e905f8b8...), headers and sequence groups from the closing-check kit. Junction filters as in the archive analyses: duplicate keys removed; |omega| > 150; finite phi/psi; standard residues; per term and next-Pro flag, values beyond 8 x MAD set missing. Next-residue identity from the exact next exported index in the same chain, with finite C-N <= 1.8 A and agreement with is_pro_np1; others excluded. A structure enters a term if it has >= 20 junctions scored by every predictor.
- C3: structure_mae.tsv from PBM_CLEAN_BENCHMARK_RESULTS_v1 (B30_primary subset), unchanged.

## Strata (by deposit header)
PHENIX-post: PHENIX build >= 2155. PHENIX-pre: PHENIX build < 2155. REFMAC-old: version <= 5.8.0189 and deposited on or after 2012-01-01. REFMAC-new: version >= 5.8.0222. Version parsing pads missing components with zero (original rule); strict parsing (abbreviated versions excluded) is reported as a sensitivity. Other programs excluded. A stratum-term cell with fewer than 30 sequence groups is "not estimable".

## Terms
Primary: Ca-C-N and O-C-N. Secondary: Ca-C-O, tau.

## Estimands
Per-structure MAE; within a stratum, structures averaged within sequence groups and groups weighted equally.
DiD(P vs Q; h vs a) = [MAE_P - MAE_Q](h) - [MAE_P - MAE_Q](a). Negative = P relatively better on its home stratum h.
- C1: P = CDL, Q = CCP4-new; h = PHENIX-post, a = REFMAC-new.
- C2: P = CCP4-new, Q = CCP4-old; h = REFMAC-new, a = REFMAC-old (within-program crossover).
- C3: P = CDL, Q = MechLib v2; h = PHENIX-post, a = REFMAC (held-out). Gate: must reproduce the existing program-difference estimates (-0.134 Ca-C-N, -0.192 O-C-N in this sign convention) within 0.001.
Secondary: C1 with a = PHENIX-pre; C1 and C2 on structure-centred MAE (each structure's mean signed deviation removed); C1 and C2 with strata reweighted to the pooled distribution of resolution bands (<=1.0, 1.0-1.5, 1.5-2.0, >2.0 A); strict version parsing.

## Uncertainty and reading rules
10,000 sequence-group bootstrap draws (multinomial, seed 20261112), identical weights for every predictor and stratum in a contrast. Two-sided bootstrap p = 2 x min(share of draws <= 0, share >= 0). Holm adjustment over the primary terms within each contrast.
- "Home advantage": DiD < 0 and Holm-adjusted p < 0.05.
- "Reversed": DiD > 0 and Holm-adjusted p < 0.05. Otherwise "not resolved".
- Report the full MAE table and every contrast, including nulls and reversals. Text may say which predictor had lower error within a stratum but may not declare an overall winner. Nothing is added or dropped after scoring.

## Outputs
STRATUM_MAE.tsv, STRUCTURE_MAE.tsv, CONTRASTS.json, COVERAGE.tsv, DICTIONARY_TARGETS.tsv, HomeAdvantage.png/pdf, RUN_LOCK.json (this file's hash, input and dictionary hashes, code hash).

## What it will and will not show
Will show whether, and by how much, agreement with deposits depends on which library refined them, for three libraries, including a within-program crossover. Will not show which library is physically correct. Strata differ in protein mix and era; the DiD cancels what is common to two predictors within a stratum, not every difference between strata. CDL v1.2 was itself derived from deposited structures, some of which may be in the archive cohort.
