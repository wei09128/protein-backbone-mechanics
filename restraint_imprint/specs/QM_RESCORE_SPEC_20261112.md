# QM re-score of the reported MechLib against the frozen Q6 panel: specification v1.0 (frozen before scoring)
12 Nov 2026, Claude. No new QM calculation. The registered contract MechLib-QM-Q1-v1 (sha d5888044...) and its frozen QM reference values (Q6_TERM_DETAIL.tsv, sha 3af8c2fe...; 80/80 cases PASS QC) are reused unchanged; only the predictor is replaced.

## Why
The original Q6 test scored a blind pre-2020 MechLib (primary JSON sha b1aefba1..., built from the pre-correction features). The paper reports one library: clean MechLib v2 with chi1 conditioning for N-Ca-Cb and C-Ca-Cb (frozen rule, run pbm_chi1.TrfkcA). This re-score applies the same registered rule to that library. The original Q6 result (FAIL) stays on record for the earlier library and is not overwritten.

## Predictor (primary)
Reported MechLib: clean v2 (target sha ef7b86b3...) for tau, Ca-C-N, C-N-Ca, Ca-C-O and the five bonds; v2 + chi1 (CHI1_MODEL.npz sha 6210897c...) for N-Ca-Cb and C-Ca-Cb. Evaluated by periodic Catmull-Rom (as in all v2 scoring) at the registered phi/psi of each case (cases_confirmatory.tsv sha 160e3672...), central residue X, next residue not Pro (ACE-X-NME). chi1 class: the registered starting rotamer (180 deg = t for every chi1-bearing residue); Gly, Ala and Pro use v2. Glycine has no C-beta terms (868 term rows, as in Q6).

## Primary rule (identical to the contract)
Weighted error e = sqrt(k_t) |q_model - q_QM| (angles in radians), k_t and q_QM and AMBER values as frozen in Q6_TERM_DETAIL. Case score delta_i = mean over available terms of [e(AMBER) - e(MechLib)]; positive favours MechLib. Success requires all of: (1) residue-clustered 95% bootstrap interval of mean delta strictly above 0 (10,000 resamples of the 20 residues, numpy default_rng seed 20260915, as reproduced exactly for Q6); (2) at least 12 of 20 residue means positive; (3) no term with a Holm-significant (over 11 terms, alpha 0.05) one-sided one-sample t-test showing MechLib worse (the Q6 test, reproduced exactly); (4) at least 68 of 80 cases pass QC (80 already pass).

## Secondary (reported; cannot rescue a failed primary)
- Plain v2 (no chi1) with the same rule.
- All 11 per-term MAEs and signed biases for AMBER, the earlier Q6 library and the reported MechLib.
- CDL as a post hoc comparator on the 10 terms CDL defines for the central residue (C-N(next) excluded: CDL assigns it from the following residue, NME, which CDL does not cover); run when the CDL table is supplied; same weighting.

## Interpretation boundary (from the contract)
Constrained local equilibrium geometry of capped dipeptides in vacuum at four phi/psi points only. Not dynamics, populations or experimental observables. A positive result supports only the narrow claim that the reported MechLib's equilibrium geometry is closer than fixed AMBER values to these DFT optima.
