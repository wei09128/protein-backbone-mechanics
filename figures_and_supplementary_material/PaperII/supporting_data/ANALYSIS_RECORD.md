# Paper II verified analysis record

Final feature input: chain_audit_features_pisces_consistent.csv, 1,636,322 residues, 7,938 chains and entries. Exact matches: 7,836 chains and 1,618,814 residues. Sequence-identical alternatives: 102 chains and 17,508 residues. PISCES nominal X-ray acquisition resolution ≤2.0 Å; strict resolution diagnostic gives 7,935 at ≤2.0 Å and 324 at ≤1.0 Å owing to three metadata values just above 2.0. The diagnostic includes Gly and Pro; main 18-type summaries exclude them.

Verified logs: 01a, 01b, four 02 angles, 03, 04, 05, 06, 07, 08, main figure 4 assembly and matched bond analysis. Main Table 1 uses 01b sample SEM; Table S1 preserves 02 printed SEM. For sparse groups these SEM conventions differ slightly. Table S1 269/288 nominal P<0.01 and 270/288 pooled-sign agreement are different statistics.

Matched bond analysis uses 1,612,919 final-feature residue keys; 23,403 unmatched. The corrected script joins only three bond-length columns, preserving geometry and descriptors from the final feature file. Correlations do not measure stiffness.

VAL: 117,438 final-feature VAL rows; 117,414 extracted with required atoms; standard τ contrasts reproduce 05. Alternative CG2 changes group assignments and does not establish atom-label errors.

Relaxation: saved results restricted by retained chain membership, 376/778 retained. No MM rerun. OLS slope 0.42008119426413776, intercept 0.6047624596919859, r 0.482443. Sign agreement 22/36 raw and 30/36 adjusted. Crystal signs use updated printed 01b means. Relaxation candidate membership in quality-filtered feature rows remains unverified and is disclosed in SI. Original runtime/platform and convergence diagnostics unavailable.

Automatic mechanistic verdicts in source logs are not scientific conclusions adopted in the manuscript. Local-model split is residue-level; residuals include training and test rows; bootstrap is residue-level; φ/ψ raw numerical Pearson comparisons are exploratory.

The large feature CSV and PDB cache remain on the author workstation and are not included in this archive. This source package is not a substitute for depositing the final feature CSV for readers.

Verified original-file join: 374 exact + 2 same_seq retained; 390 NOT_IN_FEATURES and 12 diff_seq excluded. No candidate-selection history is inferred.
