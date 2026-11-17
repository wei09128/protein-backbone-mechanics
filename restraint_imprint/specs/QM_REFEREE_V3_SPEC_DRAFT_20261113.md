# QM referee for MechLib v3: specification DRAFT v0.1 (13 Nov 2026, Claude)
Status: design draft, written before the noise-floor pilot reports and before any v3 or CDL prediction has been compared with any QM value. Numbers marked [pilot] are filled by the fixed rules below from PILOT_SUMMARY.json; nothing else may change after the pilot.

## Question
Refinement flattens the φ/ψ dependence of the carbonyl angles in deposits (de-imprint D1–D4). v3 restores it: its pattern is about 0.95 of CDL's amplitude, v2's about 0.57 (ledger V4). The referee asks the question the crystal data cannot answer: **does an independent electronic-structure calculation see the stronger (v3/CDL) or the weaker (v2) conformation dependence?**

## Why the endpoint is the pattern, not the level
Library levels come from crystals. QM levels carry method offsets: the earlier Q6 test failed on bond lengths through systematic crystal-versus-vacuum offsets. v3 changes only the φ/ψ pattern of three angles; levels are unchanged by design. The primary endpoint is therefore the amplitude of the φ/ψ pattern within each residue.

## Primary endpoint (two-sided; both outcomes publishable)
For each corrected angle (Cα–C–N, O–C–N, Cα–C–O), centre QM values and v3 predictions within each residue (× next-Pro context) across that residue's φ/ψ points. Then regress centred QM on centred v3 through the origin. The slope s says how strong the QM pattern is relative to v3's. Since v2/v3 ≈ 0.60, the outcomes are:
- **v3 amplitude supported:** the 95% residue-clustered interval of s includes 1 and excludes 0.60.
- **v2 amplitude supported:** the interval includes 0.60 and excludes 1.
- **Inconclusive:** neither, reported as such.
The three angles are reported separately. Holm adjustment is applied across them for the exclusion tests. Pattern agreement (r of QM with v3, v2, CDL) is reported alongside. A slope is meaningful only if r > 0.3; otherwise the outcome is "pattern not reproduced by QM" for that angle, which is itself the P3c result.

## Secondary
1. Paired pattern error, v3 versus v2: the mean squared difference between centred prediction and centred QM.
2. The same comparison for CDL, the 2018 CCP4 targets and fixed AMBER. The last two are flat, so their pattern error equals the QM pattern variance, which is the zero-pattern baseline.
3. Absolute (uncentred) errors for all comparators and all 12 terms.
4. **Historical panel:** v3 scored on the frozen Q6 panel (80 cases, vacuum, 4 φ/ψ points per residue). Those QM values were seen in Nov 2026 with earlier libraries; this is labelled outcome-exposed and can never decide anything.

## Molecule and points (DECISION for Wei)
- **Option A, ACE–X–NME (as Q6):** cheap and comparable. Its weakness: the residue after C is a methylamide, not an amino acid. That may be why Q6 found library Cα–C–N patterns anti-aligned with QM (centred MSE −46%; hypothesis P3c).
- **Option B, ACE–X–Ala–NME,** with the following Ala fixed at a β-region φ/ψ: this puts a real residue after the junction. It is about 1.5× the atoms of Option A and roughly 2–3× the CPU per point. **Recommended**, because it tests the angle that carries the registered de-imprint success (Cα–C–N) in a realistic context.

**Point selection uses library predictions only, never QM.** Residues: all 20 (central X). Per residue, k φ/ψ points are chosen inside populated regions (v3 own-supported cells), spread to maximise the spread of the v3 Cα–C–N pattern (greedy maximin over the three corrected angles' predicted values, seeded at the residue's most populated cell). The selection is frozen with its hash before any QM run.

## Method and medium (rule fixed now, applied to the pilot)
- **Medium:** implicit water (ddX PCM). Proteins are condensed phase, and the Q6 bond offsets point to environment. If the pilot shows S1 − R0 shape SD < 0.05° for all three angles, vacuum is used instead, to keep comparability with Q6.
- **Basis:** def2-TZVP if the pilot's B1 − R0 shape SD exceeds 0.10° in any corrected angle, otherwise def2-SVP (as Q5/Q6).
- **Functional:** ωB97X-D (the registered method). F1 (M06-2X) is used only to report how far the conclusions depend on the functional.

## Size (rule fixed now)
σ_m = the largest pilot shape SD across the three angles for the chosen method (the noise floor) [pilot]. σ_L = the SD of the v3 centred pattern at the selected points (from predictions). The number of points per residue, k, is the smallest k ≥ 4 such that 1.96 × σ_m / (σ_L × √(20k)) ≤ 0.10. That is a half-width of 0.10 from method noise alone, so that 0.60 and 1.0 are separable with room for model mismatch. The maximum is k = 8 (160 constrained optimisations); if k > 8 is needed, the panel is 160 and the expected half-width is reported in advance.

## Pass criteria and QC
QC as in Q5: φ/ψ constraints within 0.5°, connectivity audit, ≤ 2 automatic restarts. At least 90% of cases must pass QC, or the panel is reported as incomplete.

## Allowed wording after the result
v3 supported: "an independent electronic-structure calculation saw the conformation dependence restored by the correction, not the flattened one". v2 supported: "QM did not support restoring the pattern; the flattening may be partly physical or the capped-peptide model may not capture it". Inconclusive: "the QM referee could not distinguish the two amplitudes at the chosen precision". In no case: "v3 is physically correct".

## Pre-stated mechanistic prediction for the pilot (written 13 Nov 2026 ~15:00 CST, before any pilot result was seen)
Hypothesis (Claude, with Wei): the residue-centred phi/psi pattern of Ca-C-N is set by the two psi-dependent 1,4 contacts across the Ca-C bond. Near psi ~ -40 (alpha), N(i) faces N(i+1)-H; near psi ~ +130-145 (beta, ppII), N(i)-H faces O(i) (the intraresidue C5 contact). In vacuum both contacts are attractive hydrogen-bond-like interactions. The alpha contact pulls N(i+1) toward N(i), closing Ca-C-N; the C5 contact pulls O(i) toward N(i), closing Ca-C-O and, with the angle sum fixed and O-C-N stiff, opening Ca-C-N. In water (and in proteins) both are screened: the alpha contact becomes net repulsive (Ca-C-N opens) and the C5 pull weakens (Ca-C-N closes in beta/ppII).
Prediction for S1 (water) minus R0 (vacuum), residue-centred, mean over ALA/GLY/SER:
1. Ca-C-N shift at alpha > the mean shift at beta and ppII (alpha moves up relative to beta/ppII).
2. Ca-C-O shifts in the opposite sense: beta and ppII up relative to alpha.
3. O-C-N moves least of the three (|shift| smaller than Ca-C-N's and Ca-C-O's).
Each is reported as confirmed or not. None can change v3, and none changes the medium rule above. They test only whether the vacuum Ca-C-N disagreement (P3d) has the proposed cause.

Note added 13 Nov 2026 ~15:15 CST, still before any pilot result: v3's own split at the four Q6 points (residue-centred, mean of 20 residues; Ca-C-N / O-C-N / Ca-C-O) is alpha +0.94 / -0.52 / -0.40; beta +0.08 / +0.43 / -0.56; ppII -0.59 / +0.23 / +0.34; left -0.42 / -0.14 / +0.61. In crystals O-C-N is not stiff: at beta, the lowered Ca-C-O is balanced by O-C-N (+0.43), not by Ca-C-N as in vacuum (+0.51). The assumption behind prediction 3 ("O-C-N stiff") is therefore doubtful. Predictions 1-3 are kept unchanged as registered; the alternative reading (vacuum and condensed phase differ in which partner, Ca-C-N or O-C-N, compensates Ca-C-O at beta) is recorded here so that either outcome can be read.

## Interpretation caveat added 13 Nov 2026 ~17:00 CST (before any result)
The QM model (capped peptide, implicit water) measures the intrinsic, local phi/psi dependence. Libraries describe geometry in proteins, where helix and sheet hydrogen bonding is correlated with phi/psi. A QM slope near 0.6 would therefore not by itself show that v2 is right for proteins: the rest of the pattern could come from secondary-structure environment. The wording must say "intrinsic local pattern" for QM. The MD/NMR referee (in-protein) and COD (crystalline peptides) address the environment part. Two further alternatives, recorded so that results can be read either way: (a) part of the deposited pattern could be generated by phi/psi-dependent non-bonded restraints common to all refinement programs (e.g. N(i)...N(i+1)), not by the data, which would make the data pattern weaker than estimated; (b) errors in deposited phi/psi attenuate any pattern measured against them (errors in variables); this can only make the true pattern stronger than estimated, never weaker.
