# MechLib v3: specification v1.1 (approved design; frozen before any build)
13 Nov 2026, Claude. Built with PBM_MECHLIB_V3_v1, whose code checks this file's SHA-256 before reading any data.

## What v3 is
MechLib v2 with its training data corrected for refinement imprint in the three carbonyl angles (Cα–C–N, O–C–N, Cα–C–O), then refitted with v2's exact rules. Every other term (τ, C–N–Cα, the Cβ angles with their χ1 rule, all five bonds) is copied unchanged from v2. v3 therefore differs from v2 only where the imprint has been measured.

## Inputs (all hash-checked)
- v2 training geometry (blank-altloc training audit, sha a2929836…) and training chains (35f69bd2…), as used to build v2.
- v2 target file (ef7b86b3…): copied for the unchanged terms and used for the identity gate.
- Refinement program and build from the frozen headers (25f466e2…).
- CDL v1.2 (cdl_database.py, 79c197e5…), evaluated per junction by periodic Catmull–Rom, as in the de-imprint calculation.
- Transmissions from the registered de-imprint run pbm_deimprint.5tttMa: PARAMETERS.json (970aa2c9…) and BANDS.json (10dfec0c…).

## Correction (per junction, per term t ∈ {Cα–C–N, O–C–N, Cα–C–O})
The de-imprint model: deposited pattern = (1 − T_s) × data pattern + T_s × library pattern, for stratum s.

**Strata corrected and used (primary):**
- PHENIX_post (PHENIX, build ≥ 2155; library CDL): T_P from PARAMETERS.json. Pattern L = CDL prediction at the junction.
- REFMAC (library: flat CCP4 link targets): T_R from PARAMETERS.json. Pattern L = 0.

**Strata not used for the three corrected terms (primary):** PHENIX_pre (library not identified), other programs (SHELXL, CNS, X-PLOR, BUSTER, …) and unlabelled structures. Their transmission has not been measured, so they are excluded from those terms only. They stay in every other term as in v2.

**Centring unit u:** (stratum, link era, residue, next residue Pro or not). Link era is one era for PHENIX_post. For REFMAC it is "new" (version ≥ 5.8.0222, strict ≥3-part numeric parsing) or "old" (≤ 5.8.0189). A REFMAC deposit is also "old" if it was deposited before 16 Apr 2018, when the revised dictionary did not yet exist (this places missing and two-part versions). REFMAC deposits that still cannot be placed are excluded for the three terms. Centring within the link era stops the 2018 dictionary level change being amplified as if it were pattern.

**Formula:** for junction j in unit u, with q_j the deposited angle, L_j its library value, and μ_u and L̄_u the means over the unit,
  d_j = μ_u + [(q_j − μ_u) − T_s (L_j − L̄_u)] / (1 − T_s).
Levels (μ_u) are preserved; only the within-unit φ/ψ variation is rescaled and the library pattern removed. Amplification 1/(1 − T_s) is 1.5–2.7 for every term and stratum, below the spec limit of 5 set in the de-imprint run.

**Outlier guard:** before correction, the de-imprint guard is applied to the three terms: values more than 8 robust SDs (1.4826 × MAD) from the median are set missing, separately for next-Pro and other junctions. The guard limits the amplification of gross errors. The number of rows removed is reported.

## Refit
The corrected junctions of PHENIX_post and REFMAC (natural mix, unweighted) are fitted with v2's rules for the three terms: 36 × 36 grid of 10° cells per residue, next-Pro split, a cell's own mean if it has ≥ 10 rows, otherwise the unsplit cell, then the pooled cell, then the global mean; evaluation by periodic Catmull–Rom at bin centres. The output file has v2's layout (term__v2 arrays of shape (2, 20, 36, 36), so the MD builder reads it unchanged), with v3 content and a version key.

## Gates (the build stops if any fails)
1. Hashes of the spec, the code and every input.
2. **Identity:** a refit of the full v2 training set (all strata, uncorrected, no guard) reproduces the frozen v2 arrays exactly for the three terms. This proves the fitting code is v2's rules. It is not "v3 at T = 0": that library is the reduced-composition control (PHENIX_post + REFMAC, uncorrected, same guard), which is built and reported separately so that the effects of the correction and of the composition can be told apart.
3. **Synthetic test:** a known data pattern, imprinted with a known T and library, is recovered within tolerance.
4. Program labels cover ≥ 90% of training structures, and the three corrected terms keep ≥ 50% of v2's training rows.

## Reported at build, decided before the build
- v3 minus v2 per term: mean and 95th-percentile |Δ| over supported cells, and the slope of the v3 pattern on the v2 pattern.
- The same for a composition-only control: PHENIX_post + REFMAC, uncorrected, with the same rules. This separates the effect of the correction from the effect of dropping the other strata.
- Three-angle sum: the φ/ψ variation (SD over supported cells) of the v3 sum compared with v2.
- Rows by stratum and era, guard removals, fallback counts.

## Stability and support (added in v1.1, before any build; reported, cannot change v3)
- **Uncertainty in T:** V3-Tlow and V3-Thigh are built with every T at the lower or upper end of its 95% interval from the de-imprint run. Their difference from v3 per term is reported, with O–C–N and Cα–C–O, whose registered evidence is weaker, flagged.
- **Resampling stability:** 200 sequence-group bootstrap draws (seed 20261117; the groups of the de-imprint analysis), the same draws for v2 and v3. For every split cell: standard errors of v2, v3 and v3 − v2. Per term: median standard errors, their ratio (the realised noise inflation) and the fraction of supported cells where |v3 − v2| exceeds 2 SE of the difference.
- **Dominance:** for every v3 cell, the number of structures and the largest share of one structure. Reported: the fraction of supported cells where one structure contributes more than half of the rows.
- **Support map:** for every residue × next-Pro × φ/ψ cell, which v2 rule supplied the value (own cell, unsplit, pooled or global) in v2 and in v3, with counts and standard errors (V3_SUPPORT_MAP.tsv.gz). A smooth surface must not hide cells that rely on fallbacks.

## Sensitivity libraries (built and reported, never used by the referees)
- V3-band: band-specific T (≤ 1.5 Å vs > 1.5 Å; > 2.0 Å uses the 1.5–2.0 Å value).
- V3-all: PHENIX_pre corrected with T_pre = 1 − 0.70 α_R (term-specific ratio from BANDS.json); other programs and unlabelled included uncorrected.
- V3-Tlow, V3-Thigh: see above.
- V3-sumproj: after correction, the three angles of each junction are shifted equally so that their sum equals the deposited sum.

## Scalar T or a 3 × 3 matrix (decided before the build)
The primary correction uses one T per angle and program, the registered estimates. A full three-angle matrix inverse is not used, for two reasons.
- **What the controlled experiment shows.** Refinement's response to target edits was t(I − J/3), t ≈ 0.82, residual 0.037 (Figure 1). The off-diagonal terms are explained by removing the angle-sum component. On sum-free patterns, which deposited patterns are because refinement keeps the sum, this matrix acts as a scalar.
- **The matrix cannot be inverted.** It is singular in the sum direction, and its sum-free part is the scalar already used.
The remaining risk is that unequal per-angle T values (for example PHENIX_post 0.53 / 0.50 / 0.62) create an angle-sum pattern that refinement would never produce. Two outputs address this: the three-angle-sum diagnostic (v3 against v2), and the V3-sumproj sensitivity library, which applies the matrix structure exactly (scalar correction on the sum-free part, deposited sum kept). Following Wei's rule, a matrix model would be warranted only if the measured responses showed redistribution the scalar model cannot explain. Here that means: if v3's sum-pattern rms exceeds v2's by more than 0.1°, or V3-sumproj differs from v3 by more than the bootstrap SE in more than 5% of supported cells, the result is reported as a limitation of the scalar correction. v3 itself is not changed after the build.

## Freeze and referees
The v3 target file's SHA-256 is written to V3_FREEZE.json and recorded in the evidence ledger before any referee sees v3. No referee result can change v3. The referees, each with its own frozen specification:
- **MD:** the registered M4 v3 arm (add-v3 with this hash; NMR scored only when all three arms are complete).
- **QM:** a new constrained-QM panel, specified after the noise-floor pilot and frozen before any v3 prediction is scored. Comparators: v2, v3, CDL, AMBER. The endpoint includes O–C–N.
- **Atomic-resolution geometry:** held-out deposits at ≤ 1.0 Å, stratified by program. The specification is written before scoring.

## Expected, stated in advance
v3 is built to remove agreement that deposits owe to their refinement library. It should therefore agree less well than v2 with deposited geometry (B30 held-out benchmark). That benchmark is reported, but it is not a referee for v3.

## Amendment record
v1.0 → v1.1 (13 Nov 2026, before any build or any real-data read): reports added (uncertainty in T, bootstrap stability, dominance, support map), V3-sumproj sensitivity, scalar-vs-matrix rule stated, identity-gate wording clarified. The four design decisions of v1.0 are unchanged (approved by Wei).

## Allowed wording
"MechLib v3 is v2 refitted on training data corrected for measured refinement transmission in the three carbonyl angles." The model passed its registered test for Cα–C–N only; the correction is applied to all three angles because refinement conserves their sum, so correcting one alone would be incoherent. Not allowed: "v3 is the true geometry"; "de-imprinted = unrestrained"; any physical claim before the referees report. For O–C–N and Cα–C–O, say that the correction rests on the coupled model, not on an independent registered success.
