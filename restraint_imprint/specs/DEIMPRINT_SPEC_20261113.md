# De-imprinting deposited backbone geometry by calculation: specification v1.0 (frozen before any data are read)
13 Nov 2026, Claude, agreed with Wei. Revised while coding, against synthetic data only and before any real data were read: the prediction error is a noise-corrected mean squared error, because a mean absolute error penalises the model for amplifying predictor noise. Run with PBM_DEIMPRINT_v1 (checks this file's SHA256 first). Analysis only: no library is built or promoted; MechLib v2 remains the reported library.

## Question
Can the measured refinement response be used to remove the library's contribution from deposited φ/ψ patterns, so that deposits refined with different libraries yield one consistent data-supported pattern?

## Model
For a refinement stratum s with library pattern L_s (within-group, φ/ψ-dependent part of its targets), the deposited pattern is
  P_s = (1 − T_s) D + T_s L_s + noise,
where D is the pattern supported by the data and T_s is the effective transmission of a pattern change in that stratum. CDL for PHENIX post-default deposits (build ≥ 2155); flat within group (L = 0) for REFMAC deposits (all versions; CCP4 link and monomer targets do not vary with φ/ψ). Eliminating D between the two strata gives the testable relation
  P_REFMAC = γ1 P_PHENIXpost + γ2 C_PHENIXpost,  with γ1 = α_R/α_P, γ2 = −γ1 T_P, α = 1 − T.
So T_P = −γ2/γ1, α_R = γ1(1 − T_P), T_R = 1 − α_R. The de-imprinted pattern is D̂ = P_REFMAC/α_R = (P_PHENIXpost − T_P C)/α_P.

## Data (hash-checked, as the home-advantage kit)
Corrected archive features (sha e905f8b8...), same junction filters and robust guard; program and build from the frozen headers (sha 25f466e2...); sequence groups (sha 33299405...); CDL v1.2 (sha 79c197e5...), evaluated per junction by periodic Catmull–Rom (nodes −180..170). Terms: Cα–C–N, O–C–N, Cα–C–O (each analysed separately). Strata: PHENIX_post (family PHENIX, build ≥ 2155) and REFMAC (family REFMAC, any version). PHENIX_pre (build < 2155; library not identified, assumed flat) secondary only.

## Patterns
Cells: CDL residue group (8 groups: general, Ile/Val, Gly, Pro × next-Pro or not) × 10° φ × 10° ψ bin. Sequence components are split once into halves A and B (numpy default_rng(20261115), components shuffled, alternate assignment). A cell is used for a term only if it has ≥ 20 junctions in each of PHENIX_post and REFMAC in each half (set fixed before any fit). Per stratum and half, cell value = mean deposited angle; C = mean of the per-junction CDL predictions of that stratum's junctions in the cell. Patterns are centred within each group over the used cells (unweighted). Cells are weighted equally.

## Estimation
Within one half, γ is estimated by two-stage least squares to remove attenuation from noise in P_PHENIXpost: the half is split again by component (alternate assignment of its components in the shuffled order); dependent variable P_REFMAC (whole half), regressors [P_PHENIXpost from sub-half 2, C], instruments [P_PHENIXpost from sub-half 1, C]. Reported parameters (T_P, T_R per term) are the mean of the estimates from half A and half B; 95% percentile intervals from 1,000 component bootstrap draws (seed 20261116, halves kept), refitting everything in each draw.

## Primary test (out of sample; per term; Holm over the three terms)
Fit γ on one half, predict the REFMAC pattern of the other half from its PHENIX_post pattern and C; average the two directions. Error E = mean squared difference between predicted and observed REFMAC pattern over used cells, minus γ1² times the noise variance of the PHENIX_post pattern estimated from its two sub-halves ((P1 − P2)²/4, averaged over cells); the REFMAC noise is common to both models and cancels in the comparison.
Comparator: the best scaled copy (P_REFMAC = k P_PHENIXpost, k fitted on the other half with the same instrument), which ignores the library term.
Success for a term requires both: (1) E_scaled − E_model > 0 with a 95% bootstrap interval above zero (Holm-adjusted one-sided p < 0.05); (2) E_model below the E_model of each of 20 refits in which C is replaced by a CDL table permuted within group (cells shuffled; seeds 1..20), i.e. the gain needs CDL's own pattern.

## Pre-registered comparisons (reported whatever the outcome)
- Transmission agreement: T_P against the controlled, angle-sum-preserving transmission from the 19-structure panel (compensated Cα–C–N band means 0.494 ≤1.0 Å, 0.871 >1.0–1.5 Å, 0.927 >1.5–2.0 Å; 0.927 used above 2.0 Å), averaged over the resolution mix of the PHENIX_post junctions in used cells. Reading: "consistent" if this value lies in the 95% interval of T_P; otherwise the direction is reported (deposits transmit more or less than one-macrocycle, fixed-weight refinement).
- Resolution: T_P and T_R re-estimated in bands ≤1.5, >1.5–2.0, >2.0 Å (cells re-selected per band with ≥ 10 junctions per stratum per half). Expected from Figure 1: lower T at better resolution.
- Amplification: 1/α_R and 1/α_P reported; D̂ is not interpreted where 1/α > 5.
- PHENIX_pre: ratio α_pre/α_R from P_pre = (α_pre/α_R) P_REFMAC (same IV).
- Implausible parameters (T outside [0, 1)) are reported as a failure of the model for that term, not truncated.

## Outputs
CELLS.tsv (cell patterns per stratum and half), PARAMETERS.json, PRIMARY_TEST.json, BANDS.json, DEIMPRINTED_PATTERN.tsv (D̂ per term and cell from REFMAC and from PHENIX_post, with their counts), DeImprint.png/pdf, RUN_LOCK.json.

## What it will and will not show
Will show whether one transmission model with one data pattern explains PHENIX and REFMAC deposits jointly, out of sample and specifically through CDL's pattern, and what effective transmission the archive implies. Will not show that D̂ is physically correct (it is the pattern the deposited data support under a linear model), or estimate the level (mean) of any term; no library is built here.
