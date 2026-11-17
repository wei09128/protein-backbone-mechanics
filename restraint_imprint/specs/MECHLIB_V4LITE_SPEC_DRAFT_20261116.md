# v4-lite ("intrinsic-calibrated library"): specification v1.0 (16 Nov 2026, Claude, agreed with Wei; frozen in PBM_V4LITE_v1)
To be frozen, and v4 built and hash-recorded, after the pilot water arm (S1) has all 12 cases accounted for and before the confirmatory QM or the atomic-resolution referee runs. Nothing below may change after freezing.

## Definition
For each carbonyl angle c (Ca-C-N, O-C-N, Ca-C-O), on every array of the library (split and unsplit grids):
  v4_c = v2_c + lambda_c x (v3_c - v2_c),   0 <= lambda_c <= 1.
All other terms are copied from v2 (identical in v2 and v3). v4 therefore lies on the axis between the deposit-average library (lambda 0) and the de-imprinted library (lambda 1). It changes the strength of the phi/psi pattern, not its shape, and keeps v2's levels and cell structure.

## Calibration rule (fixed now)
- **Training data:** the pilot water cases (S1: wB97X-D/def2-SVP, ddX water, molecule A ACE-X-NME) that pass QC, at the time S1 has 12 cases accounted for. No other QM is used; Q6 (vacuum) is excluded.
- **Fit:** within each residue, centre the QM values and the v2 and v3 predictions (evaluated at the registered phi/psi, next residue non-Pro). Choose lambda_c by least squares of the centred QM on centred v2 + lambda (centred v3 - centred v2), then clip to [0, 1].
- **Shape gate:** if the training correlation r(centred QM, centred v3) < 0.5 for an angle, the pattern shapes disagree and a strength factor is meaningless, so lambda_c = 0 (v2 kept) for that angle.
- **Reported, never used to change lambda:** the jackknife spread of lambda_c (leave one residue out).

## Uncertainty (component D)
Each corrected cell carries sigma = sqrt(var(v2) + lambda^2 var(v3 - v2) + 2 lambda cov(v2, v3 - v2)), from the 200-draw sequence-group bootstrap of the v3 build (cov = [var(v3) - var(v2) - var(v3 - v2)] / 2). Stored as term__sigma arrays, alongside counts and a support flag (own cell / unsplit / pooled / global).
**Calibration test (before freezing; reported):** rebuild v3 (and so v4) on two halves of the training sequence groups, and check whether z = (half A - half B) / sqrt(sigma_A^2 + sigma_B^2) over supported cells has |z| < 1.96 in about 95% of cells. Reported as calibrated (90-98%), over-confident (< 90%) or conservative (> 98%). It does not block the build.

## Deferred
- B, more next-residue context (next-Gly): deferred; not in v4-lite.
- A, program- and era-specific transmission; C, a coupled-angle model (not needed: the V3-sumproj check passed); E, environment-specific geometry. These form the roadmap (Discussion, SI failure-diagnosis table).

## Referees for v4 (all fixed before v4 exists)
- **Confirmatory water QM (molecule B, new points, excluding pilot points):** tests transfer of the calibration to a new molecule and new conformations. Not independent physics, because v4 learned from water QM, and stated as such.
- **Atomic-resolution referee (claude/ATOMIC_RES_REFEREE_SPEC_20261116.md):** v4's main independent judge (in-protein, crystal).
- **COD:** if it is a referee.
- **MD:** none (M4 has three arms); stated.

## Naming and claims
v4 is the "intrinsic-calibrated library". Claim "we built a new library" only if v4 passes the confirmatory QM transfer test and is not refuted by the atomic-resolution referee. In all cases the paper may claim the concept: the deposit-average and de-imprinted libraries bound a calibration axis, and independent physics places a library on it.
