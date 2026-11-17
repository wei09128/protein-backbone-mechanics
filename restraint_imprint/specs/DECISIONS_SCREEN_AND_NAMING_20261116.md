# Decisions fixed 16 Nov 2026 ~10:30 CST, before any library has been scored against water QM or MD
Wei and Claude. Recorded before the screens below are scored, so that no outcome can shape them.

## Names
- v2 = "deposit-average library": the average of deposited geometry, imprint included.
- v3 = "de-imprinted library": deposits corrected for measured refinement transmission (spec claude/MECHLIB_V3_SPEC_20261113.md, frozen sha 8404d010).
- Not used: "average of the original data", because the data are never seen without restraints.

## Which library is called MechLib (hierarchy fixed now)
| Evidence | Role |
|---|---|
| Confirmatory water-phase QM (molecule B, frozen spec) | primary physical referee |
| MD/NMR (M4 with amendment A2, 8 seeds x 100 ns) | independent solution referee |
| Atomic-resolution deposits (spec to be frozen before scoring) | independent crystal referee |
| COD (referee only; see addendum 12:30) | additional geometry evidence |
| Deposited-structure fit | diagnostic only (home advantage) |
Outcomes: v3 supported by the primary referee and not contradicted by any other confirmatory referee -> v3 is MechLib and v2 the deposit-average comparator. v2 supported -> v2 stays MechLib and v3 is the de-imprinted candidate. Mixed or inconclusive -> no winner is forced; both are reported, and the environment dependence is discussed. Until the referees report, the manuscript keeps "MechLib" = v2.

## Confirmatory QM molecule
Molecule B, ACE-X-Ala-NME (following Ala fixed in the beta region), water (ddX). Molecule A (ACE-X-NME) is kept only as historical comparison (Q6, vacuum, outcome-exposed) and as the screen below.

## Exploratory screens (cannot decide anything; may guide a later v4, never v3)
- **QM screen:** the 12 pilot cases (ALA, GLY, SER x alpha, beta, ppII, left; molecule A; water S1 arm) are scored against v2, v3, CDL, CCP4 and AMBER once S1 is complete. Questions: the direction of v3 - v2 relative to QM; the three carbonyl angles together; consistency across residues; vacuum versus water. To keep the confirmatory panel independent, its point selection excludes phi/psi within 15 deg of the four pilot points.
- **MD screen:** a descriptive interim on seeds 1-4 at the first 50 ns of every arm (reached during the M4 run itself; no separate run). No M4 setting, stopping rule or analysis changes because of it; the primary remains 8 seeds x 100 ns.
- **Use of screens:** they may suggest where a v4 should go, or why v3 fails. Any v4 is frozen before the confirmatory QM runs and is judged on it alongside v2 and v3. Screen data are never used to judge any library. (Superseded 17 Nov: no v4 learns from QM; see DEC4.)

## Note on COD
COD is the Crystallography Open Database (open access), not the Cambridge Structural Database (no licence). Superseded at 12:30 (addendum below): COD is a referee only.

## Addendum, 16 Nov 2026 ~12:30 CST (ledger v3.27, DEC1, DEC2, V5)
Made after the COD download was verified (18,481,345,940 bytes, 11:59) and before the COD probe ran or any COD carbonyl angle was computed; after the 7-case v4-lite dry run (lambda ~0 / 0.13 / 0.01) had been seen.
- **DEC1, COD referee only.** COD never trains, selects or calibrates any library. COD role rule v1.0 -> v1.1 (kit PBM_COD_PROBE_v1_1; rule sha 74a9644b..., probe sha 1af7465c...). The count thresholds now decide only REFEREE vs REFEREE_UNDERPOWERED. The COD scoring specification is frozen before any COD angle is computed. (Probe result 13:23: alphaR / beta / ppII 212 / 189 / 118 junctions; REFEREE_UNDERPOWERED; DEC3: descriptive only.)
- **DEC2, v4-lite unchanged; v4-adaptive deferred.** (Superseded 17 Nov by DEC4: v4-lite is not built.) v4-lite (spec cb2acb4e) is built and frozen unchanged when S1 has 12 cases accounted for. A proposed v4-adaptive (QM lambda x cell-level James-Stein shrinkage, robust/penalised lambda) is not in this paper: it was proposed after the dry run was seen, at most 12 pilot QM cases cannot support its extra parameters, and no independent test available could distinguish it (deposits favour v2 by construction; pilot QM points sit in well-supported cells; the atomic referee is junction-weighted). The confirmatory QM is never used to choose between library variants. No fourth MD arm.
- **V5, diagnostic only.** Shrinkage of each own-supported cell's v3 - v2 correction toward v2 (tau^2 = var(d) - mean(s^2), B = tau^2/(tau^2 + s^2), shared 200-draw bootstrap SEs): median B 0.82 / 0.23 / 0.82 (Ca-C-N / O-C-N / Ca-C-O); junctions in cells with B > 0.8: 93% / 42% / 93%. The cell-level O-C-N correction is the least resolved; the frozen v3 targets are retained. Reported in SI Table S13 as an uncertainty-aware sensitivity analysis.

## Addendum, 17 Nov 2026 ~10:05 CST (ledger v3.28, DEC4-DEC6)
Made when S1 reached 12/12 (11 pass, 1 fail), before any v4-lite build and before the atomic-resolution referee ran.
- **DEC4, no library learns from QM or COD (Wei).** v4-lite is not built or frozen; DEC2 is superseded. QM (pilot and confirmatory) and COD stay independent referees only. The "Exploratory screens" section above no longer feeds any v4.
- **DEC5, reserve split for the next study (environment-resolved geometry).** A sequence group is RESERVED iff int(SHA-256(sequence_group, UTF-8), 16) mod 10 is 0, 1 or 2; otherwise DISCOVERY. On structure_sequence_groups.tsv (sha 33299405): 2,515 of 8,284 groups (30.4%), 2,649 of 8,697 structures (30.5%) reserved. Script make_reserve_split.py sha 20d6202a; list RESERVE_SPLIT_20261117.tsv sha b05df94b. Fixed from identifiers only. Reserved groups are never used to fit, select or tune anything in that study. Disclosure: the atomic-resolution referee computes aggregate amplitudes over all <= 1.0 A structures, reserved ones included, but reads no environment variable.
- **DEC6, atomic-resolution referee on v2 and v3 only** (spec 82f68380 unchanged; kit run without v4 arguments). Roadmap in this paper's Discussion/SI changes from v4-lite to environment-resolved geometry. First test of the next study: within-cell environment contrasts (e.g. H-bonded helix vs loop residues at the same phi/psi) in raw deposits, by program and resolution band, DISCOVERY groups only. Rationale: CDL depends only on phi/psi and residue group and flat targets on nothing, so no library writes a within-cell environment difference (caveats: secondary-structure H-bond restraints at low resolution; restraint coupling to neighbours). Unrestrained re-refinement at <= 0.8 A is a separate experiment with its own stability checks, not assumed free of model bias. The phi/psi-prediction-from-geometry test was dropped: de-imprinting is a within-unit linear rescale for flat-library deposits (prediction unchanged by construction) and is computed from phi/psi for CDL deposits (circular).
