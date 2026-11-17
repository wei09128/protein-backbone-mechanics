# MechLib MD/NMR referee (M4): protocol, fixed before production
13 Nov 2026, Claude, agreed with Wei. This document, the runner (pbm_md_m4.py), the system builder (build_m4_systems.py) and the scorer (pbm_md_m4_score.py) are hashed into M4_LOCK.json before any production frame exists. Nothing here changes after the lock except the registered addition of the MECHLIB_V3 arm.

## Question
Does replacing fixed AMBER ff14SB backbone bonded equilibria with MechLib's conformation-dependent equilibria change agreement between simulated backbone φ and experimental ³J(HN,Hα) couplings in GB3 and ubiquitin; and does the de-imprinted v3 differ from v2?

## Systems
The M3 AMBER_FIXED systems and prepared structures (GB3 1P7E chain A, 9,874 atoms; ubiquitin 1UBQ chain A, 17,248 atoms; ff14SB + TIP3P, 0.15 M NaCl, PME; M3 lock hashes ce21730a / eb99b468) are reused unchanged. MechLib arms add, for each of the 11 backbone terms of the M2-B/M3 design (τ, N–Cα–Cβ, C–Cα–Cβ, Cα–C–N, C–N–Cα, Cα–C–O; N–Cα, Cα–C, C=O, C–N, Cα–Cβ), one force replacing the AMBER equilibrium by MechLib's φ/ψ surface with AMBER's force constant, as in M3, now with MechLib's next-Pro split (40 surfaces) and merged per term. O–C–N stays AMBER (as in M3); χ1 conditioning is not represented. Builder checks: λ = 0 identity with AMBER_FIXED, finite energies, serialization identity, and an orientation check (per-term MechLib energies agree with an independent Catmull–Rom evaluation within 5% of the summed absolute per-bond energies; this denominator was set before the lock, after a ubiquitin build failed the earlier net-energy denominator, which cancellation makes unstable).

## Arms
- AMBER_FIXED (locked now)
- MECHLIB_V2: clean MechLib v2 targets (SHA-256 ef7b86b3...) (locked now)
- MECHLIB_V3: registered now; built from the frozen v3 target file and recorded in M4_LOCK.json (system and library hashes) by `add-v3` before any V3 frame exists, with the same seeds, velocities and protocol.

## Simulation protocol (M3, unchanged)
Minimize with protein heavy-atom restraints; one velocity draw per protein and seed at 298 K reused in every arm (paired); NVT 100 ps at 1 fs (restrained); NPT 1 bar, 298 K with restraints 1000 → 100 → 10 kJ mol⁻¹ nm⁻² (250 ps each, 2 fs); NPT 1 ns unrestrained; production NPT, Langevin middle integrator (1 ps⁻¹), 4 fs with hydrogen mass repartitioning (3.024 Da), 10-ns chunks with checkpoints, protein frames every 10 ps.
Seeds 2026111301–2026111308 (8 per protein and arm). Production 150 ns per lineage. 2 proteins × 3 arms × 8 seeds = 48 lineages, 7.2 µs.

## Scoring (pbm_md_m4_score.py; refuses to run until all three arms are locked and complete)
Per lineage: discard the first 20% of production; mean predicted ³J per residue (J = 6.7cos²(φ − 60°) − 1.3cos(φ − 60°) + 1.5 Hz); MSE against 50 GB3 and 63 ubiquitin non-Gly observations (M3 transcriptions).
Paired seed contrasts ΔMSE = MSE(A) − MSE(B); per protein mean, SD, 95% t interval (df = 7); equal-protein mean.
- Primary: AMBER_FIXED − MECHLIB_V3. Secondary: AMBER_FIXED − MECHLIB_V2; MECHLIB_V2 − MECHLIB_V3.
- Outcome per contrast: SUPPORTED (both protein means > 0 and both intervals exclude 0), DIRECTIONAL (both > 0, an interval includes 0), MIXED (opposite signs), NOT_SUPPORTED (both ≤ 0).
- Sensitivities (reported; cannot change the primary outcome): automated equilibration on Cα RMSD, no discard, second half only; 5-block convergence.

## Expected sensitivity (stated in advance)
With M3's seed-to-seed spread (SD of paired differences ≈ 0.09 Hz² GB3, 0.18 Hz² ubiquitin), 8 seeds give 95% half-widths of about ±0.07 and ±0.15 Hz². M3's MechLib v1 effects (+0.04, +0.18 Hz²) would not both be resolved. ³J depends on φ, whereas MechLib changes bond geometry, so small effects are expected; MD is a secondary referee, reported whatever the outcome. The Karplus relation carries its own error (≈0.4–0.6 Hz RMS), identical across arms.

## Amendment A2 (production length), registered 16 Nov 2026 before any MechLib production frame and before any scoring
Measured throughput with 16 concurrent workers was about 540 ns/day for AMBER (single-process benchmark 707), so the full design (7.2 µs) would finish around the end of November. Production per lineage is shortened from 150 to 100 ns (10 chunks). Seeds (8), arms (3), proteins (2), equilibration, timestep, paired velocities, NMR observables and the scoring rules are unchanged. Total: 2 × 3 × 8 × 100 ns = 4.8 µs. AMBER chunks already produced beyond 100 ns are kept and never scored; the scorer reads chunks 000–009 for every lineage.
Why not fewer seeds: the primary uncertainty comes from seed-to-seed spread. With M3's paired SDs (0.09 Hz² GB3, 0.18 Hz² ubiquitin), 8 seeds (t, df 7) give 95% half-widths of about ±0.07 / ±0.15 Hz²; 4 seeds (t, df 3) would give about ±0.14 / ±0.29 Hz², twice as wide. Shortening each lineage costs much less precision than dropping seeds.
Execution: the workers are restarted with 8 shards. Because pairs are taken in seed-major order, each worker finishes a seed 1–4 pair (AMBER, V2, V3) before its seed 5–8 pair, so seeds 1–4 are complete first. Stopping after seeds 1–4 would need a further amendment, decided by calendar and not by any result, with the 4-seed widths above reported.
The decision was made from throughput alone; no NMR observable or trajectory property of any arm was examined.
