# Confirmatory QM referee (molecule B): specification v1.0, FROZEN 17 Nov 2026 ~11:00 CST (approved by Wei 10:56)
History: v0.1 13 Nov; v0.2 17 Nov; v0.3 17 Nov after a design audit and operating-characteristics simulation (oc_sim.py sha 019c7460, oc_t.py sha e153fb6e); v1.0 = v0.3 + candidate-cell population rule (section 4), fixed before any point was selected.
Supersedes claude/QM_REFEREE_V3_SPEC_DRAFT_20261113.md (v0.1) where they differ. Written after the pilot arms R0 (12/12) and S1 (12/12) finished and after the exploratory screen SCR1 (7 S1 cases) was seen; written BEFORE any B1 (basis) or F1 (functional) value was seen by Claude, and before any QM value of molecule B exists. Confirmed by Wei, 17 Nov 10:56: no B1 or F1 output viewed. No external compute is available (feasibility rule applies to the local workstation). Frozen with its SHA-256 in the kit before any molecule-B calculation; amendments only before the relevant data exist, and disclosed.

## 1. Question
Is the intrinsic, local phi/psi pattern of the three carbonyl angles (Ca-C-N, O-C-N, Ca-C-O) in a capped peptide in implicit water as strong as the de-imprinted library (v3) or as weak as the deposit-average library (v2)? This is the primary physical referee of the decision hierarchy (DECISIONS doc, 16 Nov). Its answer concerns isolated residues in water; the atomic-resolution referee (ledger 4j) already answered for proteins in crystals (v2 too weak; consistent with v3). Disagreement between the two is reported as environment-dependent geometry, not forced to a winner.

## 2. Libraries scored (all frozen before this spec)
v2 (training_targets.npz, sha ef7b86b3; chi1 rule not applied: the endpoint terms do not use chi1), v3 (8404d010), CDL v1.2 (cdl_database.py, 79c197e5), the 2018 CCP4 TRANS targets (flat) and fixed AMBER ff14SB equilibria (flat). No v4 (DEC4). Predictions use the non-next-Pro arrays (the residue after the central X is Ala).

## 3. Molecule, charges, start geometry
- **Molecule B:** ACE-X-Ala-NME, all peptide bonds trans. The central X carries the measured junction (X carbonyl, N of Ala). The following Ala is fixed at phi -120, psi +130 (beta) in every case.
- **Residues:** all 20 as central X. Protonation as at pH 7 in water: Asp, Glu -1; Lys, Arg +1; His neutral (N-epsilon-H tautomer); Cys neutral. Charge and multiplicity recorded per case.
- **Side chains:** chi1 starts at the most frequent chi1 rotamer of that residue in the v2 training set (g-, t or g+; read from the training data, never from QM), chi2 etc. at standard values; side chains are free. Final chi1 is reported; a case whose chi1 leaves its starting rotamer is kept and flagged.
- **Pre-optimisation (speed only):** constrained GFN2-xTB with ALPB water if xtb is available at preflight (recorded), otherwise none. The DFT optimisation then starts from that geometry. The converged DFT geometry, not the start, is the result; the stationarity rule (section 5) guards against stopping early.

## 4. Points (from library predictions only; frozen with their hash before any QM)
- **Candidates:** non-next-Pro cell centres (10-degree grid) where v3 uses its own training mean for all three angles (V3_SUPPORT_MAP level 'own') and the cell holds >= 100 junctions in each of the three angles (cells with fewer have cell SEs above ~0.2 deg, which maximin would exploit as if it were pattern). If a residue has fewer than 8 such cells after the exclusions below, it is filled with its remaining own-supported cells in order of decreasing junction count.
- **Order:** greedy maximin on the v3 predictions of the three angles (Euclidean, degrees), seeded at the residue's most populated candidate; an ordered list of 8 points per residue is frozen, and the first k (section 8) are used. Points are cell centres, where the libraries' periodic Catmull-Rom interpolation equals the cell values.
- **Exclusion:** no point with max(|dphi|, |dpsi|) < 15 deg from any of the four pilot points (-60, -40), (-120, 130), (-75, 145), (60, 40).
- **Gly:** (phi, psi) and (-phi, -psi) are the same geometry mirrored; only one of each pair may be selected.
- **Pro:** phi is restricted by the ring; points are chosen among Pro's own cells as for any residue.
- Recorded with the points: rho = slope of centred v2 on centred v3 at the selected points, per angle (the v2-to-v3 amplitude ratio at these points; from predictions only).

## 5. Method, numerics, QC
- **Functional:** wB97X-D. **Medium:** implicit water, ddX PCM (the pilot rule is met: S1-R0 shape SD 0.18-0.48 deg > 0.05). **Basis:** by the pilot rule (def2-TZVP if the B1-R0 shape SD exceeds 0.10 deg in any of the three angles, else def2-SVP), subject to the feasibility rule (section 8).
- **Numerics:** integration grid (99, 590); SCF energy and density convergence 1e-10; ddX settings tightened from the pilot (recorded at preflight); geomeTRIC with constrained central phi, psi and the Ala phi, psi (--enforce 0.01, --conmethod 1), convergence set GAU_TIGHT.
- **Stationarity rule:** every converged case is restarted once from its final geometry with the same settings; it passes if each of the three endpoint angles moves by <= 0.02 deg and no bond by > 0.0005 A. A case failing this is re-converged once more; if it fails again it fails QC.
- **QC:** constraints within 0.5 deg; connectivity audit as in Q5; <= 2 automatic restarts; trans peptides (|omega| >= 150). At least 90% of cases must pass, or the panel is reported as incomplete.

## 6. Endpoints
- **Estimate (per angle):** centre QM values and v3 predictions within each residue (centring computed once, on the observed data); s = through-origin slope of centred QM on centred v3, pooled over all points.
- **Uncertainty:** residue-clustered bootstrap, 10,000 draws (seed 20261119), resampling whole residues so each residue's centred points stay together. Interval = s +/- t(G-1, 0.975) x SE_boot, where G is the number of residues with QC-passing points. (Simulation, 17 Nov: the percentile interval covered 88-94% with 12-20 clusters; this t-based interval 92-95%.)
- **Gate (applied first, per angle):** if r(centred QM, centred v3) < 0.3, the outcome for that angle is "pattern not reproduced by QM" and no slope test is made. (Threshold fixed in v0.1, 13 Nov, before any pilot result. Disclosure: the exploratory pilot screen, molecule A, 7 cases, showed r 0.32 for Ca-C-N.)
- **Tests:** for c in {rho, 1}: t = (s - c) / SE_boot, two-sided p from t(G-1). "Excludes c" = Holm-adjusted p < 0.05, Holm over the three angles with the family fixed at three (an angle stopped by the gate counts as p = 1). "Includes c" = unadjusted p >= 0.05.
  - v3 supported: includes 1 and excludes rho. v2 supported: includes rho and excludes 1. Otherwise inconclusive.
- **Basis allowance (only if def2-SVP is used under section 8 step 1):** delta_b = |slope of the centred (B1 - R0) shift on centred v3| at the four ALA pilot points, per angle. The interval is widened by delta_b on each side and the tests use the widened interval (equivalently, "excludes c" requires |s - c| > delta_b + the critical half-width). This is a bias band on the slope scale; it is not combined with the SE in quadrature and is not used anywhere else.
- **Secondary:** paired pattern error (centred MSE) of v3 vs v2; the same for CDL and the flat comparators (zero-pattern baseline); absolute errors for all terms; s against v2 and CDL; per-region residuals. The pilot panel (molecule A, water) is reported as outcome-exposed and decides nothing.

## 7. Size and expected operating characteristics
The v0.1 sizing rule (method noise only) is replaced: the simulation shows that precision is set by how closely the QM pattern tracks v3 (r) and by the number of points, not by method noise. Design target: 20 residues x k = 8 (160 cases); minimum k = 4. Expected probability of the correct verdict (single angle, residue heterogeneity SD 0.2 in slope; oc_sim.py, 400 replicates):
| Residues x k | r | truth s = 1 (v3): P(v3 supported) | truth s = 0.6 (v2): P(v2 supported) | worst wrong-direction rate |
|---|---|---|---|---|
| 20 x 8 | 0.75 | 0.91 | 0.94 | 0.01 |
| 20 x 8 | 0.50 | 0.73 | 0.92 | 0.03 |
| 20 x 4 | 0.75 | 0.84 | 0.92 | 0.03 |
| 20 x 4 | 0.50 | 0.45 | 0.76 | 0.04 |
| 12 x 4 | 0.75 | 0.69 | 0.85 | 0.04 |
| 12 x 4 | 0.50 | 0.28 | 0.49 | 0.03 |
The simulation used the percentile interval and one angle at alpha 0.05; the t-based interval and Holm adjustment above make all rates somewhat more conservative (fewer wrong-direction verdicts, slightly more inconclusive ones). r is the correlation of centred QM with centred v3 at the selected points. These are stated in advance so that an inconclusive result can be read as low power where r is low.
Point selection must give rho <= 0.85 per angle at the selected points (otherwise v2 and v3 are not separable there); rho is recorded with the frozen points.

## 8. Feasibility rule (fixed before any B1 value is seen)
Budget: the panel must finish within 28 days of wall-clock on the resources recorded at preflight (workers x threads), using the per-case time measured on the first 4 completed cases (one per region, residue Ala) to project the total.
If the projected time exceeds the budget, apply in this order, stopping at the first step that fits:
1. If def2-TZVP was required: use def2-SVP with the basis allowance of section 6.
2. Reduce k from 8 toward 4.
3. Reduce residues to this fixed set of 12 (fixed now): Ala, Gly, Ser, Leu, Val, Ile, Thr, Asp, Glu, Lys, Phe, Pro, with the lower power of section 7 stated.
4. If still infeasible: report the referee as not run on local resources, and request external compute; nothing is scored early.
Additional compute (cluster, cloud) may replace steps 1-3 if available before the panel starts; the method and points do not change with the resources. The final design (basis, k, residues) is recorded with its hash before the first production case.

## 8b. Interpretation, fixed before any molecule-B result (crystal referee already reported: v2 too weak, consistent with v3, all three angles)
| QM outcome (per angle) | Reading |
|---|---|
| v3 supported | The stronger pattern holds both in protein crystals and in an isolated peptide in water |
| v2 supported | Environment-dependent geometry: proteins in crystals carry the stronger pattern, the isolated peptide in water the weaker |
| Inconclusive | Crystal result stands; the intrinsic strength is unresolved at this precision (read with section 7: low r means low power) |
| Pattern not reproduced (gate) | The crystal pattern's shape does not transfer to the isolated peptide; strength cannot be compared |
| Different outcomes for different angles | Reported angle by angle; no library is declared the winner overall |
MechLib naming follows the DECISIONS hierarchy: v3 replaces v2 only if this primary referee supports v3 (for the angles concerned) and no other confirmatory referee contradicts it.

## 9. Order and timing
The panel starts after the M4 MD finishes (expected ~24-25 Nov), when all 20 cores are free, unless external compute is available earlier. A 4-case timing run (step in section 8) may run beside the MD.

## 10. Allowed wording
v3 supported: "an independent electronic-structure calculation in implicit water saw the stronger conformation dependence restored by the correction". v2 supported: "the intrinsic local pattern of an isolated capped peptide in water is weaker than the pattern in protein crystals", read with ledger 4j as environment dependence. Inconclusive: "the QM referee could not distinguish the two amplitudes at the chosen precision". Never: "v3 (or v2) is physically correct".
