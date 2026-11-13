# Supporting information for The restraint imprint: how crystallographic refinement shapes deposited protein geometry

Wei Chen and Xiaohong Lu. This file contains detailed methods, supplementary methods, Tables S1–S10 and Figure S1. Values are rounded; unrounded values are in the machine-readable tables of the code repository.

## Detailed methods

### Structural cohorts and geometry extraction

Analysis-specific manifests were used rather than treating all PBM cohorts as one dataset. PISCES-derived chain selections and PDB coordinates supplied the historical structural source; retained chain keys, training dates and junction coverage were fixed in the relevant manifests. [Wang2003,Berman2000] The corrected archive analysis read 1,698,236 exported rows and retained 1,588,519 after duplicate-key, trans-peptide and defined-torsion requirements. Term-specific robust outlier guards were applied separately. Structure-level slopes were available for 7,905 entries. These counts are specific to this export and analysis, not revised totals for the earlier atlas paper.

The primary correction used junction geometry requiring blank alternate-location identifiers for the contributing atoms. It avoids combinations assembled from independently selected alternate conformers, but also excludes valid alternate conformations. Standard-residue identity, peptide connectivity and torsion availability were checked as required by each analysis. For source-ordinal neighbor reconstruction, the next residue had to be the exact consecutive exported ordinal in the same entry and chain, have a finite C–N link no longer than 1.8 Å, and agree with the existing next-Pro flag. Missing context was excluded without gap filling. Exact feature and manifest hashes are listed in Table S1.

Sequence components were constructed from the joint MMseqs2 screen at 30% alignment identity with 80% query and target coverage. Entries connected through any clustered chain were retained in the same component, including transitive and multichain connections. [Steinegger2017] All 7,905 scored archive entries had assignments, representing 7,586 conservative structure components. Missing assignments were not replaced with artificial singleton groups.

### Refinement interventions

The expansion panel was frozen before shifted refinements, with eight structures per band at ≤1.0 Å, >1.0–1.5 Å and >1.5–2.0 Å. Coordinate, reflection and kit identities were hash checked. Deposited observations and original free-set semantics were audited before conversion and stock refinement; intensities were converted to amplitudes with the French–Wilson procedure where only intensities were deposited. [French1978] No free set was regenerated to rescue an experimental structure. Signed arms used identical starting coordinates and observation inputs.

PHENIX 2.2.1-6174 refinements used one macrocycle of individual-site refinement, fixed coordinate weight, and no weight optimization, solvent update, peptide flipping or omega-CDL. [Adams2010,Afonine2012,Liebschner2019] In-memory CDL replacements changed Cα–C and N–Cα targets by ±0.010 Å and Cα–C–N and τ targets by ±1°. Compensated arms combined ±1° ACN with ∓1° OCN. Isolated ACO and OCN edits supplied the exploratory response matrix. Native lookup tracing verified the intended library path.

Initial restraint identities, model values, ideal targets, sigmas and weights were compared across arms. Output atoms, raw amplitudes and sigmas, original flags, filtered support and map-only reflection additions were audited separately. Five structures failed identical filtered-support requirements and were excluded without replacement. Nine bond arms were recovered through a disclosed post-run correction for bounded differences between targets rounded to 0.001 Å in GEO output. Original failed states and earlier summaries were preserved.

A separate PTRANS-direction intervention re-refined 4YCA after editing the before-Pro rows of the CDL table by the documented 2018 contrast change (Cα–C–N +1.798°, O–C–N −2.453°, Cα–C–O unchanged), with one-cycle, five-cycle and geometry-only arms and an analytic planar three-restraint solution for comparison; 18 before-Pro and 531 control junctions were evaluated. This synthetic intervention tests the direction of the mean change and does not reconstruct the REFMAC dictionaries or protocols.

For target half-step h and measured quantity q, transmission was [q(+h)−q(−h)]/(2h). The primary within-structure contrast was the mean bond transmission minus the mean isolated-angle transmission. The compensated-minus-isolated ACN contrast used identical eligible proxies. Structures received equal weight. Percentile 95% intervals used 10,000 structure resamples within resolution bands with seed 20261106. Site percentiles describe variation among junctions, not independent-structure uncertainty.

### Response matrices and reference sensitivity

At each common eligible carbonyl site, isolated signed edits supplied the columns of a three-angle matrix in ACN, ACO, OCN order. The compensated response was predicted as T(1,0,−1)ᵀ and excluded from matrix fitting. Matrix predictions were compared with direct target copying and zero response. For the library-change transfer test, the PTRANS-direction edit vector (Cα–C–N +1.80°, Cα–C–O 0, O–C–N −2.45°, as applied by the kit) was multiplied by each before-Pro junction's isolated-edit matrix from the earlier 4YCA arms; comparators were the same-structure and 19-structure mean matrices, a diagonal-only matrix, target copying, the fixed-row planar restraint solution and a scalar projector, scored by the error of the mean response vector and by sitewise absolute error. The before and after dictionary snapshots were retrieved by commit (a683db68 and 2bc9cc94) with matching file hashes, and active link blocks were parsed with reversed angle atom orders normalized. Per-structure errors averaged over common sites and the three angles; structures then received equal weight. The matrix design and evaluation were exploratory additions after pilot inspection. No entries were truncated to the range zero to one.

Reference-weight experiments at full resolution were interpreted as sensitivity to analysis choices. Their endpoint spread was not treated as a calibrated experimental standard deviation or used as an inverse-variance weight. The response matrix and reference tests do not establish data-only geometry, an information fraction or a stationary-point Hessian.

### Archive analyses: CDL alignment

For each term within a structure, observed geometry q and nearest-node CDL prediction t were centered within CDL residue class c. The alignment slope was b = Σ(q−q̄c)(t−t̄c)/Σ(t−t̄c)². The primary composite was the arithmetic mean of slopes for τ, ACN, ACO and C–N–Cα, requiring all four. Class definitions distinguished general, Ile/Val, Gly and Pro residues and next-Pro status. Periodic nearest-node lookup was used for this archive score, rather than the interpolation used in the model benchmark.

Local linear regressions used PHENIX build minus 2155 as the running variable, [Imbens2008] separate trends on each side and the recorded resolution covariate. Bandwidths 400, 700 and 1100 were retained. Sequence groups supplied bootstrap resampling units. The data-driven break scan remained exploratory. Class-preserving table permutations supplied a specificity check; the finite permutation rank used a nonzero denominator correction.

The calendar event study compared PHENIX and REFMAC deposits from 2011 through 2017 with 2014 as reference and the recorded external release chronology. Grouped intervals, the joint pre-lead diagnostic and the sensitivity to horizon were retained. The original sample-derived adoption-date analysis was archived and not promoted as a release-date experiment. A release-date difference-in-differences used the documented 2 September 2015 release as cutoff, with staggered adoption treated as partial exposure. Pre-period nonsignificance was not interpreted as equivalence. [Lee2010,Bertrand2004,Callaway2021,Roth2022]

### Archive analyses: dictionary era

Reference geometry was estimated from pre-revision structures by residue class, next-residue category and 10° φ/ψ node, requiring at least five observations per cell. Within-structure context residual contrasts required at least three context junctions and 20 comparison junctions. The original recorded REFMAC version groups were reproduced before calculating sensitivity analyses. Version parsing included abbreviated versions under the original convention.

Before-Pro contrasts, before-Gly specificity contrasts and cross-program calendar comparisons were reported separately. Reference uncertainty was examined as a sensitivity analysis with sequence-group weights shared by reference fitting and evaluation (Supplementary Methods). Both TRANS and PTRANS changes were retained in the interpretation. The dictionary change was sourced to the actual versioned records rather than attributed to the AceDRG paper, which describes a different restraint-generation contribution. [Vagin2004,Long2017,Winn2011,Murshudov2011] The before and after snapshots are commits a683db684eacb84ead595e3cd97c3f055f19b595 and 2bc9cc946452592783051b781b496a044d0155a8 of the CCP4 Monomer Library repository (https://github.com/MonomerLibrary/monomers; mon_lib_list.cif SHA-256 f3b2b8ed… and 4fcb5cca…).

### MechLib specification and held-out benchmarking

MechLib, as tested throughout, has one definition. It is a table of 12 backbone terms (τ, Cα–C–N, C–N–Cα, Cα–C–O, O–C–N, N–Cα–Cβ and C–Cα–Cβ angles; N–Cα, Cα–C, C=O, C–N and Cα–Cβ bonds) on a periodic 10° φ/ψ grid per residue type, split by whether the next residue is proline, evaluated by periodic Catmull–Rom interpolation at bin centres from −175° to 175°. A cell uses its training mean when it has at least ten observations; otherwise the unsplit residue value, then a pooled supported value, then the global training mean. For N–Cα–Cβ and C–Cα–Cβ only, residues with an assigned χ1 use a χ1-conditioned cell nested within the same grid, with the same minimum count and fallback (Rotamer conditioning). This is version 2 of the library; no other version is used. Benchmarks in the main text score Cα–C–N, O–C–N, Cα–C–O, τ and the two Cβ angles; the QM test scores 11 terms.

The corrected training cohort comprised 5,926 retained entry/chain keys and 1,224,328 source rows. Term-specific connectivity, torsion and standard-residue checks yielded approximately 1.12 million comparable backbone rows; exact counts differ by term. The matched unsplit and next-Pro models used identical training data, a minimum of ten observations for supported nodes, periodic 10° grids and Catmull–Rom interpolation at centers from −175° to 175°. Unsupported split nodes fell back to matched unsplit values. Unsplit residue nodes required ten observations, then used a pooled supported node or the global training mean. Twelve target terms were exported, but the present held-out upgrade benchmark scored ACN, OCN and C–N separately.

The target array SHA-256 was ef7b86b359c84492a0be791de52a1e64112a0153142b9f9f5a0417448d2731b1 (χ1 model 6210897c…). Raw learned OCN values were scored without the separately labeled closure transformation used in refinement deployment. CDL v1.2 predictor mappings were audited term by term. In particular, C–N used the recorded incoming CDL anchor while MechLib used its outgoing junction anchor.

The held-out join contained 618,881 junction rows in 1,295 structures and 479 sequence components, deposited after the training cohort. The primary B30 subset comprised 386,705 junctions whose residue-i Cα B-factor window mean (up to three eligible residues on either side, same chain) was at most 30 Å²; its exact definition and source identities are in Table S9, and the full geometry-QC subset was retained as sensitivity.

Errors were averaged within each structure, then within sequence components, with components receiving equal weight. Paired contrasts shared component weights across models and refinement strata. Percentile intervals used 10,000 component resamples, seed 20261106. The unsplit column in the clean benchmark denotes matched clean unsplit MechLib. Source-data and model changes were not conflated.

### Context prediction sensitivity

Amplitudes of surfaces trained within each refinement group or resolution tier (Figure 4A) came from an exploratory, entry-level context-prediction analysis on corrected geometry (7,924 entries; training 4,755, validation 1,586, test 1,583), compared with the REFMAC-trained surface by slope and correlation over shared nodes, without intervals. Model details are in Supplementary Methods.

χ1 was computed from the cached coordinates (N, Cα, Cβ and Cγ, or Oγ, Oγ1, Sγ or Cγ1 as appropriate; first model, blank alternate identifiers) and assigned to the nearest of g− (−60°), t (180°) and g+ (+60°); values more than 40° from every well were treated as unassigned. A same-atom gate required the τ recomputed from these atoms to match the exported junction value, and before any χ1 surface was fitted, the χ1-free MechLib surfaces rebuilt from the training geometry had to reproduce the frozen arrays exactly (τ, 1,119,095 rows; N–Cα–Cβ and C–Cα–Cβ, 1,032,744 rows; maximum difference 0.0). χ1 passed the gate for 94.9% of χ1-bearing training rows and 96.3% of held-out B30 rows. χ1-conditioned cells were nested within MechLib's next-Pro split and used its grid, Catmull–Rom interpolation and minimum count of ten, falling back to the χ1-free value otherwise (4.0% of training χ1 rows); residues without an assigned χ1, and Gly, Ala and Pro, used the χ1-free surfaces. The primary comparison was the B30 REFMAC stratum (137 components), χ1-conditioned minus χ1-free MAE for τ, N–Cα–Cβ and C–Cα–Cβ, with 10,000 component resamples (seed 20261111) and Holm adjustment over the three terms. The reported library was defined before scoring: a term uses χ1 only if its reading rule (Holm-adjusted improvement) passes, and the χ1-free surface otherwise.

### Home-advantage contrasts

For libraries P and Q and strata of deposits refined with each, the estimand was [MAE(P) − MAE(Q)] on P's home stratum minus the same difference on Q's home stratum; negative values mean P agreed relatively better at home. The primary contrasts, fixed before data were read, were C1, CDL against the 2018 CCP4 peptide-link targets on PHENIX post-default and REFMAC post-revision deposits; and C2, the 2018 against the earlier CCP4 targets on REFMAC deposits from the two version groups. C1 and C2 used the archive cohort (1,555,072 scored junctions); C3 repeated the CDL–MechLib comparison on the held-out B30 set. CCP4 targets were taken from the hash-matched dictionary snapshots used in Figure 2, Cα–C–N and O–C–N from the ordinary or, before proline, the proline peptide link, and τ and Cα–C–O from the residue monomer or link override, all flat in φ/ψ; CDL was evaluated by periodic Catmull–Rom interpolation, as in the held-out benchmark. Errors were averaged within structures and then sequence components; intervals used 10,000 component resamples (seed 20261112) with Holm adjustment over Cα–C–N and O–C–N. Secondary analyses (centred errors, resolution reweighting, strict version parsing and PHENIX pre-default as the away stratum) are reported in Table S7. The earlier CCP4 monomer Cα–C–O target describes a free carboxylate and was not the effective peptide target, so C2 is not interpreted for Cα–C–O.

### Training-mix sensitivity

The clean training cohort was labelled by refinement program from archive headers; 96.4% of the 5,926 structures carried a label (PHENIX 1,701; REFMAC 2,904), and unlabelled or other-program structures were excluded from the program-specific fits. With the code, grid, next-Pro split, minimum count and fallback of MechLib, we fitted a natural PHENIX-plus-REFMAC mix, the same set with junction weights scaled so the two programs carried equal total weight, and PHENIX-only and REFMAC-only sets; an all-program refit had first to reproduce MechLib exactly. Cell means were weighted; minimum-count and fallback rules used unweighted counts. Surface shift was the mean absolute difference between two fits over cells supported in both. Held-out error was scored on B30 PHENIX post-default and REFMAC strata with common coverage, 10,000 component resamples (seed 20261114) and Holm adjustment over the two terms. No fit was promoted to replace MechLib.

### Prospective library-swap forecast

The specification, including the frozen comparator set, edit definition and success rule, was recorded with its SHA-256 before the first stage was run. Stage 1 computed, for every eligible carbonyl junction in the 19 complete structures, the planned edit e = MechLib − CDL for Cα–C–N, Cα–C–O and O–C–N at that junction's φ/ψ and residue class, and the predicted response T·e from that junction's isolated-edit matrix (Figure 1E); junctions whose lookup keys could not be matched between arms were excluded (47 of 5,987). The prediction table's SHA-256 (8b3f6277…; full value in Table S10) was recorded in the evidence ledger before stage 2. Stage 2 re-refined each structure with the CDL table replaced by MechLib values at those three angles only (σ unchanged), under the Figure 1 protocol and audits, and compared observed with predicted responses. The primary endpoint was D, the per-structure MAE of target copying minus that of the site matrix, averaged over structures with equal weight; success required a 95% interval above zero from 10,000 structure resamples within bands (seed 20261113). The structure-mean matrix and zero response were secondary comparators. A first stage-2 attempt stopped before any refinement because a thread setting was rejected and no CDL lookups occurred; it was rerun with the setting corrected and the predictions unchanged.

### Quantum-chemical referee

The frozen MechLib-QM-Q1-v1 comparison comprised 80 constrained local-geometry cases (ACE–X–NME; 20 residues at four φ/ψ points). Returned term errors were weighted by the square root of the recorded force constant, with angular errors converted to radians, then averaged over available terms per case. Fixed-AMBER error minus MechLib error defined the paired difference. The registered conjunction required a positive lower residue-cluster interval, at least 12 favourable residue means, no Holm-significant worsening term and at least 68 quality-passing cases. Residue-cluster intervals used 10,000 draws and seed 20260915.[Efron1979,Holm1979] The quantum-chemical reference values and AMBER predictions were reused from the frozen term table, and the original statistics for the earlier library (a blind pre-2020 fit to pre-correction features) were reproduced exactly before re-scoring. MechLib was evaluated at each case's registered φ/ψ with the next residue not proline and χ1 in its registered starting rotamer; MechLib without χ1 was scored as a secondary predictor. The separate constrained Ala–Ala calculation (E6) is described in Supplementary Methods.

### Statistical conventions

Unless stated otherwise, intervals are 95% percentile bootstrap intervals from 10,000 resamples of the independent unit of each analysis: structures, within resolution bands, for controlled refinements and the swap forecast; sequence components (MMseqs2, 30% identity, 80% coverage, connected transitively) for archive and held-out analyses, with errors averaged within structures, then components, and components weighted equally; residues for the QM test. Paired comparisons used the same resamples and weights for both members of each pair. Holm adjustment was applied within each prespecified family, as stated per analysis. Analyses are labelled as registered (specification hashed before data were read: rotamer conditioning, home advantage, training mix, swap forecast, QM re-score and the original QM contract), prespecified within a frozen panel (the bond-minus-angle and compensated transmission contrasts) or exploratory (matrix evaluation, resolution comparisons, the 4YCA transfer test, context surfaces). Exploratory and post hoc results are never presented as confirmatory.

### Reproducibility

Input, code, model and output identities were retained in manifests. Plot reconstruction, returned-table arithmetic, native raw-output inspection and independent raw-data reproduction were recorded as separate checks. Python analyses used NumPy, pandas, SciPy and Matplotlib as recorded by each run.[Harris2020,McKinney2010,Virtanen2020,Hunter2007] Source corrections and post-outcome changes were documented rather than backdated as prespecified analyses. Registered analyses were run from specifications whose SHA-256 the code checked before reading data; amendments were made only before data were read and are recorded with the specifications. A synthetic feasibility branch (refinement against simulated data) did not reach endpoint stability and is reported in Supplementary Methods and Table S8.

## Supplementary Methods

**Context prediction (Figure 4A).** The corrected E1 sensitivity retained original program/resolution labels and entry-level training, validation and test assignments while substituting corrected geometry. Newly available entries were excluded; seven original entries lost usable full context, leaving 7,924. The split contained 4,755 training, 1,586 validation and 1,583 test entries. Fourier orders, ridge scaling, seed and sample cap were retained. The baseline described residue class; subsequent models added own torsions, nearest-neighbor identities and torsions, and second-neighbor terms. Gains were expressed relative to baseline MSE, with term-specific and stratum-specific results reported separately. Surfaces trained within each refinement group or resolution tier were compared with the REFMAC-trained surface by slope and correlation over shared nodes; these comparisons carry no intervals. The program composition of each resolution tier was tabulated from the frozen metadata. These entry-level fits were exploratory context analyses rather than sequence-family-independent physical validation. The corrected run did not independently repeat the separate older SHELXL surface analysis or the E4 comparison against QM.

**Reference uncertainty in the dictionary-era contrasts.** Reference uncertainty was examined with positive exponential sequence-group weights shared by reference fitting and evaluation. Fixed-reference and refitted-reference sensitivity intervals used the same draws, with original cell-support and structure masks fixed. This Bayesian cluster-weight sensitivity was kept distinct from the original frequentist group bootstrap. [Rubin1981] Historical point estimates and fixed-reference intervals were independently reconstructed from returned structure records; the joint raw reference-refit draw distribution was not independently rebuilt from the compact review archive.

**Constrained Ala–Ala calculation (E6).** The separate E6 calculation used neutral singlet ACE–Ala–Ala–NME with ωB97X-D/def2-SVP, density-fitted RHF-reference calculations in Psi4 and constrained geomeTRIC/TRIC optimization.[Chai2008,Weigend2005,Smith2020,Wang2016] Identical initial Cartesian coordinates were used for both endpoints. φ₂ and ψ₂ were fixed at −60° and −40°, ψ₃ at −40°, central ω at 180°, and φ₃ at −90° or −45°. The measured contrast is the latter minus the former endpoint's Cα–C–N angle. Integration used a 99×590 grid with SCF energy/density thresholds 10⁻¹⁰ and optimizer limits: energy change 10⁻⁷ hartree, RMS/max projected gradients 3×10⁻⁵/4.5×10⁻⁵ hartree bohr⁻¹, RMS/max displacement 1.2×10⁻⁴/1.8×10⁻⁴ Å, and maximum torsion error 0.01°. Both endpoints converged and all five constraints held within 10⁻⁶°; earlier attempts are preserved. Earlier attempts are preserved.

**Synthetic feasibility branch.** A separate synthetic feasibility branch generated one observation set from known 4AQO coordinates and passed native preparation audits. Its original four-pass and amended six-pass protocols stopped before all required arms established endpoint stability. Retrospective comparison of saved mmCIF and PDB outputs showed that PDB rounding inflated selected-angle changes. A subsequent restart from saved mmCIF coordinates nevertheless retained substantial coordinate movement. These numerical-development outcomes were preserved separately from the completed one-cycle real-data interventions. Synthetic response prediction and recovery of known geometry remain unassessed (Table S8).

## Table S1 Analysis sources and model identities

Analysis-specific populations are retained. Counts from different exports, model fits and scoring subsets must not be combined into a revised atlas total.

| Analysis | Population or identity | Scope |
| --- | --- | --- |
| Corrected archive | 7,905 structures; 7,586 sequence components | 1,588,519 rows after common archive filters; term support differs |
| Transmission | 24 frozen; 19 complete; 6 / 7 / 6 per band | No failed structure replaced |
| Clean training | 5,926 retained training keys; 1,224,328 source rows | Term-specific eligible rows differ |
| Clean held-out join | 1,295 structures; 479 components; 618,881 rows | Before common predictor coverage and B30 selection |
| Corrected context sensitivity | 7,924 entries; train 4,755 / validation 1,586 / test 1,583 | Entry-level split, not sequence-family separation |
| QM referee | 20 residues × 4 conformations = 80; 868 term rows | Frozen reference values reused; earlier library = blind pre-2020 fit (primary JSON SHA-256 b1aefba1…) |
| Prospective swap | 19 structures; 5,940 junctions (47 excluded at the key gate) | Predictions SHA-256 8b3f6277e3428154236769bb4621c8481c53665e3fdfb6acf70b3692db01f8d3 |
| Home advantage, archive | 1,555,072 scored junctions | C1, C2; C3 on held-out B30 |
| Rotamer conditioning | χ1 gate coverage 94.9% training, 96.3% held-out B30 | CHI1_MODEL.npz SHA-256 6210897c… |
| Training mix | 5,926 training structures; 96.4% program-labelled | PHENIX 1,701; REFMAC 2,904; no refit promoted |

Clean next-Pro target array SHA-256: ef7b86b359c84492a0be791de52a1e64112a0153142b9f9f5a0417448d2731b1. Specification SHA-256 prefixes: rotamer bd10bab8, home advantage 35f61cfa, swap 92e2b8a4, training mix 22447a12, QM re-score ddfd59f1. Compact source-file hashes are recorded in SOURCE_MAP.json. The B30 generating definition and its hash chain are documented in Table S9. The CCP4 dictionary snapshots are Monomer Library commits a683db684eacb84ead595e3cd97c3f055f19b595 and 2bc9cc946452592783051b781b496a044d0155a8.

## Table S2 Transmission and matrix outcomes

The primary bond-minus-isolated-angle contrast was 0.3146, 95% structure-bootstrap interval 0.2957–0.3319. Compensated-minus-isolated ACN was 0.2947, interval 0.2791–0.3089. Estimates describe a one-cycle fixed-weight protocol. The following table preserves equal-structure values; it does not treat individual sites as independent structures. ACN is Cα–C–N; matrix MAE is in degrees.

| PDB | Band Å | Cα–C | Isolated ACN | Compensated ACN | Matrix MAE |
| --- | --- | --- | --- | --- | --- |
| 9QDV | ≤1.0 | 0.7084 | 0.3156 | 0.5085 | 0.0291 |
| 9AYN | ≤1.0 | 0.7292 | 0.3463 | 0.5707 | 0.1214 |
| 9KWV | ≤1.0 | 0.5339 | 0.1975 | 0.4412 | 0.0924 |
| 4JP6 | ≤1.0 | 0.8482 | 0.4248 | 0.6835 | 0.0302 |
| 2WFI | ≤1.0 | 0.6692 | 0.2934 | 0.4777 | 0.0358 |
| 5AL6 | ≤1.0 | 0.3988 | 0.1771 | 0.2848 | 0.0503 |
| 4YSL | >1.0–1.5 | 0.9637 | 0.5639 | 0.9387 | 0.0500 |
| 2XRH | >1.0–1.5 | 0.9373 | 0.5457 | 0.8823 | 0.0436 |
| 5FAF | >1.0–1.5 | 0.8774 | 0.4755 | 0.7711 | 0.0376 |
| 9LTB | >1.0–1.5 | 0.9518 | 0.5591 | 0.9397 | 0.0417 |
| 3BMZ | >1.0–1.5 | 0.8698 | 0.4462 | 0.7671 | 0.0493 |
| 2OPC | >1.0–1.5 | 0.9533 | 0.5748 | 0.9307 | 0.0411 |
| 1Y8A | >1.0–1.5 | 0.9068 | 0.5525 | 0.8692 | 0.0521 |
| 3LWA | >1.5–2.0 | 0.9396 | 0.5711 | 0.9048 | 0.0326 |
| 3B4Q | >1.5–2.0 | 0.9022 | 0.5384 | 0.8453 | 0.0399 |
| 8KIH | >1.5–2.0 | 1.0330 | 0.6117 | 0.9456 | 0.0594 |
| 1MG7 | >1.5–2.0 | 0.9272 | 0.5679 | 0.9093 | 0.0483 |
| 3UCE | >1.5–2.0 | 0.9696 | 0.5582 | 0.9058 | 0.1054 |
| 8CIH | >1.5–2.0 | 1.1649 | 0.7108 | 1.0529 | 0.0537 |

Five excluded frozen structures were 3Q46, 7Q0O, 6STL, 4KWD and 2GSV. Arm-dependent filtered reflection support, rather than failure of raw flag preservation, violated the comparison. Nine bond arms were recovered after a disclosed printed-target rounding correction. Earlier summaries were retained. The full values table also gives N–Cα, τ, proxy counts, direct-copy and zero-response matrix baselines. Average matrix MAE was 0.0534°, compared with 0.1702° for direct copying and 0.5467° for zero response; matrix prediction beat direct copying in 18/19 structures.

## Table S3 Archive specificity and reference sensitivity

For the primary bandwidth-700 adjusted build comparison, the estimate was +0.2732 with sequence-group interval 0.1865–0.3749. The bandwidth-400 and bandwidth-1100 adjusted estimates were +0.1277 and +0.3627. Resolution interaction was inconclusive. The short-window event study used 2011–2017 and reference year 2014; its joint pre-lead diagnostic was p≈0.285. This is not a test establishing equivalent pretrends.

The table below reproduces point estimates and jointly refitted reference sensitivity intervals. These intervals use positive exponential sequence-group weights, 1,000 draws, and fixed support masks. They are not substituted for the original frequentist intervals. The before-Gly comparison is non-null and is retained as a specificity result. Both TRANS and PTRANS targets changed.

| Context | Term | Contrast | Estimate | Joint reference interval |
| --- | --- | --- | --- | --- |
| before pro | CaCN | new minus old | 0.4716 | 0.4079 to 0.5338 |
| before pro | CaCN | late minus old | 1.1966 | 1.1211 to 1.2712 |
| before pro | CaCN | DiD | 0.2970 | 0.1795 to 0.4070 |
| before pro | O_C_N_next | new minus old | -1.9083 | -1.9785 to -1.8289 |
| before pro | O_C_N_next | late minus old | -1.6613 | -1.7407 to -1.5766 |
| before pro | O_C_N_next | DiD | -1.4797 | -1.6082 to -1.3446 |
| before pro | C_N_next | new minus old | 0.0049 | 0.0042 to 0.0055 |
| before pro | C_N_next | late minus old | 0.0003 | -0.0004 to 0.0010 |
| before pro | C_N_next | DiD | 0.0044 | 0.0033 to 0.0058 |
| before pro | CA_C_O | new minus old | 1.4396 | 1.3518 to 1.5181 |
| before pro | CA_C_O | late minus old | 0.4655 | 0.3937 to 0.5371 |
| before pro | CA_C_O | DiD | 1.1856 | 1.0606 to 1.3073 |
| before gly | CaCN | new minus old | 0.0508 | 0.0004 to 0.0979 |
| before gly | CaCN | late minus old | -0.0508 | -0.1142 to 0.0064 |
| before gly | CaCN | DiD | 0.0400 | -0.0559 to 0.1314 |
| before gly | O_C_N_next | new minus old | 0.1967 | 0.1568 to 0.2360 |
| before gly | O_C_N_next | late minus old | 0.0182 | -0.0384 to 0.0726 |
| before gly | O_C_N_next | DiD | 0.1818 | 0.1210 to 0.2546 |
| before gly | C_N_next | new minus old | 0.0013 | 0.0010 to 0.0017 |
| before gly | C_N_next | late minus old | -0.0006 | -0.0011 to -0.0001 |
| before gly | C_N_next | DiD | 0.0013 | 0.0006 to 0.0020 |
| before gly | CA_C_O | new minus old | -0.2525 | -0.2996 to -0.2093 |
| before gly | CA_C_O | late minus old | 0.0264 | -0.0328 to 0.0886 |
| before gly | CA_C_O | DiD | -0.2176 | -0.2945 to -0.1416 |

Dictionary targets in the audited snapshots changed for TRANS ACN 116.2→115.917°, OCN 123.0→123.469°, C–N 1.329→1.337 Å; PTRANS ACN 116.9→118.415°, OCN 123.0→121.016°, C–N 1.341→1.352 Å. The corresponding differences of target changes are +1.798°, −2.453° and +0.003 Å. They are not scalar transmission denominators for the observational deposit contrasts. Exact immutable dictionary-history links must be inserted from the audit record.

Figure 2 uses the grouped build and short-window event estimates above together with fixed-reference Bayesian group-weight context-sensitivity intervals. The previously reported twenty scrambled-table RDD estimates ranged from −0.0809 to +0.1051; the finite one-sided rank was 1/21. The reconstruction does not plot raw build profiles, calendar means or scrambled distributions because the corresponding current source series are not included in the compact grouped audit.

## Table S4 Corrected context prediction gains

All gains are percentage points of baseline MSE: own conformation is M0−M1, neighbors M1−M2, and second neighbors M2−M3, each divided by M0. Small negative additions are retained. Nearest-neighbor additions combine identity and torsion context.

| Term | Own φ/ψ % | Neighbors % | Second neighbors % |
| --- | --- | --- | --- |
| tau_deg | 42.2326 | 1.5370 | 0.0563 |
| angle_N_CA_CB | 5.6508 | 0.6996 | 0.1208 |
| angle_C_CA_CB | 13.4079 | 0.6522 | 0.0416 |
| angle_CaCN | 25.9775 | 9.4435 | 0.0718 |
| angle_CNCa | 22.9073 | 1.1914 | 0.1809 |
| angle_CA_C_O | 21.0600 | 2.0236 | 0.0322 |
| bond_N_CA | 2.8608 | 0.5397 | -0.0056 |
| bond_CA_C | 6.6038 | 0.5034 | 0.0090 |
| bond_C_O | 0.3673 | 0.1330 | -0.0194 |
| bond_C_N_next | 1.7972 | 5.7127 | 0.0434 |
| bond_CA_CB | 2.9073 | 0.0632 | 0.0253 |

The full summary includes held-out program/resolution strata, cross-program transfer and surface comparisons. It does not repeat the separate older SHELXL or E4 surface-versus-QM analysis.

## Table S5 Clean held-out benchmark contrasts

Clean unsplit denotes matched clean unsplit MechLib. Values below reproduce the plotted contrasts; an MAE-minus-CDL estimate below zero favors the evaluated library. Angular units are degrees. Shared sequence components receive identical weights across strata.

| Term | Model | Stratum or contrast | Estimate | 95% interval |
| --- | --- | --- | --- | --- |
| CaCN | Clean unsplit | PHENIX post | 0.2074 | 0.1890 to 0.2247 |
| CaCN | Clean unsplit | REFMAC | 0.0808 | 0.0604 to 0.1022 |
| CaCN | Clean unsplit | Post minus REFMAC | 0.1266 | 0.0986 to 0.1534 |
| CaCN | Clean next-Pro | PHENIX post | 0.1599 | 0.1438 to 0.1747 |
| CaCN | Clean next-Pro | REFMAC | 0.0257 | 0.0087 to 0.0426 |
| CaCN | Clean next-Pro | Post minus REFMAC | 0.1342 | 0.1109 to 0.1571 |
| O_C_N_next | Clean unsplit | PHENIX post | 0.1499 | 0.1398 to 0.1600 |
| O_C_N_next | Clean unsplit | REFMAC | -0.0174 | -0.0378 to 0.0044 |
| O_C_N_next | Clean unsplit | Post minus REFMAC | 0.1673 | 0.1437 to 0.1907 |
| O_C_N_next | Clean next-Pro | PHENIX post | 0.1305 | 0.1208 to 0.1404 |
| O_C_N_next | Clean next-Pro | REFMAC | -0.0619 | -0.0790 to -0.0446 |
| O_C_N_next | Clean next-Pro | Post minus REFMAC | 0.1924 | 0.1726 to 0.2132 |

Before-Pro ACN MAE was 2.3301° for clean unsplit, 1.2649° for clean next-Pro and 0.6628° for CDL, using 1,087 structures in 383 components. These are deposited-geometry errors and do not establish physical ranking. Full raw predictions were scored without a deployment angle-closure projection.

## Table S6 Quantum-chemical referee

Registered rule MechLib-QM-Q1-v1 on 80 frozen constrained geometries (ACE–X–NME). Weighted difference = mean over terms of √k|Δ| for AMBER minus the library; positive favours the library. Residue-clustered 95% intervals (10,000 draws, seed 20260915). All three libraries failed criterion 3 on the same Holm-significant worse bonds: N–Cα, C=O and C–N(next).

| Library | Mean weighted difference | 95% interval | Residues favourable | Registered rule |
| --- | --- | --- | --- | --- |
| Earlier pre-correction MechLib (blind pre-2020) | +0.0093 | 0.0026 to 0.0167 | 14/20 | Fail (3 bonds) |
| MechLib without χ1 | +0.0049 | −0.0019 to 0.0123 | 13/20 | Fail (interval, 3 bonds) |
| MechLib (χ1 for Cβ angles) | +0.0071 | 0.0003 to 0.0148 | 14/20 | Fail (3 bonds) |

MAE of MechLib against fixed AMBER values (angles in degrees, bonds in Å).

| Term | MechLib | AMBER |
| --- | --- | --- |
| τ (N–Cα–C) | 1.10 | 1.82 |
| N–Cα–Cβ | 1.65 | 2.28 |
| C–Cα–Cβ | 0.98 | 1.18 |
| Cα–C–N | 1.40 | 1.64 |
| Cα–C–O | 0.54 | 0.69 |
| C–N–Cα | 1.33 | 1.46 |
| Cα–Cβ | 0.0051 | 0.0101 |
| Cα–C | 0.0122 | 0.0144 |
| C–N(next) | 0.0257 | 0.0208 |
| C=O | 0.0166 | 0.0121 |
| N–Cα | 0.0123 | 0.0056 |

Signed mean offsets of the failing bonds: C–N(next) −0.026 Å, C=O +0.017 Å, N–Cα +0.012 Å. χ1 conditioning lowered C–Cα–Cβ MAE from 1.117° to 0.982° and N–Cα–Cβ from 1.691° to 1.649°. The earlier library had a lower τ MAE (0.88°) than MechLib (1.10°). No CDL column is yet scored. Weighted geometry errors are not potential-energy differences.

E6 (ACE–Ala–Ala–NME, ωB97X-D/def2-SVP, vacuum, central ω fixed; paired audit passed): changing φ₃ from −90° to −45° changed the preceding Cα–C–N by +0.269°, O–C–N by −0.38°, C–N–Cα by −0.38°, Cα–C–O by −0.01° and τ₂ by −0.95°. One molecule, one method, no replicate starts; not a validation of any library.

## Table S7 Home-advantage, rotamer and training-mix contrasts

Deposited-geometry endpoints. Errors averaged within structures, then sequence components; 95% intervals from 10,000 component resamples; Holm adjustment within each family. Degrees.

| Analysis | Contrast | Cα–C–N | O–C–N |
| --- | --- | --- | --- |
| C1 (primary) | CDL − CCP4 2018; PHENIX post-default vs REFMAC post-revision | −0.237 (−0.261 to −0.214) | −0.694 (−0.723 to −0.665) |
| C1 centred | same, pattern only | −0.194 | −0.156 |
| C1 resolution-reweighted | same | −0.238 | −0.701 |
| C2 (primary) | CCP4 2018 − CCP4 earlier; REFMAC post- vs pre-revision | −0.003 (−0.009 to 0.004), not resolved | −0.207 (−0.223 to −0.192) |
| C1 secondary | CDL − CCP4 2018; PHENIX post- vs pre-default | −0.743 (−0.761 to −0.725) | −0.421 (−0.436 to −0.406) |
| C3 | CDL − MechLib; held-out B30 PHENIX post-default vs REFMAC | −0.134 (−0.157 to −0.110) | −0.192 (−0.213 to −0.172) |

Group-weighted MAE, CDL vs CCP4 2018: Cα–C–N 0.450 vs 1.281 (PHENIX post-default), 0.917 vs 1.511 (REFMAC post-revision); O–C–N 0.331 vs 0.954 and 1.029 vs 0.959. C1 Cα–C–O −0.236 and τ −0.979 (descriptive). C2 is not interpreted for Cα–C–O, because the earlier monomer value describes a free carboxylate.

| Rotamer conditioning (held-out B30) | τ | N–Cα–Cβ | C–Cα–Cβ |
| --- | --- | --- | --- |
| With minus without χ1, REFMAC (primary) | +0.0048 (0.0019 to 0.0075) | −0.036 (−0.041 to −0.031) | −0.023 (−0.029 to −0.017) |
| With minus without χ1, PHENIX post-default | +0.0084 | −0.014 | −0.018 |
| MechLib − CDL, REFMAC | +0.022 (0.011 to 0.033) | −0.106 (−0.119 to −0.094) | +0.084 (0.068 to 0.100) |
| MechLib − CDL, PHENIX post-default | +0.047 | +0.058 | −0.035 |

| Training mix (held-out B30), fit minus natural PHENIX + REFMAC | Cα–C–N | O–C–N |
| --- | --- | --- |
| PHENIX-only, on PHENIX deposits | −0.024 | −0.052 |
| PHENIX-only, on REFMAC deposits | +0.005 | +0.037 |
| REFMAC-only, on REFMAC deposits | +0.016 | −0.005 |
| REFMAC-only, on PHENIX deposits | +0.033 | +0.048 |
| Balanced 50/50, on REFMAC deposits | −0.003 | +0.005 |
| Balanced 50/50, on PHENIX deposits | −0.010 | −0.016 |
| Surface shift, REFMAC-only vs PHENIX-only: mean (95th percentile) | 0.22 (0.61) | 0.21 (0.51) |
| Surface shift, balanced vs natural: mean (95th percentile) | 0.04 (0.12) | 0.04 (0.10) |

## Table S8 Numerical-development diagnostics and open items

| Item | Current evidence | Reporting rule |
| --- | --- | --- |
| Synthetic response | Native preparation passed; four-pass and six-pass stability gates stopped the pilots | Require stable, audited endpoints before response or recovery claims |
| Precision diagnostic | PDB rounding inflated angle changes; one CIF restart still moved substantially | Preserve both pilot outcomes; no precision-only rescue claim |
| Prior-swap prediction | Analysis frozen before the unshifted outcome; required arms incomplete | Use only isolated edits to fit; retain held-out status until complete |
| CDL QM comparator | Specified post hoc on the 10 terms CDL defines; needs the CDL table | Report beside MechLib; cannot change the registered result |
| E5 solvent | Exploratory numerical/environmental pilot | No equilibrium or general solvent correction claimed |
| E4 crystal surface versus QM | Source reconciliation incomplete | No positive result used as a manuscript premise |
| Figure 2 render | Reconstructed from checked grouped/sensitivity summaries | Retain source values and interval labels; figure-owner review remains appropriate |

Native synthetic preparation checked 77 carbonyl triplets and 231 edited angles. The original pilot stopped at e0p after four passes; a documented development amendment allowed up to six passes for every arm and stopped at e1m. Existing passing endpoints and prior passes were reused. Retrospective CIF scoring satisfied the operational angle/coordinate criteria at e1m passes 4 and 5, but did not change the recorded pilot verdict. The additional CIF restart had coordinate RMS movement 0.05867 Å, compared with 0.05968 Å for the existing PDB branch from the same CIF reference; both exceeded the 0.001 Å limit. The median/p95 angle limits remained 0.02°/0.05°, with two consecutive qualifying changes required. All failures and outcome exposure were retained. These are numerical-development diagnostics, not a completed prediction or recovery benchmark.

These entries are reported as incomplete rather than filled by inference.


## Table S9 B30 metadata definition and source identity

The B30 flag is attached to junction i→i+1 but is centered on residue i. It is the arithmetic mean of selected Cα B factors over up to seven entries of the same chain's eligible-residue list: the center and up to three entries on either side. Lists are ordered by integer label_seq_id within label_asym_id. Ends are clipped. No minimum window size of seven is imposed, and the window is not a fixed raw-sequence interval. The implementation does not add a new segment split when intervening positions have been removed from the eligible list.

Eligible center residues are unambiguous standard amino acids with selected N, Cα and C atoms, peptide links to adjacent integer sequence positions and finite φ/ψ. Model selection uses the first model in the coordinate table. Selected atoms require positive finite occupancy and finite coordinates. The window reconstruction follows the historical occupancy convention: greatest occupancy, then A, then blank, then alphabetical alternate identifier. This convention is used for B-factor metadata only; measured junction geometry remains subject to the conservative blank-altloc extraction.

A missing, nonfinite or negative B factor for any selected Cα in the window yields an unknown flag. Otherwise the window passes when its mean is ≤30 Å² and fails when greater. The five separately exported B fields (Cα_i, C_i, O_i, N_next and Cα_next) do not define the B30 average. The frozen metadata contained 386,705 passes and 232,176 failures, with no unknown flags, across 618,881 junctions before common predictor coverage.

| Source record | SHA-256 |
|---|---|
| Metadata producer pbm_v2_geometry_metadata.py | d56b367511bd74dc2a521f8993a935d5f555119ee7477ad33e5b77b1fe742e87 |
| Produced junction_metadata.tsv; identical locked scorer input | 2e1a820d7e6165d4c838e77d5a999b7205bf71814115065a04e61df055cbfa92 |
| Frozen junction geometry consumed by producer and scorer | 8b3ccb9ee7481288c336bae9aafff8411f207a87c6cb3ecba45bcad30f110cf6 |

The source-code hash equals metadata_summary.json's recorded producer hash; its metadata-output and geometry-input hashes match those in the locked clean-model score summary. Source inspection and bookkeeping fixtures checked window clipping, singleton windows, alternate ranking, missing-B propagation and eligible-list gap spanning. These checks did not rerun coordinate extraction, regenerate the B30 mask or rescore a model. The recovered helper source is included for reproduction; the frozen metadata summary does not separately record that helper's code hash.

## Table S10 Prospective library-swap forecast

Specification SHA-256 92e2b8a4…; predictions frozen before refinement (SHA-256 8b3f6277e3428154236769bb4621c8481c53665e3fdfb6acf70b3692db01f8d3). Edit = MechLib − CDL carbonyl-angle targets; σ unchanged. 19/19 structures passed audits; realised edit equalled the planned edit at 5,939 of 5,940 sites. D = MAE(target copying) − MAE(site matrix); positive favours the matrix. Structure resampling within bands, 10,000 draws, seed 20261113.

| Quantity | Value |
| --- | --- |
| Mean D | +0.041° (0.028 to 0.053) |
| Structures with matrix better | 16/19 (copy better: 3UCE −0.038, 9LTB −0.009, 8KIH −0.003) |
| Mean MAE: site matrix / structure-mean matrix / copy / zero | 0.064 / 0.065 / 0.105 / 0.292° |
| Mean planned edit, Cα–C–N / Cα–C–O / O–C–N | −0.232 / −0.053 / +0.301° |
| Mean predicted response | −0.198 / −0.056 / +0.253° |
| Mean observed response | −0.207 / −0.058 / +0.265° |
| Angle sum, targeted / observed | +0.016 / −0.0001° |
| Mean D by band, ≤1.0 / 1.0–1.5 / 1.5–2.0 Å | +0.115 (6/6) / +0.009 / +0.003° |
| Small edits (largest edit < 1°, 95% of sites): matrix / copy | 0.060 / 0.098° |
| Large edits: matrix / copy | 0.164 / 0.231° |
| Largest R-free change | 0.0013 |

## Figure S1 REFMAC dictionary era and the extraction correction

A, REFMAC deposits by version group: τ, Cα–C–N and Cα–C–O alignment with CDL; within the 2018 dictionary era the τ pattern fell by a third while Cα–C–N and Cα–C–O rose. B, PHENIX switch estimates before and after correcting the mixed-conformer extraction (24 term × bandwidth estimates, resolution-adjusted); no estimate moved by more than 0.07.
