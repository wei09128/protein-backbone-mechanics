# Atomic-resolution referee: specification v1.0 (16 Nov 2026, Claude; frozen with its hash in the kit; run only after v4-lite is frozen, so that it judges v2, v3 and v4 blind)

## Question
In protein crystals, how strong is the phi/psi pattern of the three carbonyl angles (Ca-C-N, O-C-N, Ca-C-O) in the data, compared with v2 (deposit average), v3 (de-imprinted), v4-lite (if frozen in time) and CDL? This is the in-protein referee. QM tests intrinsic local geometry, so if (a) the environment explanation holds, these two referees can disagree.

## The logic that makes atomic-resolution deposits a referee
A flat library (one target for every phi/psi: CCP4/REFMAC, SHELXL, pre-default PHENIX) can only shrink a phi/psi pattern; it cannot create or enlarge one. At <= 1.0 A the data dominate and transmission is lowest (controlled Ca-C-N 0.29). Flat-library deposits at <= 1.0 A therefore give a **lower bound** on the data pattern in protein crystals, which does not depend on any transmission estimate. CDL-refined deposits (PHENIX post-default) are reported separately; they can be pulled toward CDL and are not used for the primary endpoint.

## Data
- Junctions: corrected archive features (blank-altloc, sha e905f8b8...), with the v2/v3 eligibility rules (trans peptide, omega within 30 deg of 180, standard residues, next residue known) and the de-imprint outlier guard.
- Structures: resolution <= 1.0 A from the frozen headers (sha 25f466e2...).
- **Overlap with training (checked 16 Nov from headers only, no angles read):** of the <= 1.0 A flat-library structures, about 203 are v2/v3 training structures and 21 are held out by sequence group (REFMAC 16, SHELX 4, PHENIX pre 1). A held-out-only referee would be underpowered. The primary analysis therefore uses all <= 1.0 A structures. This is defensible because the endpoint is model-free: the lower-bound argument compares the in-crystal data amplitude with each library's amplitude and does not test how a library predicts unseen structures. These structures are a few percent of the 5,926 training structures, so their influence on v2's and v3's amplitudes is small; the exact row share is reported. **Sensitivity:** the 21 held-out structures alone (descriptive).
- **Primary stratum (flat libraries):** family REFMAC, SHELXL, or PHENIX with build < 2155. **Secondary:** PHENIX post-default (CDL) and other programs, each reported separately.
- Minimum: >= 100 structures in >= 50 sequence groups in the primary stratum (expected about 220 from headers). Otherwise the referee is reported as underpowered and decides nothing.

## Endpoint
For each angle, the deposited value and each library's prediction at the junction's phi/psi (periodic Catmull-Rom; next-Pro split; CDL by its groups) are centred within (stratum, residue, next-Pro). The amplitude A is the slope of the centred deposited values on the centred CDL prediction, through the origin, in CDL units. Each library's amplitude is computed the same way, as the slope of its centred prediction on CDL's, over the same junctions; ledger V4's cell-based values (v2 0.59 / 0.57 / 0.55, v3 0.95 / 0.85 / 0.95) are background only. Centring units are kept fixed in the bootstrap. Agreement r is reported alongside. 95% interval: 10,000 sequence-group bootstrap draws, seed 20261118.

## Decision rules (per angle; Holm over the three angles for the "too weak" tests)
- **v2 too weak for crystal geometry:** the lower 95% bound of A_flat exceeds v2's amplitude. Because A_flat is a lower bound, this refutes v2's amplitude as the in-crystal data pattern without any model.
- **Consistent with v3:** the interval of A_flat includes or exceeds 0.9 x v3's amplitude.
- **v3 too strong:** cannot be shown from a lower bound alone. Secondary only: the transmission-corrected A_flat / (1 - T), with the <= 1.5 A band T from the de-imprint run (Ca-C-N 0.31, O-C-N 0.30, Ca-C-O 0.34). Model-dependent; reported, never decisive.
- **v4-lite:** scored only if frozen before this referee is run; same rules with its own amplitude.
- **Pattern shape:** r(deposit, library) for each library; a library whose r is lower than another's by more than 0.05 (bootstrap-paired) is reported as "worse shape".

## Disclosed prior exposure
The 2026-11 E1 analysis (ledger P1b) already reported that an atomic-resolution tier (entry-level, all programs mixed, not held out) has 1.5-1.7x the amplitude of REFMAC-trained surfaces. This referee differs in being stratified by program with flat libraries primary, expressed in CDL units, and decided by rules fixed here. The earlier number is outcome-exposed and is cited only as background.

## Interpretation
- v2 too weak and v3 consistent: the crystal (in-protein) data support v3's strength.
- If QM supports v2-like strength at the same time: the conformation-dependent geometry differs between isolated residues and proteins (explanation a). This is reported as an environment-dependent result, not as a contradiction.
- Neither v2 too weak nor v3 consistent: crystal data do not support restoring the pattern, which points to over-correction (explanation b).
