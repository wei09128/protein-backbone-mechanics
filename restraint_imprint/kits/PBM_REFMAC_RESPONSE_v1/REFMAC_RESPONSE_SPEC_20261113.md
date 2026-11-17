# Controlled restraint response in REFMAC: specification v1.0 (frozen before any REFMAC refinement)
13 Nov 2026, Claude, agreed with Wei. Run with PBM_REFMAC_RESPONSE_v1 (checks this file's SHA256 first).

## Question
Is the coupled response measured in PHENIX (Figure 1) a property of restrained refinement in general? Specifically, in REFMAC: (1) does an angle-sum-preserving edit pass more fully than an isolated edit; (2) does refinement remove any change to the sum of the three carbonyl angles; (3) how large is REFMAC's transmission compared with PHENIX's on the same structures (the de-imprint analysis predicts lower transmission in REFMAC deposits).

## Structures (rule fixed now)
From the 19 complete structures of Figure 1 (same model.pdb and common.mtz, hash-checked against the frozen case list), in each resolution band take the first two by PDB identifier in alphabetical order that pass the REFMAC stock gate: ≤1.0 Å 2WFI, 4JP6; 1.0–1.5 Å 1Y8A, 2OPC; 1.5–2.0 Å 1MG7, 3B4Q. A structure that fails the stock gate (REFMAC stops, or stock R-free differs from the deposited-model R-free reported by REFMAC at cycle 0 by more than 0.02) is replaced by the next identifier in the same band; failures are reported, not hidden.

## Refinement protocol (identical across arms of a structure)
REFMAC5 from the installed CCP4 (version recorded), restrained refinement of coordinates and isotropic B factors, 10 cycles, no hydrogens written, no TLS, deposited free set (test value detected from the flags: the rarer value when flags are 0/1, otherwise 0). Weight: the stock arm runs with automatic weighting; its final matrix weight is then fixed for every arm of that structure, stock included (stock is rerun at the fixed weight; that run is the reference). If REFMAC rejects a keyword, one amendment is allowed before any edited arm is run, and it is recorded.

## Edits
A user dictionary (LIBIN) redefines the installed TRANS and PTRANS peptide links, copied from the installed monomer library, with angle targets shifted uniformly (all junctions):
- ACN± : Cα–C–N ±1°
- OCN± : O–C–N ±1°
- COMP± : Cα–C–N ±1° with O–C–N ∓1°
Cα–C–O targets are not edited (its response is measured).
Gate: in the first structure run, the isolated Cα–C–N response must exceed 0.10 per degree; otherwise the link override is not active and the run stops.

## Measurement
Junctions with all contributing atoms at blank alternate location, consecutive residues (C–N ≤ 1.8 Å), trans peptide (|ω| > 150°), standard residues, present in every arm. Response per degree r = [q(+) − q(−)]/2 for Cα–C–N, Cα–C–O, O–C–N; structure mean over junctions.

## Endpoints (reported whatever the outcome; 6 structures, so descriptive with structure-bootstrap intervals, 10,000 draws, seed 20261117)
- E1 compensated minus isolated Cα–C–N transmission, per structure and mean (PHENIX: positive in every structure).
- E2 angle-sum response (r_ACN + r_ACO + r_OCN) for each edit type; "conserved" if every structure mean is within ±0.02 per degree.
- E3 REFMAC minus PHENIX transmission on the same structures (isolated Cα–C–N and compensated), paired; the de-imprint prediction is negative.
- E4 the 3×3 response columns for the two edited angles, compared with PHENIX's mean matrix.

## What it will and will not show
Will show whether coupling and angle-sum conservation hold in a second refinement program, and how REFMAC's transmission compares with PHENIX's under fixed weights. Will not reproduce depositors' protocols, and edits are to link targets (all junctions), not CDL rows.

## Amendment 1 (13 Nov 2026, before any main-run refinement)
The gate (2WFI, 0.75 Å) returned an isolated Cα–C–N response of 0.023 per degree (160 junctions; REFMAC's automatic weight 9.97, final r.m.s. angle deviation 2.2°), below the 0.10 threshold, which had assumed PHENIX-like transmission. The override was nevertheless active: the response showed the coupling signature (Cα–C–O −0.012, O–C–N −0.011 per degree) with the angle sum conserved (+0.0001), which cannot occur unless the edited target is used; REFMAC reported reading the user dictionary. The gate criterion is therefore replaced by: isolated Cα–C–N response > 0.005 per degree with both Cα–C–O and O–C–N responses < 0 (the coupling signature), so that it tests the override rather than the size of the transmission. Everything else is unchanged. Disclosed: the 2WFI isolated-edit response was seen before the main run; 2WFI is rerun in the main run under the same protocol.
