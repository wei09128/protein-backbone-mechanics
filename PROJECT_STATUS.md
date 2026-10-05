---
project: "protein-backbone-mechanics"
bundle_version: "0.1.0"
document_role: "project_status"
document_revision: 1
authority_for:
  - current operational truth
  - workstream state
  - blockers and priorities
  - exact next actions
last_updated_utc: "2026-09-11T00:00:00Z"
artifact_state: "CURRENT"
---

# protein-backbone-mechanics — Project Status

## 0. Resume here

- **Objective:** Get Paper IV (MechLib) resubmitted to JCIM with a genuine, non-circular independent validation addressing the desk-rejection reasons.
- **Active workstream:** `WS-05` — MechLib static-minimization validation.
- **Current truth:** `mechlib_minimization_validation.py` is fully wired against the real repo code (`pdb_loader.py`, `features_collector.py`'s column/term conventions, `backbone_geometry_library.py`'s verified `AMBER_DEFAULTS`/`SPRING_CONSTANTS`) and the real holdout list (2,121 structures, matches the paper's reported 2,122 almost exactly). It is not yet confirmed to run end-to-end on real data — last observed failure was an OpenMM `Vec3`-vs-ndarray positions bug, now patched but **not yet re-run and confirmed** as of this bundle's creation.
- **Immediate task:** `T-001` — re-run the 3-structure smoke test (`n_pilot_structures=3`) and confirm it completes without error.
- **Completion gate:** Smoke test produces `minimization_validation_per_structure.csv` and `minimization_validation_summary.json` with 3 non-error rows.
- **Next decision gate:** If the smoke test passes, scale to the full 300-structure pilot (`R-001`, not yet run). If the RMSD-reduction result is positive and passes the shuffled-table negative control, draft the JCIM rebuttal section citing it.
- **Blocked by:** none currently — smoke-test re-run is unblocked.
- **Do not redo:** the holdout-list fetch (`RUN-20260911-01`, 2,121 structures, cached in `release_dates_cache.csv`) — already correct, matches the paper's split.
- **Last verified checkpoint:** `RUN-20260911-01` — Vec3 positions fix applied, syntax-checked, not yet executed by Wei.
- **Open next:** Explanation §9.5 (workstream chapter for WS-05) once the smoke test passes.

### 0.2 Change frontier

| Proposal ID | Type | State | Core change | Outcome exposure | Affected frozen objects | Next safe action | Governing record |
|---|---|---|---|---|---|---|---|
| `P-001` | `DESIGN` | `EXECUTING` | Add an independent static-minimization RMSD-to-truth test (perturb→minimize→score, 3 arms: AMBER-fixed, MechLib, shuffled-MechLib negative control) to answer JCIM's circularity objection. | `UNSEEN` — no pilot results exist yet | none (new workstream, not modifying existing Paper IV claims) | Run the 3-structure smoke test | `change_records/CHANGE-20260911-01_static_minimization_validation.md` |
| `P-002` | `NEW` | `CAPTURED` | (Discussed, not started) Full 5-replica MD ensemble + multi-medium RDC cross-validation on ubiquitin, matching Salmon et al. 2013 / Shen, Robertson & Bax 2023 design standards. | `NA` | none | Deliberately **not** authorized for the JCIM resubmission — flagged as a standalone follow-up paper candidate, not a rebuttal deliverable, given engineering risk and timeline (see rationale in chat; not yet in a dedicated change record) | none yet — needs its own `CHANGE-*.md` if pursued |

### 0.1 Bundle compatibility

| Document | Required bundle version | Observed version | Compatible? |
|---|---|---|---|
| `INDEX.md` | 0.1.0 | 0.1.0 | yes |
| `PROJECT_STATUS.md` | 0.1.0 | 0.1.0 | yes |
| `PROJECT_COMPLETE_EXPLANATION.md` | 0.1.0 | 0.1.0 | yes |
| `CODE_REGISTRY.md` | 0.1.0 | 0.1.0 | yes |
| `MANUSCRIPT_EVIDENCE_MATRIX.md` | 0.1.0 | 0.1.0 | yes |

## 1. Project contract

| Item | Current definition |
|---|---|
| Core question | Does conformation-dependent (φ,ψ)-binned backbone geometry (MechLib) improve on fixed force-field equilibrium values in a way that is genuinely predictive, not just a better in-sample fit? |
| Central hypothesis | Minimizing a perturbed structure toward MechLib's conformation-dependent equilibria recovers the true deposited geometry better than minimizing toward fixed AMBER ff14SB equilibria, with identical force constants in both arms. |
| Primary estimand | % reduction in backbone-atom RMSD-to-deposited-structure, MechLib arm vs. AMBER-fixed arm, after minimizing from an identically-perturbed start. |
| Primary comparator/null | Fixed AMBER ff14SB reference equilibria (verified, `backbone_geometry_library.AMBER_DEFAULTS`); shuffled-MechLib table as a stronger negative control. |
| Independent uncertainty unit | Structure (PDB entry), bootstrap-resampled at the structure level — matches Methods 2.5's convention. |
| Primary success rule | MechLib arm's RMSD-to-truth significantly lower (95% bootstrap CI excludes zero) than both the AMBER-fixed arm and the shuffled-table negative control. |
| In scope | Protein backbone (5 bonds + 6 angles, the same 11 terms as Methods 2.4), temporally-blind holdout structures (2020–2026), static single-step minimization. |
| Out of scope | DNA extension, full MD/dynamics validation, NMR/RDC validation, cross-force-field comparison for this specific test (all deferred / separate workstreams). |

## 2. Workstream state matrix

| ID | Workstream | Execution state | Evidence state | Artifact state | Access state | Governing result | Immediate task |
|---|---|---|---|---|---|---|---|
| `WS-01` | Paper I | `BLOCKED` | `NOT-ESTABLISHED` (rejected after revision) | `FROZEN` | `OPEN` | none | awaiting decision on resubmission venue — `VERIFY-LOCALLY` with Wei |
| `WS-02` | Paper II | `RUNNING` | `SUPPORTED-INTERNAL` | `DRAFT` | `OPEN` | none | both reviewers require representative MM calculations (Reviewer 1) + additional citations/DFT check (Reviewer 2) |
| `WS-03` | Paper III | `RUNNING` | `NOT-ESTABLISHED` (declined once) | `DRAFT` (v3) | `OPEN` | none | resubmitting v3 to JCIM — `VERIFY-LOCALLY` current draft state |
| `WS-04` | Paper IV / MechLib | `BLOCKED` | `SUPPORTED-INTERNAL` (headline metrics are in-sample) | `FROZEN` (desk-rejected version) | `OPEN` | `R-001` (pending) | blocked on `WS-05` completing before resubmission |
| `WS-05` | Static-minimization validation | `RUNNING` | `HYPOTHESIS` | `DRAFT` | `OPEN` | none yet | `T-001` |

## 3. Immediate task queue

### 3.1 Actionable now

| Priority | Task ID | Action | Owner | Inputs | Expected output | Completion gate |
|---:|---|---|---|---|---|---|
| 1 | `T-001` | Re-run 3-structure smoke test of `mechlib_minimization_validation.py` | Wei (human) | `CODE-004` | per-structure CSV + summary JSON, 3 rows | no unhandled exception; 3 rows written |
| 2 | `T-002` | Fill `get_refinement_software()` stub (reuse existing REMARK 3 parser) | Wei/AI | Methods 2.7 parser (location `VERIFY-LOCALLY`) | working stratification column | pilot output includes non-"UNKNOWN" refinement_software for ≥90% of structures |

### 3.2 Queued after a gate

| Task ID | Action | Prerequisite/gate | Do not start before |
|---|---|---|---|
| `T-003` | Scale to full 300-structure pilot | `T-001` passes | smoke test clean |
| `T-004` | Draft JCIM rebuttal point-by-point response | `T-003` produces a positive, CI-excludes-zero result | pilot result exists |

### 3.3 Blocked

(none currently)

### 3.4 Superseded or abandoned

| Task/object ID | State | What/previous approach | Why no longer used | Replacement | Decision/capsule | Archive | Reopen condition |
|---|---|---|---|---|---|---|---|
| — | `SUPERSEDED` | Earlier revision of `mechlib_minimization_validation.py` used placeholder flat force constants (500 kcal/mol/Å² all bonds, 200 kcal/mol/rad² all angles) instead of real per-term AMBER values | Real verified per-term constants were found in `backbone_geometry_library.py`'s `SPRING_CONSTANTS` and should be used for fidelity to Methods 2.4's cancellation design | current version (uses `SPRING_CONSTANTS_KCAL`) | see `RUN-20260911-01` | chat history / prior script version, not separately archived on disk | none — superseded version has no further use |
| — | `SUPERSEDED` | Earlier revision attributed `angle_CNCa`'s equilibrium lookup to the wrong (off-by-one) residue's (φ,ψ) bin | Cross-checking `features_collector.py`'s actual definition showed `angle_CNCa` is a backward-looking term owned by residue i+1, not i | current version (explicit owner-residue tagging in `_get_backbone_topology`) | see `RUN-20260911-01` | same | none |

### 3.5 Authorized corrections and redos

| Proposal | Authorized task | Correction class | Old object | New object/version | Frozen before protected outcome? | Completion gate |
|---|---|---|---|---|---|---|
| `P-001` | `T-001`–`T-004` | `DESIGN` | Paper IV's static-minimization "feasibility check" (attempted, not carried through, per Methods §4) | `mechlib_minimization_validation.py`, current version | yes — `NA`, no outcome inspected yet | pilot RMSD result with bootstrap CI |

## 4. Live-operation checkpoints

(none currently running)

## 5. Current claims and numerical results

No `WS-05` numerical results exist yet — pilot has not completed a run.

### 5.1 Failed primary outcomes that must remain visible

| Result ID | Prespecified rule | Observed result | Evidence state | What it prohibits |
|---|---|---|---|---|
| — | JCIM required an independent, non-circular validation | Desk rejection 31-Aug-2026: strain-reduction metric judged circular; temporal holdout judged insufficient (shows stability, not predictive value) | `FAILED-PRIMARY` for the *original* submission | Cannot resubmit citing only the original strain-reduction/temporal-holdout evidence as sufficient — must add `WS-05` (or equivalent) evidence |

### 5.2 Completed versus not completed

**Completed:** Paper IV's in-sample strain reduction (38.7% protein, 85.4% DNA), temporal holdout (39.7% vs 37.8% in-sample), cross-force-field comparison (AMBER≈OPLS, CHARMM36 CMAP correlation r=−0.25), refinement-software stratification (REFMAC-only 33.6%). `mechlib_minimization_validation.py` fully wired against real repo code (no remaining stub functions).

**Not completed:** Any actual execution of the static-minimization pilot (smoke test or full run). Full MD/RDC validation (`P-002`, deliberately deferred). `get_refinement_software()` implementation.

## 6. Decision ledger

| Decision ID | Date | Decision | Evidence/rationale | Consequence | Reopen only if |
|---|---|---|---|---|---|
| `D-001` | 2026-09-11 | Prioritize the static-minimization validation (`WS-05`) over the full MD/RDC ensemble design (`P-002`) for the JCIM resubmission | Full MD requires implementing conformation-dependent geometry inside a live dynamics trajectory — exactly the engineering risk Paper IV's own Discussion §4 already declined to take on for this submission; static minimization directly answers the stated circularity objection at much lower risk/timeline cost | `P-002` parked as a standalone follow-up paper candidate, not part of this resubmission | JCIM specifically requests dynamics-level evidence in a future review round |
| `D-002` | 2026-09-11 | Use `backbone_geometry_library.py`'s real `AMBER_DEFAULTS`/`SPRING_CONSTANTS` instead of literature-recalled AMBER values | Methods 2.2's entire point is that secondary-source AMBER values were previously wrong (caught 4 parameter errors, 31.5%→38.7%) — reusing that mistake pattern here would undermine the validation's credibility | `load_amber_reference_equilibria()` sources directly from the verified module | never, unless the verified table itself is revised |

## 7. Artifact and path map

| Artifact ID | Purpose | Absolute path or persistent file | Artifact state | Version/checksum | Generated by | Consumed by |
|---|---|---|---|---|---|---|
| `A-001` | MechLib protein correction table | `/mnt/c/Wei/PBM/protein-backbone-mechanics/library/constants_library.json` | `CURRENT` | `VERIFY-LOCALLY` | `export_mechlib_tables.py` (per repo notes) | `CODE-001`, `CODE-004` |
| `A-002` | Holdout PDB ID list (2020–2026) | `holdout_2020_2026_pdb_ids.txt` (working dir) | `CURRENT` | 2,121 IDs, no duplicates | `fetch_holdout_release_dates.py` | `CODE-004` |
| `A-003` | Release-date cache | `release_dates_cache.csv` (working dir) | `CURRENT` | 8,291/8,292 resolved (6XOK missing) | `fetch_holdout_release_dates.py` (RCSB GraphQL) | `A-002` |
| `A-004` | Protein features table | `/mnt/c/Wei/PBM/features_lj_FINAL_CLEAN.csv` | `CURRENT` | `VERIFY-LOCALLY` | `features_collector.py` | `CODE-004` |
| `A-005` | Cached protein PDB structures | `/mnt/c/Wei/PBM/pdb_cache/` (lowercase ids, e.g. `9ywn.pdb`) | `CURRENT` | `VERIFY-LOCALLY` | `VERIFY-LOCALLY` (likely `pdb_loader.py`'s caller) | `CODE-004` |

### 7.1 Storage policy

- Code root: `/mnt/c/Wei/PBM/protein-backbone-mechanics/` (packaged) — but note working scripts (`mechlib_minimization_validation.py`, `fetch_holdout_release_dates.py`, `features_collector.py`'s dependencies) currently run from `/mnt/c/Wei/PBM/` top level. **This split is a known documentation/organization gap — see `K-001`.**
- Large-data root: `/mnt/c/Wei/PBM/pdb_cache/`, `/mnt/c/Wei/PBM/dna_pdb_cache/`.
- Scratch/temp root: wherever `mechlib_minimization_validation_out/` lands (working dir).
- Environment/startup: `conda activate mechlib_val`.
- Deprecated roots: `/mnt/c/Wei/PBM/old/` — contains superseded copies of multiple scripts (`add_lj_torques.py`, `paper2_features_collector.py`, etc.). Do not import from here.
- Never guess a path. Mark an uncertain path `VERIFY-LOCALLY`.

### 7.2 Acquisition ledger

(No external acquisition currently pending.)

### 7.3 Code and manuscript routing

- Authoritative executable selection: `CODE_REGISTRY.md`.
- Manuscript inclusion and readiness: `MANUSCRIPT_EVIDENCE_MATRIX.md`.

## 8. Risks, assumptions, and abstention

| Risk ID | Assumption/risk | Diagnostic | Failure signature | Response/abstention | Governing section |
|---|---|---|---|---|---|
| `K-001` | Working scripts import bare module names (`pdb_loader`, `hbond_finder`, `molcore`) relying on cwd/sys.path rather than the packaged repo structure | Run from a different working directory and see if imports break | `ModuleNotFoundError` when run from `protein-backbone-mechanics/scripts/` instead of `/mnt/c/Wei/PBM/` | Treat `/mnt/c/Wei/PBM/` top level as the de facto code root for now; resolve the split before archiving/publishing the repo | Explanation §9.5 |
| `K-002` | `load_mechlib_lookup()`'s nearest-bin fallback (used when the exact (φ,ψ) bin is empty for a residue) approximates but does not exactly replicate `GeometryLibrary`'s real fallback chain (residue bin → `'ALL'` pooled bin → `AMBER_DEFAULTS`) | Compare a sample of lookups between the reimplementation and the real `GeometryLibrary` class | Systematic small discrepancies in pilot RMSD results vs. what the real library would produce | Prefer importing `GeometryLibrary` directly over the reimplementation once its module path is confirmed importable from the validation script's environment | §5.2 non-equivalences (Index) |
| `K-003` | The pilot's negative control (shuffled-MechLib table) is a reasonable but not the only possible falsification design | none run yet | shuffled control fails to distinguish MechLib from noise even when MechLib genuinely helps, or vice versa | Interpret pilot result alongside, not instead of, the AMBER-fixed comparison | `PROJECT_STATUS.md` §5 |

## 9. Exact next-session commands or drafting actions

### 9.1 Safe read-only verification

```bash
cd /mnt/c/Wei/PBM
conda activate mechlib_val
cat mechlib_minimization_validation_out/minimization_validation_summary.json 2>/dev/null || echo "not yet run"
```

### 9.2 Next authorized execution

```bash
# with n_pilot_structures temporarily set to 3 in the __main__ block
python mechlib_minimization_validation.py
```

Expected output: `mechlib_minimization_validation_out/minimization_validation_per_structure.csv` (3 rows) and `minimization_validation_summary.json`.
Completion gate: script exits without traceback; JSON contains both `mechlib_vs_amber_reference` and `mechlib_vs_shuffled_control` keys.
Stop/abstain condition: any `res_name` mismatch exception (indicates a real chain/loader misalignment — do not suppress or retry blindly).

## 10. Recent update ledger

### 2026-09-11 — `RUN-20260911-01` — Wire mechlib_minimization_validation.py to real repo code

- **Changed:** Filled all previously-stubbed functions (`load_mechlib_lookup`, `_get_backbone_topology`, `_to_atom_indexed`, `load_deposited_structure`, `load_amber_reference_equilibria`) against the real `pdb_loader.py`, `features_collector.py`, and `backbone_geometry_library.py`. Fixed an `angle_CNCa` off-by-one-residue attribution bug and a flat-placeholder-force-constant issue. Fixed an OpenMM `Vec3`-vs-ndarray positions bug.
- **Evidence:** No numerical results yet — smoke test not yet successfully executed.
- **Interpretation:** none yet.
- **Status-axis changes:** `WS-05` execution state `QUEUED` → `RUNNING`.
- **Next task:** `T-001`.
- **Explanation synchronized:** no — §9.5 (WS-05 chapter) still needs writing once smoke test passes.
- **Index synchronized:** yes (this bundle's creation).

## 11. Documentation synchronization checklist

- [x] Immediate task and completion gate reflect reality.
- [x] Execution, evidence, artifact, and access states are not conflated.
- [x] New claims/results/decisions/tasks/artifacts have stable IDs.
- [x] `P-001` and `P-002` recorded with outcome-exposure classification.
- [ ] `CODE_REGISTRY.md` fully reconciled — several entries still `VERIFY-LOCALLY`.
- [x] Failed primary results remain visible (§5.1).
- [ ] Paths and artifacts were verified rather than remembered — several marked `VERIFY-LOCALLY`, pending Wei's confirmation.
- [x] A run record captures exact execution and outputs (`RUN-20260911-01`).
- [ ] Explanation was updated — §9.5 still pending.

**Human explanation sync:** UPDATE REQUIRED
**Explanation sections changed or still required:** §9.5 (WS-05 workstream chapter), §8 (mathematical framework for the minimization test), §10 (results ledger once pilot runs)
**New human "why?" questions captured:** none this session beyond what's in chat

## 12. Compact generated handoff source

See §0 **Resume here** plus §3 task queue plus §9 exact commands for everything a new chat needs.


## Paper II PISCES submission update

Final analysis: 7,938 chains and 1,636,322 residues. Current article and SI: manuscript/PaperII. Figures and supporting code: figures_and_supplementary_material/PaperII. Relaxation: 376 retained; 390 absent feature-chain pairs and 12 diff_seq excluded. Old Paper II files are backed up outside the repository. Publication and co-author approval are not recorded by this preparation step.
