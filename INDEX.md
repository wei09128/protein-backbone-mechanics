---
project: "protein-backbone-mechanics"
bundle_version: "0.1.0"
document_role: "index"
document_revision: 1
authority_for:
  - document hierarchy
  - read order
  - canonical names and roots
  - status vocabularies
  - task routing
  - change-control routing
last_updated_utc: "2026-09-11T00:00:00Z"
artifact_state: "CURRENT"
---

# protein-backbone-mechanics — Master Index

> This is INDEX.md. Keep it short. It is the router and authority contract, not a status report or scientific textbook.

## 0. Mandatory entry protocol

Before answering a substantive question, interpreting a result, changing code, or running an analysis, an AI must:

1. Read this file and verify that all active documents share the same `bundle_version`.
2. Read `PROJECT_STATUS.md`, beginning with **Resume here**.
3. Read only the relevant sections of `PROJECT_COMPLETE_EXPLANATION.md`.
4. Inspect the governing run record, manifest, source file, or frozen artifact before making an evidence-dependent claim.
5. Consult `CODE_REGISTRY.md` before choosing or rerunning an executable entry point.
6. Verify volatile paths, processes, partial downloads, and external state rather than trusting an old snapshot.
7. Preserve failed primary results, settled decisions, unrelated user changes, and archive files.
8. Resume the first actionable task unless the user gives a different instruction.
9. Synchronize the documents after the work block according to Section 8.
10. Treat a human question that exposes an unexplained equation or mechanism as a documentation defect: answer it, then add the durable explanation to PROJECT_COMPLETE_EXPLANATION.md before closeout.
11. Treat every new idea, correction, or redo as a proposal until its impact is mapped and its execution authority is explicit. Never silently mutate a frozen object because a newer idea sounds better.

## 1. Read order and authority

| Order | Document/artifact | Authority | Editing frequency |
|---:|---|---|---|
| 1 | `INDEX.md` | hierarchy, vocabulary, routing, canonical roots | rare |
| 2 | `PROJECT_STATUS.md` | current operations, blockers, priorities, next actions | frequent |
| 3 | `PROJECT_COMPLETE_EXPLANATION.md` | concepts, symbols, mathematics, scientific interpretation, claim boundaries | when meaning changes |
| 4 | `run_records/RUN-*.md` | exact execution record and evidence-grade conclusion for one work block | one record per meaningful run |
| 5 | frozen outputs + manifests + logs | exact computational evidence | every frozen run |
| optional | `NEW_CHAT_HANDOFF.md` | generated transport snapshot only | conversation boundary |

## 2. Project in one paragraph

The protein-backbone-mechanics series (Papers I–IV) reframes the Ramachandran plot as a mechanical force field: local (φ,ψ) conformations reflect dynamic equilibria among steric, electrostatic, and hydrogen-bonding forces that shift backbone bond lengths and angles away from the fixed values used by standard molecular mechanics force fields. Paper IV packages this as MechLib, a conformation-dependent geometry correction library, reporting 38.7% phantom-strain reduction on ~1.7M protein residues (verified AMBER ff14SB parameters), a temporally blind holdout validation, cross-force-field agreement, and an 85.4% reduction on a DNA backbone extension. JCIM desk-rejected Paper IV (31-Aug-2026) on the grounds that the strain-reduction metric is circular (bin averages fit their own data by construction) and untested against an independent observable, and that Papers I–III (its methodological foundation) remain unpublished. Current work: building a static-minimization validation (perturb→minimize→RMSD-to-truth) as an independent, non-circular test, ahead of resubmission.

## 3. Task router

| Question | Read first | Then inspect |
|---|---|---|
| What should happen next? | `PROJECT_STATUS.md` §§0–3 | active task and live-operation records |
| What is completed, running, blocked, or queued? | `PROJECT_STATUS.md` | relevant run record |
| Which code produces or validates an object? | `CODE_REGISTRY.md` | governing run, code hash/commit, artifact manifest |
| Is a result ready for a manuscript? | `MANUSCRIPT_EVIDENCE_MATRIX.md` | linked claim/result/artifact/run/code |
| Why was an idea/method/result discarded? | Explanation §12 discard capsules | decision and governing run record/archive |
| Should a new idea/correction trigger a redo? | active `CHANGE-*.md` / Status change frontier | impact map, outcome exposure, governing contract |
| What did a run produce exactly? | relevant `RUN-*.md` | manifest, log, and frozen outputs |
| Where is a file or dataset? | Status artifact map | verify the path locally — this repo has known duplicate-file risk, see §5.2 |
| How do I restart in a new chat? | Status **Resume here** | generated handoff only if supplied |

## 4. Workstream map

| Workstream ID | Name | Scientific question | Governing Explanation section | Operational owner/path |
|---|---|---|---|---|
| `WS-01` | Paper I — Cβ proxy / MLP basin model | Does a pre-LJ Cβ proxy predict backbone (φ,ψ)-dependent geometry? | §9.1 | ACS Omega ms `ao-2026-06964g.R1` |
| `WS-02` | Paper II — three-channel Cα angle hierarchy | Does basin→sidechain-class→χ1-rotamer explain Cα bond-angle variance? | §9.2 | `paper2_gd_v2.docx`; both reviewers "major revisions" |
| `WS-03` | Paper III — GAM φ×ψ coupling decomposition | Is protein backbone geometry non-additively coupled across φ and ψ? | §9.3 | ACS Omega ms `ao-2026-06966t`, declined 21-Jul-2026; v3 resubmitting to JCIM |
| `WS-04` | Paper IV — MechLib + DNA extension | Does a conformation-dependent geometry library reduce phantom strain vs. fixed force-field values, genuinely (not circularly)? | §9.4 | JCIM ms `ci-2026-02643y`, desk-rejected 31-Aug-2026; `VERIFY-LOCALLY`: `/mnt/c/Wei/PBM/` |
| `WS-05` | MechLib static-minimization validation | Does minimizing a perturbed structure toward MechLib equilibria recover deposited geometry better than toward fixed AMBER equilibria? | §9.5 | `VERIFY-LOCALLY`: `/mnt/c/Wei/PBM/mechlib_minimization_validation.py` |

Current status and numerical results belong in `PROJECT_STATUS.md`, not here.

## 5. Canonical names, roots, and permanent distinctions

### 5.1 Canonical names and roots

| Item | Canonical value | Deprecated or prohibited alternatives |
|---|---|---|
| Project name | `protein-backbone-mechanics` | — |
| GitHub repo | `github.com/wei09128/protein-backbone-mechanics` | — |
| Packaged library root | `/mnt/c/Wei/PBM/protein-backbone-mechanics/` (`library/`, `scripts/`) | — |
| Working/scratch root | `/mnt/c/Wei/PBM/` (top level) | never treat as durable; contains multiple script versions and an `old/` tree — see §5.2 |
| Environment | conda env `mechlib_val` (WSL, `rnaseq_tools` is a **different**, unrelated env used for RNA-seq work) | do not run MechLib validation work in `rnaseq_tools` |

Large immutable data remain at the declared data root. Never record secrets, credentials, protected identifiers, or sensitive raw data in Markdown.

### 5.2 AI-critical non-equivalences

| Incorrect shortcut | Correct distinction |
|---|---|
| filename/mtime = current code | This repo has confirmed duplicate/superseded copies under `/mnt/c/Wei/PBM/old/` and inconsistent top-level vs. packaged locations (e.g. `pdb_loader.py` exists at repo top level; `features_collector.py`'s canonical copy is under `protein-backbone-mechanics/scripts/` but imports `pdb_loader` by bare module name, meaning either a second copy exists there or the working directory happens to satisfy the import — **VERIFY-LOCALLY, do not assume**). Never select an entry point by filename or directory date alone — resolve via `CODE_REGISTRY.md`. |
| strain-reduction % = non-circular validation | The headline 38.7%/85.4% strain-reduction metrics are **guaranteed positive** whenever geometry varies with conformation (bin averages fit their own training data by construction). They are NOT independent validation. Only the temporally-blind holdout, cross-force-field agreement, and the in-progress static-minimization RMSD-to-truth test carry independent evidentiary weight — and even the temporal holdout was explicitly judged insufficient by the JCIM editor (it shows stability across deposition dates, not physical/predictive value). |
| companion paper "submitted" = citable foundation | Papers I–III are cited in Paper IV as refs 14–16 ("manuscript submitted for publication") but Papers I and III have each been rejected/declined at least once as of this bundle's creation. Do not treat their results as settled external validation without checking `PROJECT_STATUS.md` for current state. |
| MechLib's 11 corrected terms = all force-field-independent | Per `backbone_geometry_library.py`, only tau/N-Cα-Cβ/C-Cα-Cβ have per-force-field-verified baselines (AMBER/CHARMM/OPLS). The other 8 terms (3 angles + 5 bonds) are AMBER-referenced regardless of requested force field — not independently verified for CHARMM/OPLS in the current library release. |
| static minimization test = MD/dynamics validation | The in-progress validation (`WS-05`) is a single-step energy minimization with fixed force constants, not a dynamics trajectory. It does not address vibrational spectra, free-energy landscapes, or NMR/RDC agreement — those remain future work per Paper IV's own Discussion §4. |

## 6. Orthogonal status vocabularies

Never compress these axes into one ambiguous label.

### 6.1 Execution state
`QUEUED`, `RUNNING`, `BLOCKED`, `COMPLETE`, `ARCHIVED`.

### 6.2 Evidence state
`HYPOTHESIS`, `EXPLORATORY`, `SUPPORTED-INTERNAL`, `SUPPORTED-EXTERNAL-COMPONENT`, `VALIDATED-EXTERNAL`, `VALIDATED-PROSPECTIVE`, `FAILED-PRIMARY`, `NOT-ESTABLISHED`, `UNIDENTIFIABLE`.

### 6.3 Artifact state
`DRAFT`, `CURRENT`, `FROZEN`, or `SUPERSEDED`.

### 6.4 Access state
`OPEN`, `CONTROLLED-PENDING`, `CONTROLLED-DEFERRED`, or `UNAVAILABLE`.

## 7. Stable identifiers

| Prefix | Object |
|---|---|
| `WS-###` | workstream |
| `C-###` | scientific claim |
| `R-###` | numerical result |
| `D-###` | durable decision |
| `T-###` | actionable task |
| `A-###` | artifact |
| `K-###` | risk or failure mode |
| `P-###` | proposal, correction, amendment, or redo request |
| `CODE-###` | executable entry point or reproducible pipeline stage |
| `M-###` | manuscript sentence, figure, table, panel, or supplement item |
| `RUN-YYYYMMDD-##` | dated execution record |

## 8. Documentation update contract

See `DOCUMENTATION_WORKFLOW_v2.md` (shipped alongside this bundle) for the full trigger table. In short: update `PROJECT_STATUS.md` after every meaningful command or decision; update `CODE_REGISTRY.md` when an entry point is created/moved/superseded; update `MANUSCRIPT_EVIDENCE_MATRIX.md` when a claim's evidence state changes; update this file only when routing, authority, vocabulary, names, or roots change.

## 9. Conflict and staleness protocol

If active sources disagree: stop substantive execution; identify the conflicting claim/task/path/version/status; resolve from the primary artifact, manifest, or user decision; assign/update the relevant `R-`, `D-`, or `T-` identifier; synchronize affected documents; record the resolution in a run record.

## 10. New-chat startup prompt

> Read `INDEX.md`, verify the active bundle version, then read `PROJECT_STATUS.md` beginning with **Resume here**. Read only the relevant sections of `PROJECT_COMPLETE_EXPLANATION.md` and the governing run record. Verify volatile state and paths before execution — this repo has known path/duplicate-file ambiguity (§5.2). Preserve failed primary outcomes, settled decisions, frozen artifacts, and unrelated user changes. Resume the first actionable task unless instructed otherwise, then synchronize documentation after the work block.

## 11. Activation checklist

- [x] Rename the three master files (done — this bundle).
- [x] Give all three the same `bundle_version` (`0.1.0`).
- [ ] Replace remaining `VERIFY-LOCALLY` placeholders with confirmed paths.
- [x] Create `run_records/` and `change_records/`.
- [x] Create `CODE_REGISTRY.md` with initial known entry points.
- [x] Create `MANUSCRIPT_EVIDENCE_MATRIX.md` for Paper IV.
- [ ] Populate `PROJECT_COMPLETE_EXPLANATION.md` §§8, 10, 11 (mathematical framework, results ledger, failure map) — currently skeletal, needs a dedicated pass.
- [x] Set `T-001` as one executable action (see `PROJECT_STATUS.md`).
- [ ] Verify all canonical roots locally (several are marked `VERIFY-LOCALLY`).


## Paper II PISCES submission update

Final analysis: 7,938 chains and 1,636,322 residues. Current article and SI: manuscript/PaperII. Figures and supporting code: figures_and_supplementary_material/PaperII. Relaxation: 376 retained; 390 absent feature-chain pairs and 12 diff_seq excluded. Old Paper II files are backed up outside the repository. Publication and co-author approval are not recorded by this preparation step.
