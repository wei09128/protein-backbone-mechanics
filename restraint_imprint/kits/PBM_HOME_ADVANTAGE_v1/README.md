# PBM home-advantage test (spec v1.1), 12 Nov 2026

Does each backbone restraint library agree best with deposits refined using that library? Four predictors (CDL v1.2, CCP4 dictionary before and after the 2018 revision, MechLib v2) are scored on deposits grouped by refinement program and version. The result is a set of difference-in-differences ("home advantage"), not a ranking. No refinement, MD or QM is run.

## Run (WSL, the environment that ran the earlier RDD / closing checks; needs numpy, pandas, matplotlib)

```bash
cd /mnt/c/Wei/PBM
bash PBM_HOME_ADVANTAGE_v1/run.sh
```

Upload the printed `pbm_home_advantage.*.tar.gz`. Expected run time: tens of minutes (about 45 bootstrap analyses of 10,000 draws).

The script finds by itself:
- `pbm_altloc_full_rdd_audit_v1/blank_only_geometry.csv.gz` (hash-checked, as in the closing checks);
- the CDL file `cdl_database.py` with the frozen hash (the same file score.py used). If it is not found, give it: `CDL=/path/to/cdl_database.py bash PBM_HOME_ADVANTAGE_v1/run.sh`;
- `structure_mae.tsv` from `PBM_CLEAN_BENCHMARK_RESULTS_v1` (for C3). Override with `BENCH=/path/...`.

The 20 CCP4 monomer files for the two dictionary commits are downloaded once from GitHub (MonomerLibrary/monomers, pinned commits) into `dictionaries/`. The list files and ALA/PRO monomers are included and hash-checked; the link targets parsed from the downloads must equal the values recovered earlier, or the run stops.

## Safeguards built in
- The spec's SHA256 is checked before any geometry is read; every input is hash-checked (no silent substitution).
- A synthetic self-test runs first: a home advantage built into fake data must be recovered, and a null must stay near zero.
- C3 must reproduce the existing CDL-vs-v2 program difference (-0.134 / -0.192) or the run stops.
- All contrasts, nulls and reversals are written; no winner is declared.

## Outputs (in `results/`)
STRATUM_MAE.tsv, STRUCTURE_MAE.tsv, COVERAGE.tsv, CONTRASTS.json, DICTIONARY_TARGETS.tsv, HomeAdvantage.png/pdf, RUN_LOCK.json; plus SELFTEST.log and RUN.log.

## Not claimed
Physical correctness of any library; effects in refinement or simulation. The old CCP4 Ca-C-O target (monomer value) is uncertain as a description of what depositors used; Ca-C-O is secondary.
