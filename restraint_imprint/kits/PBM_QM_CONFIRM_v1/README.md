# PBM_QM_CONFIRM_v1: confirmatory QM referee, molecule B (spec QM_CONFIRMATORY_SPEC_v1_0.md, sha 31edca32)
Frozen spec and frozen points (inputs/QM_POINTS_v1_0.tsv, sha b8e2cedd) are checked by the code before anything runs.

    cd /mnt/c/Wei/PBM && unzip -o /mnt/c/Users/*/Downloads/PBM_QM_CONFIRM_v1.zip
    bash PBM_QM_CONFIRM_v1/run.sh selftest
    bash PBM_QM_CONFIRM_v1/run.sh preflight     # tools, ddX settings, chi1 start rotamers, pilot basis rule (prints the rule's decision)
    bash PBM_QM_CONFIRM_v1/run.sh build         # tleap geometries for all 160 points (minutes)
    bash PBM_QM_CONFIRM_v1/run.sh timing        # 4 ALA timing cases (8 if the rule says def2-TZVP); start after the pilot F1 arm finishes
    bash PBM_QM_CONFIRM_v1/run.sh status
    bash PBM_QM_CONFIRM_v1/run.sh design 5      # once timing is complete: frozen feasibility rule -> DESIGN.json (once; 5 = production workers after the MD)
    bash PBM_QM_CONFIRM_v1/run.sh production 5  # after the M4 MD finishes
    bash PBM_QM_CONFIRM_v1/run.sh analyse       # after every design case is accounted for

Environments: QM in mechlib_qm_solv (psi4, geomeTRIC, pyddx, xtb); tleap from the amber env (AMBERHOME). Output ~/pbm_qm_confirm.
Notes: timing cases run beside the MD, so their times are, if anything, slower than production (conservative projection). The ALA points 1-4 at the chosen basis count as production cases. v1.1 adds the secondary comparators (CDL, 2018 CCP4 TRANS, AMBER ff14SB) and all-term absolute errors; the code that runs each case is unchanged from v1.0.2.
