# PBM_MECHLIB_V3_v1: build MechLib v3
v2 refitted on training junctions corrected for measured refinement transmission (Ca-C-N, O-C-N, Ca-C-O); every other term copied from v2. See MECHLIB_V3_SPEC.md. Do not run until the spec is approved (its hash is checked).

    cd /mnt/c/Wei/PBM && unzip -o /mnt/c/Users/*/Downloads/PBM_MECHLIB_V3_v1.zip
    conda activate <the numpy/pandas env used for the de-imprint kit>
    bash PBM_MECHLIB_V3_v1/run.sh

Upload the printed pbm_mechlib_v3.*.tar.gz. Outputs: MECHLIB_V3_TARGETS.npz (v2 layout; term__v2 keys hold v3 content, term__v3 aliases), V3_FREEZE.json (its SHA-256), V3_BUILD_REPORT.json, sensitivity and control files (not referee libraries).
Stops if: spec/code/input hashes differ; the T=0 refit does not reproduce v2 exactly; v3 keeps < 50% of v2's rows; amplification > 5.
