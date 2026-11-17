# PBM_QM_PILOT_v1: QM noise-floor pilot

How much do constrained-QM backbone angles and bonds move when the method changes? 12 registered cases (ALA, GLY, SER x 4 regions), each re-optimised from its frozen Q5 geometry with 4 arms:
R0 wB97X-D/def2-SVP gas (reproduction), S1 + implicit water, B1 def2-TZVP, F1 another functional/def2-TZVP (wB97M-V, else M06-2X, else PBE0; Amendment 1).
No library is read. See QM_PILOT_SPEC.md.

    cd /mnt/c/Wei/PBM && unzip -o ~/Downloads/PBM_QM_PILOT_v1.zip
    bash PBM_QM_PILOT_v1/run.sh selftest
    bash PBM_QM_PILOT_v1/run.sh preflight     # finds work_q5, checks it against Q6, tests each arm on water
    bash PBM_QM_PILOT_v1/run.sh run           # background, nice 19, 2 workers x 4 threads
    bash PBM_QM_PILOT_v1/run.sh status

If preflight cannot find the Q5 output: `bash PBM_QM_PILOT_v1/run.sh preflight --q5-root /path/to/work_q5`.
If the conda env is elsewhere: `QM_ENV=/path/to/mechlib_qm_clean bash ...`.
When done: `tar czf ~/pbm_qm_pilot_results.tgz -C ~ pbm_qm_pilot --exclude='*.tmp'` and upload.
