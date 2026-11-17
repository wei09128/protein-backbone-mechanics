# PBM MD referee M4 (v1.1.0: amendment A2, production 100 ns), 13 Nov 2026
Reuses the M3 GB3/ubiquitin AMBER_FIXED systems; adds MechLib v2 now (next-Pro split, merged forces) and v3 later; 8 paired seeds x 150 ns. NMR is not scored until all three arms are complete. See M4_PROTOCOL.md.

Use the same conda environment as M3 (OpenMM with CUDA, pdbfixer, mdtraj). Keep this folder where you unzip it: the lock records the runner's hash.
```bash
cd /mnt/c/Wei/PBM
K=PBM_MD_M4_v1
(cd $K && sha256sum -c SHA256SUMS.txt)
python $K/pbm_md_m4.py preflight
python $K/pbm_md_m4.py build            # ~1-2 min; prints BUILD PASS
python $K/pbm_md_m4.py bench            # ~5 min; real ns/day for each arm
python $K/pbm_md_m4.py lock --production-ns 150 --timestep-fs 4
for i in 0 1 2 3 4 5; do nohup python $K/pbm_md_m4.py run --shard $i --nshards 6 > ~/pbm_md_m4/worker_$i.log 2>&1 & done
python $K/pbm_md_m4.py status           # any time
```
Later, when v3 is frozen:  python $K/pbm_md_m4.py add-v3 --v3-targets FILE.npz --v3-sha256 HASH  and start the workers again (finished lineages are skipped).
Do not run `wsl --shutdown` or reboot while workers run (they resume from checkpoints, but the current chunk is lost).

## v1.1 (amendment A2): shorten production to 100 ns
When every AMBER lineage shows at least 10/15 in `status`: stop the workers, copy the v1.1 files over the v1 folder, apply A2, restart with 8 workers.
```bash
pkill -f PBM_MD_M4_v1/pbm_md_m4.py; sleep 5
cp PBM_MD_M4_v1_1/pbm_md_m4.py PBM_MD_M4_v1_1/M4_PROTOCOL.md PBM_MD_M4_v1_1/README.md PBM_MD_M4_v1_1/SHA256SUMS.txt PBM_MD_M4_v1/
~/miniconda3/envs/pbm_md_gpu/bin/python PBM_MD_M4_v1/pbm_md_m4.py amend-length --production-ns 100 --confirm-workers-stopped
for i in 0 1 2 3 4 5 6 7; do nohup ~/miniconda3/envs/pbm_md_gpu/bin/python PBM_MD_M4_v1/pbm_md_m4.py run --shard $i --nshards 8 > ~/pbm_md_m4/worker8_$i.log 2>&1 & done
```
