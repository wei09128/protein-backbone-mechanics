#!/usr/bin/env bash
# Stage 1: predictions only (no refinement). Usage: bash PBM_SWAP_v1/run_stage1.sh [/mnt/c/Wei/PBM]
set -eo pipefail
kit=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=${1:-/mnt/c/Wei/PBM}
(cd "$kit" && sha256sum -c --quiet SHA256SUMS.txt) || { echo "Kit files changed; stop."; exit 1; }
CDL=${CDL:-$HOME/phenix-2.2.1-6174/lib/python3.11/site-packages/mmtbx/conformation_dependent_library/cdl_database.py}
out="$root/pbm_swap_stage1"
python3 -u "$kit/stage1_predict.py" --batch "$root/pbm_transmission_batch.j1pKgh" --cdl "$CDL" \
  --targets "$root/PBM_CLEAN_REBUILD_v1/results/training_targets.npz" --out "$out" 2>&1 | tee "$root/pbm_swap_stage1.log"
tar -czf "$root/pbm_swap_stage1.tar.gz" -C "$root" pbm_swap_stage1 pbm_swap_stage1.log
echo "Return $root/pbm_swap_stage1.tar.gz and send the SHA256 above BEFORE running stage 2."
