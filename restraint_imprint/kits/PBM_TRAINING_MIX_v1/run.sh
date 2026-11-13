#!/usr/bin/env bash
# Training-mix sensitivity (spec v1.0). Usage: bash PBM_TRAINING_MIX_v1/run.sh [/mnt/c/Wei/PBM]
set -eo pipefail
kit=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=${1:-/mnt/c/Wei/PBM}
(cd "$kit" && sha256sum -c --quiet SHA256SUMS.txt) || { echo "Kit files changed; stop."; exit 1; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
out=$(mktemp -d "$root/pbm_training_mix.XXXXXX")
trap 'cp -a "$kit" "$out/kit"; tar -czf "$out.tar.gz" -C "$(dirname "$out")" "$(basename "$out")"; echo "Return archive: $out.tar.gz"' EXIT
python -u "$kit/training_mix.py" --selftest 2>&1 | tee "$out/SELFTEST.log"
python -u "$kit/training_mix.py" --root "$root" --out "$out/results" 2>&1 | tee "$out/RUN.log"
