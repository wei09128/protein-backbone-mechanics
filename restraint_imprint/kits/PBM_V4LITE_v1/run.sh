#!/usr/bin/env bash
# Build and freeze v4-lite. Run only when the pilot water arm (S1) has 12 cases accounted for. Any numpy/pandas env.
set -eo pipefail
kit=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); root=/mnt/c/Wei/PBM
(cd "$kit" && sha256sum -c --quiet SHA256SUMS.txt) || { echo "kit changed; stop"; exit 1; }
out=$root/pbm_v4lite.$(date +%Y%m%d_%H%M)
python "$kit/build_v4lite.py" --pilot ~/pbm_qm_pilot --v2 $root/PBM_CLEAN_REBUILD_v1/results/training_targets.npz \
  --v3dir $root/pbm_mechlib_v3.s37m8h --out "$out"
cp -a "$kit" "$out/kit"; tar -czf "$out.tar.gz" -C "$root" "$(basename "$out")"; echo "Return: $out.tar.gz"
