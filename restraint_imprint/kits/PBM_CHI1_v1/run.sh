#!/usr/bin/env bash
# chi1 test (spec v1.1). Usage: bash PBM_CHI1_v1/run.sh [/mnt/c/Wei/PBM]   Optional: CDL=/path/cdl_database.py WORKERS=4
set -eo pipefail
kit=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=${1:-/mnt/c/Wei/PBM}
(cd "$kit" && sha256sum -c --quiet SHA256SUMS.txt) || { echo "Kit files changed; stop."; exit 1; }
CDL=${CDL:-$HOME/phenix-2.2.1-6174/lib/python3.11/site-packages/mmtbx/conformation_dependent_library/cdl_database.py}
[ -f "$CDL" ] || { echo "CDL file not found at $CDL. Set CDL=/path/to/cdl_database.py"; exit 1; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
out=$(mktemp -d "$root/pbm_chi1.XXXXXX")
trap 'cp -a "$kit" "$out/kit"; tar -czf "$out.tar.gz" -C "$(dirname "$out")" "$(basename "$out")"; echo "Return archive: $out.tar.gz"' EXIT
cd "$root"
python -u "$kit/chi1_test.py" --selftest 2>&1 | tee "$out/SELFTEST.log"
python -u "$kit/chi1_test.py" --root "$root" --cdl "$CDL" --workers "${WORKERS:-4}" --out "$out/results" 2>&1 | tee "$out/RUN.log"
