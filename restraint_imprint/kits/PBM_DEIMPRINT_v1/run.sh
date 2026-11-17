#!/usr/bin/env bash
# De-imprint calculation (spec v1.0). Usage: bash PBM_DEIMPRINT_v1/run.sh [/mnt/c/Wei/PBM]
# Optional override: CDL=/path/to/cdl_database.py
set -eo pipefail
kit=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=${1:-/mnt/c/Wei/PBM}
(cd "$kit" && sha256sum -c --quiet SHA256SUMS.txt) || { echo "Kit files changed; stop."; exit 1; }
want_cdl=79c197e5692986cde26db9aa492cecfb5b8b7d5b1f3cba4dc222bb2e04d23c68
if [ -z "$CDL" ]; then
  while IFS= read -r f; do
    if [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$want_cdl" ]; then CDL=$f; break; fi
  done < <(find "$root" "$HOME" /opt /usr/local -name cdl_database.py 2>/dev/null)
fi
[ -n "$CDL" ] || { echo "No cdl_database.py with the frozen hash found. Set CDL=/path/to/cdl_database.py"; exit 1; }
echo "CDL: $CDL"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
out=$(mktemp -d "$root/pbm_deimprint.XXXXXX")
trap 'cp -a "$kit" "$out/kit"; tar -czf "$out.tar.gz" -C "$(dirname "$out")" "$(basename "$out")"; echo "Return archive: $out.tar.gz"' EXIT
python -u "$kit/deimprint.py" --selftest 2>&1 | tee "$out/SELFTEST.log"
python -u "$kit/deimprint.py" --root "$root" --cdl "$CDL" --out "$out/results" 2>&1 | tee "$out/RUN.log"
