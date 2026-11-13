#!/usr/bin/env bash
# Home-advantage test (spec v1.1). Usage: bash PBM_HOME_ADVANTAGE_v1/run.sh [/mnt/c/Wei/PBM]
# Optional overrides: CDL=/path/to/cdl_database.py BENCH=/path/to/structure_mae.tsv
set -eo pipefail
kit=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=${1:-/mnt/c/Wei/PBM}
(cd "$kit" && sha256sum -c --quiet SHA256SUMS.txt) || { echo "Kit files changed; stop."; exit 1; }
want_cdl=79c197e5692986cde26db9aa492cecfb5b8b7d5b1f3cba4dc222bb2e04d23c68
if [ -z "$CDL" ]; then
  echo "Looking for the frozen CDL database (cdl_database.py, sha ${want_cdl:0:8}...)"
  while IFS= read -r f; do
    if [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$want_cdl" ]; then CDL=$f; break; fi
  done < <(find "$root" "$HOME" /opt /usr/local -name cdl_database.py 2>/dev/null)
fi
[ -n "$CDL" ] || { echo "No cdl_database.py with the frozen hash found. Set CDL=/path/to/cdl_database.py"; exit 1; }
if [ -z "$BENCH" ]; then
  BENCH=$(find "$root" -iname structure_mae.tsv -ipath '*clean_benchmark_results_v1*' 2>/dev/null | head -1)
fi
[ -n "$BENCH" ] || { echo "No PBM_CLEAN_BENCHMARK_RESULTS_v1 structure_mae.tsv found. Set BENCH=/path/to/structure_mae.tsv"; exit 1; }
echo "CDL:   $CDL"; echo "BENCH: $BENCH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
out=$(mktemp -d "$root/pbm_home_advantage.XXXXXX")
trap 'cp -a "$kit" "$out/kit"; tar -czf "$out.tar.gz" -C "$(dirname "$out")" "$(basename "$out")"; echo "Return archive: $out.tar.gz"' EXIT
python -u "$kit/home_advantage.py" --selftest 2>&1 | grep -v '^  C' | tee "$out/SELFTEST.log"
python -u "$kit/home_advantage.py" --root "$root" --cdl "$CDL" --benchmark "$BENCH" --out "$out/results" 2>&1 | tee "$out/RUN.log"
