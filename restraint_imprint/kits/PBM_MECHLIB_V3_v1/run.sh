#!/usr/bin/env bash
# Build MechLib v3 (spec MECHLIB_V3_SPEC.md v1.1). Usage: bash PBM_MECHLIB_V3_v1/run.sh [/mnt/c/Wei/PBM]
# Optional overrides: CDL=/path/to/cdl_database.py  DEIMP=/path/to/pbm_deimprint.5tttMa/results
set -eo pipefail
kit=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=${1:-/mnt/c/Wei/PBM}
(cd "$kit" && sha256sum -c --quiet SHA256SUMS.txt) || { echo "Kit files changed (spec or code); stop."; exit 1; }
want_cdl=79c197e5692986cde26db9aa492cecfb5b8b7d5b1f3cba4dc222bb2e04d23c68
want_par=970aa2c9991edb80a1479a32efcd32f68ac2df8b4959633897630920ad9b2f45
if [ -z "$CDL" ]; then
  while IFS= read -r f; do [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$want_cdl" ] && { CDL=$f; break; }; done \
    < <(find "$root" "$HOME" -maxdepth 6 -name cdl_database.py 2>/dev/null)
fi
if [ -z "$DEIMP" ]; then
  while IFS= read -r f; do [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$want_par" ] && { DEIMP=$(dirname "$f"); break; }; done \
    < <(find "$root" "$HOME" -maxdepth 4 -path '*pbm_deimprint*' -name PARAMETERS.json 2>/dev/null)
fi
[ -n "$CDL" ] || { echo "No cdl_database.py with the frozen hash found; set CDL=/path"; exit 1; }
[ -n "$DEIMP" ] || { echo "No de-imprint PARAMETERS.json (run pbm_deimprint.5tttMa) found; set DEIMP=/path/to/results"; exit 1; }
echo "CDL: $CDL"; echo "De-imprint results: $DEIMP"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
out=$(mktemp -d "$root/pbm_mechlib_v3.XXXXXX"); rmdir "$out"
mkdir -p "$out.log"
trap 'cp -a "$kit" "$out.log/kit"; tar -czf "$out.tar.gz" -C "$(dirname "$out")" "$(basename "$out")" "$(basename "$out").log"; echo "Return archive: $out.tar.gz"' EXIT
nice -n 19 python -u "$kit/build_v3.py" --selftest 2>&1 | tee "$out.log/SELFTEST.log"
nice -n 19 python -u "$kit/build_v3.py" --root "$root" --cdl "$CDL" --deimprint "$DEIMP" --out "$out" 2>&1 | tee "$out.log/RUN.log"
