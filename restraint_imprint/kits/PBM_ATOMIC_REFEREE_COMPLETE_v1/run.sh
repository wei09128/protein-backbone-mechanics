#!/usr/bin/env bash
# Completeness outputs for the atomic-resolution referee. Usage: bash PBM_ATOMIC_REFEREE_COMPLETE_v1/run.sh  (rnaseq_tools env: numpy, pandas)
set -eo pipefail
kit=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); root=/mnt/c/Wei/PBM
(cd "$kit" && sha256sum -c --quiet SHA256SUMS.txt) || { echo "kit changed; stop"; exit 1; }
REF=${REF:-$root/PBM_ATOMIC_REFEREE_v1}
RESULT=${RESULT:-$root/pbm_atomic_referee.20261117_1008/ATOMIC_REFEREE_RESULT.json}
want=79c197e5692986cde26db9aa492cecfb5b8b7d5b1f3cba4dc222bb2e04d23c68
[ -n "$CDL" ] || while IFS= read -r f; do [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$want" ] && { CDL=$f; break; }; done < <(find "$root" "$HOME" -maxdepth 6 -name cdl_database.py 2>/dev/null)
[ -n "$CDL" ] || { echo "set CDL=/path/to/cdl_database.py"; exit 1; }
V3=${V3:-$root/pbm_mechlib_v3.s37m8h/MECHLIB_V3_TARGETS.npz}
out=$root/pbm_atomic_referee_complete.$(date +%Y%m%d_%H%M)
nice -n 19 python -I "$kit/complete.py" --kit "$REF" --result "$RESULT" --root "$root" --cdl "$CDL" --v3 "$V3" --out "$out"
cp -a "$kit" "$out/kit"; tar -czf "$out.tar.gz" -C "$root" "$(basename "$out")"; echo "Return: $out.tar.gz"
