#!/usr/bin/env bash
# Atomic-resolution referee. Usage: bash PBM_ATOMIC_REFEREE_v1/run.sh [V4.npz V4_SHA256]
# Run only after v4-lite is frozen (or with v2/v3 only if v4 is abandoned). Any numpy/pandas env.
set -eo pipefail
kit=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); root=/mnt/c/Wei/PBM
(cd "$kit" && sha256sum -c --quiet SHA256SUMS.txt) || { echo "kit changed; stop"; exit 1; }
want=79c197e5692986cde26db9aa492cecfb5b8b7d5b1f3cba4dc222bb2e04d23c68
[ -n "$CDL" ] || while IFS= read -r f; do [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$want" ] && { CDL=$f; break; }; done < <(find "$root" "$HOME" -maxdepth 6 -name cdl_database.py 2>/dev/null)
[ -n "$CDL" ] || { echo "set CDL=/path/to/cdl_database.py"; exit 1; }
V3=${V3:-$root/pbm_mechlib_v3.s37m8h/MECHLIB_V3_TARGETS.npz}
out=$root/pbm_atomic_referee.$(date +%Y%m%d_%H%M)
extra=(); [ -n "$1" ] && extra=(--v4 "$1" --v4-sha "$2")
nice -n 19 python "$kit/atomic_referee.py" --root "$root" --cdl "$CDL" --v3 "$V3" "${extra[@]}" --spec-sha 82f68380 --out "$out"
cp -a "$kit" "$out/kit"; tar -czf "$out.tar.gz" -C "$root" "$(basename "$out")"; echo "Return: $out.tar.gz"
