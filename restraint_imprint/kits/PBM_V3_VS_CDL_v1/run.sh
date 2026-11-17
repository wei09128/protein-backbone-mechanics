#!/usr/bin/env bash
# v3 vs CDL (descriptive). Usage: bash PBM_V3_VS_CDL_v1/run.sh   (any python env with numpy)
set -eo pipefail
kit=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); root=/mnt/c/Wei/PBM
want=79c197e5692986cde26db9aa492cecfb5b8b7d5b1f3cba4dc222bb2e04d23c68
if [ -z "$CDL" ]; then
  while IFS= read -r f; do [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$want" ] && { CDL=$f; break; }; done \
    < <(find "$root" "$HOME" -maxdepth 6 -name cdl_database.py 2>/dev/null)
fi
[ -n "$CDL" ] || { echo "set CDL=/path/to/cdl_database.py"; exit 1; }
V3DIR=${V3DIR:-$root/pbm_mechlib_v3.s37m8h}
V2=${V2:-$root/PBM_CLEAN_REBUILD_v1/results/training_targets.npz}
python "$kit/v3_vs_cdl.py" --cdl "$CDL" --v3dir "$V3DIR" --v2 "$V2" --out "$root/V3_VS_CDL.json"
