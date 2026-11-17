#!/usr/bin/env bash
# PBM_QM_CONFIRM_v1. Usage: bash run.sh selftest|preflight|build|timing|design W|production W|status|analyse
set -eo pipefail
kit=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
(cd "$kit" && sha256sum -c --quiet SHA256SUMS.txt) || { echo "kit changed; stop"; exit 1; }
ENV=${QM_ENV:-$HOME/miniconda3/envs/mechlib_qm_solv}
export AMBERHOME=${AMBERHOME:-$HOME/miniconda3/envs/amber}
TLEAP=${TLEAP:-$AMBERHOME/bin/tleap}
export PATH="$ENV/bin:$PATH"
OUT=${OUT:-$HOME/pbm_qm_confirm}; mkdir -p "$OUT"
PY="$ENV/bin/python -I $kit/qmc.py"
THREADS=${THREADS:-4}
case "${1:-status}" in
  selftest)   $PY selftest ;;
  preflight)  $PY preflight --out "$OUT" --tleap "$TLEAP" ;;
  build)      $PY build --out "$OUT" ;;
  timing)     nohup nice -n 19 $PY run --set timing --out "$OUT" --workers "${WORKERS:-2}" --threads "$THREADS" >> "$OUT/run.log" 2>&1 &
              echo "timing run started (pid $!); progress: bash run.sh status   log: $OUT/run.log" ;;
  design)     [ -n "$2" ] || { echo "usage: bash run.sh design <production workers>"; exit 1; }
              $PY design --out "$OUT" --workers "$2" --threads "$THREADS" ;;
  production) [ -n "$2" ] || { echo "usage: bash run.sh production <workers>"; exit 1; }
              nohup nice -n 19 $PY run --set production --out "$OUT" --workers "$2" --threads "$THREADS" >> "$OUT/run.log" 2>&1 &
              echo "production started (pid $!); progress: bash run.sh status" ;;
  status)     $PY status --out "$OUT" ;;
  analyse)    $PY analyse --out "$OUT" ;;
  *) echo "unknown command"; exit 1 ;;
esac
