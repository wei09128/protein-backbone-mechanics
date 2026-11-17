#!/usr/bin/env bash
# QM noise-floor pilot. Usage: bash run.sh selftest | preflight | run | status | analyse
# Runs at lowest CPU priority (nice 19) so the MD keeps the CPU. Never shares an output directory.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
ENV="${QM_ENV:-$HOME/miniconda3/envs/mechlib_qm_clean}"
[ -x "$ENV/bin/psi4" ] || { echo "psi4 not found in $ENV (set QM_ENV=/path/to/mechlib_qm_clean)"; exit 1; }
export PATH="$ENV/bin:$PATH"
OUT="${OUT:-$HOME/pbm_qm_pilot}"; mkdir -p "$OUT"
WORKERS="${WORKERS:-2}"; THREADS="${THREADS:-4}"
cmd="${1:-status}"; shift || true
case "$cmd" in
  run)
    nohup nice -n 19 "$ENV/bin/python" -I "$KIT/qm_pilot.py" run --out "$OUT" --workers "$WORKERS" --threads "$THREADS" "$@" \
      >> "$OUT/run.log" 2>&1 &
    echo "started in background (pid $!); progress: bash run.sh status   log: $OUT/run.log" ;;
  preflight)
    nice -n 19 "$ENV/bin/python" -I "$KIT/qm_pilot.py" preflight --out "$OUT" --threads "$THREADS" "$@" ;;
  *)
    "$ENV/bin/python" -I "$KIT/qm_pilot.py" "$cmd" --out "$OUT" "$@" ;;
esac
