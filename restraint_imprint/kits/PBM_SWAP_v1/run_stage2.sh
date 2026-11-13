#!/usr/bin/env bash
# Stage 2: refinements + scoring. Usage: bash PBM_SWAP_v1/run_stage2.sh <registered PREDICTIONS sha256> [/mnt/c/Wei/PBM]
# Optional: JOBS=4 (parallel refinements)
set -eo pipefail
kit=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
reg=${1:?give the registered PREDICTIONS.tsv SHA256}
root=${2:-/mnt/c/Wei/PBM}
(cd "$kit" && sha256sum -c --quiet SHA256SUMS.txt) || { echo "Kit files changed; stop."; exit 1; }
stage1="$root/pbm_swap_stage1"
[ "$(sha256sum "$stage1/PREDICTIONS.tsv" | cut -d' ' -f1)" = "$reg" ] || { echo "PREDICTIONS.tsv does not match the registered hash; stop."; exit 1; }
source "$HOME/phenix-2.2.1-6174/phenix_env.sh"
entry=$(find "$HOME/phenix-2.2.1-6174" -type f -name phenix_refine.py -print -quit); test -f "$entry"
batch="$root/pbm_transmission_batch.j1pKgh"
phil="$batch/kit/protocol.phil"
[ "$(sha256sum "$phil" | cut -d' ' -f1)" = c510fc195b30586a8c646be4ba5217b61b632b34a570de86068e6fa4ec8b691d ] || { echo "protocol.phil hash differs; stop."; exit 1; }
targets="$root/PBM_CLEAN_REBUILD_v1/results/training_targets.npz"
run=$(mktemp -d "$root/pbm_swap_stage2.XXXXXX"); echo "Run directory $run"
cp -a "$kit" "$run/kit"; phenix.version > "$run/PHENIX_VERSION.txt" 2>&1 || true
trap 'tar -czf "$run.tar.gz" --exclude="*.mtz" --exclude="*.map" -C "$(dirname "$run")" "$(basename "$run")"; echo "Return archive $run.tar.gz"' EXIT
ids=$(cut -f1 "$stage1/PREDICTIONS.tsv" | tail -n +2 | sort -u)
jobs_file="$run/JOBS.txt"; : > "$jobs_file"
for id in $ids; do
  lid=$(echo "$id" | tr 'A-Z' 'a-z')
  for arm in stock swap; do echo "$lid $arm" >> "$jobs_file"; done
done
export run kit entry phil targets batch
xargs -P "${JOBS:-4}" -L 1 bash -c '
  lid=$0; arm=$1; d="$run/runs/$lid/$arm"; mkdir -p "$d"; cd "$d"
  phenix.python "$kit/run_swap.py" --arm "$arm" --model "$batch/inputs/$lid/model.pdb" --mtz "$batch/inputs/$lid/common.mtz" \
    --phil "$phil" --entry "$entry" --targets "$targets" > RUN.log 2>&1 && echo "done $lid $arm" || echo "FAILED $lid $arm (see $d/RUN.log)"
' < "$jobs_file"
phenix.python "$kit/stage2_score.py" --stage1 "$stage1" --runs "$run/runs" --registered "$reg" --out "$run/results" 2>&1 | tee "$run/SCORE.log"
