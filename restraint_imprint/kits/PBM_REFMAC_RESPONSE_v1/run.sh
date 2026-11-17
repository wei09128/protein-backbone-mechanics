#!/usr/bin/env bash
# REFMAC controlled response (spec v1.0). Usage: bash PBM_REFMAC_RESPONSE_v1/run.sh [/mnt/c/Wei/PBM]
# Optional: CCP4_SETUP=/path/to/ccp4-X.Y/bin/ccp4.setup-sh  JOBS=3
set -eo pipefail
kit=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=${1:-/mnt/c/Wei/PBM}
(cd "$kit" && sha256sum -c --quiet SHA256SUMS.txt) || { echo "Kit files changed; stop."; exit 1; }
if ! command -v refmac5 >/dev/null 2>&1; then
  setup=${CCP4_SETUP:-$(find "$HOME" /opt /usr/local -maxdepth 4 -name ccp4.setup-sh 2>/dev/null | sort | tail -1)}
  [ -n "$setup" ] || { echo "CCP4 not found: no refmac5 on PATH and no ccp4.setup-sh under ~, /opt, /usr/local. Install CCP4 or set CCP4_SETUP."; exit 1; }
  echo "Sourcing $setup"; source "$setup"
fi
command -v refmac5 >/dev/null || { echo "refmac5 still not on PATH; stop."; exit 1; }
py=$(command -v ccp4-python || command -v python3)
"$py" -c "import gemmi, numpy" || { echo "$py lacks gemmi/numpy; stop."; exit 1; }
echo "refmac5: $(command -v refmac5)   python: $py"
out=$(mktemp -u "$root/pbm_refmac_response.XXXXXX")
trap 'cp -a "$kit" "$out/kit" 2>/dev/null; tar -czf "$out.tar.gz" --exclude="*.mtz" --exclude="*.map" -C "$(dirname "$out")" "$(basename "$out")"; echo "Return archive: $out.tar.gz"' EXIT
"$py" -u "$kit/refmac_response.py" --selftest
"$py" -u "$kit/refmac_response.py" --root "$root" --out "$out" --jobs "${JOBS:-3}" 2>&1 | tee "$out.log"
mv "$out.log" "$out/RUN.log" 2>/dev/null || true
