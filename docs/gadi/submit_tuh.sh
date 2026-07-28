#!/bin/bash
# Submit TUH generation, batching around Gadi's max_array_size=10.
#   bash docs/gadi/submit_tuh.sh docs/tuh/manifest.csv [chunk]
set -euo pipefail
MANIFEST="${1:?manifest}"
CHUNK="${2:-24}"
N=$(grep -vc '^#' "$MANIFEST")
SUBJOBS=$(( (N + CHUNK - 1) / CHUNK ))
echo "[INFO] $N sessions, chunk=$CHUNK -> $SUBJOBS subjobs"
b=0
while [ $((b * 10)) -lt $SUBJOBS ]; do
  lo=$(( b * 10 + 1 )); hi=$(( (b + 1) * 10 )); [ $hi -gt $SUBJOBS ] && hi=$SUBJOBS
  echo "[INFO] batch $((b+1)): subjobs $lo-$hi"
  qsub -J "${lo}-${hi}" -v "MANIFEST=${MANIFEST},CHUNK=${CHUNK}" \
       docs/gadi/tuh_generate_full.pbs
  b=$((b + 1))
done
echo "[INFO] monitor: qstat -tu \$USER"
