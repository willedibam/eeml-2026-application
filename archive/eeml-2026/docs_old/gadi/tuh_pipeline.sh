#!/bin/bash
# Unattended TUH chain: stage -> generate -> train -> report.
#
# Run inside tmux on a LOGIN node (staging needs internet; compute nodes have
# none). Detach with ctrl-b then d; it keeps running on Gadi after you
# disconnect.
#
#   tmux new -s tuh
#   bash docs/gadi/tuh_pipeline.sh docs/tuh/manifest.csv
#   # ctrl-b d
#
# Stages every session in the manifest, then submits the generation array job,
# then chains the training job to start only if generation succeeds
# (qsub -W depend=afterok). Each stage writes its own log so any one of them can
# be pasted back on its own.
set -uo pipefail

MANIFEST="${1:-docs/tuh/manifest.csv}"
CHUNK="${CHUNK:-8}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
mkdir -p logs

STAGE_LOG="logs/tuh_stage.log"
echo "=== [1/3] staging $(grep -vc '^#' "$MANIFEST") sessions -> $STAGE_LOG"
echo "    (~7 GB at ~1 MB/s; this is the slow part)"
bash docs/gadi/stage_data.sh "$MANIFEST" > "$STAGE_LOG" 2>&1
N_EDF=$(find /scratch/ql44/tusz/edf -name '*.edf' 2>/dev/null | wc -l)
echo "=== staged $N_EDF EDFs"
if [ "$N_EDF" -lt 50 ]; then
  echo "!!! too few EDFs staged; stopping. see $STAGE_LOG"; exit 1
fi

echo "=== [2/3] submitting generation"
GEN_IDS=$(bash docs/gadi/submit_tuh.sh "$MANIFEST" "$CHUNK" \
          | grep -oE '^[0-9]+\[\]\.gadi-pbs|^[0-9]+\.gadi-pbs' || true)
if [ -z "$GEN_IDS" ]; then
  # submit_tuh prints qsub output directly; capture job ids from qstat instead
  GEN_IDS=$(qstat -u "$USER" | awk '/tuh_gen/{print $1}')
fi
echo "    generation jobs: $GEN_IDS"

# Chain training behind ALL generation jobs. afterok fires only if they succeed,
# so a failed generation will not silently train on a partial dataset.
DEP=""
for j in $GEN_IDS; do DEP="${DEP}:${j}"; done
if [ -n "$DEP" ]; then
  echo "=== [3/3] chaining training with depend=afterok${DEP}"
  qsub -W "depend=afterok${DEP}" docs/gadi/tuh_train.pbs
else
  echo "!!! no generation job ids captured; submit training manually:"
  echo "    qsub docs/gadi/tuh_train.pbs"
fi

echo
echo "=== all submitted. monitor with: qstat -u \$USER"
echo "=== logs: $STAGE_LOG, logs/tuh_train.txt, tuh_gen.o*, tuh_train.o*"
