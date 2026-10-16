#!/bin/bash
# Stage TUSZ EDFs from ISIP to Gadi /scratch.
# RUN ON A LOGIN NODE or copyq — compute nodes have NO internet.
#
#   ./stage_data.sh manifest.csv
#
# manifest.csv lines: edf_relpath,csv_relpath,split
#   where *_relpath are relative to the corpus root
#   data/tuh_eeg/tuh_eeg_seizure/v2.0.6/ on the ISIP server.
#
# TODO(verify): PROJECT, DEST path, that the key is authorised from login node.
set -euo pipefail

PROJECT="${PROJECT:-ql44}"
DEST="${DEST:-/scratch/${PROJECT}/${USER}/tusz/edf}"
REMOTE="nedc-tuh-eeg@www.isip.piconepress.com"
ROOT="data/tuh_eeg/tuh_eeg_seizure/v2.0.6"
KEY="${KEY:-${HOME}/.ssh/id_ed25519}"
[ -f "$KEY" ] || { echo "FAIL: no SSH key at $KEY. Copy it from the machine that
  is registered with NEDC:  scp ~/.ssh/id_ed25519 gadi:~/.ssh/  &&  chmod 600
  ~/.ssh/id_ed25519  (NEDC uses publickey only; a password will not work)."; exit 1; }

[ $# -eq 1 ] || { echo "usage: $0 <manifest.csv>"; exit 1; }
mkdir -p "$DEST"

# Pull EDF + both label files for each session token.
tail -n +1 "$1" | while IFS=, read -r edf csv split label; do
  [ -z "${edf:-}" ] && continue
  d="$DEST/$(dirname "$edf")"
  mkdir -p "$d"
  rsync -auvxL -e "ssh -i ${KEY}" \
    "${REMOTE}:${ROOT}/${edf}"      "$d/" || { echo "FAIL edf $edf"; continue; }
  rsync -auvxL -e "ssh -i ${KEY}" \
    "${REMOTE}:${ROOT}/${csv}"      "$d/" || echo "WARN no csv $csv"
  # .csv_bi is optional (discovery already used .csv for the type)
  rsync -auvxL -e "ssh -i ${KEY}" \
    "${REMOTE}:${ROOT}/${csv}_bi"   "$d/" 2>/dev/null || true
done
echo "staged to $DEST"
