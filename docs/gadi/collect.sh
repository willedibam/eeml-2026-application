#!/bin/bash
# Compact, paste-able summary of a finished job. Read-only.
#
#   bash docs/gadi/collect.sh tuh_train.o174960000
#   bash docs/gadi/collect.sh                        # newest .o file
#
# PBS only copies a job's .o file back when the job ENDS, so this is for after.
# Per-shard logs under logs/ are written live and are readable during the run.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
# Prefer the training job's log. Falling back to "newest .o file" picked a
# tuh_gen log while training was still running and its .o did not exist yet.
O="${1:-$(ls -t ./tuh_train.o[0-9]* 2>/dev/null | head -1)}"
O="${O:-$(ls -t ./*.o[0-9]* 2>/dev/null | head -1)}"

echo "=============== JOB LOG: ${O:-none} ==============="
[ -n "${O:-}" ] && [ -f "$O" ] && head -25 "$O" || echo "  (no .o file yet -- job still running?)"
echo
echo "=============== FAILURE MARKERS ==============="
if [ -n "${O:-}" ] && [ -f "$O" ]; then
  # Anchored: the job echoes a shard script that CONTAINS the literal
  # '### FAIL' inside its `|| echo` clause, which matched as a false positive.
  grep -E "^\[FATAL\]|^### FAIL|^### merge FAILED|too long|cannot be assembled" "$O" \
    || echo "  none"
fi
echo
echo "=============== SHARD LOGS WITH ERRORS ==============="
found=0
for f in logs/tuh_*_s[0-9]*.log; do
  [ -f "$f" ] || continue
  if grep -qE "Traceback|Error:|error:|Killed" "$f" 2>/dev/null; then
    found=1; echo "--- $f"; tail -12 "$f"; echo
  fi
done
[ "$found" = 0 ] && echo "  none"
echo
echo "=============== SHARD COMPLETION ==============="
ok=$(ls results/*tuh_*_s[0-9]*_results.json 2>/dev/null | wc -l)
lg=$(ls logs/tuh_*_s[0-9]*.log 2>/dev/null | wc -l)
echo "  $ok result files from $lg shard logs"
echo
echo "=============== REPORT ==============="
if [ -f logs/tuh_report.txt ]; then
  age=$(( $(date +%s) - $(stat -c %Y logs/tuh_report.txt 2>/dev/null || echo 0) ))
  [ "$age" -gt 1800 ] && echo "  [WARN] report is $((age/60)) min old -- likely STALE from a previous job"
  cat logs/tuh_report.txt
else
  echo "  logs/tuh_report.txt not written"
fi
