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
O="${1:-$(ls -t ./*.o[0-9]* 2>/dev/null | head -1)}"

echo "=============== JOB LOG: ${O:-none} ==============="
[ -n "${O:-}" ] && [ -f "$O" ] && head -25 "$O" || echo "  (no .o file yet -- job still running?)"
echo
echo "=============== FAILURE MARKERS ==============="
if [ -n "${O:-}" ] && [ -f "$O" ]; then
  grep -E "FATAL|### FAIL|### merge FAILED|too long|cannot be assembled" "$O" || echo "  none"
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
[ -f logs/tuh_report.txt ] && cat logs/tuh_report.txt || echo "  logs/tuh_report.txt not written"
