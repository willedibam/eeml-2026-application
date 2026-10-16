#!/bin/bash
# One-screen status. Read-only, runs in seconds on a login node, safe any time.
#
#   cd /scratch/ql44/we2614/eeml-2026-application && bash docs/gadi/status.sh
#
# Exists because the useful state is spread over qstat, two data trees and half
# a dozen logs, and PBS only copies a job's .o file back when the job ENDS --
# so "is it working?" cannot be answered by reading the log of a running job.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
TUH="${TUH:-/scratch/ql44/we2614/mts-spi-data/260728_tuh}"

echo "=============== JOBS ==============="
qstat -tu "$USER" 2>/dev/null | tail -n +6 | awk '{printf "  %-22s %-11s %s %s\n",$1,$4,$10,$11}' \
  || echo "  (none)"

echo
echo "=============== TUH GENERATION ==============="
tot=0
for c in fnsz gnsz; do
  n=$(find "$TUH/$c" -name spi_mpis.npz 2>/dev/null | wc -l | tr -d ' ')
  tot=$((tot + n)); printf "  %-6s %5d windows\n" "$c" "$n"
done
echo "  total  $tot"
# 'X' means a subjob has EXITED. Counting it as still-running made the progress
# line read 14 when three shards were already done.
act=$(qstat -tu "$USER" 2>/dev/null | grep "tuh_gen" | awk '$10!="X"' | wc -l | tr -d ' ')
fin=$(qstat -tu "$USER" 2>/dev/null | grep "tuh_gen" | awk '$10=="X"' | wc -l | tr -d ' ')
echo "  tuh_gen shards: $act still running, $fin finished   (0 running = done)"

echo
echo "=============== REPORTS PRESENT ==============="
for f in logs/r1b_lam0.0002_report.txt logs/r1b_lam0.005_report.txt \
         logs/r1b_panel_report.txt logs/audit_report.txt logs/tuh_report.txt; do
  if [ -f "$f" ]; then printf "  %-34s %6s bytes\n" "$f" "$(wc -c <"$f")"
  else printf "  %-34s --\n" "$f"; fi
done

echo
echo "=============== TUH CONTROLS (read these first) ==============="
# Guard against the stale report: an early chained run produced a complete-
# looking tuh_report.txt from ZERO windows. If the file has no per-n rows, say
# so rather than printing a legend that implies results exist.
if [ -f logs/tuh_report.txt ] && grep -qE "^  [0-9]+ +0\.[0-9]" logs/tuh_report.txt; then
  # node-only near chance keeps the coupling claim alive; high means the classes
  # differ in per-channel activity and the SPI vocabulary is mismatched.
  grep -E "^-{8}|node-only|shuffled|spi-mpnn|^  [0-9]+ " logs/tuh_report.txt | head -30
elif [ -f logs/tuh_report.txt ]; then
  echo "  logs/tuh_report.txt exists but has NO result rows -- this is the stale"
  echo "  report from the early zero-window run. Ignore until tuh_train reruns."
else
  echo "  not yet -- tuh_train runs automatically when every shard exits 0"
fi

echo
echo "=============== FAILURES IN RECENT LOGS ==============="
# Anchored: the training job echoes a generated shard script whose text
# contains the literal "### FAIL" inside its own || echo clause.
grep -lE "^\[FATAL\]|Traceback|^### FAIL" ./*.o[0-9]* logs/*.log 2>/dev/null | head -8 \
  || echo "  none found"
