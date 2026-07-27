#!/bin/bash
# Bundled regime battery. Runs unattended (hours) and prints one compact report.
#
#   bash docs/run_all.sh [R0_DIR] [R1_DIR] [SEEDS]
#
# On Gadi:
#   bash docs/run_all.sh \
#     /scratch/ql44/we2614/mts-spi-data/260727_r0_297 \
#     /scratch/ql44/we2614/mts-spi-data/260726_r1_nonlinear 5
#
# Paste the FINAL REPORT section back; the result JSONs stay on disk.
#
# What it tests, and why each row is here:
#
#  1. R0 3-class      positive control. Linear VAR -> expect directed AND
#                     LINEAR enrichment, nonlinear DEPLETED. (Measured at
#                     K=284: 2.24x / 2.85x / 0.50x.)
#  2. R0 binary       NEGATIVE control. "collider vs {chain,fork}" is solvable
#                     by SYMMETRIC statistics -- a collider's two parents are
#                     independent, so corr(p1,p2)~0 separates it without any
#                     directional information. Same instances, same vocabulary,
#                     same lambda; only the labels change. Directed enrichment
#                     should therefore be much weaker than row 1. This is the
#                     control that makes the claim two-sided, and it is tighter
#                     than a separate symmetric-coupling dataset (e.g. Kuramoto)
#                     because it changes ONLY the label structure.
#  3. R1 3-class      THE EXPERIMENT. Quadratic coupling; linear GC is blind
#                     (generator: |pearson| 0.020 vs MI 0.296 nats). Prediction:
#                     signature moves from directed&linear to directed&NONLINEAR.
#  4. R1 binary       negative control for R1.
#  5. R1 fixed-spi    CLAUDE.md open issue #3. At M=20, top_d=5 keeps 26% of
#                     edges (vs 56% at M=10), so this asks whether the
#                     fixed-spi ~ spi-mpnn tie breaks when sparsification
#                     actually sparsifies.
#  6. R1 lambda sweep signature/accuracy trade-off at the new K.
set -uo pipefail

R0="${1:-/scratch/ql44/we2614/mts-spi-data/260727_r0_297}"
R1="${2:-/scratch/ql44/we2614/mts-spi-data/260726_r1_nonlinear}"
SEEDS="${3:-5}"
LAM="${LAM:-0.01}"
cd "$(dirname "$0")/.."
mkdir -p results logs

# Runs are independent and each is single-threaded (BLAS pinned to 1), so they
# are executed in PARALLEL. Running them sequentially wastes the node: an
# earlier version used 1 of 12 cores and would not have finished r0_baselines
# inside an 8 h wall. JOBS caps concurrency -- memory, not CPU, is the limit,
# since each process holds the whole SPI tensor (~0.4 GB for R0 at M=10,
# ~2.1 GB for R1 at M=20).
JOBS="${JOBS:-5}"
JOBFILE="$(mktemp)"

queue () {  # tag, data, classes, labels_or_-, models, lambda, ntrain
  local tag="$1" data="$2" cls="$3" lbl="$4" mdl="$5" lam="$6" nt="$7"
  local extra=""
  [[ "$lbl" != "-" ]] && extra="--class-labels ${lbl//,/ }"
  printf '%s\t%s\n' "$tag" \
    "python -u -m src.run_pipeline --data-dir $data --class-names ${cls//,/ } $extra \
     --mode sample-efficiency --n-train $nt --test-per-class 200 --val-per-class 100 \
     --seeds $SEEDS --models ${mdl//,/ } --group-lambda $lam --spi-groups literature \
     --device cpu --tag $tag" >> "$JOBFILE"
}

R0C="var-chain,var-fork,var-collider"
R1C="r1-nl-chain,r1-nl-fork,r1-nl-collider"

echo "### integrity" >&2
for d in "$R0" "$R1"; do
  echo "  $d: $(find "$d" -name spi_mpis.npz 2>/dev/null | wc -l | tr -d ' ') complete, $(find "$d" -name spi_mpis.npz -size -1k 2>/dev/null | wc -l | tr -d ' ') truncated" >&2
done

# THE decisive baseline comparison: does the vocabulary buy anything beyond
# interpretability, or does a fair fully-latent model (temporal encoder over the
# raw series, NRI/MTGNN-style) match it? If latent-directed ties at every n, the
# accuracy argument is gone and the contribution is interpretability alone.
queue r0_baselines "$R0" "$R0C" - spi-mpnn,fixed-spi,latent-directed,latent,node-only "$LAM" "20 50 100 200 400 700"
queue r1_baselines "$R1" "$R1C" - spi-mpnn,fixed-spi,latent-directed,latent,node-only "$LAM" "20 50 100 200 400 700"
queue r0_3class "$R0" "$R0C" -      spi-mpnn "$LAM" "100 400 700"
queue r0_binary "$R0" "$R0C" 0,0,1  spi-mpnn "$LAM" "100 400 700"
queue r1_3class "$R1" "$R1C" -      spi-mpnn "$LAM" "100 400 700"
queue r1_binary "$R1" "$R1C" 0,0,1  spi-mpnn "$LAM" "100 400 700"
queue r1_fixed  "$R1" "$R1C" -      spi-mpnn,fixed-spi "$LAM" "100 400 700"
for gl in 0.002 0.005 0.02; do
  queue "r1_gl$gl" "$R1" "$R1C" - spi-mpnn "$gl" "400 700"
done

echo "### launching $(wc -l < "$JOBFILE") runs, $JOBS at a time" >&2
cut -f2 "$JOBFILE" | nl -ba | while read -r i cmd; do
  tag=$(sed -n "${i}p" "$JOBFILE" | cut -f1)
  echo "$tag|$cmd"
done | xargs -d'\n' -P "$JOBS" -I{} bash -c '
  t="${1%%|*}"; c="${1#*|}"
  echo "### start $t $(date +%H:%M)" >&2
  eval "$c" > "logs/$t.log" 2>&1 && echo "### done  $t $(date +%H:%M)" >&2 \
    || echo "### FAIL  $t (see logs/$t.log)" >&2
' _ {}
rm -f "$JOBFILE"

echo
echo "======================== FINAL REPORT ========================"
for t in r0_baselines r1_baselines r0_3class r0_binary r1_3class r1_binary r1_fixed r1_gl0.002 r1_gl0.005 r1_gl0.02; do
  f="results/sample_efficiency_${t}_results.json"
  [[ -f "$f" ]] || { echo "-- $t: MISSING"; continue; }
  echo
  echo "-------- $t --------"
  python - "$f" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
for n in sorted(r['results'],key=int):
    ms=r['results'][n]['models']
    print("  n=%-4s " % n + "  ".join(f"{m}={ms[m]['f1_mean']:.4f}+/-{ms[m]['f1_std']:.3f}" for m in ms))
PY
  PYTHONPATH=. python docs/enrichment_2x2.py "$f" 2>/dev/null | grep -vE "^\s*$"
  PYTHONPATH=. python docs/compare_resolutions.py "$f" 2>/dev/null | grep -E "^  (families|modules|axes) " 
done
echo
echo "======================== END REPORT ========================"
