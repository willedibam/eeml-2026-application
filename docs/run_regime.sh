#!/bin/bash
# One-command regime analysis. Trains spi-mpnn on a generated dataset and
# prints a compact text report -- no figures, no large files, so the output can
# be pasted back over a slow link while the result JSONs stay on the cluster.
#
#   bash docs/run_regime.sh <data-dir> <class1,class2,class3> <tag> [lambda] [seeds]
#
# e.g. on Gadi:
#   bash docs/run_regime.sh \
#     /scratch/ql44/we2614/mts-spi-data/260726_r1_nonlinear \
#     r1-nl-chain,r1-nl-fork,r1-nl-collider r1 0.01 5
#
# Everything it needs is in the repo (src/spi_labels.json, docs/enrichment_2x2.py).
set -euo pipefail

DATA="${1:?data dir}"
CLASSES="${2:?comma-separated class dir names}"
TAG="${3:?tag}"
LAMBDA="${4:-0.01}"
SEEDS="${5:-5}"
NTRAIN="${NTRAIN:-100 400 700}"

CLS="${CLASSES//,/ }"
cd "$(dirname "$0")/.."

echo "=== dataset integrity ==="
for c in $CLS; do
  n=$(find "$DATA/$c" -name spi_mpis.npz 2>/dev/null | wc -l | tr -d ' ')
  b=$(find "$DATA/$c" -name spi_mpis.npz -size -1k 2>/dev/null | wc -l | tr -d ' ')
  echo "  $c: $n complete, $b truncated"
done

echo
echo "=== training (lambda=$LAMBDA, seeds=$SEEDS, n_train=$NTRAIN) ==="
python -u -m src.run_pipeline \
  --data-dir "$DATA" \
  --class-names $CLS \
  --mode sample-efficiency \
  --n-train $NTRAIN \
  --test-per-class 200 --val-per-class 100 \
  --seeds "$SEEDS" \
  --models spi-mpnn \
  --group-lambda "$LAMBDA" \
  --spi-groups literature \
  --device cpu --tag "$TAG" 2>&1 | grep -E "Retained|SPI-MODULES|F1=" | tail -20

RES="results/sample_efficiency_${TAG}_results.json"
echo
echo "=== accuracy ==="
python - "$RES" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
for n in sorted(r['results'],key=int):
    b=r['results'][n]['models']['spi-mpnn']
    print(f"  n={n:>4}  F1={b['f1_mean']:.4f} +/- {b['f1_std']:.4f}")
PY

echo
echo "=== 2x2 enrichment (directed x nonlinear, permutation null) ==="
PYTHONPATH=. python docs/enrichment_2x2.py "$RES"

echo
echo "=== stability / uncertainty ==="
PYTHONPATH=. python - "$RES" <<'PY'
import json,sys
from src.analysis import (report_weight_stability, report_family_uncertainty,
                          report_directed_enrichment)
r=json.load(open(sys.argv[1]))
report_weight_stability(r); report_directed_enrichment(r); report_family_uncertainty(r)
PY
echo
echo "=== paste everything above back ==="
