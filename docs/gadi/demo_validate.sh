#!/bin/bash
# End-to-end pipeline smoke test. Run this BEFORE committing cluster hours.
#
# Proves the full chain works: generator -> pyspi -> per-instance dirs ->
# graph_build loads them -> every model forward/backward runs -> results JSON
# -> analysis reads it. It does NOT produce meaningful accuracy (too few
# instances); it proves the plumbing.
#
#   bash docs/gadi/demo_validate.sh                 # assumes data already generated
#   bash docs/gadi/demo_validate.sh --with-generate # generate the demo data first
#
# Expected wall time: ~6 min generation (9 datasets) + ~1 min training.
set -euo pipefail

EEML_DIR="${EEML_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
GEN_DIR="${GEN_DIR:-$EEML_DIR/../mts-spi-study-cluster}"
DEMO_DATA="$GEN_DIR/data/260726_r1_demo"
PY="${PY:-$EEML_DIR/.venv/bin/python}"

if [[ "${1:-}" == "--with-generate" ]]; then
    echo "=== [1/3] generating demo data (9 datasets) ==="
    cd "$GEN_DIR"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
    unset CUDA_VISIBLE_DEVICES || true
    seq 1 9 | xargs -n1 -P6 -I{} "$GEN_DIR/.venv/bin/python" -m src.run_experiments \
        --job-index {} --skip-existing \
        --experiment-config configs/generate/eeml/eeml-r1-demo.yaml \
        --n-jobs 1 --parquet
fi

echo "=== [2/3] checking generated structure ==="
n=$(find "$DEMO_DATA" -name 'spi_mpis.npz' 2>/dev/null | wc -l | tr -d ' ')
echo "  instances with SPIs: $n"
if [[ "$n" -lt 3 ]]; then
    echo "  FAIL: need >=3 instances (1/class). Run with --with-generate."
    exit 1
fi
for f in timeseries.npy spi_mpis.npz meta.json; do
    c=$(find "$DEMO_DATA" -name "$f" | wc -l | tr -d ' ')
    echo "  $f: $c"
done

echo "=== [3/3] training smoke (all models, 2 epochs) ==="
cd "$EEML_DIR"
"$PY" -u -m src.run_pipeline \
    --data-dir "$DEMO_DATA" \
    --class-names r1-nl-chain r1-nl-fork r1-nl-collider \
    --mode standard \
    --train-ratio 0.6 --val-ratio 0.2 \
    --seeds 1 --max-epochs 2 --warmup-epochs 1 --restarts 1 \
    --models spi-mpnn fixed-spi correlation latent latent-directed node-only \
    --spi-groups modules \
    --device cpu --output-dir results --tag demo_validate

echo
echo "=== DEMO PASSED: generation -> SPI -> graph -> all models -> results JSON ==="
echo "Accuracy here is meaningless (3 instances/class); this validates plumbing only."
