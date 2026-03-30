# Cluster setup for eeml-2026-application

## 1. Clone

```bash
cd /import/taiji1/wedi0306
git clone https://github.com/willedibam/eeml-2026-application.git
cd eeml-2026-application
```

## 2. Create venv and install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

If torch/torch-geometric need CPU-only wheels on the cluster:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install -e .
```

## 3. Verify

```bash
python -m src.run_pipeline --help
```

## 4. Run

Data lives in the sibling repo. Use absolute path:

```bash
DATA=/import/taiji1/wedi0306/mts-spi-study-cluster/data/eeml_bciciv2a

# Pilot (5 seeds, core models) — Terminal 1
python -m src.run_pipeline \
    --data-dir $DATA \
    --class-names feet left-hand right-hand tongue \
    --mode sample-efficiency \
    --n-train 20 50 100 200 \
    --test-per-class 200 --val-per-class 100 \
    --seeds 5 \
    --models spi-mpnn fixed-spi correlation latent node-only \
    --warmup-epochs 60 --top-d 5 --group-lambda 0.02 --l1-lambda 0.001 --lr 1e-3 --restarts 2 \
    --device cpu \
    --output-dir results --tag bciciv2a_pilot_5seeds

# Ablation models — Terminal 2
python -m src.run_pipeline \
    --data-dir $DATA \
    --class-names feet left-hand right-hand tongue \
    --mode sample-efficiency \
    --n-train 20 50 100 200 \
    --test-per-class 200 --val-per-class 100 \
    --seeds 5 \
    --models edge-ablation shuffled \
    --warmup-epochs 60 --top-d 5 --group-lambda 0.02 --l1-lambda 0.001 --lr 1e-3 --restarts 2 \
    --device cpu \
    --output-dir results --tag bciciv2a_ablation_5seeds

# MLP-Mix — Terminal 3
python -m src.run_pipeline \
    --data-dir $DATA \
    --class-names feet left-hand right-hand tongue \
    --mode sample-efficiency \
    --n-train 20 50 100 200 \
    --test-per-class 200 --val-per-class 100 \
    --seeds 5 \
    --models mlp-mix \
    --warmup-epochs 60 --top-d 5 --group-lambda 0.02 --l1-lambda 0.001 --lr 1e-3 --restarts 2 \
    --device cpu \
    --output-dir results --tag bciciv2a_mlpmix_5seeds
```

## Notes

- All hyperparameters match the VAR 30-seed runs exactly
- Results write to `results/` inside this repo (eeml-2026-application)
- `scp` or `git add` the result JSONs back to local when done
- Data stays in mts-spi-study-cluster — nothing is copied
