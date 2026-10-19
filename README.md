# Statistical interactions for transferable time-series learning

Research reset of the EEML 2026 application. The current deliverable is an audited research foundation and a literature-grounded agenda, **not a submission-ready paper or a validated new forecasting method**.

Start with the [research assessment](docs/research/assessment.md), the lead [candidate question](docs/research/information-lattice.md) (pending decision), the earlier [agenda and literature map](docs/research/agenda.md), and the [durable context index](docs/context/INDEX.md).

## Layout

| Path | Purpose |
| --- | --- |
| `src/spigraph/` | Reusable preprocessing and data/evaluation contracts |
| `research/premise_audit/` | Executable check of the historical symmetry claim |
| `research/information_lattice/` | Exploratory toy check of the candidate group-surrogate decomposition |
| `tests/` | Fast, data-independent correctness tests |
| `scripts/` | Repository/data audit utilities |
| `configs/` | Current extraction policy; future experiment configurations |
| `docs/` | Research decisions, evidence, operations and figure conventions |
| `archive/eeml-2026/` | Historical implementation, configurations, notebooks, manuscript and figures |
| `data/`, `results/` | Local scientific inputs and generated outputs, excluded from Git |

## Run the foundation

```bash
uv sync --locked
uv run --locked pytest -q
uv run --locked ruff check src tests scripts research
uv run --locked python -m research.premise_audit.symmetry --output results/premise/analytic.json
uv run --locked python -m research.premise_audit.symmetry --simulate --n-time 500 --output results/premise/t500.json
uv run --locked python -m research.premise_audit.symmetry --simulate --n-time 5000 --output results/premise/t5000.json
```

Audit commands refuse to overwrite an existing output. The simulation uses no pyspi or GPU. The original universal 2/3 ceiling for symmetric statistics is false; the [audit](research/premise_audit/README.md) explains the counterexample and its finite-sample limits.

The existing local `.venv` was preserved. This cleanup was verified using `UV_PROJECT_ENVIRONMENT=.venv-foundation uv sync --locked`; use that prefix locally to retain both environments. If local certificate configuration requires it, use uv's `--native-tls` option.

pyspi extraction belongs to the versioned runner in `mts-spi-study-cluster`; this package deliberately does not install an old PyPI release or silently depend on a mutable sibling checkout. See [extraction and reproduction](docs/operations/reproduction.md) before producing a new bank. Historical GNN training is archived and excluded from the active package and tests.
