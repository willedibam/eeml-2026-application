# Local scientific inputs

The old datasets remain in place and are ignored by Git. They are incomplete local copies, not validated current feature banks. The [research assessment](../docs/research/assessment.md#local-data-and-result-provenance) records the inventory and its limits. Do not infer the condition of Gadi data from this local inventory.

Keep future inputs and feature banks under `data/<study>/<bank-id>/`, with a manifest and computation identity; generated analyses belong under `results/<study>/<run-id>/`. Choose a new bank ID when estimator computation, configuration or extraction semantics change. Never extend the legacy directories with current pyspi. Do not delete historical data merely because it is incomplete.
