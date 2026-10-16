# Current configuration policy

New extraction should use pyspi 3.0's **bundled** `benchmarked_p90`, with an explicitly recorded pyspi revision, computation version, resolved catalogue/config hash and extraction recipe. On 2026-10-16 the inspected sibling checkout was `c2d723f9422617d668bca1688cb06ada67395aaa`, computation `3.0.0.r9`, with 283 p90 statistics. These are an inspection snapshot, not a promise that a future checkout is identical.

Do not copy `archive/eeml-2026/configs/` into new runs. Its `pyspi-v2` paths are stale, one path starts with `cconfigs`, and the catalogue/estimator semantics differ. Even the sibling study's retained 289-statistic YAML is explicitly historical; its README instructs new runs to use the bundled name. The bundled `sonnet` set (14 representatives) is a useful cost/coverage comparator, not a scientifically equivalent replacement for p90.

Place runnable study configs under `research/<study>/configs/` when that study has a settled protocol. Do not create placeholder production configurations that look validated. [Reproduction contract](../docs/operations/reproduction.md).
