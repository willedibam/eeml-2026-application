"""Gate 1 runner: simulate planted effects, build every lattice cell, fit one learner.

One task = (block, effect, theta, n_train, T, M, seed, learner) and writes one
JSON under <output>/tasks/. Datasets depend only on (effect, theta, n, T, M,
seed), so learners see identical records. Resumes by skipping finished tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def plan(config: dict, block: str, learner: str) -> list[dict]:
    spec = config["blocks"][block]
    if learner not in spec["seeds"]:
        raise ValueError(f"Block {block} does not run learner {learner}")
    skip = {tuple(pair) for pair in spec.get("exclude", [])}
    tasks = []
    for effect, theta, n_train, (length, channels), seed in product(
        spec["effects"],
        spec["thetas"],
        spec["n_train"],
        spec["shapes"],
        range(spec["seeds"][learner]),
    ):
        if (theta, n_train) in skip:
            continue
        task = dict(
            block=block,
            effect=effect,
            theta=theta,
            n_train=n_train,
            length=length,
            channels=channels,
            seed=seed,
            learner=learner,
            cells=spec["cells"],
            n_test=config["n_test"],
            root_seed=config["root_seed"],
        )
        task["id"] = "__".join(
            [
                block,
                effect,
                f"th{theta:g}",
                f"n{n_train}",
                f"T{length}",
                f"M{channels}",
                f"s{seed}",
                learner,
            ]
        )
        tasks.append(task)
    return tasks


def _streams(task: dict):
    from research.information_lattice.effects import EFFECTS

    key = (
        EFFECTS.index(task["effect"]),
        round(task["theta"] * 1000),
        task["n_train"],
        task["length"],
        task["channels"],
        task["seed"],
    )
    root = task["root_seed"]

    def seq(*extra):
        return np.random.SeedSequence(root, spawn_key=key + extra)

    return seq


def run_task(task: dict, out_dir: str) -> str:
    from research.information_lattice.effects import simulate
    from research.information_lattice.groups import CELLS, cell
    from research.information_lattice.learners import LEARNERS, fit_predict

    target = Path(out_dir) / "tasks" / f"{task['id']}.json"
    if target.exists():
        return "skipped"
    seq = _streams(task)
    data = np.random.default_rng(seq(0))
    n_train, n_test = task["n_train"], task["n_test"]
    labels = [data.permutation(np.repeat([0, 1], n // 2)) for n in (n_train, n_test)]
    raw = simulate(
        task["effect"],
        task["theta"],
        np.concatenate(labels),
        task["channels"],
        task["length"],
        data,
    )
    parts = ((raw[:n_train], labels[0]), (raw[n_train:], labels[1]))
    learner_seed = int(seq(2, LEARNERS.index(task["learner"])).generate_state(1)[0])
    result = {k: v for k, v in task.items() if k != "cells"}
    result["cells"] = {}
    for name in task["cells"]:
        rng = np.random.default_rng(seq(1, CELLS.index(name)))
        started = time.perf_counter()
        x_train, x_test = (cell(name, x, y, task["length"], rng) for x, y in parts)
        predicted = fit_predict(task["learner"], x_train, labels[0], x_test, learner_seed)
        correct = predicted == labels[1]
        result["cells"][name] = {
            "accuracy": float(correct.mean()),
            "n_correct": int(correct.sum()),
            "correct_bits": np.packbits(correct).tobytes().hex(),
            "seconds": round(time.perf_counter() - started, 3),
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".tmp")
    partial.write_text(json.dumps(result))
    os.replace(partial, target)
    return "done"


def _init_worker(threads: int, uses_torch: bool):
    if uses_torch:
        import torch

        torch.set_num_threads(threads)


def main():
    from spigraph.contracts import sha256_file
    from spigraph.provenance import run_provenance

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--learner", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--threads", type=int, default=1, help="threads per worker")
    parser.add_argument("--limit", type=int, help="run only the first N pending tasks (smoke)")
    parser.add_argument("--allow-dirty", action="store_true", help="smoke runs only")
    args = parser.parse_args()

    config = tomllib.loads(args.config.read_text())
    files = sorted(STUDY.glob("*.py")) + [REPO / "src/spigraph/provenance.py"]
    provenance = run_provenance(REPO, files, ("numpy", "scipy", "scikit-learn", "aeon", "torch"))
    provenance["config_sha256"] = sha256_file(args.config)
    if provenance["source"]["dirty"] and not args.allow_dirty:
        parser.error("Commit first: results must bind to a clean source commit")
    manifest = args.output / "manifest.json"
    identity = {
        k: provenance[k] for k in ("source", "lock_sha256", "files_sha256", "config_sha256")
    }
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if {k: previous["provenance"][k] for k in identity} != identity:
            parser.error("Output was produced by different source/config/lock; use a new directory")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"config": config, "provenance": provenance}, indent=2))

    tasks = [
        t
        for t in plan(config, args.block, args.learner)
        if not (args.output / "tasks" / f"{t['id']}.json").exists()
    ]
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"{len(tasks)} pending tasks for {args.block}/{args.learner}", flush=True)
    for name in THREAD_VARS + ("NUMBA_NUM_THREADS",):
        os.environ[name] = str(args.threads)
    started, done = time.time(), 0
    with ProcessPoolExecutor(
        args.workers,
        mp_context=get_context("spawn"),
        initializer=_init_worker,
        initargs=(args.threads, args.learner == "cnn"),
    ) as pool:
        futures = [pool.submit(run_task, task, str(args.output)) for task in tasks]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 10 == 0 or done == len(tasks):
                rate = (time.time() - started) / done
                print(
                    f"{done}/{len(tasks)} done, {rate:.1f} s/task, "
                    f"eta {rate * (len(tasks) - done) / 60:.0f} min",
                    flush=True,
                )


if __name__ == "__main__":
    main()
