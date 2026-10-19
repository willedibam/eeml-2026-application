"""Gate 1 analysis: per-dataset tests, rates over seeds, pass criteria, power curves.

Reads <input>/tasks/*.json from gate1.py and writes summary.md, summary.json and
figures to <input>/analysis/. Criteria and predictions are those pre-specified
in README.md; nothing here is tuned to the results.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binom, fisher_exact

ALPHA = 0.05
PAIRED = {"X", "CS", "IS", "IO", "G1", "G0"}
CONTRASTS = {
    "dep": ("IS", "XR"),
    "time": ("CS", "IS"),
    "time2": ("G1", "G0"),
    "high": ("CS", "G1"),
    "highU": ("XR", "U0"),
    "dep2": ("G0", "U0"),
    "seam": ("IS", "IO"),
}
CELL_ORDER = ["X", "CS", "IS", "IO", "XR", "G1", "G0", "U0"]
_P = dict(zip(CELL_ORDER, "++++++++"))
PREDICTED = {
    "U2": _P,
    "U4": {**_P, "G1": "~0", "G0": "~0", "U0": "~0"},
    "R2": {**_P, "XR": "0", "U0": "0"},
    "R2dir": {**_P, "IS": "0", "IO": "0", "XR": "0", "G0": "0", "U0": "0"},
    "R4env": {**_P, "XR": "0", "G1": "~", "G0": "~", "U0": "0"},
}
PREDICTED["R2dirP"] = PREDICTED["R2dir"]
PREDICTED["R4quad"] = PREDICTED["R4pac"] = PREDICTED["R4env"]
PRIMARY = {
    "U2": ["cell:U0"],
    "U4": ["highU"],
    "R2": ["dep"],
    "R2dir": ["time", "time2"],
    "R4env": ["high", "dep"],
    "R4quad": ["high", "dep"],
    "R4pac": ["high", "dep"],
}
SEEDS_MAX_NULL = {"lp": 4, "mr": 4, "cnn": 1}


def _bits(cell: dict, n: int) -> np.ndarray:
    raw = np.frombuffer(bytes.fromhex(cell["correct_bits"]), np.uint8)
    return np.unpackbits(raw)[:n].astype(bool)


def task_tests(task: dict) -> dict:
    """One-sided p-values: each cell above chance, each available contrast A > B."""
    n = task["n_test"]
    cells = task["cells"]
    out = {f"cell:{c}": binom.sf(v["n_correct"] - 1, n, 0.5) for c, v in cells.items()}
    for name, (a, b) in CONTRASTS.items():
        if a not in cells or b not in cells:
            continue
        if a in PAIRED and b in PAIRED:
            ca, cb = _bits(cells[a], n), _bits(cells[b], n)
            wins, losses = int((ca & ~cb).sum()), int((~ca & cb).sum())
            p = binom.sf(wins - 1, wins + losses, 0.5) if wins + losses else 1.0
        else:
            ka, kb = cells[a]["n_correct"], cells[b]["n_correct"]
            p = fisher_exact([[ka, n - ka], [kb, n - kb]], alternative="greater").pvalue
        out[name] = p
    return out


def aggregate(tasks: list[dict]) -> dict:
    """Rates over seeds keyed by (block, learner, effect, theta, n, T, M)."""
    groups = defaultdict(lambda: {"p": defaultdict(list), "acc": defaultdict(list)})
    for task in tasks:
        key = (
            task["block"],
            task["learner"],
            task["effect"],
            task["theta"],
            task["n_train"],
            task["length"],
            task["channels"],
        )
        for test, p in task_tests(task).items():
            groups[key]["p"][test].append(p)
        for c, v in task["cells"].items():
            groups[key]["acc"][c].append(v["accuracy"])
    summary = {}
    for key, g in groups.items():
        summary[key] = {
            "seeds": len(next(iter(g["acc"].values()))),
            "reject": {t: int((np.array(ps) < ALPHA).sum()) for t, ps in g["p"].items()},
            "accuracy": {c: (float(np.mean(a)), float(np.std(a))) for c, a in g["acc"].items()},
        }
    return summary


def criteria(summary: dict) -> list[dict]:
    rows = []
    validity = {k: v for k, v in summary.items() if k[0] in ("validity", "validity-cnn")}
    for learner in ("lp", "mr"):  # P1: pooled null calibration
        null = [v for k, v in validity.items() if k[1] == learner and k[3] == 0]
        if not null:
            continue
        tests = sorted({t for v in null for t in v["reject"]})
        for test in tests:
            k = sum(v["reject"].get(test, 0) for v in null if test in v["reject"])
            total = sum(v["seeds"] for v in null if test in v["reject"])
            rows.append(
                dict(
                    criterion="P1",
                    learner=learner,
                    item=test,
                    value=f"{k}/{total}",
                    passed=k / total <= 0.09,
                )
            )
    for (block, learner, effect, theta, *_), v in validity.items():
        if theta != 1:
            continue
        for cell, mark in PREDICTED[effect].items():
            if f"cell:{cell}" not in v["reject"]:
                continue
            k = v["reject"][f"cell:{cell}"]
            if mark == "0":  # P2: specificity on hard nulls
                rows.append(
                    dict(
                        criterion="P2",
                        learner=learner,
                        item=f"{effect}/{cell}",
                        value=f"{k}/{v['seeds']}",
                        passed=k <= SEEDS_MAX_NULL[learner],
                    )
                )
            if mark == "+" and learner == "lp":  # P3: ceiling sensitivity
                rows.append(
                    dict(
                        criterion="P3",
                        learner=learner,
                        item=f"{effect}/{cell}",
                        value=f"{k}/{v['seeds']}",
                        passed=k >= 16,
                    )
                )
        if effect == "R2dirP" and "seam" in v["reject"]:  # P4: seam leakage
            k = v["reject"]["seam"]
            rows.append(
                dict(
                    criterion="P4",
                    learner=learner,
                    item="R2dirP/seam",
                    value=f"{k}/{v['seeds']}",
                    passed=k <= SEEDS_MAX_NULL[learner],
                )
            )
    return rows


def _style():
    import shutil

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "text.usetex": bool(shutil.which("latex") and shutil.which("dvipng")),
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 1.7,
            "lines.markersize": 2.7,
            "figure.constrained_layout.use": True,
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
        }
    )


def _save(fig, path: Path):
    for suffix in (".pdf", ".svg", ".png"):
        fig.savefig(path.with_suffix(suffix), dpi=180 if suffix == ".png" else None)


def plot_validity(summary: dict, out: Path):
    import matplotlib.pyplot as plt

    learners = [
        lr
        for lr in ("lp", "mr", "cnn")
        if any(k[1] == lr and k[3] == 1 and k[0].startswith("validity") for k in summary)
    ]
    effects = list(PREDICTED)
    fig, axes = plt.subplots(
        1, len(learners), figsize=(2.4 * len(learners) + 0.6, 3.0), squeeze=False
    )
    for ax, learner in zip(axes[0], learners):
        grid = np.full((len(effects), len(CELL_ORDER)), np.nan)
        for (block, lr, effect, theta, *_), v in summary.items():
            if lr == learner and theta == 1 and block.startswith("validity"):
                for j, c in enumerate(CELL_ORDER):
                    if c in v["accuracy"]:
                        grid[effects.index(effect), j] = v["accuracy"][c][0]
        image = ax.imshow(grid, vmin=0.5, vmax=1, cmap="viridis", aspect="auto")
        for i, effect in enumerate(effects):
            for j, c in enumerate(CELL_ORDER):
                if PREDICTED[effect][c] == "0":
                    ax.text(j, i, "0", ha="center", va="center", color="white", fontsize=7)
        ax.set_xticks(range(len(CELL_ORDER)), CELL_ORDER, rotation=90)
        ax.set_yticks(range(len(effects)), effects if learner == learners[0] else [])
        ax.set_title(learner.upper())
    fig.colorbar(
        image, ax=axes[0].tolist(), label=r"test accuracy ($\theta=1$, mean over seeds)", shrink=0.8
    )
    _save(fig, out / "validity")
    plt.close(fig)


def plot_power(summary: dict, out: Path):
    import matplotlib.pyplot as plt

    panels = [(e, t) for e, tests in PRIMARY.items() for t in tests]
    fig, axes = plt.subplots(
        2, (len(panels) + 1) // 2, figsize=(9.0, 3.8), sharex=True, sharey=True
    )
    colours = dict(zip((0.125, 0.25, 0.5, 1.0), plt.cm.viridis(np.linspace(0.15, 0.85, 4))))
    for ax, (effect, test) in zip(axes.flat, panels):
        for learner, style in (("lp", "-"), ("mr", "--")):
            for theta, colour in colours.items():
                points = sorted(
                    (k[4], v["reject"][test] / v["seeds"])
                    for k, v in summary.items()
                    if k[1] == learner
                    and k[2] == effect
                    and k[3] == theta
                    and k[5] == 1024
                    and k[6] == 4
                    and test in v["reject"]
                )
                if points:
                    ax.plot(
                        *zip(*points),
                        style,
                        color=colour,
                        marker="o",
                        label=rf"$\theta={theta:g}$" if learner == "lp" else None,
                    )
        ax.set_title(f"{effect}: {test.replace('cell:', '')}", fontsize=9)
        ax.set_xscale("log")
        ax.axhline(ALPHA, color="0.6", lw=0.8, ls=":")
    for ax in axes[-1]:
        ax.set_xlabel(r"$n_\mathrm{train}$")
    for ax in axes[:, 0]:
        ax.set_ylabel("rejection rate")
    axes.flat[0].legend(loc="lower right")
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)
    _save(fig, out / "power")
    plt.close(fig)


def markdown(summary: dict, rows: list[dict]) -> str:
    lines = ["# Gate 1 summary", "", "## Criteria", ""]
    for crit in ("P1", "P2", "P3", "P4"):
        subset = [r for r in rows if r["criterion"] == crit]
        failed = [r for r in subset if not r["passed"]]
        lines.append(
            f"- **{crit}**: {len(subset) - len(failed)}/{len(subset)} pass"
            + (
                ""
                if not failed
                else "; failures: "
                + ", ".join(f"{r['learner']} {r['item']} {r['value']}" for r in failed)
            )
        )
    for theta in (1.0, 0.0):
        lines += ["", f"## Validity, theta={theta:g}: mean accuracy (rejections/seeds)", ""]
        lines.append("| learner | effect | " + " | ".join(CELL_ORDER) + " |")
        lines.append("|---" * (len(CELL_ORDER) + 2) + "|")
        for (block, learner, effect, th, *_), v in sorted(summary.items()):
            if not block.startswith("validity") or th != theta:
                continue
            cells = [
                f"{v['accuracy'][c][0]:.3f} ({v['reject'][f'cell:{c}']}/{v['seeds']})"
                if c in v["accuracy"]
                else ""
                for c in CELL_ORDER
            ]
            lines.append(f"| {learner} | {effect} | " + " | ".join(cells) + " |")
    lines += ["", "## Validity, theta=1: contrast rejections/seeds", ""]
    lines.append("| learner | effect | " + " | ".join(CONTRASTS) + " |")
    lines.append("|---" * (len(CONTRASTS) + 2) + "|")
    for (block, learner, effect, th, *_), v in sorted(summary.items()):
        if block.startswith("validity") and th == 1:
            lines.append(
                f"| {learner} | {effect} | "
                + " | ".join(
                    f"{v['reject'][c]}/{v['seeds']}" if c in v["reject"] else "" for c in CONTRASTS
                )
                + " |"
            )
    lines += ["", "## Scale block (theta=0.5, n=300): primary-test rejections/seeds", ""]
    lines.append("| learner | effect | test | T=256,M=4 | T=4096,M=4 | T=1024,M=16 | T=1024,M=32 |")
    lines.append("|---|---|---|---|---|---|---|")
    for learner in ("lp", "mr"):
        for effect, tests in PRIMARY.items():
            for test in tests:
                row = []
                for shape in ((256, 4), (4096, 4), (1024, 16), (1024, 32)):
                    v = summary.get(("scale", learner, effect, 0.5, 300, *shape))
                    row.append(
                        f"{v['reject'][test]}/{v['seeds']}" if v and test in v["reject"] else ""
                    )
                if any(row):
                    lines.append(f"| {learner} | {effect} | {test} | " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    tasks = [json.loads(p.read_text()) for p in sorted((args.input / "tasks").glob("*.json"))]
    if not tasks:
        parser.error("No finished tasks")
    summary = aggregate(tasks)
    rows = criteria(summary)
    out = args.input / "analysis"
    out.mkdir(exist_ok=True)
    (out / "summary.md").write_text(markdown(summary, rows))
    (out / "summary.json").write_text(
        json.dumps(
            {
                "n_tasks": len(tasks),
                "criteria": rows,
                "groups": [{"key": list(k), **v} for k, v in summary.items()],
            },
            indent=1,
        )
    )
    _style()
    plot_validity(summary, out)
    if any(k[0] == "power" for k in summary):
        plot_power(summary, out)
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
