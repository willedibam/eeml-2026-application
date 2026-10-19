"""Exactness of lattice cells, planted-effect matching and the Gate 1 plan."""

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from research.information_lattice import effects, gate1, groups
from research.information_lattice.learners import lattice_probe_features

CONFIG = Path("research/information_lattice/configs/gate1.toml")


@pytest.fixture
def records():
    rng = np.random.default_rng(1)
    return groups.zscore(np.cumsum(rng.standard_normal((6, 3, 128)), axis=-1)), rng


def _xcorr(x):
    f = np.fft.rfft(x, axis=-1)
    return np.fft.irfft(np.conj(f[:, 0]) * f[:, 1], n=x.shape[-1], axis=-1)


def test_phase_cells_preserve_their_invariants_exactly(records):
    x, rng = records
    f = np.fft.rfft(x, axis=-1)
    cross = np.einsum("njf,nkf->njkf", f, np.conj(f))
    g1 = np.fft.rfft(groups.common_phase(x, rng), axis=-1)
    assert np.allclose(np.einsum("njf,nkf->njkf", g1, np.conj(g1)), cross, atol=1e-9)
    g0 = groups.independent_phase(x, rng)
    assert np.allclose(np.abs(np.fft.rfft(g0, axis=-1)), np.abs(f), atol=1e-9)
    assert not np.allclose(np.fft.rfft(g0, axis=-1), f)
    for out in (g0, groups.common_phase(x, rng)):
        assert np.allclose(out.mean(-1), 0, atol=1e-9) and np.allclose(out.std(-1), 1)


def test_shift_cells(records):
    x, rng = records
    cs = groups.common_shift(x, rng)
    assert np.allclose(_xcorr(cs), _xcorr(x))
    shifted = groups.independent_shift(x, rng)
    assert np.allclose(np.sort(shifted, -1), np.sort(x, -1))
    assert np.allclose(np.abs(_xcorr(shifted)).max(-1), np.abs(_xcorr(x)).max(-1))


def test_reference_cells(records):
    x, rng = records
    y = np.array([0, 0, 0, 1, 1, 1])
    xr = groups.within_class_shuffle(x, y, rng)
    for label in (0, 1):
        for channel in range(3):
            original = {r.tobytes() for r in x[y == label, channel]}
            assert {r.tobytes() for r in xr[y == label, channel]} == original
    io = groups.independent_offset(np.concatenate([x, x], -1), 128, rng)
    assert io.shape == x.shape and np.allclose(io.std(-1), 1)
    with pytest.raises(ValueError):
        groups.cell("nope", x, y, 128, rng)


@pytest.mark.parametrize("effect", ["R2", "R2dir", "R4quad", "R4env", "R4pac"])
def test_relational_effects_keep_channel_laws(effect):
    rng = np.random.default_rng(7)
    y = np.repeat([0, 1], 300)
    x = effects.simulate(effect, 1.0, y, 3, 512, rng)
    z = groups.zscore(x)
    stats = [x.var(-1), (z[..., 1:] * z[..., :-1]).mean(-1), (z**4).mean(-1)]
    for stat in stats:
        a, b = stat[y == 0], stat[y == 1]
        se = np.sqrt(a.var(0) / len(a) + b.var(0) / len(b))
        assert np.all(np.abs(a.mean(0) - b.mean(0)) < 4.5 * se), effect


def test_planted_linear_coupling_is_present():
    rng = np.random.default_rng(3)
    y = np.repeat([0, 1], 50)
    z = groups.zscore(effects.simulate("R2", 1.0, y, 2, 1024, rng))
    lag5 = (z[:, 0, :-5] * z[:, 1, 5:]).mean(-1)
    assert lag5[y == 1].mean() > 0.5 and abs(lag5[y == 0].mean()) < 0.05


def test_lattice_probe_features_are_finite():
    rng = np.random.default_rng(0)
    feats = lattice_probe_features(groups.zscore(rng.standard_normal((5, 3, 256))))
    assert feats.shape[0] == 5 and np.isfinite(feats).all()


def test_plan_matches_protocol():
    config = tomllib.loads(CONFIG.read_text())
    counts = {
        (b, lr): len(gate1.plan(config, b, lr))
        for b, lr in [
            ("validity", "lp"),
            ("validity", "mr"),
            ("validity-cnn", "cnn"),
            ("power", "lp"),
            ("power", "mr"),
            ("scale", "lp"),
            ("scale", "mr"),
        ]
    }
    assert counts == {
        ("validity", "lp"): 320,
        ("validity", "mr"): 320,
        ("validity-cnn", "cnn"): 40,
        ("power", "lp"): 1540,
        ("power", "mr"): 770,
        ("scale", "lp"): 560,
        ("scale", "mr"): 280,
    }


def test_run_task_writes_paired_results(tmp_path):
    task = dict(
        id="smoke",
        block="smoke",
        effect="R2",
        theta=1.0,
        n_train=40,
        n_test=40,
        length=128,
        channels=3,
        seed=0,
        learner="lp",
        root_seed=1,
        cells=list(groups.CELLS),
    )
    assert gate1.run_task(task, str(tmp_path)) == "done"
    result = json.loads((tmp_path / "tasks" / "smoke.json").read_text())
    assert set(result["cells"]) == set(groups.CELLS)
    bits = np.unpackbits(
        np.frombuffer(bytes.fromhex(result["cells"]["X"]["correct_bits"]), np.uint8)
    )[:40]
    assert bits.sum() == result["cells"]["X"]["n_correct"]
    assert gate1.run_task(task, str(tmp_path)) == "skipped"
