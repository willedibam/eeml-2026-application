import dataclasses

import numpy as np
import pytest

from research.premise_audit.symmetry import (
    MOTIFS,
    simulate,
    stationary_symmetric_lags,
    symmetric_lag_features,
)
from spigraph.contracts import (
    ComputationIdentity,
    SplitRecord,
    require_same_computation,
    validate_classification_split,
)
from spigraph.preprocessing import SPIPreprocessor, load_spi_tensor


def tensor():
    data = np.arange(27, dtype=float).reshape(3, 3, 3)
    data[np.arange(3), np.arange(3)] = np.nan
    return data


def test_missingness_survives_loading_and_selection(tmp_path):
    data = tensor()
    data[0, 1, 1] = np.inf
    path = tmp_path / "spi.npz"
    np.savez(path, a=data[:, :, 0], b=data[:, :, 1], c=data[:, :, 2])
    loaded = load_spi_tensor(path, ["a", "b", "c"])
    assert np.isinf(loaded[0, 1, 1])
    fitted = SPIPreprocessor(["a", "b", "c"]).fit([loaded])
    assert fitted.indices_.tolist() == [0, 2]
    assert fitted.missing_rate_[1] == pytest.approx(1 / 6)
    with pytest.raises(ValueError, match="catalogue"):
        load_spi_tensor(path, ["a", "b"])


def test_test_values_do_not_change_fit_and_missing_is_training_median():
    fitted = SPIPreprocessor(["a", "b", "c"]).fit([tensor()])
    state = fitted.state()
    test = tensor() * 1e5
    test[0, 1, 0] = np.nan
    scaled, valid = fitted.transform(test)
    assert fitted.state() == state
    assert np.isfinite(scaled).all() and np.max(np.abs(scaled)) <= 10
    assert scaled[0, 1, 0] == 0 and not valid[0, 1, 0]
    assert not valid[np.arange(3), np.arange(3)].any()
    assert scaled[1, 0, 0] != scaled[0, 1, 0]


def test_selection_is_refit_on_each_training_budget():
    small = tensor()
    small[:, :, 1] = 1
    fitted = SPIPreprocessor(["a", "b", "c"]).fit([small])
    assert 1 not in fitted.indices_
    fitted.fit([small, tensor()])
    assert 1 in fitted.indices_
    with pytest.raises(ValueError, match="No valid"):
        fitted.fit([np.ones((3, 3, 3))])
    with pytest.raises(RuntimeError):
        fitted.transform(tensor())


def test_permutation_equivariance_and_variable_system_size():
    fitted = SPIPreprocessor(["a", "b", "c"]).fit([tensor()])
    perm = [2, 0, 1]
    original, mask = fitted.transform(tensor())
    transformed, perm_mask = fitted.transform(tensor()[perm][:, perm])
    np.testing.assert_allclose(transformed, original[perm][:, perm])
    np.testing.assert_array_equal(perm_mask, mask[perm][:, perm])
    assert fitted.transform(np.ones((5, 5, 3)))[0].shape == (5, 5, 3)


def records():
    return [
        SplitRecord(str(i), f"group{i}", str(i % 2), split, f"{i:064x}")
        for i, split in enumerate(["train"] * 2 + ["validation"] * 2 + ["test"] * 2)
    ]


def test_valid_grouped_manifest():
    assert validate_classification_split(records())["n_groups"] == 6


@pytest.mark.parametrize("field,value", [("group_id", "group0"), ("source_sha256", "0" * 64)])
def test_cross_split_overlap_rejected(field, value):
    rows = records()
    rows[-1] = dataclasses.replace(rows[-1], **{field: value})
    with pytest.raises(ValueError, match="overlap"):
        validate_classification_split(rows)


def test_absent_groups_and_classes_are_not_silently_accepted():
    rows = records()
    with pytest.raises(ValueError, match="Every split"):
        validate_classification_split(rows[:-1])
    rows[0] = dataclasses.replace(rows[0], group_id="")
    with pytest.raises(ValueError, match="no fallback"):
        validate_classification_split(rows)


def identity():
    return ComputationIdentity(
        "3.0.0", "3.0.0.r9", "a" * 40, "b" * 64, "c" * 64, ("first", "second")
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("computation_version", "3.0.0.r7"),
        ("catalogue", ("second", "first")),
        ("recipe_sha256", "d" * 64),
    ],
)
def test_banks_cannot_mix_versions_catalogues_or_recipes(field, value):
    first = identity()
    require_same_computation([first, first])
    with pytest.raises(ValueError, match="Mixed"):
        require_same_computation([first, dataclasses.replace(first, **{field: value})])


def test_unknown_identity_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        dataclasses.replace(identity(), computation_version="unknown")


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7])
def test_symmetric_lagged_statistics_distinguish_all_three_motifs(alpha):
    for motif in ("chain", "fork", "collider"):
        matrices = stationary_symmetric_lags(motif, alpha)
        np.testing.assert_allclose(matrices, matrices.transpose(0, 2, 1))
        lag0, lag2 = matrices[[0, 2]].sum(axis=(1, 2))
        assert (lag0 > 1e-10) == (motif == "fork")
        assert (lag2 > 1e-10) == (motif == "chain")


def test_empirical_counterexample_features_do_not_use_channel_identity():
    series = np.random.default_rng(17).normal(size=(100, 10))
    np.testing.assert_allclose(
        symmetric_lag_features(series), symmetric_lag_features(series[:, ::-1]), atol=1e-12
    )


@pytest.mark.parametrize("motif", ["chain", "fork", "collider"])
def test_vectorised_simulator_matches_recursive_var(motif):
    rng = np.random.default_rng(29)
    alpha = rng.uniform(0.3, 0.7)
    noise = rng.normal(0, 0.1, size=(220, 10))
    transition = np.diag([0.0] * 3 + [0.8] * 7)
    for source, target in MOTIFS[motif]:
        transition[target, source] = alpha
    reference = noise.copy()
    for t in range(1, 220):
        reference[t] += transition @ reference[t - 1]
    reference = reference[200:, rng.permutation(10)]
    reference = (reference - reference.mean(axis=0)) / reference.std(axis=0)
    np.testing.assert_allclose(simulate(29, motif, n_time=20), reference, atol=1e-12)
