import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from cplx_gmm import GaussianMixtureCplx


def test_invalid_covariance_type_raises() -> None:
    X = np.ones((10, 2), dtype=complex)
    model = GaussianMixtureCplx(covariance_type="invalid")

    with pytest.raises(ValueError, match="Invalid covariance_type"):
        model.fit(X)


def test_block_circulant_requires_blocks() -> None:
    X = np.ones((10, 4), dtype=complex)
    model = GaussianMixtureCplx(covariance_type="block-circulant")

    with pytest.raises(ValueError, match="blocks must be provided"):
        model.fit(X)


def test_block_shape_must_match_n_features() -> None:
    X = np.ones((10, 5), dtype=complex)
    model = GaussianMixtureCplx(covariance_type="block-circulant", blocks=(2, 2))

    with pytest.raises(ValueError, match="n_1 \\* n_2 == n_features"):
        model.fit(X)


@pytest.mark.parametrize(
    "attribute",
    [
        "covariances",
        "converged",
        "means",
        "precisions",
        "precisions_cholesky",
        "weights",
    ],
)
def test_fitted_properties_raise_before_fit(attribute: str) -> None:
    model = GaussianMixtureCplx()

    with pytest.raises(NotFittedError):
        getattr(model, attribute)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_components": 0}, "n_components must be >= 1"),
        ({"tol": -1.0}, "tol must be non-negative"),
        ({"reg_covar": -1.0}, "reg_covar must be non-negative"),
        ({"max_iter": 0}, "max_iter must be >= 1"),
        ({"n_init": 0}, "n_init must be >= 1"),
        ({"init_params": "invalid"}, "init_params must be"),
    ],
)
def test_invalid_hyperparameters_raise(kwargs: dict, match: str) -> None:
    X = np.ones((10, 2), dtype=complex)
    model = GaussianMixtureCplx(**kwargs)

    with pytest.raises(ValueError, match=match):
        model.fit(X)


def test_one_dimensional_input_raises() -> None:
    X = np.ones(10, dtype=complex)
    model = GaussianMixtureCplx()

    with pytest.raises(ValueError, match="Expected X to be a 2D array"):
        model.fit(X)


def test_too_few_samples_raise() -> None:
    X = np.ones((2, 3), dtype=complex)
    model = GaussianMixtureCplx(n_components=3)

    with pytest.raises(ValueError, match="Expected n_samples >= n_components"):
        model.fit(X)
