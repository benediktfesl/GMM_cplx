"""Tests for warm-start behavior."""

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from cplx_gmm import GaussianMixtureCplx


@pytest.mark.parametrize("covariance_type", ["full", "diag", "spherical"])
def test_warm_start_reuses_previous_solution(covariance_type: str) -> None:
    """Test that warm_start continues from the previous fitted solution."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3)) + 1j * rng.normal(size=(200, 3))

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type=covariance_type,
        warm_start=True,
        random_state=0,
        max_iter=2,
        n_init=3,
        init_params="random",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)

        model.fit(x)
        first_lower_bound = model.lower_bound_
        first_weights = model.weights_.copy()
        first_means = model.means_.copy()
        first_covariances = model.covariances_.copy()

        model.fit(x)
        second_lower_bound = model.lower_bound_

    assert second_lower_bound >= first_lower_bound - 1e-10
    assert model.n_iter_ >= 1
    assert model.weights_.shape == first_weights.shape
    assert model.means_.shape == first_means.shape
    assert model.covariances_.shape == first_covariances.shape
    assert np.isfinite(model.lower_bound_)
    assert np.all(np.isfinite(model.weights_))
    assert np.all(np.isfinite(model.means_))
    assert np.all(np.isfinite(model.covariances_))


@pytest.mark.parametrize(
    "covariance_type",
    ["circulant", "block-circulant", "toeplitz", "block-toeplitz"],
)
def test_warm_start_rejected_for_structured_covariance_types(
    covariance_type: str,
) -> None:
    """Test that warm_start is rejected for structured covariance types."""
    rng = np.random.default_rng(0)

    blocks = (
        (2, 3) if covariance_type in {"block-circulant", "block-toeplitz"} else None
    )
    x = rng.normal(size=(40, 6)) + 1j * rng.normal(size=(40, 6))

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type=covariance_type,
        blocks=blocks,
        warm_start=True,
        random_state=0,
        max_iter=20,
        init_params="random",
    )

    with pytest.raises(ValueError, match="warm_start=True"):
        model.fit(x)

    assert not hasattr(model, "weights_")
    assert not hasattr(model, "means_")
    assert not hasattr(model, "covariances_")
