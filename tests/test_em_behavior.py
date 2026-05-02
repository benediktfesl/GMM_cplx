import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from cplx_gmm import GaussianMixtureCplx


def _complex_data(n_samples: int, n_features: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(n_samples, n_features))
        + 1j * rng.normal(size=(n_samples, n_features))
    ) / np.sqrt(2)


def _fit_and_collect_lower_bounds(
    model: GaussianMixtureCplx,
    X: np.ndarray,
) -> list[float]:
    lower_bounds = []

    original_m_step = model._m_step

    def wrapped_m_step(X_step, log_resp):
        original_m_step(X_step, log_resp)
        log_prob_norm, _ = model._estimate_log_prob_resp(X_step)
        lower_bounds.append(float(np.mean(log_prob_norm)))

    model._m_step = wrapped_m_step

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X)

    return lower_bounds


@pytest.mark.parametrize(
    ("covariance_type", "blocks", "n_features"),
    [
        ("full", None, 5),
        ("diag", None, 5),
        ("spherical", None, 5),
        ("circulant", None, 5),
        ("block-circulant", (3, 3), 9),
        ("toeplitz", None, 5),
        ("block-toeplitz", (3, 3), 9),
    ],
)
def test_em_lower_bound_is_monotonic_for_covariance_types(
    covariance_type: str,
    blocks: tuple[int, int] | None,
    n_features: int,
) -> None:
    X = _complex_data(n_samples=300, n_features=n_features)

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type=covariance_type,
        blocks=blocks,
        random_state=0,
        max_iter=20,
        n_init=1,
        init_params="random",
    )

    lower_bounds = _fit_and_collect_lower_bounds(model, X)

    assert len(lower_bounds) >= 2

    differences = np.diff(lower_bounds)
    assert np.all(differences >= -1e-8)


@pytest.mark.parametrize(
    ("covariance_type", "blocks", "n_features"),
    [
        ("full", None, 5),
        ("diag", None, 5),
        ("spherical", None, 5),
        ("circulant", None, 5),
        ("block-circulant", (3, 3), 9),
        ("toeplitz", None, 5),
        ("block-toeplitz", (3, 3), 9),
    ],
)
def test_zero_mean_enforces_zero_component_means(
    covariance_type: str,
    blocks: tuple[int, int] | None,
    n_features: int,
) -> None:
    X = _complex_data(n_samples=300, n_features=n_features)

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type=covariance_type,
        blocks=blocks,
        zero_mean=True,
        random_state=0,
        max_iter=50,
        n_init=1,
        init_params="random",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X)

    assert model.means_.shape == (2, n_features)
    assert np.allclose(model.means_, 0.0, atol=1e-10)

    labels = model.predict(X)
    proba = model.predict_proba(X)

    assert labels.shape == (300,)
    assert proba.shape == (300, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)