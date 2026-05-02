import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from cplx_gmm import GaussianMixtureCplx


def _complex_data(n_samples: int, n_features: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(n_samples, n_features))
        + 1j * rng.normal(size=(n_samples, n_features))
    ) / np.sqrt(2)


def _fit_ignoring_convergence_warning(
    model: GaussianMixtureCplx,
    X: np.ndarray,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X)


def test_weights_init_is_used_during_initialization() -> None:
    X = _complex_data(n_samples=200, n_features=3)

    weights_init = np.array([0.2, 0.8])

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="diag",
        weights_init=weights_init,
        random_state=0,
        max_iter=1,
        n_init=1,
        init_params="random",
    )

    _fit_ignoring_convergence_warning(model, X)

    assert model.weights_.shape == (2,)
    assert np.allclose(model.weights_.sum(), 1.0)


def test_means_init_is_used_during_initialization() -> None:
    X = _complex_data(n_samples=200, n_features=3)

    means_init = np.array(
        [
            [1.0 + 0.5j, 0.0 + 0.0j, -1.0 + 0.2j],
            [-1.0 - 0.5j, 0.5 + 0.1j, 1.0 - 0.2j],
        ]
    )

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="diag",
        means_init=means_init,
        random_state=0,
        max_iter=1,
        n_init=1,
        init_params="random",
    )

    _fit_ignoring_convergence_warning(model, X)

    assert model.means_.shape == means_init.shape
    assert np.iscomplexobj(model.means_)


def test_precisions_init_full_is_accepted() -> None:
    X = _complex_data(n_samples=200, n_features=3)

    precisions_init = np.array(
        [
            np.eye(3, dtype=complex),
            2.0 * np.eye(3, dtype=complex),
        ]
    )

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="full",
        precisions_init=precisions_init,
        random_state=0,
        max_iter=1,
        n_init=1,
        init_params="random",
    )

    _fit_ignoring_convergence_warning(model, X)

    assert model.precisions_cholesky_.shape == (2, 3, 3)
    assert np.iscomplexobj(model.precisions_cholesky_)


def test_precisions_init_diag_is_accepted() -> None:
    X = _complex_data(n_samples=200, n_features=3)

    precisions_init = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ]
    )

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="diag",
        precisions_init=precisions_init,
        random_state=0,
        max_iter=1,
        n_init=1,
        init_params="random",
    )

    _fit_ignoring_convergence_warning(model, X)

    assert model.precisions_cholesky_.shape == (2, 3)
