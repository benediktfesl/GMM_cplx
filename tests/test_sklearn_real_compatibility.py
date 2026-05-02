import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

from cplx_gmm import GaussianMixtureCplx


def _real_data(n_samples: int, n_features: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_samples, n_features))


@pytest.mark.parametrize("covariance_type", ["full", "diag", "spherical"])
def test_single_component_real_valued_fit_matches_sklearn_parameters(
    covariance_type: str,
) -> None:
    X = _real_data(n_samples=300, n_features=5)

    model = GaussianMixtureCplx(
        n_components=1,
        covariance_type=covariance_type,
        random_state=0,
        max_iter=20,
        n_init=1,
        init_params="random",
    )
    sklearn_model = GaussianMixture(
        n_components=1,
        covariance_type=covariance_type,
        random_state=0,
        max_iter=20,
        n_init=1,
        init_params="random",
    )

    model.fit(X)
    sklearn_model.fit(X)

    assert np.allclose(model.weights_, sklearn_model.weights_, atol=1e-12)
    assert np.allclose(np.real(model.means_), sklearn_model.means_, atol=1e-12)
    assert np.allclose(np.imag(model.means_), 0.0, atol=1e-12)

    assert np.allclose(
        np.real(model.covariances_),
        sklearn_model.covariances_,
        atol=1e-10,
    )
    assert np.allclose(np.imag(model.covariances_), 0.0, atol=1e-12)


@pytest.mark.parametrize("covariance_type", ["full", "diag", "spherical"])
def test_single_component_real_valued_log_likelihood_matches_sklearn_transform(
    covariance_type: str,
) -> None:
    X = _real_data(n_samples=300, n_features=5)

    model = GaussianMixtureCplx(
        n_components=1,
        covariance_type=covariance_type,
        random_state=0,
        max_iter=20,
        n_init=1,
        init_params="random",
    )
    sklearn_model = GaussianMixture(
        n_components=1,
        covariance_type=covariance_type,
        random_state=0,
        max_iter=20,
        n_init=1,
        init_params="random",
    )

    model.fit(X)
    sklearn_model.fit(X)

    n_features = X.shape[1]
    expected_complex_scores = 2.0 * sklearn_model.score_samples(
        X
    ) + n_features * np.log(2.0)

    assert np.allclose(
        model.score_samples(X),
        expected_complex_scores,
        atol=1e-10,
    )


@pytest.mark.parametrize("covariance_type", ["full", "diag", "spherical"])
def test_multi_component_real_valued_fit_matches_sklearn_api_shapes(
    covariance_type: str,
) -> None:
    X = _real_data(n_samples=300, n_features=5)

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type=covariance_type,
        random_state=0,
        max_iter=50,
        n_init=1,
        init_params="random",
    )
    sklearn_model = GaussianMixture(
        n_components=2,
        covariance_type=covariance_type,
        random_state=0,
        max_iter=50,
        n_init=1,
        init_params="random",
    )

    model.fit(X)
    sklearn_model.fit(X)

    assert model.weights_.shape == sklearn_model.weights_.shape
    assert model.means_.shape == sklearn_model.means_.shape
    assert model.covariances_.shape == sklearn_model.covariances_.shape

    assert model.predict(X).shape == sklearn_model.predict(X).shape
    assert model.predict_proba(X).shape == sklearn_model.predict_proba(X).shape
    assert model.score_samples(X).shape == sklearn_model.score_samples(X).shape

    assert np.allclose(np.imag(model.means_), 0.0, atol=1e-12)
    assert np.allclose(np.imag(model.covariances_), 0.0, atol=1e-12)
    assert np.allclose(model.weights_.sum(), 1.0)
    assert np.all(model.weights_ > 0)
