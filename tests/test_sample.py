import numpy as np

from cplx_gmm import GaussianMixtureCplx


def _complex_data(n_samples: int, n_features: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(n_samples, n_features))
        + 1j * rng.normal(size=(n_samples, n_features))
    ) / np.sqrt(2)


def test_sample_shapes() -> None:
    X = _complex_data(n_samples=200, n_features=3)

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="full",
        random_state=0,
        max_iter=100,
        init_params="random",
    )
    model.fit(X)

    X_sampled, labels = model.sample(20)

    assert X_sampled.shape == (20, 3)
    assert labels.shape == (20,)
    assert np.iscomplexobj(X_sampled)
    assert set(labels).issubset({0, 1})


def test_sample_component_frequencies_match_weights_approximately() -> None:
    X = _complex_data(n_samples=500, n_features=3)

    model = GaussianMixtureCplx(
        n_components=3,
        covariance_type="diag",
        random_state=0,
        max_iter=100,
        init_params="random",
    )
    model.fit(X)

    X_sampled, labels = model.sample(5000)

    observed_weights = np.bincount(labels, minlength=model.n_components) / labels.size

    assert X_sampled.shape == (5000, 3)
    assert np.allclose(observed_weights, model.weights_, atol=0.04)


def test_one_component_sample_mean_and_covariance_match_fitted_parameters() -> None:
    X = _complex_data(n_samples=1000, n_features=3)

    model = GaussianMixtureCplx(
        n_components=1,
        covariance_type="full",
        random_state=0,
        max_iter=100,
        init_params="random",
    )
    model.fit(X)

    X_sampled, labels = model.sample(10000)

    empirical_mean = X_sampled.mean(axis=0)
    centered = X_sampled - empirical_mean
    empirical_covariance = centered.T @ centered.conj() / X_sampled.shape[0]

    assert labels.shape == (10000,)
    assert np.all(labels == 0)
    assert np.allclose(empirical_mean, model.means_[0], atol=0.05)
    assert np.allclose(empirical_covariance, model.covariances_[0], atol=0.08)