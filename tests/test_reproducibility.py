import numpy as np

from cplx_gmm import GaussianMixtureCplx


def _complex_data(n_samples: int, n_features: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(n_samples, n_features))
        + 1j * rng.normal(size=(n_samples, n_features))
    ) / np.sqrt(2)


def test_same_random_state_produces_same_fit() -> None:
    X = _complex_data(n_samples=300, n_features=4)

    kwargs = dict(
        n_components=2,
        covariance_type="full",
        random_state=42,
        max_iter=20,
        n_init=1,
        init_params="random",
    )

    model_1 = GaussianMixtureCplx(**kwargs)
    model_2 = GaussianMixtureCplx(**kwargs)

    model_1.fit(X)
    model_2.fit(X)

    assert np.allclose(model_1.weights_, model_2.weights_)
    assert np.allclose(model_1.means_, model_2.means_)
    assert np.allclose(model_1.covariances_, model_2.covariances_)
    assert np.allclose(model_1.lower_bound_, model_2.lower_bound_)
