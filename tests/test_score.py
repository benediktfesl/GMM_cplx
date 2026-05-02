import numpy as np

from cplx_gmm import GaussianMixtureCplx


def test_score_is_mean_score_samples() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3)) + 1j * rng.normal(size=(100, 3))

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="full",
        random_state=0,
        max_iter=50,
        init_params="random",
    )
    model.fit(X)

    assert np.allclose(model.score(X), model.score_samples(X).mean())
