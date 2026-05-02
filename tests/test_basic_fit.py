import numpy as np

from cplx_gmm import GaussianMixtureCplx


def test_fit_predict_proba_shapes() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2)) + 1j * rng.normal(size=(100, 2))

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="full",
        random_state=0,
        max_iter=50,
    )

    model.fit(X)

    labels = model.predict(X)
    proba = model.predict_proba(X)

    assert labels.shape == (100,)
    assert proba.shape == (100, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
