import numpy as np

from cplx_gmm import GaussianMixtureCplx


def test_sample_shapes() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2)) + 1j * rng.normal(size=(100, 2))

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="full",
        random_state=0,
        max_iter=50,
    )
    model.fit(X)

    X_sampled, labels = model.sample(10)

    assert X_sampled.shape == (10, 2)
    assert labels.shape == (10,)
    assert np.iscomplexobj(X_sampled)