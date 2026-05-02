import numpy as np
import pytest

from cplx_gmm import GaussianMixtureCplx


@pytest.mark.parametrize(
    ("covariance_type", "blocks"),
    [
        ("full", None),
        ("diag", None),
        ("spherical", None),
        ("circulant", None),
        ("block-circulant", (2, 2)),
        ("toeplitz", None),
        ("block-toeplitz", (2, 2)),
    ],
)
def test_fit_predict_for_covariance_types(covariance_type, blocks) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4)) + 1j * rng.normal(size=(80, 4))

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type=covariance_type,
        blocks=blocks,
        random_state=0,
        max_iter=50,
        init_params="random",
    )

    model.fit(X)

    labels = model.predict(X)
    proba = model.predict_proba(X)

    assert labels.shape == (80,)
    assert proba.shape == (80, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert model.means_.shape == (2, 4)
