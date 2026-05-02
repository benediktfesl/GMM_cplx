import numpy as np
import pytest

from cplx_gmm import GaussianMixtureCplx


def _complex_data(n_samples: int, n_features: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(n_samples, n_features))
        + 1j * rng.normal(size=(n_samples, n_features))
    ) / np.sqrt(2)


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
def test_weights_sum_to_one_after_fit(
    covariance_type: str,
    blocks: tuple[int, int] | None,
    n_features: int,
) -> None:
    X = _complex_data(n_samples=300, n_features=n_features)

    model = GaussianMixtureCplx(
        n_components=3,
        covariance_type=covariance_type,
        blocks=blocks,
        random_state=0,
        max_iter=100,
        n_init=1,
        init_params="random",
    )

    model.fit(X)

    assert model.weights_.shape == (3,)
    assert np.all(model.weights_ >= 0.0)
    assert np.allclose(model.weights_.sum(), 1.0, atol=1e-12)
