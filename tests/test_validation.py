import numpy as np
import pytest

from cplx_gmm import GaussianMixtureCplx


def test_invalid_covariance_type_raises() -> None:
    X = np.ones((10, 2), dtype=complex)
    model = GaussianMixtureCplx(covariance_type="invalid")

    with pytest.raises(ValueError, match="Invalid covariance_type"):
        model.fit(X)


def test_block_circulant_requires_blocks() -> None:
    X = np.ones((10, 4), dtype=complex)
    model = GaussianMixtureCplx(covariance_type="block-circulant")

    with pytest.raises(ValueError, match="blocks must be provided"):
        model.fit(X)


def test_block_shape_must_match_n_features() -> None:
    X = np.ones((10, 5), dtype=complex)
    model = GaussianMixtureCplx(covariance_type="block-circulant", blocks=(2, 2))

    with pytest.raises(ValueError, match="n_1 \\* n_2 == n_features"):
        model.fit(X)
