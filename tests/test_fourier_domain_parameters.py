"""Tests for retained Fourier-domain fitted parameters."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from cplx_gmm import GaussianMixtureCplx
from cplx_gmm._transforms import (
    diagonal_fft2_parameters_to_full_covariance,
    diagonal_fft_parameters_to_full_covariance,
)


def test_circulant_fit_stores_fft_domain_parameters() -> None:
    """Test that circulant fitting retains diagonal FFT-domain parameters."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 5)) + 1j * rng.normal(size=(40, 5))

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="circulant",
        random_state=0,
        max_iter=50,
        init_params="random",
    )
    model.fit(x)

    assert model.means_fft_.shape == (2, 5)
    assert model.covariances_fft_.shape == (2, 5)
    assert np.iscomplexobj(model.means_fft_)
    assert np.all(np.real(model.covariances_fft_) > 0)
    assert np.allclose(np.imag(model.covariances_fft_), 0.0)

    means, covariances = diagonal_fft_parameters_to_full_covariance(
        model.means_fft_,
        model.covariances_fft_,
    )
    np.testing.assert_allclose(model.means_, means)
    np.testing.assert_allclose(model.covariances_, covariances)


def test_block_circulant_fit_stores_fft2_domain_parameters() -> None:
    """Test that block-circulant fitting retains diagonal 2D FFT parameters."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 6)) + 1j * rng.normal(size=(50, 6))
    blocks = (2, 3)

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="block-circulant",
        blocks=blocks,
        random_state=0,
        max_iter=50,
        init_params="random",
    )
    model.fit(x)

    assert model.means_fft2_.shape == (2, 6)
    assert model.covariances_fft2_.shape == (2, 6)
    assert np.iscomplexobj(model.means_fft2_)
    assert np.all(np.real(model.covariances_fft2_) > 0)
    assert np.allclose(np.imag(model.covariances_fft2_), 0.0)

    means, covariances = diagonal_fft2_parameters_to_full_covariance(
        model.means_fft2_,
        model.covariances_fft2_,
        *blocks,
    )
    np.testing.assert_allclose(model.means_, means)
    np.testing.assert_allclose(model.covariances_, covariances)


def test_circulant_fit_predict_stores_fft_domain_parameters() -> None:
    """Test that public fit_predict keeps circulant FFT-domain parameters."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 5)) + 1j * rng.normal(size=(40, 5))

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="circulant",
        random_state=0,
        max_iter=50,
        init_params="random",
    )

    labels = model.fit_predict(x)

    assert labels.shape == (40,)
    assert hasattr(model, "means_fft_")
    assert hasattr(model, "covariances_fft_")
    assert model.means_fft_.shape == (2, 5)
    assert model.covariances_fft_.shape == (2, 5)
    assert model.means_.shape == (2, 5)
    assert model.covariances_.shape == (2, 5, 5)


def test_block_circulant_fit_predict_stores_fft2_domain_parameters() -> None:
    """Test that public fit_predict keeps block-circulant FFT-domain parameters."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 6)) + 1j * rng.normal(size=(50, 6))

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="block-circulant",
        blocks=(2, 3),
        random_state=0,
        max_iter=50,
        init_params="random",
    )

    labels = model.fit_predict(x)

    assert labels.shape == (50,)
    assert hasattr(model, "means_fft2_")
    assert hasattr(model, "covariances_fft2_")
    assert model.means_fft2_.shape == (2, 6)
    assert model.covariances_fft2_.shape == (2, 6)
    assert model.means_.shape == (2, 6)
    assert model.covariances_.shape == (2, 6, 6)


def test_fft_domain_properties_return_copies() -> None:
    """Test that FFT-domain convenience properties return copies."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 5)) + 1j * rng.normal(size=(40, 5))

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="circulant",
        random_state=0,
        max_iter=50,
        init_params="random",
    )
    model.fit(x)

    means_fft = model.means_fft
    covariances_fft = model.covariances_fft

    means_fft[0, 0] = 123.0 + 456.0j
    covariances_fft[0, 0] = 789.0

    assert not np.allclose(means_fft, model.means_fft_)
    assert not np.allclose(covariances_fft, model.covariances_fft_)


def test_fft2_domain_properties_return_copies() -> None:
    """Test that 2D FFT-domain convenience properties return copies."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 6)) + 1j * rng.normal(size=(50, 6))

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="block-circulant",
        blocks=(2, 3),
        random_state=0,
        max_iter=50,
        init_params="random",
    )
    model.fit(x)

    means_fft2 = model.means_fft2
    covariances_fft2 = model.covariances_fft2

    means_fft2[0, 0] = 123.0 + 456.0j
    covariances_fft2[0, 0] = 789.0

    assert not np.allclose(means_fft2, model.means_fft2_)
    assert not np.allclose(covariances_fft2, model.covariances_fft2_)


def test_fft_domain_properties_require_matching_fit_type() -> None:
    """Test that FFT-domain properties are unavailable for non-FFT fits."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 5)) + 1j * rng.normal(size=(40, 5))

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="full",
        random_state=0,
        max_iter=50,
        init_params="random",
    )
    model.fit(x)

    with pytest.raises(NotFittedError):
        _ = model.means_fft

    with pytest.raises(NotFittedError):
        _ = model.covariances_fft

    with pytest.raises(NotFittedError):
        _ = model.means_fft2

    with pytest.raises(NotFittedError):
        _ = model.covariances_fft2


def test_refit_clears_stale_fourier_domain_parameters() -> None:
    """Test that refitting with a non-FFT covariance type clears stale attributes."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 5)) + 1j * rng.normal(size=(40, 5))

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="circulant",
        random_state=0,
        max_iter=50,
        init_params="random",
    )
    model.fit(x)

    assert hasattr(model, "means_fft_")
    assert hasattr(model, "covariances_fft_")

    model.covariance_type = "full"
    model.fit(x)

    assert not hasattr(model, "means_fft_")
    assert not hasattr(model, "covariances_fft_")
    assert not hasattr(model, "means_fft2_")
    assert not hasattr(model, "covariances_fft2_")


@pytest.mark.parametrize(
    "covariance_type",
    ["circulant", "block-circulant", "toeplitz", "block-toeplitz"],
)
def test_warm_start_rejected_for_structured_covariance_types(
    covariance_type: str,
) -> None:
    """Test that warm_start is rejected for structured covariance types."""
    rng = np.random.default_rng(0)

    blocks = (
        (2, 3) if covariance_type in {"block-circulant", "block-toeplitz"} else None
    )
    x = rng.normal(size=(40, 6)) + 1j * rng.normal(size=(40, 6))

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type=covariance_type,
        blocks=blocks,
        warm_start=True,
        random_state=0,
        max_iter=50,
        init_params="random",
    )

    with pytest.raises(ValueError, match="warm_start=True"):
        model.fit(x)
