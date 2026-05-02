"""Structured covariance transforms for complex-valued Gaussian mixtures."""

import numpy as np
from scipy import linalg as scilinalg


def diagonal_fft_parameters_to_full_covariance(means_dft, covariances_dft):
    """Transform diagonal FFT-domain parameters to full covariance form.

    Parameters
    ----------
    means_dft : array-like of shape (n_components, n_features)
        Component means in the one-dimensional FFT domain.

    covariances_dft : array-like of shape (n_components, n_features)
        Diagonal covariance parameters in the one-dimensional FFT domain.

    Returns
    -------
    means : array-like of shape (n_components, n_features)
        Component means in the original domain.

    covariances : array-like of shape (n_components, n_features, n_features)
        Full circulant covariance matrices in the original domain.
    """
    n_components, n_features = means_dft.shape

    means = np.fft.ifft(means_dft, axis=1) * np.sqrt(n_features)

    covariances = np.empty(
        (n_components, n_features, n_features),
        dtype=complex,
    )

    for k in range(n_components):
        first_col = np.fft.ifft(covariances_dft[k])
        covariances[k] = scilinalg.circulant(first_col)

    return means, covariances


def diagonal_fft2_parameters_to_full_covariance(means_dft, covariances_dft, n_1, n_2):
    """Transform diagonal 2D FFT-domain parameters to full covariance form.

    Parameters
    ----------
    means_dft : array-like of shape (n_components, n_1 * n_2)
        Component means in the two-dimensional FFT domain.

    covariances_dft : array-like of shape (n_components, n_1 * n_2)
        Diagonal covariance parameters in the two-dimensional FFT domain.

    n_1 : int
        Number of blocks in the block-circulant structure.

    n_2 : int
        Block size in the block-circulant structure.

    Returns
    -------
    means : array-like of shape (n_components, n_1 * n_2)
        Component means in the original domain.

    covariances : array-like of shape (n_components, n_1 * n_2, n_1 * n_2)
        Full block-circulant covariance matrices in the original domain.
    """
    n_components = means_dft.shape[0]
    n_features = n_1 * n_2

    means_grid = means_dft.reshape(n_components, n_1, n_2)
    means = (np.fft.ifft2(means_grid, axes=(1, 2)) * np.sqrt(n_features)).reshape(
        n_components, n_features
    )

    covariances = np.empty(
        (n_components, n_features, n_features),
        dtype=complex,
    )

    eye = np.eye(n_features, dtype=complex).reshape(n_features, n_1, n_2)

    for k in range(n_components):
        spectral_variances = covariances_dft[k].reshape(n_1, n_2)

        cov_columns = (
            np.fft.ifft2(
                spectral_variances[np.newaxis, :, :] * np.fft.fft2(eye, axes=(1, 2)),
                axes=(1, 2),
            )
        ).reshape(n_features, n_features)

        covariances[k] = cov_columns.T

    return means, covariances
