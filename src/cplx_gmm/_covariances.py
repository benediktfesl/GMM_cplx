"""Covariance utilities for complex-valued Gaussian mixture models."""

import numpy as np
from scipy import linalg as scilinalg


def compute_precision_cholesky(covariances, covariance_type):
    """Compute the Cholesky decomposition of the precisions.

    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    """
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar."
    )

    if covariance_type == "full":
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty(
            (n_components, n_features, n_features), dtype=complex
        )
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = scilinalg.cholesky(covariance, lower=True)
            except scilinalg.LinAlgError as exc:
                raise ValueError(estimate_precision_error_message) from exc
            precisions_chol[k] = scilinalg.solve_triangular(
                cov_chol, np.eye(n_features), lower=True
            ).T.conj()
    else:
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1.0 / np.sqrt(covariances).conj()
    return precisions_chol


def compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.

    Parameters
    ----------
    matrix_chol : array-like
        Cholesky decompositions of the matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
    n_features : int
        Number of features.

    Returns
    -------
    log_det_precision_chol : array-like of shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    if covariance_type == "full":
        n_components, _, _ = matrix_chol.shape
        log_det_chol = np.sum(
            np.log(matrix_chol.reshape(n_components, -1)[:, :: n_features + 1]), 1
        )
    elif covariance_type == "diag":
        log_det_chol = np.sum(np.log(matrix_chol), axis=1)
    else:
        log_det_chol = n_features * (np.log(matrix_chol))
    return log_det_chol


def estimate_gaussian_parameters(
    X,
    resp,
    reg_covar,
    covariance_type,
    zero_mean=False,
    covariance_inv=None,
    fourier_embedding=None,
    sigma=None,
):
    """Estimate the Gaussian distribution parameters.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data array.

    resp : array-like of shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.

    covariance_type : {'full', 'diag', 'spherical', 'inv-em'}
        The type of covariance estimator.

    zero_mean : bool, default=False
        If True, enforce zero component means.

    covariance_inv : array-like, optional
        Inverse covariance matrices from the previous EM iteration. Required
        for ``covariance_type='inv-em'``.

    fourier_embedding : array-like, optional
        Fourier-domain embedding matrix. Required for
        ``covariance_type='inv-em'``.

    sigma : array-like, optional
        Current diagonal spectral covariance parameters. Required for
        ``covariance_type='inv-em'``.

    Returns
    -------
    nk : array-like of shape (n_components,)
        The numbers of data samples in the current components.

    means : array-like of shape (n_components, n_features)
        The centers of the current components.

    covariances : array-like
        The covariance matrix of the current components.
        The shape depends on ``covariance_type``.

    sigma : array-like or None
        The updated diagonal spectral covariance parameters for
        ``covariance_type='inv-em'``. Otherwise unchanged.
    """
    nk = np.real(resp).sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]

    if zero_mean:
        means = np.zeros_like(means)

    if covariance_type == "full":
        covariances = estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar)
    elif covariance_type == "diag":
        covariances = estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar)
    elif covariance_type == "spherical":
        covariances = estimate_gaussian_covariances_spherical(
            resp, X, nk, means, reg_covar
        )
    elif covariance_type == "inv-em":
        if covariance_inv is None or fourier_embedding is None or sigma is None:
            raise ValueError(
                "covariance_inv, fourier_embedding, and sigma must be provided "
                "for covariance_type='inv-em'."
            )
        covariances, sigma = estimate_gaussian_covariances_inv(
            resp,
            X,
            nk,
            means,
            reg_covar,
            covariance_inv,
            fourier_embedding,
            sigma,
        )
    else:
        raise ValueError(f"Unsupported covariance_type={covariance_type!r}.")

    return nk, means, covariances, sigma


def estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
    """Estimate the full covariance matrices.

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
    """
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features), dtype=complex)

    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff.conj()) / nk[k]
        covariances[k].flat[:: n_features + 1] += reg_covar

    return covariances


def estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar):
    """Estimate diagonal complex covariance vectors."""
    avg_X2 = np.dot(resp.T, X * X.conj()) / nk[:, np.newaxis]
    avg_means2 = np.abs(means) ** 2
    avg_X_means = means.conj() * np.dot(resp.T, X) / nk[:, np.newaxis]

    return np.real(avg_X2 - 2.0 * np.real(avg_X_means) + avg_means2) + reg_covar + 0j


def estimate_gaussian_covariances_spherical(resp, X, nk, means, reg_covar):
    """Estimate spherical complex covariance values."""
    diagonal_covariances = estimate_gaussian_covariances_diag(
        resp,
        X,
        nk,
        means,
        reg_covar,
    )
    return np.real(diagonal_covariances.mean(1)) + 0j


def estimate_gaussian_covariances_inv(
    resp,
    X,
    nk,
    means,
    reg_covar,
    covariance_inv,
    fourier_embedding,
    sigma,
):
    """Estimate the Toeplitz-structured covariance matrices.

    Uses the EM-based inverse covariance update from:

    T. A. Barton and D. R. Fuhrmann, "Covariance estimation for
    multidimensional data using the EM algorithm," Proceedings of the 27th
    Asilomar Conference on Signals, Systems and Computers, 1993,
    pp. 203-207 vol. 1.

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    covariance_inv : array-like of shape (n_components, n_features, n_features)
        The inverse covariance matrices from the previous EM iteration.

    fourier_embedding : array-like
        The Fourier-domain embedding matrix used for the structured covariance
        update.

    sigma : array-like of shape (n_components, n_embedding_features)
        The current diagonal spectral covariance parameters.

    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.

    sigma : array-like of shape (n_components, n_embedding_features)
        The updated diagonal spectral covariance parameters.
    """
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features), dtype=complex)

    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff.conj()) / nk[k]

        theta = np.real(
            fourier_embedding
            @ (
                covariance_inv[k] @ covariances[k] @ covariance_inv[k]
                - covariance_inv[k]
            )
            @ fourier_embedding.conj().T
        )

        sigma[k] = sigma[k] + np.diag(sigma[k] * theta * sigma[k])
        sigma[k][sigma[k] < reg_covar] = reg_covar

        covariances[k] = np.multiply(fourier_embedding.conj().T, sigma[k]) @ (
            fourier_embedding
        )
        covariances[k].flat[:: n_features + 1] += reg_covar

    return covariances, sigma
