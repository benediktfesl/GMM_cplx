# Original code from scikit-learn:
# Author: Wei Xue <xuewei4d@gmail.com>
# Modified by Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

# Extension to the complex-valued case with (block-)Toeplitz and
# (block-)circulant covariances:
# Author: Benedikt Fesl <benedikt.fesl@tum.de>
# License: BSD 3 clause

import warnings

import numpy as np
from scipy import linalg as scilinalg
from scipy.special import logsumexp
from sklearn import cluster
from sklearn.base import BaseEstimator, ClusterMixin, DensityMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_is_fitted

from . import _utils as ut


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


def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
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


class GaussianMixtureCplx(BaseEstimator, ClusterMixin, DensityMixin):
    """Complex-valued Gaussian mixture model.

    The model estimates mixtures of circularly symmetric complex Gaussian
    distributions using expectation-maximization.
    """

    _valid_covariance_types = {
        "full",
        "diag",
        "spherical",
        "circulant",
        "block-circulant",
        "toeplitz",
        "block-toeplitz",
    }

    def __init__(
        self,
        n_components=1,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
        zero_mean=False,
        blocks=None,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.zero_mean = zero_mean
        self.blocks = blocks

    def _validate_parameters(self, X):
        """Validate estimator hyperparameters against the input data."""
        if self.n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}.")

        if self.covariance_type not in self._valid_covariance_types:
            raise ValueError(
                f"Invalid covariance_type={self.covariance_type!r}. "
                f"Expected one of {sorted(self._valid_covariance_types)}."
            )

        if self.tol < 0:
            raise ValueError(f"tol must be non-negative, got {self.tol}.")

        if self.reg_covar < 0:
            raise ValueError(f"reg_covar must be non-negative, got {self.reg_covar}.")

        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}.")

        if self.n_init < 1:
            raise ValueError(f"n_init must be >= 1, got {self.n_init}.")

        if self.init_params not in {"kmeans", "random"}:
            raise ValueError(
                f"init_params must be 'kmeans' or 'random', got {self.init_params!r}."
            )

        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}."
            )

    def _check_X(self, X, reset):
        """Convert input data to a valid 2D complex-valued array."""
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                f"Expected X to be a 2D array, got array with shape {X.shape}."
            )

        if not np.iscomplexobj(X):
            X = X.astype(complex)

        n_features = X.shape[1]

        if reset:
            self.n_features_in_ = n_features
        elif n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} features, but GaussianMixtureCplx "
                f"is expecting {self.n_features_in_} features."
            )

        return X

    def _verbose_msg_init_beg(self, init):
        """Print a message at the start of an EM initialization if enabled."""
        if self.verbose:
            print(f"Initialization {init + 1}")

    def _verbose_msg_iter_end(self, n_iter, change):
        """Print EM iteration progress if verbose output is enabled."""
        if self.verbose and n_iter % self.verbose_interval == 0:
            print(f"  Iteration {n_iter}, change {change:.6f}")

    def _verbose_msg_init_end(self, lower_bound):
        """Print a message after an EM initialization if enabled."""
        if self.verbose:
            print(f"Initialization converged with lower bound {lower_bound:.6f}")

    def _get_parameters(self):
        """Return the current fitted EM parameters as a tuple."""
        return (
            self.weights_,
            self.means_,
            self.covariances_,
            self.precisions_cholesky_,
        )

    def _effective_covariance_type(self):
        """Return the covariance type used for likelihood and precision evaluation."""
        if self._em_covariance_type == "inv-em":
            return "full"
        return self._em_covariance_type

    @property
    def covariances(self):
        """Return a copy of the fitted component covariance parameters."""
        check_is_fitted(self)
        return self.covariances_.copy()

    @property
    def converged(self):
        """Return whether the EM algorithm converged during fitting."""
        check_is_fitted(self)
        return self.converged_

    @property
    def means(self):
        """Return a copy of the fitted component means."""
        check_is_fitted(self)
        return self.means_.copy()

    @property
    def precisions(self):
        """Return a copy of the fitted precision parameters."""
        check_is_fitted(self)
        return self.precisions_.copy()

    @property
    def precisions_cholesky(self):
        """Return a copy of the fitted precision Cholesky factors."""
        check_is_fitted(self)
        return self.precisions_cholesky_.copy()

    @property
    def weights(self):
        """Return a copy of the fitted mixture weights."""
        check_is_fitted(self)
        return self.weights_.copy()

    def _fit_em(self, X, y=None):
        """Run the direct EM fit without covariance-structure preprocessing."""
        self.fit_predict(X, y=y)
        return self

    def fit(self, X, y=None):
        """Fit the complex-valued Gaussian mixture model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Complex-valued input samples.

        y : Ignored
            Present for sklearn API compatibility.

        Returns
        -------
        self : GaussianMixtureCplx
            Fitted estimator.
        """
        del y  # unused, kept for sklearn API compatibility

        X = self._check_X(X, reset=True)
        self._validate_parameters(X)

        requested_covariance_type = self.covariance_type

        self._em_covariance_type = requested_covariance_type
        self._zero_mean = self.zero_mean
        self._F2 = None
        self._sigma = None

        if requested_covariance_type in {"full", "diag", "spherical"}:
            self._em_covariance_type = requested_covariance_type
            self._fit_em(X)
            self._store_public_fit_attributes()

        elif requested_covariance_type == "circulant":
            n_features = X.shape[1]

            self._em_covariance_type = "diag"
            X_dft = np.fft.fft(X, axis=1) / np.sqrt(n_features)
            self._fit_em(X_dft)

            self._convert_diagonal_fft_fit_to_full_covariance()

        elif requested_covariance_type == "block-circulant":
            if self.blocks is None:
                raise ValueError(
                    "blocks must be provided for covariance_type='block-circulant'."
                )

            n_1, n_2 = self.blocks
            if n_1 * n_2 != X.shape[1]:
                raise ValueError(
                    "For covariance_type='block-circulant', blocks must satisfy "
                    f"n_1 * n_2 == n_features, got {n_1} * {n_2} != {X.shape[1]}."
                )

            self._em_covariance_type = "diag"

            X_grid = X.reshape(X.shape[0], n_1, n_2)
            X_dft = np.fft.fft2(X_grid, axes=(1, 2)) / np.sqrt(n_1 * n_2)
            X_dft = X_dft.reshape(X.shape[0], n_1 * n_2)

            self._fit_em(X_dft)

            self._convert_diagonal_fft2_fit_to_full_covariance(n_1, n_2)

        elif requested_covariance_type == "toeplitz":
            n_features = X.shape[1]

            self._em_covariance_type = "inv-em"
            self._F2 = np.fft.fft(np.eye(2 * n_features, dtype=complex))[
                :, :n_features
            ] / np.sqrt(2 * n_features)

            self._fit_em(X)
            self._store_public_fit_attributes()

        elif requested_covariance_type == "block-toeplitz":
            if self.blocks is None:
                raise ValueError(
                    "blocks must be provided for covariance_type='block-toeplitz'."
                )

            n_1, n_2 = self.blocks
            if n_1 * n_2 != X.shape[1]:
                raise ValueError(
                    "For covariance_type='block-toeplitz', blocks must satisfy "
                    f"n_1 * n_2 == n_features, got {n_1} * {n_2} != {X.shape[1]}."
                )

            F2_1 = np.fft.fft(np.eye(2 * n_1, dtype=complex))[:, :n_1] / np.sqrt(
                2 * n_1
            )
            F2_2 = np.fft.fft(np.eye(2 * n_2, dtype=complex))[:, :n_2] / np.sqrt(
                2 * n_2
            )

            self._em_covariance_type = "inv-em"
            self._F2 = np.kron(F2_1, F2_2)

            self._fit_em(X)
            self._store_public_fit_attributes()

        else:
            raise NotImplementedError(
                f"Fitting for covariance_type={requested_covariance_type!r} "
                "is not implemented."
            )

        return self

    def _store_public_fit_attributes(self):
        """Store backward-compatible aliases for fitted complex parameters."""
        self.means_cplx = self.means_.copy()
        self.covs_cplx = self.covariances_.copy()
        self.chol = self.precisions_cholesky_.copy()

    def _convert_diagonal_fft_fit_to_full_covariance(self):
        """Transform diagonal FFT-domain parameters to full covariance form."""
        means_dft = self.means_.copy()
        covariances_dft = self.covariances_.copy()

        n_components, n_features = means_dft.shape

        self.means_ = np.fft.ifft(means_dft, axis=1) * np.sqrt(n_features)

        self.covariances_ = np.empty(
            (n_components, n_features, n_features),
            dtype=complex,
        )

        for k in range(n_components):
            first_col = np.fft.ifft(covariances_dft[k])
            self.covariances_[k] = scilinalg.circulant(first_col)

        self.precisions_cholesky_ = compute_precision_cholesky(
            self.covariances_,
            covariance_type="full",
        )

        self.precisions_ = np.empty_like(self.precisions_cholesky_)
        for k, prec_chol in enumerate(self.precisions_cholesky_):
            self.precisions_[k] = prec_chol @ prec_chol.T.conj()

        self._em_covariance_type = "full"
        self._store_public_fit_attributes()

    def _convert_diagonal_fft2_fit_to_full_covariance(self, n_1, n_2):
        """Transform diagonal 2D FFT-domain parameters to full block-circulant form."""
        means_dft = self.means_.copy()
        covariances_dft = self.covariances_.copy()

        n_components = means_dft.shape[0]
        n_features = n_1 * n_2

        means_grid = means_dft.reshape(n_components, n_1, n_2)
        self.means_ = (
            np.fft.ifft2(means_grid, axes=(1, 2)) * np.sqrt(n_features)
        ).reshape(n_components, n_features)

        self.covariances_ = np.empty(
            (n_components, n_features, n_features),
            dtype=complex,
        )

        eye = np.eye(n_features, dtype=complex).reshape(n_features, n_1, n_2)

        for k in range(n_components):
            spectral_variances = covariances_dft[k].reshape(n_1, n_2)

            # Apply the block-circulant covariance operator to each basis vector.
            cov_columns = (
                np.fft.ifft2(
                    spectral_variances[np.newaxis, :, :]
                    * np.fft.fft2(eye, axes=(1, 2)),
                    axes=(1, 2),
                )
            ).reshape(n_features, n_features)

            self.covariances_[k] = cov_columns.T

        self.precisions_cholesky_ = compute_precision_cholesky(
            self.covariances_,
            covariance_type="full",
        )

        self.precisions_ = np.empty_like(self.precisions_cholesky_)
        for k, prec_chol in enumerate(self.precisions_cholesky_):
            self.precisions_[k] = prec_chol @ prec_chol.T.conj()

        self._em_covariance_type = "full"
        self._store_public_fit_attributes()

    def sample(self, n_samples=1):
        """Generate random samples from the fitted Gaussian distribution.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample.
        y : array, shape (nsamples,)
            Component labels.
        """
        check_is_fitted(self)

        if n_samples < 1:
            raise ValueError(
                f"Invalid value for 'n_samples': {n_samples}. The sampling requires "
                "at least one sample."
            )

        rng = ut.check_random_state(self.random_state)
        n_samples_comp = rng.multinomial(n_samples, self.weights_)

        X = np.vstack(
            [
                ut.multivariate_normal_cplx(
                    mean,
                    covariance,
                    int(sample),
                    self._effective_covariance_type(),
                )
                for mean, covariance, sample in zip(
                    self.means_cplx,
                    self.covs_cplx,
                    n_samples_comp,
                    strict=True,
                )
                if sample > 0
            ]
        )

        y = np.concatenate(
            [
                np.full(sample, component, dtype=int)
                for component, sample in enumerate(n_samples_comp)
                if sample > 0
            ]
        )

        return X, y

    def predict(self, X):
        """Predict component labels for complex-valued input samples."""
        check_is_fitted(self)
        X = self._check_X(X, reset=False)
        return self._estimate_weighted_log_prob(X).argmax(axis=1)

    def predict_proba(self, X):
        """Predict posterior component probabilities for input samples."""
        check_is_fitted(self)
        X = self._check_X(X, reset=False)
        _, log_resp = self._estimate_log_prob_resp(X)
        return np.exp(log_resp)

    def score_samples(self, X):
        """Compute per-sample log-likelihoods under the fitted model."""
        check_is_fitted(self)
        X = self._check_X(X, reset=False)
        return logsumexp(self._estimate_weighted_log_prob(X), axis=1)

    def score(self, X, y=None):
        """Compute the mean log-likelihood under the fitted model."""
        del y
        return self.score_samples(X).mean()

    def predict_cplx(self, X):
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        return self.predict(X)

    def predict_proba_cplx(self, X):
        """Predict posterior probability of each component given the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        resp : array, shape (n_samples, n_components)
            Returns the probability each Gaussian (state) in
            the model given each sample.
        """
        return self.predict_proba(X)

    def _estimate_weighted_log_prob(self, X):
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        weighted_log_prob : array, shape (n_samples, n_component)
        """
        return self._estimate_log_prob(X) + self._estimate_log_weights()

    def _estimate_log_weights(self):
        """Estimate log mixture weights."""
        return np.log(self.weights_)

    def _estimate_log_prob(self, X):
        """Estimate component-wise log probabilities for input samples."""
        return self._estimate_log_gaussian_prob(
            X,
            self.means_,
            self.precisions_cholesky_,
            self._effective_covariance_type(),
        )

    def _estimate_log_gaussian_prob(self, X, means, precisions_chol, covariance_type):
        """Estimate the log Gaussian probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        means : array-like of shape (n_components, n_features)
        precisions_chol : array-like
            Cholesky decompositions of the precision matrices.
            'full' : shape of (n_components, n_features, n_features)
            'diag' : shape of (n_components, n_features)
            'spherical' : shape of (n_components,)
        covariance_type : {'full', 'tied', 'diag', 'spherical'}

        Returns
        -------
        log_prob : array, shape (n_samples, n_components)
        """
        n_samples, n_features = X.shape
        n_components, _ = means.shape
        # The determinant of the precision matrix from the Cholesky decomposition
        # corresponds to the negative half of the determinant of the full precision
        # matrix.
        # In short: det(precision_chol) = - det(precision) / 2
        log_det = np.real(
            _compute_log_det_cholesky(precisions_chol, covariance_type, n_features)
        )

        if covariance_type == "full":
            log_prob = np.empty((n_samples, n_components))
            for k, (mu, prec_chol) in enumerate(
                zip(means, precisions_chol, strict=True)
            ):
                y = np.dot(X, prec_chol.conj()) - np.dot(mu, prec_chol.conj())
                log_prob[:, k] = np.sum(np.abs(y) ** 2, axis=1)

        elif covariance_type == "diag":
            precisions = np.abs(precisions_chol) ** 2
            log_prob = (
                np.sum((np.abs(means) ** 2 * precisions), 1)
                - 2.0 * np.real(np.dot(X, (means.conj() * precisions).T))
                + np.dot(np.abs(X) ** 2, precisions.T)
            )
        elif covariance_type == "spherical":
            precisions = np.abs(precisions_chol) ** 2
            log_prob = (
                np.sum(np.abs(means) ** 2, 1) * precisions
                - 2 * np.real(np.dot(X, means.conj().T) * precisions)
                + np.outer((X.conj() * X).sum(axis=1), precisions)
            )
        # Since we are using the precision of the Cholesky decomposition,
        # `- log_det_precision` becomes `+ 2 * log_det_precision_chol`
        return -(n_features * np.log(np.pi) + np.real(log_prob)) + 2 * log_det

    def fit_cplx(self, X, y=None):
        """Fit the direct complex-valued EM model.

        This method is kept for backward compatibility. New code should use
        fit(X).
        """
        self._fit_em(X, y=y)
        return self

    def fit_predict(self, X, y=None):
        """Estimate model parameters using X and predict the labels for X.

        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.

        .. versionadded:: 0.20

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        del y  # unused, kept for sklearn API compatibility

        X = self._check_X(X, reset=not hasattr(self, "n_features_in_"))
        self._validate_parameters(X)

        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.inf
        self.converged_ = False

        random_state = ut.check_random_state(self.random_state)

        best_params = None
        best_n_iter = 0

        for init in range(n_init):
            self._verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state)

            lower_bound = -np.inf if do_init else self.lower_bound_

            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = lower_bound

                log_prob_norm, log_resp = self._e_step(X)
                self._m_step(X, log_resp)
                lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)

                change = lower_bound - prev_lower_bound
                self._verbose_msg_iter_end(n_iter, change)

                if abs(change) < self.tol:
                    self.converged_ = True
                    break

            self._verbose_msg_init_end(lower_bound)

            if lower_bound > max_lower_bound:
                max_lower_bound = lower_bound
                best_params = self._get_parameters()
                best_n_iter = n_iter

        if not self.converged_:
            warnings.warn(
                "Initialization did not converge. Try different init parameters, "
                "or increase max_iter, tol, or check for degenerate data.",
                ConvergenceWarning,
                stacklevel=2,
            )

        if best_params is None:
            raise RuntimeError("Failed to fit GaussianMixtureCplx.")

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        _, log_resp = self._e_step(X)
        return log_resp.argmax(axis=1)

    def _initialize_parameters(self, X, random_state):
        """Initialize responsibilities using the configured initialization strategy."""
        n_samples, _ = X.shape

        if self.init_params == "kmeans":
            resp = np.zeros((n_samples, self.n_components))
            X_real = ut.cplx2real(X, axis=1)
            labels = (
                cluster.KMeans(
                    n_clusters=self.n_components,
                    n_init=1,
                    random_state=random_state,
                )
                .fit(X_real)
                .labels_
            )
            resp[np.arange(n_samples), labels] = 1

        elif self.init_params == "random":
            resp = random_state.rand(n_samples, self.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]

        else:
            raise ValueError(
                f"Unimplemented initialization method {self.init_params!r}."
            )

        self._initialize(X, resp)

    def _initialize(self, X, resp):
        """Initialize the Gaussian mixture parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        resp : array-like of shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        init_covariance_type = (
            "full" if self._em_covariance_type == "inv-em" else self._em_covariance_type
        )

        weights, means, covariances = self.estimate_gaussian_parameters(
            X,
            resp,
            self.reg_covar,
            init_covariance_type,
        )
        weights /= n_samples

        self.weights_ = weights if self.weights_init is None else self.weights_init
        self.means_ = means if self.means_init is None else self.means_init

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = compute_precision_cholesky(
                covariances,
                self._effective_covariance_type(),
            )

            if self._em_covariance_type == "inv-em":
                self._sigma = np.zeros((self.n_components, self._F2.shape[0]))

                for k in range(self.n_components):
                    self._sigma[k] = np.real(
                        np.diag(self._F2 @ covariances[k] @ self._F2.conj().T)
                    )
                    self._sigma[k][self._sigma[k] < self.reg_covar] = self.reg_covar

        elif self._effective_covariance_type() == "full":
            self.precisions_cholesky_ = np.array(
                [
                    scilinalg.cholesky(prec_init, lower=True)
                    for prec_init in self.precisions_init
                ]
            )

        else:
            self.precisions_cholesky_ = np.sqrt(self.precisions_init)

    def _e_step(self, X):
        """E step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp

    def _compute_lower_bound(self, log_resp, log_prob_norm):
        """Compute the EM lower bound from the mean log probability."""
        del log_resp  # kept for API similarity with sklearn's internal implementation
        return log_prob_norm

    def _estimate_log_prob_resp(self, X):
        """Estimate log probabilities and responsibilities for each sample.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape

        weights, means, covariances = self.estimate_gaussian_parameters(
            X,
            np.exp(log_resp),
            self.reg_covar,
            self._em_covariance_type,
        )

        self.weights_ = weights / n_samples
        self.means_ = means
        self.covariances_ = covariances
        self.precisions_cholesky_ = compute_precision_cholesky(
            self.covariances_,
            self._effective_covariance_type(),
        )

    def _set_parameters(self, params):
        (
            self.weights_,
            self.means_,
            self.covariances_,
            self.precisions_cholesky_,
        ) = params

        if self._effective_covariance_type() == "full":
            self.precisions_ = np.empty(self.precisions_cholesky_.shape, dtype=complex)

            for k, prec_chol in enumerate(self.precisions_cholesky_):
                self.precisions_[k] = prec_chol @ prec_chol.T.conj()

        else:
            self.precisions_ = np.abs(self.precisions_cholesky_) ** 2

    def estimate_gaussian_parameters(self, X, resp, reg_covar, covariance_type):
        """Estimate the Gaussian distribution parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data array.

        resp : array-like of shape (n_samples, n_components)
            The responsibilities for each data sample in X.

        reg_covar : float
            The regularization added to the diagonal of the covariance matrices.

        covariance_type : {'full', 'tied', 'diag', 'spherical'}
            The type of precision matrices.

        Returns
        -------
        nk : array-like of shape (n_components,)
            The numbers of data samples in the current components.

        means : array-like of shape (n_components, n_features)
            The centers of the current components.

        covariances : array-like
            The covariance matrix of the current components.
            The shape depends of the covariance_type.
        """
        nk = np.real(resp).sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        means = np.dot(resp.T, X) / nk[:, np.newaxis]
        if self._zero_mean:
            means = np.zeros_like(means)
        covariances = {
            "full": self.estimate_gaussian_covariances_full,
            "diag": self.estimate_gaussian_covariances_diag,
            "inv-em": self.estimate_gaussian_covariances_inv,
            "spherical": self.estimate_gaussian_covariances_spherical,
        }[covariance_type](resp, X, nk, means, reg_covar)
        return nk, means, covariances

    def estimate_gaussian_covariances_full(self, resp, X, nk, means, reg_covar):
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

    def estimate_gaussian_covariances_diag(self, resp, X, nk, means, reg_covar):
        """Estimate the diagonal covariance vectors.

        Parameters
        ----------
        responsibilities : array-like of shape (n_samples, n_components)

        X : array-like of shape (n_samples, n_features)

        nk : array-like of shape (n_components,)

        means : array-like of shape (n_components, n_features)

        reg_covar : float

        Returns
        -------
        covariances : array, shape (n_components, n_features)
            The covariance vector of the current components.
        """
        avg_X2 = np.dot(resp.T, X * X.conj()) / nk[:, np.newaxis]
        avg_means2 = np.abs(means) ** 2
        avg_X_means = means.conj() * np.dot(resp.T, X) / nk[:, np.newaxis]
        return (
            np.real(avg_X2 - 2.0 * np.real(avg_X_means) + avg_means2) + reg_covar + 0j
        )

    def estimate_gaussian_covariances_inv(self, resp, X, nk, means, reg_covar):
        """Estimate the Topelitz-structured covariance matrices.

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

        Returns
        -------
        covariances : array, shape (n_components, n_features, n_features)
            The covariance matrix of the current components.
        """
        n_components, n_features = means.shape
        covariances = np.empty((n_components, n_features, n_features), dtype=complex)
        covariance_inv = np.linalg.pinv(self.covariances_, hermitian=True)

        for k in range(n_components):
            diff = X - means[k]
            covariances[k] = np.dot(resp[:, k] * diff.T, diff.conj()) / nk[k]

            theta = np.real(
                self._F2
                @ (
                    covariance_inv[k] @ covariances[k] @ covariance_inv[k]
                    - covariance_inv[k]
                )
                @ self._F2.conj().T
            )

            self._sigma[k] = self._sigma[k] + np.diag(
                self._sigma[k] * theta * self._sigma[k]
            )
            self._sigma[k][self._sigma[k] < reg_covar] = reg_covar

            covariances[k] = np.multiply(self._F2.conj().T, self._sigma[k]) @ self._F2
            covariances[k].flat[:: n_features + 1] += reg_covar

        return covariances

    def estimate_gaussian_covariances_spherical(self, resp, X, nk, means, reg_covar):
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
        return (
            np.real(
                self.estimate_gaussian_covariances_diag(
                    resp, X, nk, means, reg_covar
                ).mean(1)
            )
            + 0j
        )
