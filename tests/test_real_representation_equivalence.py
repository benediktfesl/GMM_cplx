import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from cplx_gmm import GaussianMixtureCplx


def _complex_data(n_samples: int, n_features: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(n_samples, n_features))
        + 1j * rng.normal(size=(n_samples, n_features))
    ) / np.sqrt(2)


def _complex_data_to_real(X: np.ndarray) -> np.ndarray:
    return np.concatenate([X.real, X.imag], axis=1)


def _complex_mean_to_real(mean: np.ndarray) -> np.ndarray:
    return np.concatenate([mean.real, mean.imag])


def _complex_covariance_to_real(covariance: np.ndarray) -> np.ndarray:
    return 0.5 * np.block(
        [
            [covariance.real, -covariance.imag],
            [covariance.imag, covariance.real],
        ]
    )


@pytest.mark.parametrize("covariance_type", ["full", "diag", "spherical"])
def test_complex_gmm_likelihood_matches_doubled_real_representation(
    covariance_type: str,
) -> None:
    X = _complex_data(n_samples=200, n_features=4)

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type=covariance_type,
        random_state=0,
        max_iter=100,
        n_init=1,
        init_params="random",
    )
    model.fit(X)

    X_real = _complex_data_to_real(X)

    component_log_probs = []
    for weight, mean, covariance in zip(
        model.weights_,
        model.means_,
        model.covariances_,
        strict=True,
    ):
        if covariance_type == "diag":
            covariance = np.diag(covariance)
        elif covariance_type == "spherical":
            covariance = covariance * np.eye(X.shape[1], dtype=complex)

        real_mean = _complex_mean_to_real(mean)
        real_covariance = _complex_covariance_to_real(covariance)

        component_log_probs.append(
            np.log(weight)
            + multivariate_normal.logpdf(
                X_real,
                mean=real_mean,
                cov=real_covariance,
                allow_singular=False,
            )
        )

    manual_real_scores = logsumexp(np.column_stack(component_log_probs), axis=1)

    assert np.allclose(model.score_samples(X), manual_real_scores, atol=1e-8)