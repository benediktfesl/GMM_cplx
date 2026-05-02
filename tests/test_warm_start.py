import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from cplx_gmm import GaussianMixtureCplx


def test_warm_start_reuses_previous_solution() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3)) + 1j * rng.normal(size=(200, 3))

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="diag",
        warm_start=True,
        random_state=0,
        max_iter=2,
        n_init=3,
        init_params="random",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X)
        first_lower_bound = model.lower_bound_

        model.fit(X)
        second_lower_bound = model.lower_bound_

    assert second_lower_bound >= first_lower_bound - 1e-10
