import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from cplx_gmm import GaussianMixtureCplx


@pytest.mark.parametrize(
    "method_name",
    ["predict", "predict_proba", "score_samples", "score", "sample"],
)
def test_public_methods_raise_before_fit(method_name: str) -> None:
    X = np.ones((10, 2), dtype=complex)
    model = GaussianMixtureCplx()

    method = getattr(model, method_name)

    with pytest.raises(NotFittedError):
        if method_name == "sample":
            method()
        else:
            method(X)
