from cplx_gmm import GaussianMixtureCplx


def test_get_params_contains_constructor_parameters() -> None:
    model = GaussianMixtureCplx(n_components=3, covariance_type="diag")

    params = model.get_params()

    assert params["n_components"] == 3
    assert params["covariance_type"] == "diag"
    assert params["zero_mean"] is False


def test_set_params_updates_constructor_parameters() -> None:
    model = GaussianMixtureCplx()

    returned = model.set_params(n_components=4, covariance_type="spherical")

    assert returned is model
    assert model.n_components == 4
    assert model.covariance_type == "spherical"
