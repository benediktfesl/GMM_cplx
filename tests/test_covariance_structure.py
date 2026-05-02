import numpy as np
import pytest

from cplx_gmm import GaussianMixtureCplx


def _complex_data(n_samples: int, n_features: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(n_samples, n_features))
        + 1j * rng.normal(size=(n_samples, n_features))
    ) / np.sqrt(2)


def _assert_hermitian(matrix: np.ndarray, atol: float = 1e-8) -> None:
    assert np.allclose(matrix, matrix.conj().T, atol=atol)


def _assert_circulant(matrix: np.ndarray, atol: float = 1e-8) -> None:
    first_row = matrix[0]
    for row_idx in range(matrix.shape[0]):
        assert np.allclose(
            matrix[row_idx],
            np.roll(first_row, row_idx),
            atol=atol,
        )


def _assert_toeplitz(matrix: np.ndarray, atol: float = 1e-8) -> None:
    n_rows, n_cols = matrix.shape

    for row in range(n_rows):
        for col in range(n_cols):
            diagonal = np.diag(matrix, k=col - row)
            assert np.allclose(matrix[row, col], diagonal[0], atol=atol)


def _as_blocks(matrix: np.ndarray, block_shape: tuple[int, int]) -> np.ndarray:
    n_blocks, block_size = block_shape
    return matrix.reshape(n_blocks, block_size, n_blocks, block_size).transpose(
        0, 2, 1, 3
    )


def _assert_block_circulant(
    matrix: np.ndarray,
    block_shape: tuple[int, int],
    atol: float = 1e-8,
) -> None:
    n_blocks, _ = block_shape
    blocks = _as_blocks(matrix, block_shape)

    first_block_row = blocks[0]
    for row_idx in range(n_blocks):
        for col_idx in range(n_blocks):
            expected_block = first_block_row[(col_idx - row_idx) % n_blocks]
            assert np.allclose(blocks[row_idx, col_idx], expected_block, atol=atol)


def _assert_block_toeplitz(
    matrix: np.ndarray,
    block_shape: tuple[int, int],
    atol: float = 1e-8,
) -> None:
    n_blocks, _ = block_shape
    blocks = _as_blocks(matrix, block_shape)

    for row in range(n_blocks):
        for col in range(n_blocks):
            diagonal_blocks = [
                blocks[i, i + col - row]
                for i in range(n_blocks)
                if 0 <= i + col - row < n_blocks
            ]
            assert np.allclose(blocks[row, col], diagonal_blocks[0], atol=atol)


@pytest.mark.parametrize("n_features", [3, 5, 7, 9, 12])
def test_diag_covariance_yields_diagonal_variances(n_features: int) -> None:
    X = _complex_data(n_samples=300, n_features=n_features)

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="diag",
        random_state=0,
        max_iter=100,
        init_params="random",
    )
    model.fit(X)

    assert model.covariances_.shape == (2, n_features)

    for covariance in model.covariances_:
        assert np.iscomplexobj(covariance)
        assert np.all(np.real(covariance) > 0)
        assert np.allclose(np.imag(covariance), 0.0)


@pytest.mark.parametrize("n_features", [3, 5, 7, 9, 12])
def test_full_covariance_yields_hermitian_matrices(n_features: int) -> None:
    X = _complex_data(n_samples=300, n_features=n_features)

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="full",
        random_state=0,
        max_iter=100,
        init_params="random",
    )
    model.fit(X)

    assert model.covariances_.shape == (2, n_features, n_features)

    for covariance in model.covariances_:
        assert np.iscomplexobj(covariance)
        _assert_hermitian(covariance)


@pytest.mark.parametrize("n_features", [3, 5, 7, 9, 12])
def test_spherical_covariance_yields_scalar_variances(n_features: int) -> None:
    X = _complex_data(n_samples=300, n_features=n_features)

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="spherical",
        random_state=0,
        max_iter=100,
        init_params="random",
    )
    model.fit(X)

    assert model.covariances_.shape == (2,)
    assert np.iscomplexobj(model.covariances_)
    assert np.all(np.real(model.covariances_) > 0)
    assert np.allclose(np.imag(model.covariances_), 0.0)


@pytest.mark.parametrize("n_features", [3, 5, 7, 9, 12])
def test_circulant_covariance_yields_circulant_matrices(n_features: int) -> None:
    X = _complex_data(n_samples=300, n_features=n_features)

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="circulant",
        random_state=0,
        max_iter=100,
        init_params="random",
    )
    model.fit(X)

    assert model.covariances_.shape == (2, n_features, n_features)

    for covariance in model.covariances_:
        assert np.iscomplexobj(covariance)
        _assert_hermitian(covariance)
        _assert_circulant(covariance)


@pytest.mark.parametrize("blocks", [(3, 3), (4, 3), (3, 4), (4, 4), (5, 3), (3, 5)])
def test_block_circulant_covariance_yields_block_circulant_matrices(
    blocks: tuple[int, int],
) -> None:
    n_features = blocks[0] * blocks[1]
    X = _complex_data(n_samples=400, n_features=n_features)

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="block-circulant",
        blocks=blocks,
        random_state=0,
        max_iter=100,
        init_params="random",
    )
    model.fit(X)

    assert model.covariances_.shape == (2, n_features, n_features)

    for covariance in model.covariances_:
        assert np.iscomplexobj(covariance)
        _assert_hermitian(covariance)
        _assert_block_circulant(covariance, blocks)


@pytest.mark.parametrize("n_features", [3, 5, 7, 9, 12])
def test_toeplitz_covariance_yields_toeplitz_matrices(n_features: int) -> None:
    X = _complex_data(n_samples=300, n_features=n_features)

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="toeplitz",
        random_state=0,
        max_iter=100,
        init_params="random",
    )
    model.fit(X)

    assert model.covariances_.shape == (2, n_features, n_features)

    for covariance in model.covariances_:
        assert np.iscomplexobj(covariance)
        _assert_hermitian(covariance)
        _assert_toeplitz(covariance)


@pytest.mark.parametrize("blocks", [(3, 3), (4, 3), (3, 4), (4, 4), (5, 3), (3, 5)])
def test_block_toeplitz_covariance_yields_block_toeplitz_matrices(
    blocks: tuple[int, int],
) -> None:
    n_features = blocks[0] * blocks[1]
    X = _complex_data(n_samples=400, n_features=n_features)

    model = GaussianMixtureCplx(
        n_components=2,
        covariance_type="block-toeplitz",
        blocks=blocks,
        random_state=0,
        max_iter=100,
        init_params="random",
    )
    model.fit(X)

    assert model.covariances_.shape == (2, n_features, n_features)

    for covariance in model.covariances_:
        assert np.iscomplexobj(covariance)
        _assert_hermitian(covariance)
        _assert_block_toeplitz(covariance, blocks)