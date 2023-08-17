import gmm_cplx
import time
import numpy as np
import utils as ut

if __name__ == '__main__':
    """
    Test script for the complex-valued GMM implementation.
    """
    rng = np.random.default_rng(1235428719812346)

    n_train = 1_000
    n_val = 100
    n_dim = 32
    # covariance types: {'full', 'diag', 'spherical', (block-)circulant, (block-)toeplitz}
    covariance_type = 'spherical'
    # Dimensions of matrix blocks (only necessary for block-circulant and block-toeplitz), e.g., 4 blocks of size 8x8
    blocks = (4, 8)
    # Enforce zero mean of all GMM components
    zero_mean = False

    # Create toy data
    h_train = (rng.standard_normal((n_train, n_dim)) + 1j * rng.standard_normal((n_train, n_dim))) / np.sqrt(2)
    h_val = (rng.standard_normal((n_val, n_dim)) + 1j * rng.standard_normal((n_val, n_dim))) / np.sqrt(2)

    #
    # GMM training
    #
    tic = time.time()
    gm_full = gmm_cplx.GaussianMixtureCplx(
        n_components=16,
        random_state=2,
        max_iter=100,
        n_init=1,
        covariance_type=covariance_type,
    )
    gm_full.fit(h_train, blocks=blocks, zero_mean=zero_mean)
    toc = time.time()
    print(f'Training done: {toc - tic} sec.')

    # Covariances & means & weights
    means = gm_full.means
    covs = gm_full.covariances
    weights = gm_full.weights
    print(f'Sum of weights: {np.real(np.sum(weights))}')

    #
    # Responsibility evaluation
    #
    # soft responsibilities for all components
    proba_soft = gm_full.predict_proba_cplx(h_val)
    # components with max responsibilities
    proba_max = gm_full.predict_cplx(h_val)

    #
    # Generate new samples
    #
    samples, comps = gm_full.sample(n_samples=100)
    # check generated samples by computing max responsibility
    proba_max_samples = gm_full.predict_cplx(samples)