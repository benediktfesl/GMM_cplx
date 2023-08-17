import numpy as np

def check_random_state(seed):
    import numbers
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def cplx2real(vec: np.ndarray, axis=0):
    """
    Concatenate real and imaginary parts of vec along axis=axis.
    """
    return np.concatenate([vec.real, vec.imag], axis=axis)


def multivariate_normal_cplx(mean, covariance, n_samples, covariance_type):
    if covariance_type == 'diag':
        cov_sqrt = np.diag(np.sqrt(covariance))
    elif covariance_type == 'spherical':
        cov_sqrt = np.sqrt(covariance) * np.eye(mean.shape[0])
    else:
        cov_sqrt = np.linalg.cholesky(covariance)
    h = np.squeeze(cov_sqrt @ crandn(n_samples, covariance.shape[0], 1))
    h += np.expand_dims(mean, 0)
    return h


def crandn(*arg, rng=np.random.default_rng()):
    return np.sqrt(0.5) * (rng.standard_normal(arg) + 1j * rng.standard_normal(arg))