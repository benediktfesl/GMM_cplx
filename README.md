# cplx-gmm

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![Package](https://img.shields.io/badge/package-PyPI-informational.svg)](https://pypi.org/project/cplx-gmm/)

Complex-valued Gaussian mixture models with structured covariance matrices.

`cplx-gmm` provides a scikit-learn-style estimator for fitting Gaussian mixture models (GMMs) to complex-valued data using expectation-maximization (EM). It supports circularly symmetric complex Gaussian components and structured covariance types that are not available in the standard scikit-learn `GaussianMixture`, including circulant, block-circulant, Toeplitz, and block-Toeplitz covariance matrices.

## ✨ Highlights

- Complex-valued Gaussian mixture models for data in complex vector spaces
- scikit-learn-style estimator API
- Structured covariance models beyond standard scikit-learn GMMs
- Full, diagonal, spherical, circulant, block-circulant, Toeplitz, and block-Toeplitz covariance types
- Optional zero-mean component constraint
- Sampling from fitted complex-valued mixture models
- Tested covariance-structure correctness for non-trivial dimensions
- Modern Python packaging with `pyproject.toml`, `uv`, `pytest`, and `ruff`

## 📌 Citation

If you use `cplx-gmm` in academic work, please cite the package directly:

```bibtex
@software{fesl_cplx_gmm,
  author = {Fesl, Benedikt},
  title = {{cplx-gmm}: Complex-valued Gaussian mixture models with structured covariance matrices},
  year = {2026},
  url = {https://github.com/benediktfesl/GMM_cplx},
  version = {0.1.0}
}
```

Plain-text citation:

> B. Fesl, `cplx-gmm`: Complex-valued Gaussian mixture models with structured covariance matrices, version 0.1.0. Available: https://github.com/benediktfesl/GMM_cplx

## 📦 Installation

Install from PyPI:

```bash
pip install cplx-gmm
```

or with `uv`:

```bash
uv add cplx-gmm
```

## 🧩 Covariance Structures

The covariance models are one of the main reasons to use this package. In addition to the usual `full`, `diag`, and `spherical` covariance types, `cplx-gmm` supports structured covariance matrices that are common in signal processing and wireless channel modeling.

| `covariance_type` | Description |
|---|---|
| `"full"` | Full covariance matrix for each component. |
| `"diag"` | Diagonal covariance for each component. |
| `"spherical"` | One scalar variance per component. |
| `"circulant"` | Circulant covariance matrix for each component. |
| `"block-circulant"` | Block-circulant covariance matrix. Requires `blocks=(n_1, n_2)`. |
| `"toeplitz"` | Toeplitz covariance matrix for each component. |
| `"block-toeplitz"` | Block-Toeplitz covariance matrix. Requires `blocks=(n_1, n_2)`. |

For block-structured covariance types, pass the block dimensions in the constructor:

```python
model = GaussianMixtureCplx(
    n_components=4,
    covariance_type="block-circulant",
    blocks=(4, 8),
    random_state=0,
)

model.fit(X)
```

Here, `blocks=(4, 8)` means that the feature dimension must satisfy `n_features = 4 * 8`.

## 🚀 Quick Start

```python
import numpy as np

from cplx_gmm import GaussianMixtureCplx

rng = np.random.default_rng(0)

X = (
    rng.normal(size=(1_000, 8))
    + 1j * rng.normal(size=(1_000, 8))
) / np.sqrt(2)

model = GaussianMixtureCplx(
    n_components=4,
    covariance_type="full",
    random_state=0,
    max_iter=100,
    n_init=1,
)

model.fit(X)

labels = model.predict(X)
responsibilities = model.predict_proba(X)
log_likelihood = model.score(X)

samples, component_labels = model.sample(n_samples=100)
```

The estimator follows the usual scikit-learn pattern: model configuration is passed to the constructor, and `fit(X)` receives the data.


## 📚 Research Background

This implementation was developed in the context of complex-valued Gaussian mixture modeling for wireless channel estimation and related signal processing applications.

The results of the following works are, in parts, based on the complex-valued implementation:

- M. Koller, B. Fesl, N. Turan, and W. Utschick, “An Asymptotically MSE-Optimal Estimator Based on Gaussian Mixture Models,” *IEEE Transactions on Signal Processing*, vol. 70, pp. 4109–4123, 2022.  
  [[IEEE](https://ieeexplore.ieee.org/abstract/document/9842343)] [[arXiv](https://arxiv.org/abs/2112.12499)]

- N. Turan, B. Fesl, M. Grundei, M. Koller, and W. Utschick, “Evaluation of a Gaussian Mixture Model-based Channel Estimator using Measurement Data,” *International Symposium on Wireless Communication Systems (ISWCS)*, 2022.  
  [[IEEE](https://ieeexplore.ieee.org/abstract/document/9940363)] [[arXiv](https://arxiv.org/abs/2207.14150)]

- B. Fesl, M. Joham, S. Hu, M. Koller, N. Turan, and W. Utschick, “Channel Estimation based on Gaussian Mixture Models with Structured Covariances,” *56th Asilomar Conference on Signals, Systems, and Computers*, 2022, pp. 533–537.  
  [[IEEE](https://ieeexplore.ieee.org/abstract/document/10051921)] [[arXiv](https://arxiv.org/abs/2205.03634)]

- B. Fesl, N. Turan, M. Joham, and W. Utschick, “Learning a Gaussian Mixture Model from Imperfect Training Data for Robust Channel Estimation,” *IEEE Wireless Communications Letters*, 2023.  
  [[IEEE](https://ieeexplore.ieee.org/abstract/document/10078293)] [[arXiv](https://arxiv.org/abs/2301.06488)]

- M. Koller, B. Fesl, N. Turan, and W. Utschick, “An Asymptotically Optimal Approximation of the Conditional Mean Channel Estimator Based on Gaussian Mixture Models,” *IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)*, 2022, pp. 5268–5272.  
  [[IEEE](https://ieeexplore.ieee.org/abstract/document/9747226)] [[arXiv](https://arxiv.org/abs/2111.11064)]

- B. Fesl, A. Faika, N. Turan, M. Joham, and W. Utschick, “Channel Estimation with Reduced Phase Allocations in RIS-Aided Systems,” *IEEE 24th International Workshop on Signal Processing Advances in Wireless Communications (SPAWC)*, 2023, pp. 161–165.  
  [[IEEE](https://ieeexplore.ieee.org/document/10304464)] [[arXiv](https://arxiv.org/abs/2211.07552)]

- N. Turan, B. Fesl, M. Koller, M. Joham, and W. Utschick, “A Versatile Low-Complexity Feedback Scheme for FDD Systems via Generative Modeling,” *IEEE Transactions on Wireless Communications*, 2023.  
  [[IEEE](https://ieeexplore.ieee.org/document/10318056)] [[arXiv](https://arxiv.org/abs/2304.14373)]

- N. Turan, B. Fesl, and W. Utschick, “Enhanced Low-Complexity FDD System Feedback with Variable Bit Lengths via Generative Modeling,” *57th Asilomar Conference on Signals, Systems, and Computers*, 2023.  
  [[IEEE](https://ieeexplore.ieee.org/document/10477075)] [[arXiv](https://arxiv.org/abs/2305.03427)]

- N. Turan, M. Koller, B. Fesl, S. Bazzi, W. Xu, and W. Utschick, “GMM-based Codebook Construction and Feedback Encoding in FDD Systems,” *56th Asilomar Conference on Signals, Systems, and Computers*, 2022, pp. 37–42.  
  [[IEEE](https://ieeexplore.ieee.org/abstract/document/10052020)] [[arXiv](https://arxiv.org/abs/2205.12002)]

- ... and more

## 🧠 Estimator API

The main class is:

```python
from cplx_gmm import GaussianMixtureCplx
```

Core methods:

| Method | Description |
|---|---|
| `fit(X, y=None)` | Fit the complex-valued GMM. |
| `fit_predict(X, y=None)` | Fit the model and return component labels. |
| `predict(X)` | Predict the most likely component for each sample. |
| `predict_proba(X)` | Return posterior component probabilities. |
| `score_samples(X)` | Return per-sample log-likelihoods. |
| `score(X, y=None)` | Return the mean log-likelihood. |
| `sample(n_samples=1)` | Draw samples from the fitted mixture model. |

Fitted parameters follow scikit-learn-style trailing-underscore names such as `weights_`, `means_`, `covariances_`, `precisions_`, `precisions_cholesky_`, `converged_`, `n_iter_`, and `lower_bound_`.

## 🔒 Zero-Mean Components

Some signal processing models assume zero-mean Gaussian components. This can be enforced with:

```python
model = GaussianMixtureCplx(
    n_components=4,
    covariance_type="full",
    zero_mean=True,
)

model.fit(X)
```

When `zero_mean=True`, all component means are fixed to zero during fitting.

## 🔁 Circulant Covariances

Circulant covariance matrices are diagonalized by the discrete Fourier transform (DFT). For `"circulant"` and `"block-circulant"`, the estimator fits a diagonal covariance model in the Fourier domain and transforms the fitted parameters back to the original domain.

```python
model = GaussianMixtureCplx(
    n_components=4,
    covariance_type="circulant",
    random_state=0,
)

model.fit(X)
```

For block-circulant covariances, a two-dimensional FFT representation is used.

## 📐 Toeplitz Covariances

Toeplitz and block-Toeplitz covariance fitting uses an EM-based inverse covariance update inspired by:

> T. A. Barton and D. R. Fuhrmann, “Covariance Estimation for Multidimensional Data using the EM Algorithm,” *Proceedings of the 27th Asilomar Conference on Signals, Systems and Computers*, 1993, pp. 203–207.

Example:

```python
model = GaussianMixtureCplx(
    n_components=4,
    covariance_type="toeplitz",
    random_state=0,
)

model.fit(X)
```

## 🧪 Development

Clone the repository and install the development environment with `uv`:

```bash
git clone https://github.com/benediktfesl/GMM_cplx.git
cd GMM_cplx
uv sync
```

Run tests:

```bash
uv run pytest
```

## ✅ Test Coverage

The test suite covers:

- package imports
- sklearn-style estimator API
- validation behavior
- all supported covariance types
- structural covariance correctness
- EM lower-bound monotonicity
- zero-mean fitting
- initialization options
- warm-start behavior
- reproducibility with fixed `random_state`
- sampling behavior
- real-valued compatibility checks
- doubled real-valued likelihood equivalence
- example execution

## 📄 License

This project is licensed under the [BSD 3-Clause License](LICENSE).

The implementation is based on ideas and portions of the original scikit-learn Gaussian mixture implementation, which is also distributed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).

See [`LICENSE`](LICENSE) for details.
