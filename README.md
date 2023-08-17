# GMM_cplx
Python implementation of a complex-valued version of the expectation-maximization (EM) algorithm for fitting Gaussian Mixture Models (GMMs). 
The implementation is an extension to the scikit learn implementation for GMMs from
https://scikit-learn.org/stable/modules/mixture.html
for complex-valued data and with different options for structured covariance matrices.
The EM algorithm maximizes the likelihood of a circularly symmetric Gaussian distribution.

## Instructions
The main implementation is contained in `gmm_cplx.py` with the class `GaussianMixtureCplx` which has the same arguments as `sklearn.mixture.GaussianMixture`. 
The file `examples_gmm_cplx.py` provides useful examples of how to use the code, e.g., with the different covariance structures.

## Requirements
This code is written in *Python*. It uses the *numpy*, *scipy*, *sklearn*, and *time* packages. The code was tested with Python 3.7.

## Methods of `GaussianMixtureCplx`
- `fit(X, blocks=None, zero_mean=False)`: Fitting the GMM parameters to the provided complex-valued dataset X of shape `(n_samples, n_dim)`.
  
  `zero_mean=True` enforces that every GMM component has mean zero.
  
  `blocks=(dim1, dim2)` is necessary for block-matrices (see below).
  
- `predict_cplx(X)`: Predict the labels for the data samples in X using trained model.

- `predict_proba_cplx(X)`: Predict posterior probability of each component given the data.

- `sample(n_samples)`: Generate random samples from the fitted Gaussian mixture distribution.

## Possible Covariance Structures
The following covariance structures are supported:
- 'full' (full covariance matrix with no structural constraints for each GMM component)
- 'diag' (diagonal covariance matrix for each GMM component)
- 'spherical' (scaled identity covariance matrix for each GMM component)
- 'circulant' (Circulant covariance matrix for each GMM component
- 'block-circulant' (Block-circulant covariance matrix with circulant blocks for each GMM component, use keyword 'blocks' in 'fit')
- 'toeplitz' (Toeplitz covariance matrix for each GMM component)
- 'block-toeplitz' (Block-Toeplitz covariance matrix with Toeplitz blocks for each GMM component, use keyword 'blocks' in 'fit')

## Research work
The results of the following works are (in parts) based on the complex-valued implementation:
- M. Koller, B. Fesl, N. Turan, and W. Utschick, “An Asymptotically MSE-Optimal Estimator Based on Gaussian Mixture Models,” *IEEE Trans. Signal Process.*, vol. 70, pp. 4109–4123, 2022.
- N. Turan, B. Fesl, M. Grundei, M. Koller, and W. Utschick, “Evaluation of a Gaussian Mixture Model-based Channel Estimator using Measurement Data,” in *Int. Symp. Wireless Commun. Syst. (ISWCS)*, 2022.
- B. Fesl, M. Joham, S. Hu, M. Koller, N. Turan, and W. Utschick, “Channel Estimation based on Gaussian Mixture Models with Structured Covariances,” in *56th Asilomar Conf. Signals, Syst., Comput.*, 2022, pp. 533–537.
- B. Fesl, N. Turan, M. Joham, and W. Utschick, “Learning a Gaussian Mixture Model from Imperfect Training Data for Robust Channel Estimation,” *IEEE Wireless Commun. Lett.*, 2023.
- M. Koller, B. Fesl, N. Turan and W. Utschick, "An Asymptotically Optimal Approximation of the Conditional Mean Channel Estimator Based on Gaussian Mixture Models," *IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP)*, 2022, pp. 5268-5272.
- B. Fesl, A. Faika, N. Turan, M. Joham, and W. Utschick, “Channel Estimation with Reduced Phase Allocations in RIS-Aided Systems,” in *IEEE 24th Int. Workshop Signal Process. Adv. Wireless Commun. (SPAWC)*, 2023.
- N. Turan, B. Fesl, M. Koller, M. Joham, and W. Utschick, “A Versatile Low-Complexity Feedback Scheme for FDD Systems via Generative Modeling,” 2023, arXiv preprint: 2304.14373.
- N. Turan, B. Fesl, and W. Utschick, "Enhanced Low-Complexity FDD System Feedback with Variable Bit Lengths via Generative Modeling," in *57th Asilomar Conf. Signals, Syst., Comput.*, 2023.
