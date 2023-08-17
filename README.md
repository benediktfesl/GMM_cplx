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

## Toeplitz Covariance Matrices
The implementation of the EM algorithm to enforce a GMM with (block-)Toeplitz-structured covariances is based on the paper 
> T. Barton and D. Fuhrmann, “Covariance Estimation for Multidimensional Data using the EM Algorithm,” in *Proc. of 27th Asilomar Conf. on Signals, Syst. and Comput.*, 1993, pp. 203–207.

Further reading can be found in
> B. Fesl, M. Joham, S. Hu, M. Koller, N. Turan, and W. Utschick, “Channel Estimation based on Gaussian Mixture Models with Structured Covariances,” in *56th Asilomar Conf. Signals, Syst., Comput.*, 2022, pp. 533–537.

## Circulant Covariance Matrices
Since a (block-)circulant covariance matrix is diagonalized by the (two-dimensional) discrete Fourier transform matrix (DFT), we simply transform the training data to the Fourier domain and fit a diagonal covariance matrix as described in 
> M. Koller, B. Fesl, N. Turan, and W. Utschick, “An Asymptotically MSE-Optimal Estimator Based on Gaussian Mixture Models,” *IEEE Trans. Signal Process.*, vol. 70, pp. 4109–4123, 2022.

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

## Original License
The original code from https://scikit-learn.org/stable/modules/mixture.html is covered by the following license:

> BSD 3-Clause License
>
> Copyright (c) 2007-2023 The scikit-learn developers.
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
>modification, are permitted provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice, this
>  list of conditions and the following disclaimer.
>
> * Redistributions in binary form must reproduce the above copyright notice,
>  this list of conditions and the following disclaimer in the documentation
>  and/or other materials provided with the distribution.
>
> * Neither the name of the copyright holder nor the names of its
>  contributors may be used to endorse or promote products derived from
>  this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
> FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
> CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
> OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
## Licence of Contributions
The contributions and extensions are also covered by the BSD 3-Clause License:

> BSD 3-Clause License
>
> Copyright (c) 2023 Benedikt Fesl.
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
>modification, are permitted provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice, this
>  list of conditions and the following disclaimer.
>
> * Redistributions in binary form must reproduce the above copyright notice,
>  this list of conditions and the following disclaimer in the documentation
>  and/or other materials provided with the distribution.
>
> * Neither the name of the copyright holder nor the names of its
>  contributors may be used to endorse or promote products derived from
>  this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
> FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
> CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
> OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
