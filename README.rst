TensorLy-Gcp
===============================================
Generalized Parafac with Tensorly library

This library  includes generalized decomposition [1] and its stochastic version (SGCP) [2] which allow using different kind of losses rather than only Euclidean. Both algorithms aim at decreasing loss between the input tensor and estimated tensor by using gradient which is calculated according to the selected loss by the user.While GCP implementation uses LBFGS method for optimization, SGCP uses ADAM algorithm as in [2].
Following table gives existing losses in Tensorly with their gradients and constraints:

| Distribution   | Loss          | Gradient |Constraints|
|:----------------:|:-------------:|:---------:|-----------|
|Rayleigh       | $2\log(m) + (\pi/4)(x/(m + \epsilon))^2$| 2\times(m) + (\pi/4)(x/(m))^2             |x>0, m>0|
| Bernoulli odds | log(m + 1) - xlog(m + \epsilon)       | 1 / (m + 1) - x/(m + \epsilon)          |x\in(0,1), m>0|
| Bernoulli logit| log(1 + e^m) - xm                     | e^m / (e^m+1) - x                         |x\in(0,1), $m>0$|
| Gamma          |x / (m + \epsilon) + log(m + \epsilon)|-x / ((m + \epsilon)^2) + 1/(m + \epsilon)|x>0, m>0      |
| Poisson count  | m - xlog(m + \epsilon)           | 1 - x/(m + \epsilon)                     |$m>0$       |
| Poisson log    | e^m - xm                              | e^m - x                             |                      |
| Gaussian       | (x - m)^2                           | 2\times(m - x)                            |                      |

Installing TensorLy-GCP
-------------------------



References
----------

[1] Hong, D., Kolda, T. G., & Duersch, J. A. (2020).
Generalized canonical polyadic tensor decomposition.
SIAM Review, 62(1), 133-163.
[Link](https://arxiv.org/abs/1808.07452)

[2] Kolda, T. G., & Hong, D. (2020). Stochastic gradients for
large-scale tensor decomposition.
SIAM Journal on Mathematics of Data Science, 2(4), 1066-1095.
[Link](https://arxiv.org/abs/1906.01687)