TensorLy-Gcp
===============================================
Generalized Parafac with Tensorly library

This library  includes generalized decomposition [1] and its stochastic version (SGCP) [2] which allow using different kind of losses rather than only Euclidean. Both algorithmsaim at decreasing loss between the input tensor and estimated tensor by usinggradient which is calculated according to the selected loss by the user.While GCP implementation uses LBFGS method for optimization, SGCP uses ADAM algorithm as in [2].
Following table gives existing losses in Tensorly with their gradients and constraints:

# +----------------+-----------------------------------------------+---------------------------------------------------+----------------------+
# | Distribution   | Loss                                          | Gradient                                          |Constraints           |
# |                |                                               |                                                   |                      |
# +================+===============================================+===================================================+======================+
# | Rayleigh       | :math:`2\log(m) + (\pi/4)(x/(m + \epsilon))^2`| :math:`2\times(m) + (\pi/4)(x/(m))^2`             |:math:`x>0, m>0`      |
# +----------------+-----------------------------------------------+---------------------------------------------------+----------------------+
# | Bernoulli odds | :math:`log(m + 1) - xlog(m + \epsilon)`       | :math:`1 / (m + 1) - x/(m + \epsilon)`            |:math:`x\in(0,1), m>0`|
# +----------------+-----------------------------------------------+---------------------------------------------------+----------------------+
# | Bernoulli logit| :math:`log(1 + e^m) - xm`                     | :math:`e^m / (e^m+1) - x`                         |:math:`x\in(0,1), m>0`|
# +----------------+-----------------------------------------------+---------------------------------------------------+----------------------+
# | Gamma          | :math:`x / (m + \epsilon) + log(m + \epsilon)`| :math:`-x / ((m + \epsilon)^2) + 1/(m + \epsilon)`|:math:`x>0, m>0`      |
# +----------------+-----------------------------------------------+---------------------------------------------------+----------------------+
# | Poisson count  | :math:`m - xlog(m + \epsilon)`                | :math:`1 - x/(m + \epsilon)`                      |:math:`m>0`           |
# +----------------+-----------------------------------------------+---------------------------------------------------+----------------------+
# | Poisson log    | :math:`e^m - xm`                              | :math:`e^m - x`                                   |                      |
# +----------------+-----------------------------------------------+---------------------------------------------------+----------------------+
# | Gaussian       | :math:`(x - m)^2`                             | :math:`2\times(m - x)`                            |                      |
# +----------------+-----------------------------------------------+---------------------------------------------------+----------------------+
