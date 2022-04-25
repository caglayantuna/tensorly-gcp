"""
Generalized Parafac in Tensorly
===============================================
On this page, you will find examples showing how to use Generalized CP (GCP).
"""

##############################################################################
# Introduction
# -----------------------
# Tensorly includes generalized decomposition [1] and its stochastic version (SGCP) [2]
# which allow using different kind of losses rather than only Euclidean. Both algorithms
# aim at decreasing loss between the input tensor and estimated tensor by using
# gradient which is calculated according to the selected loss by the user.
# While GCP implementation uses FISTA method for optimization, SGCP uses ADAM
# algorithm as in [2].
# Following table gives existing losses in Tensorly with their gradients and constraints:
#
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

import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_parafac_hals
from tlgcp import generalized_parafac, stochastic_generalized_parafac
import matplotlib.pyplot as plt
from tensorly.metrics import RMSE
from tlgcp.utils import loss_operator
import time

np.set_printoptions(precision=2)

##############################################################################
# Example with Bernoulli loss
# --------------------------------------------
# To use GCP decomposition efficiently, loss should be selected according to the input tensor.
# Here, we will report an example with Bernoulli odds loss. Let us note that
# we suggest to use random init rather than svd while using GCP decomposition.

# Parameters
init = 'random'
rank = 5
loss = 'bernoulli_odds'
shape = [60, 80, 50]

##############################################################################
# To create a synthetic tensor wih Bernoulli distribution, we use random cp and numpy
# binomial functions:

cp_tensor = tl.cp_to_tensor(tl.random.random_cp(shape, rank))
array = np.random.binomial(1, cp_tensor / (cp_tensor + 1), size=shape)
tensor = tl.tensor(array, dtype='float')

##############################################################################
# GCP decomposition function requires loss and learning rate (LR) as differ from
# existing tensorly decomposition functions. It should be noted that LR
# should be tuned by the user since the algorithm is sensitive to its value.

# GCP
tic = time.time()
tensor_gcp, errors_gcp = generalized_parafac(tensor, rank=rank, init=init, return_errors=True, loss=loss, n_iter_max=500)
cp_reconstruction_gcp = tl.cp_to_tensor((tensor_gcp))
time_gcp = time.time() - tic

##############################################################################
# Stochastic GCP (SGCP) decomposition function requires batch size, epochs and beta
# parameters (for ADAM) as input in addition to GCP decomposition inputs. Fortunately,
# LR and beta parameters could be fixed thanks to the literature who works with
# ADAM optimization. Besides, in case of badly chosen LR, SGCP updates the LR by dividing
# LR by 10 after each failed iteration until reaching 20 successive bad iteration.

# SGCP
tic = time.time()
tensor_sgcp, errors_sgcp = stochastic_generalized_parafac(tensor, rank=rank, init=init,
                                                          return_errors=True, loss=loss, lr=1e-3,
                                                          n_iter_max=1000, batch_size=50, epochs=100)
cp_reconstruction_sgcp = tl.cp_to_tensor((tensor_sgcp))
time_sgcp = time.time() - tic

##############################################################################
# To compare GCP decompositions, we choose non-negative CP with HALS (NN-CP)
# since Bernoulli odds has a non-negative constraint.

# NN-Parafac with HALS result
tic = time.time()
tensor_cp, errors = non_negative_parafac_hals(tensor, rank=rank, n_iter_max=100, init=init, return_errors=True)
cp_reconstruction = tl.cp_to_tensor((tensor_cp))
time_cp = time.time() - tic

##############################################################################
# In the example, we use binary tensor `tensor` as an input. It is possible to
# have binary result by using numpy binomial function on reconstructed cp tensors.
# Besides, we could compare the results with initial `cp_tensor` and reconstructed tensors
# without calculating it.


print("RMSE for GCP:", RMSE(cp_tensor, cp_reconstruction_gcp))
print("RMSE for SGCP:", RMSE(cp_tensor, cp_reconstruction_sgcp))
print("RMSE for NN-CP:", RMSE(cp_tensor, cp_reconstruction))

print("Loss for GCP:", tl.sum(loss_operator(cp_tensor, cp_reconstruction_gcp, loss)))
print("Loss for SGCP:", tl.sum(loss_operator(cp_tensor, cp_reconstruction_sgcp, loss)))
print("Loss for NN-CP:", tl.sum(loss_operator(cp_tensor, cp_reconstruction, loss)))

print("GCP time:", time_gcp)
print("SGCP time:", time_sgcp)
print("NN-CP time:", time_cp)

##############################################################################
# We compare the results according to processing time, root mean square error and
# the selected loss. According to the final Bernoulli loss,
# both GCP and SGCP give better results than NN-CP. Since SGCP requires many
# iteration inside each epoch, processing time is much more than the others.
# On the other hand, NN-CP is better in terms of root mean square error as it is
# expected.

##############################################################################
# References
# ----------
#
# [1] Hong, D., Kolda, T. G., & Duersch, J. A. (2020).
# Generalized canonical polyadic tensor decomposition.
# SIAM Review, 62(1), 133-163.
# `(Online version)
# <https://arxiv.org/abs/1808.07452>`_
#
# [2] Kolda, T. G., & Hong, D. (2020). Stochastic gradients for
# large-scale tensor decomposition.
# SIAM Journal on Mathematics of Data Science, 2(4), 1066-1095.
# `(Online version)
# <https://arxiv.org/abs/1906.01687>`_
