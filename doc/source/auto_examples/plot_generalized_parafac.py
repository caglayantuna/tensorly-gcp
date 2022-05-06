"""
Example use of Generalized CP for integer-valued tensor
=======================================================
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

def each_iteration(a, b):
    fig=plt.figure()
    fig.set_size_inches(10, fig.get_figheight(), forward=True)
    plt.plot(a)
    plt.plot(b)
    plt.yscale('log')
    plt.legend(['GCP', 'S-GCP'], loc='upper right')

##############################################################################
# Example with Bernoulli loss
# --------------------------------------------
# In this example, a tensor containing integer values is decomposed in the CP
# format using Generalized CP [Hong, Kolda, Duersch 2019]. To use GCP decomposition
# efficiently, the correct loss should be selected according to the input tensor.
# Here, we use Bernoulli odds loss, which stems from the maximum likelihood
# estimator when the data is generated as a Bernoulli process (integer values)
# with low-rank CP parameters. Let us note that we suggest to use random init rather
# than Singular Value Decomposition with using GCP decomposition.

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
# Running GCP is quite simple, and boils down to calling the
# `generalized_parafac` routine as follows.

# GCP
tic = time.time()
tensor_gcp, errors_gcp = generalized_parafac(tensor, rank=rank, init=init, return_errors=True, loss=loss, n_iter_max=500)
cp_reconstruction_gcp = tl.cp_to_tensor(tensor_gcp)
time_gcp = time.time() - tic

##############################################################################
# Stochastic GCP (SGCP) decomposition function requires learning rate (LR),
# batch size, epochs and beta parameters (for ADAM) as input in addition to GCP
# decomposition inputs. Fortunately, LR and beta parameters can be fixed following
# the literature on ADAM optimization. Besides, in case of badly chosen LR,
# SGCP updates the LR by dividing LR by 10 after each failed iteration until
# reaching 20 successive bad iteration.

# SGCP
tic = time.time()
tensor_sgcp, errors_sgcp = stochastic_generalized_parafac(tensor, rank=rank, init=init,
                                                          return_errors=True, loss=loss, lr=1e-3,
                                                          n_iter_max=10, batch_size=50, epochs=100)
cp_reconstruction_sgcp = tl.cp_to_tensor(tensor_sgcp)
time_sgcp = time.time() - tic

##############################################################################
# To compare GCP decompositions, we choose non-negative CP with HALS (NN-CP)
# since Bernoulli odds imply a nonnegativity constraint on the CP tensor..

# NN-Parafac with HALS result
tic = time.time()
tensor_cp, errors = non_negative_parafac_hals(tensor, rank=rank, n_iter_max=100, init=init, return_errors=True)
cp_reconstruction = tl.cp_to_tensor((tensor_cp))
time_cp = time.time() - tic

##############################################################################
# In the example, we use binary tensor `tensor` as an input. It is possible
# to have binary reconstructed tensor by using numpy binomial function on the
# estimated cp tensors. Below instead we compare the estimated cp tensors with
# the true `cp_tensor` which is the main goal of GCP.


print("RMSE for GCP:", "%.2f" % RMSE(cp_tensor, cp_reconstruction_gcp))
print("RMSE for SGCP:", "%.2f" % RMSE(cp_tensor, cp_reconstruction_sgcp))
print("RMSE for NN-CP:", "%.2f" % RMSE(cp_tensor, cp_reconstruction))

print("Loss for GCP:", "%.2f" % tl.sum(loss_operator(cp_tensor, cp_reconstruction_gcp, loss)))
print("Loss for SGCP:", "%.2f" % tl.sum(loss_operator(cp_tensor, cp_reconstruction_sgcp, loss)))
print("Loss for NN-CP:", "%.2f" % tl.sum(loss_operator(cp_tensor, cp_reconstruction, loss)))

print("GCP time:", "%.2f" % time_gcp)
print("SGCP time:", "%.2f" % time_sgcp)
print("NN-CP time:", "%.2f" % time_cp)

##############################################################################
# We compare the results according to processing time, root mean square error and
# the selected loss. According to the final Bernoulli loss,
# both GCP and SGCP give better results than NN-CP. Since SGCP requires many
# iteration inside each epoch, processing time is much more than the others.
# We can also compare the methods by error per iteration plot:

each_iteration(errors_gcp, errors_sgcp)

##############################################################################
# References
# ----------
#
# [1] Hong, D., Kolda, T. G., & Duersch, J. A. (2020).
# Generalized canonical polyadic tensor decomposition.
# SIAM Review, 62(1), 133-163.
# `(Link 1)
# <https://arxiv.org/abs/1808.07452>`_
#
# [2] Kolda, T. G., & Hong, D. (2020). Stochastic gradients for
# large-scale tensor decomposition.
# SIAM Journal on Mathematics of Data Science, 2(4), 1066-1095.
# `(Link 2)
# <https://arxiv.org/abs/1906.01687>`_
