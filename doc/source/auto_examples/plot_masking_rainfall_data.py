"""
Generalized Parafac with missing values
===============================================
On this page, you will find examples showing how to handle missing data with Generalized CP (GCP).
"""

##############################################################################
# Introduction
# -----------------------
# Missing values can be handled through GCP decomposition by masking them.

from tlgcp import generalized_parafac
from tlgcp.data import get_tensor
from tensorly.metrics import RMSE
import numpy as np
import tensorly as tl
import time
from tensorly.decomposition import non_negative_parafac_hals
import matplotlib.pyplot as plt


def each_iteration(a, title):
    fig = plt.figure()
    fig.set_size_inches(10, fig.get_figheight(), forward=True)
    plt.plot(a)
    plt.title(str(title))
    plt.yscale('log')
    plt.legend(['gcp'], loc='upper right')


def plot_components(f, title):
    fig, axs = plt.subplots(5, 3)
    for j in range(5):
        fig.set_size_inches(15, fig.get_figheight(), forward=True)
        fig.suptitle(str(title))
        axs[j, 0].bar(np.arange(36), height=f[0][:, j], color='r')
        axs[j, 1].plot(f[1][:, j], 'o-')
        axs[j, 2].bar(np.arange(12), height=f[2][:, j], color='b')

##############################################################################
# Here, we use india rainfall dataset which has some missing values in it.
# If data doesn't come with a mask, we need to create it ourselves e.g. by searching
# the nan values in data.


tensor = get_tensor("rainfall")
mask = tl.ones(tl.shape(tensor))
mask[np.isnan(tensor)] = 0
tensor[np.isnan(tensor)] = 0

# Parameters
rank = 5
init = 'random'
loss = 'gaussian'

##############################################################################
# Both GCP and SGCP allow us to use mask. Here, we will use only GCP.

# GCP
tic = time.time()
tensorgcp, errorsgcp = generalized_parafac(tensor, rank=rank, init=init, return_errors=True, loss=loss,
                                           mask=mask, n_iter_max=100)
weightsgcp, factorsgcp = tensorgcp
cp_reconstructiongcp = tl.cp_to_tensor((weightsgcp, factorsgcp))
time_gcp = time.time() - tic

# NN-Parafac
tic = time.time()
tensorcp, errors = non_negative_parafac_hals(tensor, rank=rank, n_iter_max=100, init=init, return_errors=True)
weights, factors = tensorcp
cp_reconstruction = tl.cp_to_tensor((weights, factors))
time_cp = time.time() - tic

each_iteration(errorsgcp, "GCP")

each_iteration(errors, "NN-HALS")


print("RMSE for GCP:", RMSE(tensor, cp_reconstructiongcp))
print("RMSE for NN-CP:", RMSE(tensor, cp_reconstruction))

print("GCP time:", time_gcp)
print("NN-CP time:", time_cp)

############################################################
# Here, we plot components of the factors for interpretation.
# Here GCP components,

plot_components(factorsgcp, 'GCP')

############################################################
# and NN-CP components;

plot_components(factors, 'NN-Parafac')
