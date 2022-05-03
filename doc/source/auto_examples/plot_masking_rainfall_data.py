"""
Generalized Parafac with missing values
===============================================
On this page, you will find examples showing how to handle missing data with Generalized CP (GCP).
"""

##############################################################################
# Introduction
# -----------------------
# Some data could have missing values in it.

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
    for j in range(5):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_size_inches(15, fig.get_figheight(), forward=True)
        fig.suptitle(str(title) + ' ' + 'Rank' + ' ' + str(j+1))
        ax1.bar(np.arange(36),height=f[0][:, j], color='r')
        ax2.plot(f[1][:, j], 'o-')
        ax3.bar(np.arange(12), height=f[2][:, j], color='b')


tensor = get_tensor("rainfall")
mask = tl.ones(tl.shape(tensor))
mask[np.isnan(tensor)] = 0
tensor[np.isnan(tensor)] = 0

# Parameters
rank = 5
init = 'random'
loss = 'gaussian'

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

plot_components(factorsgcp, 'GCP')
plot_components(factors, 'NN-Parafac')
