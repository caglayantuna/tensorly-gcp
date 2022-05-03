TensorLy-Gcp  
===============================================  
TensorLy-Gcp is a Python library for fitting generalized parafac decomposition (GCP) [1] and its stochastic version (SGCP) [2] which allows using different losses rather than only Euclidean. Tensorly-Gcp builds on top of `TensorLy <http://tensorly.org/dev/installation.html>`_. Both GCP and SGCP algorithms fit generalized parafac given an input tensor using a gradient-based algorithm. Gradients are computed according to a loss selected by the user, and may be provided by the user, allowing customization. While GCP implementation uses Limited-memory BFGS (LBFGS) method for optimization, SGCP uses ADAM optimization as in [2]. 

The following losses are implemented in Tensorly-Gcp:

- Rayleigh
- Bernoulli odds
- Bernoulli logit
- Gamma
- Poisson count
- Poisson log
- Gaussian

Contributing
============
At the moment, only Numpy backend is supported to implement GCP and SGCP. This library can be compatible with other backends (Pytorch, Tensorflow, Jax, Mxnet) by improving LBFGS with the given information in `here <https://github.com/caglayantuna/tensorly-gcp/blob/master/tlgcp/utils/_lbfgs.py>`_. Then, it could eventually be merged in TensorLy.

Usage
============
It is possible select one of the losses from the list above and fit generalized decomposition easily:

.. code:: python
    
    from tlgcp import generalized_parafac
    rank = 3
    loss = 'rayleigh'
    tensorgcp, errorsgcp = generalized_parafac(tensor, rank=rank, loss=loss)


Installing TensorLy-GCP  
=========================
Through pip
-----------

.. code:: 

   pip install tensorly-gcp   
   
From source
-----------

.. code::

  git clone https://github.com/tensorly/gcp
  cd gcp
  pip install -e .
  
  
  
References  
----------  
  
.. [1] Hong, D., Kolda, T. G., & Duersch, J. A. (2020). Generalized canonical polyadic tensor decomposition. SIAM Review, 62(1), 133-163. `Link-1 <https://arxiv.org/abs/1808.07452>`_  
  
.. [2] Kolda, T. G., & Hong, D. (2020). Stochastic gradients for large-scale tensor decomposition. SIAM Journal on Mathematics of Data Science, 2(4), 1066-1095. `Link-2 <https://arxiv.org/abs/1906.01687>`_
