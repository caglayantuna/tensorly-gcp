Optimization in Tensorly-Gcp
===============================

LBFG-S
-----------------

Standard generalized parafac decomposition uses LBFG-S optimization. We use scipy minimize lbfgs function
when numpy backend is used.

.. code-block:: python

   >>> from tlgcp.utils import lbfgs

To be able to use these function, input variable should be 1D vector. To do this, we defined a function
which vectorizes all the factors according to rank and shape variables.


ADAM
-----------------
Stochastic generalized parafac uses ADAM optimization. This optimization requires learning rate, batch size
and beta parameters as input. Suggested values from the literature are in the signature
as default for these parameters.
