:no-toc:
:no-localtoc:
:no-pagination:

.. only:: latex

   TensorLy-Gcp
   ================


.. TensorLy-Gcp documentation

.. only:: html

   .. raw:: html

      <div class="container content">
      <br/><br/>

.. image:: _static/logos/logo_tensorly.png
   :align: center
   :width: 1000

.. only:: html

   .. raw:: html 
   
      <div class="has-text-centered">
         <h3> Generalized Tensor Decomposition </h3>
      </div>
      <br/><br/>

.. toctree::
   :maxdepth: 1
   :hidden:

   install
   user_guide/index
   modules/api
   auto_examples/index
   Notebooks <https://github.com/caglayantuna/tensorly-gcp/tree/master/tlgcp/notebooks>
   about

TensorLy-Gcp is a Python library for Generalized Parafac Decomposition Learning that
builds on top of `TensorLy <https://github.com/tensorly/tensorly/>`_.

With TensorLy-Gcp, you can easily: 

- **Losses**: You can use built-in losses such as gamma, poisson, rayleigh, gaussian and binomial, or you define your own losses.
- **Stochastic**: It is also possible to implement stochastic version with selecting the batch size.
- **Different backends**: TensorLy-Gcp uses LBFGS-S optimization which works for all backends (except Mxnet) of TensorLy with some constraints.

.. only:: html

   .. raw:: html

      <br/> <br/>
      <br/>

      <div class="container has-text-centered">
      <a class="button is-large is-dark is-primary" href="install.html">
         Start Generalizing!
      </a>
      </div>
      
      </div>
