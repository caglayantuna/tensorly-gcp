.. _user_guide-backend:

TensorLy's backend system
=========================

.. note::

   In short, you can write your code using TensorLy and you can transparently combine it and execute with any of the backends. 
   Currently we support NumPy PyTorch, MXNet, JAX, TensorFlow and CuPy as backends.

Why backends?
-------------
The goal of TensorLy is to make tensor methods accessible.
While NumPy needs no introduction, other backends such as MXNet and PyTorch backends are especially useful as they allows to perform transparently computation on CPU or GPU. 
Last but not least, using MXNet or PyTorch as a backend, we are able to combine tensor methods and deep learning easily!



How do I change the backend?
----------------------------
To change the backend, e.g. to NumPy, you can change the value of ``default_backend`` in tensorly/__init__.
Alternatively during the execution, assuming you have imported TensorLy as ``import tensorly as tl``, you can change the backend in your code by calling ``tl.set_backend('numpy')``.

.. important::
   
   NumPy is installed by default with TensorLy if you haven't already installed it. 
   However, to keep dependencies as minimal as possible, and to not complexify installation, neither MXNet nor PyTorch are installed.  If you want to use them as backend, you will have to install them first. 
   It is easy however, simply refer to their respective installation instructions:

   * `PyTorch <http://pytorch.org>`_
   * `MXNet <https://mxnet.apache.org/install/index.html>`_
   * `JAX <https://jax.readthedocs.io/en/latest/developer.html#building-or-installing-jaxlib>`_ 
   * `CuPy <https://docs.cupy.dev/en/stable/install.html>`_
   * `TensorFlow <https://www.tensorflow.org/install>`_ 


Once you change the backend, all the computation is done using that backend.

Backend in Tensorly-Gcp?
------------------------
