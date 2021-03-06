
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_using_your_own_loss.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_using_your_own_loss.py>`
        to download the full example code

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_using_your_own_loss.py:


How to use custom loss and gradient function
===============================================
On this page, you will find examples showing how to use your own loss with Generalized CP.

.. GENERATED FROM PYTHON SOURCE LINES 8-14

Introduction
-----------------------
Tensorly-Gcp allows you to make use of your own loss within GCP. Under the
hood, GCP relies on `scipy.minimize.lbfgs` which requires a vector (1D) input.
Therefore your custom loss and gradient functions must take a vector input.
To write the loss at the tensor level more easily, we provide some reshaping functions.

.. GENERATED FROM PYTHON SOURCE LINES 14-24

.. code-block:: default


    from tlgcp import generalized_parafac
    from tensorly.metrics import RMSE
    import numpy as np
    import tensorly as tl
    import time
    from tlgcp.utils import loss_operator
    from tlgcp.generalized_parafac._generalized_parafac import vectorized_factors_to_tensor, vectorized_mttkrp
    import matplotlib.pyplot as plt








.. GENERATED FROM PYTHON SOURCE LINES 25-32

In GCP, we vectorize all the factors, however estimated tensor should be
reconstructed inside the loss and gradient. Therefore, you should use
vectorized_factors_to_tensor(x, shape, rank) to define estimated tensor
in your function. Also, vectorized_mttkrp(gradient, x , rank) is necessary
in your gradient function to follow GCP formulization, it simply computes
and unfolds the MTTKRP. Knowing this, you can define your callable function,
here e.g. as fun_loss and fun_gradient.

.. GENERATED FROM PYTHON SOURCE LINES 32-36

.. code-block:: default


    fun_loss = lambda x: tl.sum((tensor - vectorized_factors_to_tensor(x, shape, rank)) ** 2) / size
    fun_gradient = lambda x: vectorized_mttkrp(2 * (vectorized_factors_to_tensor(x, shape, rank) - tensor), x, rank) / size








.. GENERATED FROM PYTHON SOURCE LINES 37-40

In these functions, x represents vectorized factors. In our loss functions,
we use the size (total number of entries) of the tensor for normalization as
it is suggested in Hong, Kolda and Duersch's paper, but it is not mandatory.

.. GENERATED FROM PYTHON SOURCE LINES 40-48

.. code-block:: default


    rank = 5
    shape = [60, 80, 50]
    cp_tensor = tl.cp_to_tensor(tl.random.random_cp(shape, rank))
    array = np.random.binomial(1, cp_tensor / (cp_tensor + 1), size=shape)
    tensor = tl.tensor(array, dtype='float')
    size = tl.prod(tl.shape(tensor))








.. GENERATED FROM PYTHON SOURCE LINES 49-52

After we create a tensor and define necessary variables, we can apply GCP
with our defined loss and gradient functions. It should be noted that, loss
should be None to be able to use your functions.

.. GENERATED FROM PYTHON SOURCE LINES 52-59

.. code-block:: default


    tic = time.time()
    tensor_gcp, errors_gcp = generalized_parafac(tensor, rank=rank, return_errors=True,
                                                 loss=None, fun_loss=fun_loss, fun_gradient=fun_gradient)
    cp_reconstruction_gcp = tl.cp_to_tensor(tensor_gcp)
    time_gcp = time.time() - tic








.. GENERATED FROM PYTHON SOURCE LINES 60-63

As you may have noticed, the custom loss we used here is the Gaussian loss
and its associated gradient. We can compare the results with the built-in
Gaussian loss inside Tensorly-gcp.

.. GENERATED FROM PYTHON SOURCE LINES 63-80

.. code-block:: default


    loss = "gaussian"
    tic = time.time()
    tensor_gcp_gaussian, errors_gcp_gaussian = generalized_parafac(tensor, rank=rank,return_errors=True,
                                                                   loss=loss)
    cp_reconstruction_gcp_gaussian = tl.cp_to_tensor(tensor_gcp)
    time_gcp_gaussian = time.time() - tic

    print("RMSE for GCP:", RMSE(cp_tensor, cp_reconstruction_gcp))
    print("RMSE for GCP:", RMSE(cp_tensor, cp_reconstruction_gcp_gaussian))

    print("Loss for GCP:", tl.sum(loss_operator(cp_tensor, cp_reconstruction_gcp, loss)))
    print("Loss for GCP:", tl.sum(loss_operator(cp_tensor, cp_reconstruction_gcp_gaussian, loss)))

    print("GCP time:", time_gcp)
    print("GCP time:", time_gcp_gaussian)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    RMSE for GCP: 0.32711796011609506
    RMSE for GCP: 0.32711796011609506
    Loss for GCP: 0.10700615983051519
    Loss for GCP: 0.10700615983051519
    GCP time: 0.32782697677612305
    GCP time: 0.4455714225769043




.. GENERATED FROM PYTHON SOURCE LINES 81-84

As expected, results are very similar except processing time, the variation
being due to random initialization. We can also observe this behaviour by
plotting error per iteration for each method.

.. GENERATED FROM PYTHON SOURCE LINES 84-95

.. code-block:: default


    def each_iteration(a, b):
        fig=plt.figure()
        fig.set_size_inches(10, fig.get_figheight(), forward=True)
        plt.plot(a)
        plt.plot(b)
        plt.yscale('log')
        plt.legend(['GCP-defined','GCP-gaussian'], loc='upper right')


    each_iteration(errors_gcp, errors_gcp_gaussian)



.. image-sg:: /auto_examples/images/sphx_glr_plot_using_your_own_loss_001.png
   :alt: plot using your own loss
   :srcset: /auto_examples/images/sphx_glr_plot_using_your_own_loss_001.png
   :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.953 seconds)


.. _sphx_glr_download_auto_examples_plot_using_your_own_loss.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_using_your_own_loss.py <plot_using_your_own_loss.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_using_your_own_loss.ipynb <plot_using_your_own_loss.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
