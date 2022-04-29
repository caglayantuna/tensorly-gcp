=============
API reference
=============

:mod:`tlgcp`: Generalized CP decomppsition
============================================


.. automodule:: tlgcp
    :no-members:
    :no-inherited-members:

Available functions
-------------------

.. automodule:: tlgcp.utils
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: function.rst

    lbfgs
    loss_operator
    gradient_operator

.. automodule:: tlgcp.generalized_parafac._generalized_parafac
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: function.rst

    vectorize_factors
    vectorized_factors_to_tensor
    vectorized_mttkrp
    loss_operator_func
    gradient_operator_func

.. automodule:: tlgcp.generalized_parafac._stochastic_generalized_parafac
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: function.rst

    stochastic_gradient

.. automodule:: tlgcp.data.tensor_dataset
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: function.rst

    get_tensor

Decompositions
-------------------
.. automodule:: tlgcp.generalized_parafac
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: function.rst

    generalized_parafac
    stochastic_generalized_parafac

Classes
-------------------
.. automodule:: tlgcp.generalized_parafac
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    GCP
    Stochastic_GCP
