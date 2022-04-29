.. _user_guide-backend:

TensorLy's backend system
=========================
Tensorly supports NumPy PyTorch, MXNet, JAX, TensorFlow and CuPy as backends. Our target is to
make Tensorly-GCP compatible for all these backends.

How do I change the backend?
----------------------------
To change the backend, you can start coding with ``tl.set_backend('backend_name')``.
Once you change the backend, all the computation is done using that backend.

Backend in Tensorly-Gcp?
------------------------
Tensorly-Gcp has two main features: generalized parafac and its stochastic version. Since LBFG-S solver has
some issues with the backends as it is explained in its documentation, this decomposition works for only NUmpy backend
for now. On the other hand, stochastic generalized decomposition can be used with all backends except Tensorflow because of
its indexing issue.