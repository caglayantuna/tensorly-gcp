import tensorly as tl
import scipy
import numpy as np

def lbfgs(loss, x0, gradient=None, n_iter_max=100, non_negative=None, norm=1):
    if tl.get_backend()=="numpy":
        from scipy.optimize import minimize
        if non_negative:
            bound = scipy.optimize.Bounds(0, np.inf)
        else:
            bound = scipy.optimize.Bounds(-np.inf, np.inf)
        error = []
        error_func = lambda X:error.append(loss(X) / norm)
        return minimize(loss, x0, method='L-BFGS-B', jac=gradient, callback= error_func, options={'maxiter': n_iter_max}, bounds=bound).x, error

    elif tl.get_backend()=="pytorch":
        import torch
        x0.requires_grad = True
        optimizer = torch.optim.LBFGS([x0], line_search_fn="strong_wolfe")
        error = []
        for i in range(n_iter_max):
            optimizer.zero_grad()
            objective = loss(x0)
            objective.backward()
            if non_negative:
                  with torch.no_grad():
                      x0.clamp(min=0)
            optimizer.step(lambda: loss(x0))
            error.append(objective.item() / norm)
        return x0, error

    elif tl.get_backend() == "tensorflow":
        import tensorflow_probability as tfp
        def quadratic_loss_and_gradient(x):
            return tfp.math.value_and_gradient(loss, x)
        error = []
        for i in range(n_iter_max):
            optim_results = tfp.optimizer.lbfgs_minimize(quadratic_loss_and_gradient,
                                                         initial_position=x0,
                                                         max_iterations=1)
            error.append(optim_results.objective_value / norm)
            x0 = optim_results.position
        return optim_results.position, error

    elif tl.get_backend() == "jax":
        from jax.scipy.optimize import minimize
        method = 'l-bfgs-experimental-do-not-rely-on-this'
        error = []
        result = minimize(loss, x0, method=method, options={'maxiter': n_iter_max})
        return result.x, error


    elif tl.get_backend() == "mxnet":
        raise("There is no LBFGS method in Mxnet library")