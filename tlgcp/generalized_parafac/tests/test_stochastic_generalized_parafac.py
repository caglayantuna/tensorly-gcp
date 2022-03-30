from .._stochastic_generalized_parafac import stochastic_generalized_parafac, stochastic_gradient, Stochastic_GCP
from tensorly.testing import assert_, assert_class_wrapper_correctly_passes_arguments
from tensorly.cp_tensor import cp_to_tensor
from tensorly.random import random_cp
import tensorly as tl
from ...utils import loss_operator


def test_stochastic_gradient():
    """Test for the Stochastic gradient
    """
    shape = [8, 10, 6]
    rank = 3
    weights, factors = random_cp(shape, rank)
    tensor = tl.cp_to_tensor((weights, factors))*100
    gradient = stochastic_gradient(tensor, factors, batch_size=10)
    assert_(tl.shape(gradient[0])[0] == shape[0])
    assert_(tl.shape(gradient[1])[0] == shape[1])
    assert_(tl.shape(gradient[2])[0] == shape[2])


def test_stochastic_generalized_parafac(monkeypatch):
    """Test for the Stochastic Generalized Parafac decomposition
    """
    tol_norm_2 = 0.3
    rank = 3
    shape = [8, 10, 6]
    init = 'random'
    rng = tl.check_random_state(1234)
    initial_tensor = cp_to_tensor(random_cp(shape, rank=rank))
    batch_size = 8

    # Gaussian
    loss = 'gaussian'
    gcp_result = stochastic_generalized_parafac(initial_tensor, loss=loss, rank=rank, n_iter_max=100, init=init)
    reconstructed_tensor = cp_to_tensor(gcp_result)
    error = loss_operator(initial_tensor, reconstructed_tensor, loss)
    error = tl.sum(error) / tl.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Gamma
    loss = 'gamma'
    array = rng.gamma(1, initial_tensor, size=shape)
    tensor = tl.tensor(array)
    gcp_result = stochastic_generalized_parafac(tensor, loss=loss, rank=rank, init=init, n_iter_max=100, batch_size=batch_size)
    reconstructed_tensor = cp_to_tensor(gcp_result)
    error = loss_operator(initial_tensor, reconstructed_tensor, loss)
    error = tl.sum(error) / tl.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Rayleigh
    loss = 'rayleigh'
    array = rng.rayleigh(initial_tensor, size=shape)
    tensor = tl.tensor(array)
    gcp_result = stochastic_generalized_parafac(tensor, loss=loss, rank=rank, init=init, n_iter_max=100, batch_size=batch_size)
    reconstructed_tensor = cp_to_tensor(gcp_result)
    error = loss_operator(initial_tensor, reconstructed_tensor, loss)
    error = tl.sum(error) / tl.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Poisson-count
    loss = 'poisson_count'
    array = 1.0 * rng.poisson(initial_tensor, size=shape)
    tensor = tl.tensor(array)
    gcp_result = stochastic_generalized_parafac(tensor, loss=loss, rank=rank, init=init, n_iter_max=500, batch_size=batch_size)
    reconstructed_tensor = cp_to_tensor(gcp_result)
    error = loss_operator(initial_tensor, reconstructed_tensor, loss)
    error = tl.sum(error) / tl.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Poisson-log
    loss = 'poisson_log'
    array = 1.0 * rng.poisson(tl.exp(initial_tensor), size=shape)
    tensor = tl.tensor(array)
    gcp_result = stochastic_generalized_parafac(tensor, loss=loss, rank=rank, init=init, n_iter_max=500, batch_size=batch_size)
    reconstructed_tensor = cp_to_tensor(gcp_result)
    error = loss_operator(initial_tensor, reconstructed_tensor, loss)
    error = tl.sum(error) / tl.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Bernoulli-odds
    loss = 'bernoulli_odds'
    array = 1.0 * rng.binomial(1, initial_tensor / (initial_tensor + 1), size=shape)
    tensor = tl.tensor(array)
    gcp_result = stochastic_generalized_parafac(tensor, loss=loss, rank=rank, init=init, n_iter_max=100, batch_size=batch_size)
    reconstructed_tensor = cp_to_tensor(gcp_result)
    error = loss_operator(initial_tensor, reconstructed_tensor, loss)
    error = tl.sum(error) / tl.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Bernoulli-logit
    loss = 'bernoulli_logit'
    array = 1.0 * rng.binomial(1, tl.exp(initial_tensor) / (tl.exp(initial_tensor) + 1), size=shape)
    tensor = tl.tensor(array)
    gcp_result = stochastic_generalized_parafac(tensor, loss=loss, rank=rank, init=init, batch_size=batch_size,
                                                epochs=100, n_iter_max=100)
    reconstructed_tensor = cp_to_tensor(gcp_result)
    error = loss_operator(initial_tensor, reconstructed_tensor, loss)
    error = tl.sum(error) / tl.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')
    assert_class_wrapper_correctly_passes_arguments(monkeypatch, stochastic_generalized_parafac, Stochastic_GCP, rank=3)