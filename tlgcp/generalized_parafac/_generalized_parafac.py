from tensorly.random import random_cp
from tensorly.decomposition._base_decomposition import DecompositionMixin
import tensorly as tl
from tensorly.cp_tensor import CPTensor, validate_cp_rank, unfolding_dot_khatri_rao
import math
from ..utils import lbfgs


def vectorize_factors(factors):
    """
    Vectorizes each factor, then concatenates them to return one vector.

    Parameters
    ----------
    factors : list of ndarray

    Returns
    -------
    vectorized_factors: vector
    """
    vectorized_factors = []
    for i in range(len(factors)):
        vectorized_factors.append(tl.tensor_to_vec(factors[i]))
    vectorized_factors = tl.concatenate(vectorized_factors, axis=0)
    return vectorized_factors


def vectorized_factors_to_tensor(vectorized_factors, shape, rank, return_factors=False):
    """
    Transforms vectorized factors of a CP decomposition into a reconstructed full tensor.

    Parameters
    ----------
    vectorized_factors : 1darray, a vector of length :math:`\prod(shape) \times rank^\{\len(shape)\}`
    shape : tuple, contains the row dimensions of the factors
    rank : int, number of components in the CP decomposition
    return_factors : bool, if True returns factors list instead of full tensor 
        Default: False


    Returns
    -------
    tensor: ndarray or list of ndarrays
    """
    n_factors = len(shape)
    factors = []

    cursor = 0
    for i in range(n_factors):
        factors.append(tl.reshape(vectorized_factors[cursor: cursor + shape[i]*rank], [shape[i], rank]))
        cursor += shape[i] * rank

    if return_factors:
        return CPTensor((None, factors))
    else:
        return tl.cp_to_tensor((None, factors))


def vectorized_mttkrp(tensor, vectorized_factors, rank):
    """
    Computes the Matricized Tensor Times Khatri-Rao Product (MTTKRP) for all modes between a tensor and vectorized factors. Returns a vectorized stack of MTTKRPs.

    Parameters
    ----------
    tensor : ndarray, data tensor
    vectorized_factors : 1darray, factors of the CP decomposition stored in one vector 
    rank : int, number of components in the CP decomposition

    Returns
    -------
    vectorized_mttkrp :
        vector of length vectorized_factors containing the mttkrp for all modes
    """
    _, factors = vectorized_factors_to_tensor(vectorized_factors, tl.shape(tensor), rank, return_factors=True)
    all_mttkrp = []
    for i in range(len(factors)):
        all_mttkrp.append(tl.tensor_to_vec(unfolding_dot_khatri_rao(tensor, (None, factors), i)))
    return tl.concatenate(all_mttkrp, axis=0)


def loss_operator_func(tensor, rank, loss):
    """
    Various loss functions for generalized parafac decomposition, see [1] for more details.
    The returned function maps a vectorized factors input x to the loss (todo maths)
    1/len(x)*L(T,x)
    where L is the maximum likelihood estimator when tensor is generated from x using one of the following distributions:
    - Gaussian
    - Gamma
    - Rayleigh
    - Poisson (count or log)
    - Bernoulli (odds or log)

    Parameters
    ----------
    tensor : ndarray, input tensor data
    rank : int, rank of
    loss : string, choices are {'gaussian', 'gamma', 'rayleigh', 'poisson_count', 'poisson_log', 'bernoulli_odds', 'bernoulli_log'}

    Returns
    -------
    function to calculate loss
         Size based normalized loss for each entry

    References
    ----------
    .. [1] Hong, D., Kolda, T. G., & Duersch, J. A. (2020).
           Generalized canonical polyadic tensor decomposition. SIAM Review, 62(1), 133-163.
    """
    shape = tl.shape(tensor)
    size = tl.prod(tl.tensor(shape, **tl.context(tensor)))
    epsilon = 1e-8
    # should we not create the reconstructed tensor here in advance? we sometimes recreate it twice in the losses, this seens like a waste of time?
    # rec_tensor = vectorized_factors_to_tensor(x,shape,rank)
    if loss == 'gaussian':
        return lambda x: tl.sum((tensor - vectorized_factors_to_tensor(x, shape, rank)) ** 2) / size
    elif loss == 'bernoulli_odds':
        return lambda x: tl.sum(tl.log(vectorized_factors_to_tensor(x, shape, rank) + 1) - (tensor * tl.log(vectorized_factors_to_tensor(x, shape, rank) + epsilon))) / size
    elif loss == 'bernoulli_logit':
        return lambda x: tl.sum(tl.log(tl.exp(vectorized_factors_to_tensor(x, shape, rank)) + 1) - (tensor * vectorized_factors_to_tensor(x, shape, rank))) / size
    elif loss == 'rayleigh':
        return lambda x: tl.sum(2 * tl.log(vectorized_factors_to_tensor(x, shape, rank) + epsilon) + (math.pi / 4) * ((tensor / (vectorized_factors_to_tensor(x, shape, rank) + epsilon)) ** 2)) / size
    elif loss == 'poisson_count':
        return lambda x: tl.sum(vectorized_factors_to_tensor(x, shape, rank) - tensor * tl.log(vectorized_factors_to_tensor(x, shape, rank) + epsilon)) / size
    elif loss == 'poisson_log':
        return lambda x: tl.sum(tl.exp(vectorized_factors_to_tensor(x, shape, rank)) - (tensor * vectorized_factors_to_tensor(x, shape, rank))) / size
    elif loss == 'gamma':
        return lambda x: tl.sum(tensor / (vectorized_factors_to_tensor(x, shape, rank) + epsilon) + tl.log(vectorized_factors_to_tensor(x, shape, rank) + epsilon)) / size
    else:
        raise ValueError('Loss "{}" not recognized'.format(loss))


def gradient_operator_func(tensor, rank, loss):
    """
    Operator to use loss functions from [1] in order to compute gradient for
    generalized parafac decomposition.

    Parameters
    ----------
    tensor : ndarray
    rank : int, rank of
    loss : {'gaussian', 'gamma', 'rayleigh', 'poisson_count', 'poisson_log', 'bernoulli_odds', 'bernoulli_log'}

    Returns
    -------
    function to calculate gradient
         Size based normalized loss for each entry
    References
    ----------
    .. [1] Hong, D., Kolda, T. G., & Duersch, J. A. (2020).
           Generalized canonical polyadic tensor decomposition. SIAM Review, 62(1), 133-163.
    """
    shape = tl.shape(tensor)
    size = tl.prod(tl.tensor(shape, **tl.context(tensor)))
    epsilon = 1e-8
    if loss == 'gaussian':
        return lambda x: vectorized_mttkrp(2 * (vectorized_factors_to_tensor(x, shape, rank) - tensor), x, rank) / size
    elif loss == 'bernoulli_odds':
        return lambda x: vectorized_mttkrp(1 / (vectorized_factors_to_tensor(x, shape, rank) + 1) - (tensor / (vectorized_factors_to_tensor(x, shape, rank) + epsilon)), x, rank) / size
    elif loss == 'bernoulli_logit':
        return lambda x: vectorized_mttkrp(tl.exp(vectorized_factors_to_tensor(x, shape, rank)) / (tl.exp(vectorized_factors_to_tensor(x, shape, rank)) + 1) - tensor, x, rank) / size
    elif loss == 'rayleigh':
        return lambda x: vectorized_mttkrp(2 / (vectorized_factors_to_tensor(x, shape, rank) + epsilon) - (math.pi / 2) * (tensor ** 2) / ((vectorized_factors_to_tensor(x, shape, rank) + epsilon) ** 3), x, rank) / size
    elif loss == 'poisson_count':
        return lambda x: vectorized_mttkrp(1 - tensor / (vectorized_factors_to_tensor(x, shape, rank) + epsilon), x, rank) / size
    elif loss == 'poisson_log':
        return lambda x: vectorized_mttkrp(tl.exp(vectorized_factors_to_tensor(x, shape, rank)) - tensor, x, rank) / size
    elif loss == 'gamma':
        return lambda x: vectorized_mttkrp(-tensor / ((vectorized_factors_to_tensor(x, shape, rank) + epsilon) ** 2) + (1 / (vectorized_factors_to_tensor(x, shape, rank) + epsilon)), x, rank) / size
    else:
        raise ValueError('Loss "{}" not recognized'.format(loss))


def initialize_generalized_parafac(tensor, rank, init='random', svd='numpy_svd', loss='gaussian', random_state=None):
    r"""Initialize factors used in `generalized parafac`.

    Parameters
    ----------
    The type of initialization is set using `init`. If `init == 'random'` then
    initialize factor matrices with uniform distribution using `random_state`. If `init == 'svd'` then
    initialize the `m`th factor matrix using the `rank` left singular vectors
    of the `m`th unfolding of the input tensor. If init is a previously initialized `cp tensor`, all
    the weights are pulled in the last factor and then the weights are set to "1" for the output tensor.

    Parameters
    ----------
    tensor : ndarray
    rank : int
    init : {'svd', 'random', cptensor}, optional
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    loss : {'gaussian', 'gamma', 'rayleigh', 'poisson_count', 'poisson_log', 'bernoulli_odds', 'bernoulli_log'}
        Some loss functions require positive factors, which is enforced by clipping
    random_state : {None, int, np.random.RandomState}
    Returns
    -------
    factors : CPTensor
        An initial cp tensor.
    """
    rng = tl.check_random_state(random_state)
    if init == 'random':
        kt = random_cp(tl.shape(tensor), rank, random_state=rng, normalise_factors=False, **tl.context(tensor))

    elif init == 'svd':
        try:
            svd_fun = tl.SVD_FUNS[svd]
        except KeyError:
            message = 'Got svd={}. However, for the current backend ({}), the possible choices are {}'.format(
                      svd, tl.get_backend(), tl.SVD_FUNS)
            raise ValueError(message)

        factors = []
        for mode in range(tl.ndim(tensor)):
            U, S, _ = svd_fun(tl.unfold(tensor, mode), n_eigenvecs=rank)

            # Put SVD initialization on the same scaling as the tensor in case normalize_factors=False
            if mode == 0:
                idx = min(rank, tl.shape(S)[0])
                U = tl.index_update(U, tl.index[:, :idx], U[:, :idx] * S[:idx])

            if tensor.shape[mode] < rank:
                random_part = tl.tensor(rng.random_sample((U.shape[0], rank - tl.shape(tensor)[mode])), **tl.context(tensor))
                U = tl.concatenate([U, random_part], axis=1)

            factors.append(U[:, :rank])
        kt = CPTensor((None, factors))
    elif isinstance(init, (tuple, list, CPTensor)):
        try:
            weights, factors = CPTensor(init)

            if tl.all(weights == 1):
                weights, factors = CPTensor((None, factors))
            else:
                weights_avg = tl.prod(weights) ** (1.0 / tl.shape(weights)[0])
                for i in range(len(factors)):
                    factors[i] = factors[i] * weights_avg
            kt = CPTensor((None, factors))
            return kt
        except ValueError:
            raise ValueError(
                'If initialization method is a mapping, then it must '
                'be possible to convert it to a CPTensor instance'
            )
    else:
        raise ValueError('Initialization method "{}" not recognized'.format(init))
    if loss == 'gamma' or loss == 'rayleigh' or loss == 'poisson_count' or loss == 'bernoulli_odds':
        kt.factors = [tl.abs(f) for f in kt[1]]
    return kt


def generalized_parafac(tensor, rank, n_iter_max=100, init='random', svd='numpy_svd',
                        random_state=None, return_errors=False, loss='gaussian', fun_loss=None, fun_gradient=None):
    """ Generalized PARAFAC decomposition by using LBFGS optimization
    Computes a rank-`rank` decomposition of `tensor` [1]_ such that::
        tensor = [|weights; factors[0], ..., factors[-1] |].

    Parameters
    ----------
    tensor : ndarray
    rank  : int
        Number of components.
    n_iter_max : int
        Maximum number of iteration
    init : {'svd', 'random', CPTensor}, optional
        Type of factor matrix initialization.
        If a CPTensor is passed, this is directly used for initialization.
        See `initialize_factors`.
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    random_state : {None, int, np.random.RandomState}
    return_errors : bool, optional
        Activate return of iteration errors
    loss : {'gaussian', 'bernoulli_odds', 'bernoulli_logit', 'rayleigh', 'poisson_count', 'poisson_log', 'gamma'}
        Default : 'gaussian'
    fun_loss : callable, optional. You can use your own loss function here if its signature is (x: 1darray vectorized cp factors) and return a scalar value.
        Default :None
    fun_gradient : callable, optional. Use this if you defined a custom loss function, this should map x to the gradient of the loss (1darray of size x.shape).
        Default :None

    Returns
    -------
    CPTensor : (weight, factors)
        * weights : 1D array of shape (rank, )
          * all ones if normalize_factors is False (default)
          * weights of the (normalized) factors otherwise
        * factors : List of factors of the CP decomposition element `i` is of shape ``(tensor.shape[i], rank)``
    errors : list
        A list of reconstruction errors at each iteration of the algorithms.
    References
    ----------
    .. [1] Hong, D., Kolda, T. G., & Duersch, J. A. (2020).
           Generalized canonical polyadic tensor decomposition. SIAM Review, 62(1), 133-163.
    .. [2] Kolda, T. G., & Hong, D. (2020). Stochastic gradients for large-scale tensor decomposition.
           SIAM Journal on Mathematics of Data Science, 2(4), 1066-1095.
    """

    rank = validate_cp_rank(tl.shape(tensor), rank=rank)
    rng = tl.check_random_state(random_state)
    norm = tl.norm(tensor, 2)
    # Initial tensor
    weights, factors = initialize_generalized_parafac(tensor, rank, init=init, svd=svd, loss=loss,
                                                      random_state=rng)
    vectorized_factors = vectorize_factors(factors)

    if loss == 'gamma' or loss == 'rayleigh' or loss == 'poisson_count' or loss == 'bernoulli_odds':
        non_negative = True
    else:
        non_negative = False

    if loss is not None:
        fun_loss = loss_operator_func(tensor, rank, loss=loss)
        fun_gradient = gradient_operator_func(tensor, rank, loss=loss)

    vectorized_factors, rec_errors = lbfgs(fun_loss, vectorized_factors, fun_gradient, n_iter_max=n_iter_max,
                                           non_negative=non_negative, norm=norm)
    _, factors = vectorized_factors_to_tensor(vectorized_factors, tl.shape(tensor), rank, return_factors=True)

    cp_tensor = CPTensor((weights, factors))
    if return_errors:
        return cp_tensor, rec_errors
    else:
        return cp_tensor


class GCP(DecompositionMixin):
    """ Generalized PARAFAC decomposition by using Fista optimization
    Computes a rank-`rank` decomposition of `tensor` [1]_ such that::
        tensor = [|weights; factors[0], ..., factors[-1] |].

    Parameters
    ----------
    tensor : ndarray
    rank  : int
        Number of components.
    n_iter_max : int
        Maximum number of iteration
    init : {'svd', 'random', CPTensor}, optional
        Type of factor matrix initialization.
        If a CPTensor is passed, this is directly used for initialization.
        See `initialize_factors`.
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    tol : float, optional
        (Default: 1e-6) Relative reconstruction error tolerance. The
        algorithm is considered to have found the global minimum when the
        reconstruction error is less than `tol`.
    verbose : int, optional
        Level of verbosity
    return_errors : bool, optional
        Activate return of iteration errors
    loss : {'gaussian', 'bernoulli_odds', 'bernoulli_logit', 'rayleigh', 'poisson_count', 'poisson_log', 'gamma'}
    cvg_criterion : {'abs_rec_error', 'rec_error'}, optional
       Stopping criterion for ALS, works if `tol` is not None.
       If 'rec_error',  ALS stops at current iteration if ``(previous rec_error - current rec_error) < tol``.
       If 'abs_rec_error', ALS terminates when `|previous rec_error - current rec_error| < tol`.
    fun_loss : callable, optional
        Default :None
    fun_gradient : callable, optional
        Default :None

    Returns
    -------
    CPTensor : (weight, factors)
        * weights : 1D array of shape (rank, )
          * all ones if normalize_factors is False (default)
          * weights of the (normalized) factors otherwise
        * factors : List of factors of the CP decomposition element `i` is of shape ``(tensor.shape[i], rank)``
        * sparse_component : nD array of shape tensor.shape. Returns only if `sparsity` is not None.
    errors : list
        A list of reconstruction errors at each iteration of the algorithms.
    References
    ----------
    .. [1] Hong, D., Kolda, T. G., & Duersch, J. A. (2020).
           Generalized canonical polyadic tensor decomposition. SIAM Review, 62(1), 133-163.
    .. [2] Kolda, T. G., & Hong, D. (2020). Stochastic gradients for large-scale tensor decomposition.
           SIAM Journal on Mathematics of Data Science, 2(4), 1066-1095.
    """

    def __init__(self, rank, n_iter_max=100, init='svd', svd='numpy_svd', loss='gaussian',
                 random_state=None, return_errors=True, fun_loss=None, fun_gradient=None):
        self.rank = rank
        self.n_iter_max = n_iter_max
        self.init = init
        self.svd = svd
        self.return_errors = return_errors
        self.loss = loss
        self.random_state = random_state
        self.fun_loss = fun_loss
        self.fun_gradient = fun_gradient

    def fit_transform(self, tensor):
        """Decompose an input tensor
        Parameters
        ----------
        tensor : tensorly tensor
            input tensor to decompose
        Returns
        -------
        CPTensor
            decomposed tensor
        """
        cp_tensor, errors = generalized_parafac(
            tensor,
            rank=self.rank,
            n_iter_max=self.n_iter_max,
            init=self.init,
            svd=self.svd,
            loss=self.loss,
            random_state=self.random_state,
            return_errors=self.return_errors,
            fun_loss=self.fun_loss,
            fun_gradient=self.fun_gradient
        )
        self.decomposition_ = cp_tensor
        self.errors_ = errors
        return self.decomposition_
