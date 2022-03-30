import tensorly as tl
from tensorly.testing import assert_array_almost_equal
from .._lbfgs import lbfgs
import numpy as np

def test_lbfgs():
    a = tl.tensor(np.random.rand(10, 10))
    true_res = tl.tensor(np.random.rand(10, 10))
    b = tl.dot(a, true_res)
    x_init = tl.tensor(np.random.rand(tl.shape(true_res)[0], tl.shape(true_res)[1]))
    loss = lambda x: tl.sum((a @ tl.reshape(x, tl.shape(x_init)) - b)**2)
    result, _ = lbfgs(loss, x_init.flatten())
    result = tl.reshape(result, tl.shape(x_init))
    assert_array_almost_equal(true_res, result, decimal=2)
