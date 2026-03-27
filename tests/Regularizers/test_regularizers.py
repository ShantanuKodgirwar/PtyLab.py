import pytest
import numpy as np
from numpy.testing import assert_allclose
from PtyLab.Regularizers import divergence, grad_TV


@pytest.fixture
def complex_object():
    shape = (1, 1, 1, 1, 380, 380)
    obj = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    return obj - (0.5 + 0.5j)


def test_grad_tv_matches_divergence(complex_object):
    epsilon = 1e-2
    gradient = np.gradient(complex_object, axis=(4, 5))
    norm = (gradient[0] + gradient[1]) ** 2
    temp = [gradient[0] / np.sqrt(norm + epsilon), gradient[1] / np.sqrt(norm + epsilon)]
    TV_update = divergence(temp)
    TV_update_2 = grad_TV(complex_object, epsilon)
    assert_allclose(TV_update, TV_update_2)
