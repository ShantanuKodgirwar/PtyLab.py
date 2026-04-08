import numpy as np
from numpy.testing import assert_almost_equal
from PtyLab.utils.utils import fft2c, ifft2c


def test_fft2c_ifft2c_unitary():
    E_in = np.random.rand(5, 100, 100) + 1j * np.random.rand(5, 100, 100) - 0.5 - 0.5j
    assert_almost_equal(ifft2c(fft2c(E_in)), E_in)
    assert_almost_equal(fft2c(ifft2c(E_in)), E_in)
