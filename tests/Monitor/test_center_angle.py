import pytest
import matplotlib
matplotlib.use("Agg")
import numpy as np
pytest.importorskip("tensorflow")
from PtyLab.Monitor.TensorboardMonitor import center_angle
from scipy import ndimage


def test_center_angle():
    N = 128
    Ein = np.fft.fftshift(ndimage.fourier_shift(np.ones((N, N)), [5, 0]))
    Ein_c = center_angle(Ein)
    assert Ein_c is not None
    assert Ein_c.shape == Ein.shape
