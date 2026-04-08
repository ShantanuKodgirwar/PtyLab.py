import pytest
import numpy as np
from numpy.testing import assert_almost_equal
import unittest
from PtyLab.Operators.Operators import aspw, scaledASP, fresnelPropagator
from PtyLab.utils.utils import circ


@pytest.fixture
def wave_params():
    dx = 5e-6
    N = 100
    x = np.arange(-N / 2, N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return {
        "dx": dx,
        "E_in": circ(X, Y, N / 2 * dx).astype(float),
        "wavelength": 600e-9,
        "z": 1e-4,
        "L": dx * N,
    }


def test_aspw(wave_params):
    p = wave_params
    E_1, _ = aspw(p["E_in"], 0, p["wavelength"], p["L"], is_FT=False)
    E_2, _ = aspw(E_1, p["z"], p["wavelength"], p["L"], is_FT=False)
    E_3, _ = aspw(E_2, -p["z"], p["wavelength"], p["L"], is_FT=False)
    assert_almost_equal(p["E_in"], E_1)
    assert_almost_equal(abs(E_1), abs(E_3))


def test_scaledASP(wave_params):
    p = wave_params
    E_1, _, _ = scaledASP(p["E_in"], p["z"], p["wavelength"], p["dx"], p["dx"])
    E_2, _, _ = scaledASP(E_1, -p["z"], p["wavelength"], p["dx"], p["dx"])
    assert_almost_equal(abs(E_2), abs(p["E_in"]))


@pytest.mark.skip(reason="fresnelPropagator not yet validated")
def test_fresnelPropagator(wave_params):
    p = wave_params
    E_out = fresnelPropagator(p["E_in"], 0, p["wavelength"], p["L"])
    assert_almost_equal(p["E_in"], E_out)
