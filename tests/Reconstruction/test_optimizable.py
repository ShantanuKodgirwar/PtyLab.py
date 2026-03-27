import logging
import pytest
from numpy.testing import assert_array_almost_equal
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.Params.Params import Params

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def reconstruction():
    data = ExperimentalData("test:nodata")
    data.wavelength = 1234
    return data, Reconstruction(data, Params())


def test_copy_scalar_attribute(reconstruction):
    data, optimizable = reconstruction
    assert optimizable.wavelength == data.wavelength
    optimizable.wavelength = 4321
    assert optimizable.wavelength != data.wavelength


@pytest.mark.skip(reason="positions is a computed property, not a settable attribute")
def test_copy_array_attribute(reconstruction):
    data, optimizable = reconstruction
    optimizable.positions += 1
    assert_array_almost_equal(optimizable.positions - 1, data.positions)
