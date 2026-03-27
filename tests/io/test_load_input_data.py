import pytest
from PtyLab.io import getExampleDataFolder
from PtyLab.io.readHdf5 import loadInputData


@pytest.mark.skip(reason="missing example data file: fourier_simulation.hdf5")
def test_load_input_data():
    example_data_folder = getExampleDataFolder()
    filename = example_data_folder / "fourier_simulation.hdf5"
    result = loadInputData(filename)
    assert result["ptychogram"].shape == (49, 256, 256)
