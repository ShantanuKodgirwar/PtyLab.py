import pytest
from PtyLab.io import getExampleDataFolder


def test_example_folder_exists():
    example_data_folder = getExampleDataFolder()
    assert example_data_folder.exists(), (
        "example data folder returned does not exist on the filesystem"
    )


@pytest.mark.skip(reason="missing example data file: simulationTiny.hdf5")
def test_simulation_tiny_in_example_data():
    example_data_folder = getExampleDataFolder()
    matlabfile = example_data_folder / "simulationTiny.hdf5"
    assert matlabfile.exists(), "`simulationTiny.hdf5` is not present in the example data folder"
