import pytest
import numpy as np
from PtyLab.io import readExample


@pytest.mark.skip(reason="missing example data file: LungCarcinomaSmallFPM.hdf5")
def test_read_example_fpm():
    readExample.listExamples()
    archive = readExample.loadExample("simulation_fpm")
    assert np.array(archive["Nd"], int) == 256


@pytest.mark.skip(reason="missing example data file: simulationTiny.hdf5")
def test_read_example_simulation_tiny():
    archive = readExample.loadExample("simulationTiny")
    assert np.array(archive["Nd"], int) == 64
