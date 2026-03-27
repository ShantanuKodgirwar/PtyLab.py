import pytest
from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Params.Params import Params
from PtyLab.Monitor.Monitor import Monitor
from PtyLab.Engines.ePIE import ePIE


@pytest.fixture
def epie_engine():
    experimentalData = ExperimentalData("test:nodata")
    params = Params()
    monitor = Monitor()
    optimizable = Reconstruction(experimentalData, params)
    return ePIE(optimizable, experimentalData, params, monitor)


def test_init(epie_engine):
    pass
