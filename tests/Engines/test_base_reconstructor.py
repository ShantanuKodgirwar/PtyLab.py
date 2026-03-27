import pytest
from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Params.Params import Params
from PtyLab.Monitor.Monitor import Monitor
from PtyLab.Engines.BaseEngine import BaseEngine


@pytest.fixture
def engine():
    experimentalData = ExperimentalData("test:nodata")
    params = Params()
    monitor = Monitor()
    optimizable = Reconstruction(experimentalData, params)
    return BaseEngine(optimizable, experimentalData, params, monitor), experimentalData


def test_change_optimizable(engine):
    BR, experimentalData = engine
    optimizable2 = Reconstruction(experimentalData, Params())
    BR.changeOptimizable(optimizable2)
    assert BR.reconstruction is optimizable2


def test_set_position_order(engine):
    pass


def test_get_error_metrics(engine):
    pass
