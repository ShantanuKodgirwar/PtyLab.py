import pytest
from numpy.testing import assert_almost_equal
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.Params.Params import Params
from PtyLab.Engines import ePIE
from PtyLab.Monitor.Monitor import Monitor


@pytest.mark.skip(reason="missing example data file: simulation_ptycho")
def test_fresnel_propagator_round_trip():
    exampleData = ExperimentalData()
    exampleData.loadData("example:simulation_ptycho")
    exampleData.operationMode = "CPM"

    optimizable = Reconstruction(exampleData, Params())
    optimizable.npsm = 1
    optimizable.nosm = 1
    optimizable.nlambda = 1
    optimizable.initializeObjectProbe()

    monitor = Monitor()
    ePIE_engine = ePIE.ePIE(optimizable, exampleData, monitor)
    ePIE_engine.propagatorType = "ASP"
    ePIE_engine.numIterations = 1
    ePIE_engine.reconstruct()

    A = optimizable.esw
    ePIE_engine.object2detector()
    ePIE_engine.detector2object()

    assert_almost_equal(A, optimizable.esw)
