import pytest
import numpy as np
import cupy as cp
from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Params.Params import Params
from PtyLab.Monitor.Monitor import DummyMonitor
from PtyLab.Engines.OPR import OPR

@pytest.fixture
def opr_engine():
    # Use minimal data for fast tests
    experimentalData = ExperimentalData("test:nodata")
    # Overwrite ptychogram and encoder to have more than one frame
    experimentalData.ptychogram = np.zeros((5, 16, 16), dtype=np.float32)
    experimentalData.encoder = np.zeros((5, 2), dtype=np.float64)
    experimentalData.numFrames = 5

    params = Params()
    # Configure OPR specific params
    params.OPR_modes = np.array([0, 1])
    params.OPR_subspace = 2

    monitor = DummyMonitor()
    optimizable = Reconstruction(experimentalData, params)

    # Initialize probe and object for OPR
    optimizable.initializeObjectProbe()
    # Ensure probe has enough modes for OPR_modes
    # Reconstruction.initializeProbe usually sets npsm = 1.
    # OPR expects a probe with multiple modes.
    optimizable.npsm = 2
    optimizable.probe = np.zeros((1, 1, 2, 1, optimizable.Np, optimizable.Np), dtype=np.complex64)

    return OPR(optimizable, experimentalData, params, monitor)

def test_opr_init(opr_engine):
    # Initialize probe_stack by running reconstruct with 0 iterations
    opr_engine.numIterations = 0
    opr_engine.reconstruct()

    # Check if probe_stack is initialized as a NumPy array (not CuPy)
    # and has the correct 4D shape: (Nmodes, Np, Np, Nframes)
    stack = opr_engine.reconstruction.probe_stack
    assert isinstance(stack, np.ndarray)
    assert stack.shape == (2, opr_engine.reconstruction.Np, opr_engine.reconstruction.Np, 5)

def test_opr_reconstruct_loop(opr_engine):
    # Run reconstruction for a few iterations to verify CPU <-> GPU transfers
    # and ensure no crashes occur.
    # Overwrite numIterations to be small for testing
    opr_engine.numIterations = 2

    try:
        opr_engine.reconstruct()
    except Exception as e:
        pytest.fail(f"OPR reconstruction loop failed: {e}")

def test_opr_orthogonalization(opr_engine):
    # Initialize probe_stack by running reconstruct with 0 iterations
    opr_engine.numIterations = 0
    opr_engine.reconstruct()

    # Test orthogonalizeProbeStack
    n_dim = 2
    stack = opr_engine.reconstruction.probe_stack
    try:
        updated_stack = opr_engine.orthogonalizeProbeStack(stack, n_dim)
        assert updated_stack.shape == stack.shape
        assert isinstance(updated_stack, np.ndarray)
    except Exception as e:
        pytest.fail(f"orthogonalizeProbeStack failed: {e}")

    # Test orthogonalizeIncoherentModes
    try:
        opr_engine.orthogonalizeIncoherentModes()
    except Exception as e:
        pytest.fail(f"orthogonalizeIncoherentModes failed: {e}")
