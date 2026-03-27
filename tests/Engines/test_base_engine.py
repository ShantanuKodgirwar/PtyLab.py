import pytest
import numpy as np
import logging
import PtyLab

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False


@pytest.fixture
def engine_setup():
    experimentalData, reconstruction, params, monitor, ePIE_engine = PtyLab.easyInitialize(
        "example:simulation_cpm", operationMode="CPM"
    )
    return reconstruction, ePIE_engine


def test_move_data_to_cpu(engine_setup):
    reconstruction, ePIE_engine = engine_setup
    ePIE_engine.reconstruction.logger.setLevel(logging.DEBUG)
    ePIE_engine._move_data_to_cpu()
    ePIE_engine._move_data_to_cpu()
    assert type(ePIE_engine.reconstruction.object) is np.ndarray


@pytest.mark.skipif(not HAS_GPU, reason="no GPU available")
def test_move_data_to_gpu(engine_setup):
    reconstruction, ePIE_engine = engine_setup
    ePIE_engine.reconstruction.logger.setLevel(logging.DEBUG)
    ePIE_engine._move_data_to_gpu()
    ePIE_engine._move_data_to_gpu()
    assert type(ePIE_engine.reconstruction.object) is cp.ndarray


@pytest.mark.skip(reason="incomplete/experimental test")
def test_position_correction(engine_setup):
    pass
