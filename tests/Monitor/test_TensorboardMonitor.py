import pytest
import numpy as np
from PtyLab.Monitor.TensorboardMonitor import TensorboardMonitor

imageio = pytest.importorskip("imageio")


@pytest.fixture
def monitor():
    return TensorboardMonitor(name="testing purposes only")


def test_diffraction_data(monitor):
    estimated = imageio.imread("imageio:camera.png")
    measured = np.fliplr(estimated)
    for i in range(10):
        monitor.i = i
        monitor.updateDiffractionDataMonitor(estimated, measured)


def test_update_object_probe(monitor):
    object_estimate = np.random.rand(1, 1280, 1280)
    probe_estimate = np.random.rand(1, 64, 64) * 1j
    for i in range(10):
        monitor.updatePlot(object_estimate, probe_estimate)


def test_update_error(monitor):
    errors = np.random.rand(100).cumsum()[::-1]
    for e in errors:
        monitor.i += 1
        monitor.updateObjectProbeErrorMonitor(e)
        monitor.updateObjectProbeErrorMonitor([e])


def test_update_z(monitor):
    z = np.random.rand(100) / 100
    z[0] = 10
    z[30:50] = 0
    z[80:120] = 0
    z = z.cumsum()
    for zi in z:
        monitor.i += 1
        monitor.update_z(zi)


def test_update_positions(monitor):
    positions = np.random.rand(100, 2).cumsum(axis=1)
    positions = np.cos(positions)
    scaling = 1.5
    other_positions = positions + np.random.rand(*positions.shape) * 1e-2
    monitor.update_positions(positions, scaling * other_positions, scaling)
