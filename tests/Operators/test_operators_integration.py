import pytest
import numpy as np
from numpy.testing import assert_allclose

from PtyLab import Operators, easyInitialize

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False


@pytest.mark.skip(reason="requires cupyx and GPU; performance benchmark only")
def test_caching_aspw():
    xp = cp if HAS_GPU else np
    E = xp.random.rand(10, 1, 3, 512, 512)
    z = 1e-3
    wl = 512e-9
    pixel_pitch = 10e-6
    L = pixel_pitch * E.shape[-1]

    E_prop = Operators.Operators.aspw_cached(E, z, wl, L)
    if HAS_GPU:
        E_prop = xp.asnumpy(E_prop)

    E_prop2 = Operators.Operators.aspw(E, z, wl, L)[0]
    if HAS_GPU:
        E_prop2 = E_prop2.get()

    assert_allclose(E_prop, E_prop2)


def test_object2detector():
    experimentalData, reconstruction, params, monitor, engine = easyInitialize(
        "example:simulation_cpm"
    )
    params.gpuSwitch = False
    reconstruction._move_data_to_cpu()
    for operator_name in Operators.Operators.forward_lookup_dictionary:
        params.propagatorType = operator_name
        reconstruction.esw = reconstruction.probe
        Operators.Operators.object2detector(reconstruction.esw, params, reconstruction)


def test_propagate_fresnel():
    experimentalData, reconstruction, params, monitor, engine = easyInitialize(
        "example:simulation_cpm"
    )
    reconstruction.initializeObjectProbe()
    reconstruction.esw = 2
    params.gpuSwitch = False
    reconstruction._move_data_to_cpu()

    for operator in [
        Operators.Operators.propagate_fresnel,
        Operators.Operators.propagate_ASP,
        Operators.Operators.propagate_scaledASP,
        Operators.Operators.propagate_twoStepPolychrome,
        Operators.Operators.propagate_scaledPolychromeASP,
    ]:
        operator(reconstruction.probe, params, reconstruction)


@pytest.mark.skip(reason="placeholder - not implemented")
def test_aspw_cached():
    pass


def test_propagate_asp_fft_equivalence():
    experimentalData, reconstruction, params, monitor, engine = easyInitialize(
        "example:simulation_cpm"
    )
    reconstruction.esw = None
    a = reconstruction.probe
    P1 = Operators.Operators.propagate_ASP(a, params, reconstruction, z=1e-3, fftflag=False)[1]
    P2 = Operators.Operators.propagate_ASP(a, params, reconstruction, z=1e-3, fftflag=True)[1]
    assert_allclose(P1, P2)
