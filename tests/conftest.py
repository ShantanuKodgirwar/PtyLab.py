import numpy as np
import h5py
import pytest
from pathlib import Path


@pytest.fixture(scope="session", autouse=True)
def generate_simu_hdf5():
    """Generate a minimal simu.hdf5 if it doesn't exist (CI-friendly).

    Tests that call easyInitialize("example:simulation_cpm") resolve to
    example_data/simu.hdf5. This fixture ensures that file exists before
    any test runs, so no real dataset needs to be committed to the repo.
    If the file already exists (e.g. locally), it is left untouched.
    """
    example_data_dir = Path(__file__).parent.parent / "example_data"
    example_data_dir.mkdir(exist_ok=True)
    hdf5_path = example_data_dir / "simu.hdf5"

    if hdf5_path.exists():
        yield hdf5_path
        return

    rng = np.random.default_rng(42)
    Nd, N_frames = 64, 20

    ptychogram = rng.random((N_frames, Nd, Nd)).astype(np.float32)
    encoder = rng.uniform(-1e-3, 1e-3, (N_frames, 2))

    with h5py.File(hdf5_path, "w") as hf:
        hf.create_dataset("ptychogram", data=ptychogram, dtype="f")
        hf.create_dataset("encoder", data=encoder, dtype="f")
        hf.create_dataset("dxd", data=np.array(75e-6))
        hf.create_dataset("zo", data=np.array(0.1))
        hf.create_dataset("wavelength", data=np.array(632e-9))
        hf.create_dataset("entrancePupilDiameter", data=np.array(170e-6))

    yield hdf5_path
