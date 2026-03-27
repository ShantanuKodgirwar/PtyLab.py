import pytest
import PtyLab


@pytest.mark.skip(reason="missing example data file: ptyLab_helical_beam.h5")
def test_initial_probe_circ_smooth():
    experimentalData, reconstruction, params, monitor, ePIE_engine = PtyLab.easyInitialize(
        "example:helicalbeam", operationMode="CPM", engine=PtyLab.Engines.mPIE
    )
    experimentalData.setOrientation(4)
    experimentalData.entrancePupilDiameter = 30 * reconstruction.dxo

    reconstruction = PtyLab.Reconstruction(experimentalData, params)
    reconstruction.copyAttributesFromExperiment(experimentalData)
    reconstruction.initialProbe = "circ_smooth"
    reconstruction.initializeObjectProbe()

    engine = PtyLab.Engines.mPIE(
        reconstruction, experimentalData, params=params, monitor=monitor
    )
    engine.numIterations = 50
    monitor.probeZoom = None
    engine.reconstruct()
