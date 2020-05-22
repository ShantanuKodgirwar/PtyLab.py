from fracPy.monitors.default_visualisation import DefaultMonitor,DiffractionDataMonitor
import numpy as np
import logging

# fracPy imports
from fracPy.utils.initializationFunctions import initialProbeOrObject
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.utils.utils import ifft2c, fft2c, orthogonalizeModes
from fracPy.operators.operators import aspw, scaledASP
from fracPy.monitors.Monitor import Monitor

class BaseReconstructor(object):
    """
    Common properties for any reconstruction ePIE_engine are defined here.

    Unless you are testing the code, there's hardly any need to create this object. For your own implementation,
    inherit from this object

    """
    def __init__(self, optimizable: Optimizable, experimentalData:ExperimentalData, monitor:Monitor):
        # These statements don't copy any data, they just keep a reference to the object
        self.optimizable = optimizable
        self.experimentalData = experimentalData
        self.monitor = monitor
        self.monitor.optimizable = optimizable

        # datalogger
        self.logger = logging.getLogger('BaseReconstructor')

        # Default settings
        # settings that involve how things are computed
        self.fftshiftSwitch = False
        self.FourierMaskSwitch = False
        self.CPSCswitch = False
        self.fontSize = 17
        self.intensityConstraint = 'standard'  # standard or sigmoid
        self.propagator = 'Fraunhofer' # 'Fresnel' 'ASP'


        # Specific reconstruction settings that are the same for all engines
        self.absorbingProbeBoundary = False

        # This only makes sense on a GPU, not there yet
        self.saveMemory = False

        self.objectUpdateStart = 1
        self.positionOrder = 'random'  # 'random' or 'sequential'

        ## Swtiches used in applyConstraints method:
        self.orthogonalizationFrequency = 10  # probe orthogonalization frequency
        # object regularization
        self.objectSmoothenessSwitch = False
        self.objectSmoothenessWidth = 2  # # pixels over which object is assumed fairly smooth
        self.objectSmoothnessAleph = 1e-2  # relaxation constant that determines strength of regularization
        self.absObjectSwitch = False  # force the object to be abs-only
        self.absObjectBeta = 1e-2  # relaxation parameter for abs-only constraint
        self.objectContrastSwitch = False  # pushes object to zero outside ROI
        # probe regularization
        self.probeSmoothenessSwitch = False # enforce probe smootheness
        self.probeSmoothnessAleph = 5e-2  # relaxation parameter for probe smootheness
        self.probeSmoothenessWidth = 3  # loose object support diameter
        self.absorbingProbeBoundary = False  # controls if probe has period boundary conditions (zero)
        self.probePowerCorrectionSwitch = False  # probe normalization to measured PSD
        self.modulusEnforcedProbeSwitch = False  # enforce empty beam
        self.comStabilizationSwitch = False  # center of mass stabilization for probe
        # other
        self.backgroundModeSwitch = False  # background estimate
        self.comStabilizationSwitch = False # center of mass stabilization for probe
        self.PSDestimationSwitch = False
        self.objectContrastSwitch = False # pushes object to zero outside ROI



        # initialize detector error matrices
        if self.saveMemory:
            self.detectorError = 0
        else:
            self.detectorError = np.zeros((self.experimentalData.numFrames,
                                          self.experimentalData.Nd, self.experimentalData.Nd))

        # initialize energy at each scan position
        if not hasattr(self, 'errorAtPos'):
            self.errorAtPos = np.zeros((self.experimentalData.numFrames, 1), dtype=np.float32)

        if len(self.experimentalData.ptychogram) != 0:
            self.energyAtPos = np.sum(np.sum(abs(self.experimentalData.ptychogram), axis=-1), axis=-1)
        else:
            raise Exception('ptychogram is empty')

        # initialize quadraticPhase term
        # todo multiwavelength implementation
        # todo check why fraunhofer also need quadraticPhase term

        if self.propagator == 'Fresnel':
            self.propagator.quadraticPhase = np.exp(1.j * np.pi/(self.experimentalData.wavelength * self.experimentalData.zo)
                                         * (self.experimentalData.Xp**2 + self.experimentalData.Yp**2))
        elif self.propagator == 'ASP':
            _, self.propagator.transferFunction = aspw(np.squeeze(self.optimizable.probe[0, 0, 0, 0, :, :]),
                                                       self.experimentalData.zo, self.experimentalData.wavelength,
                                                       self.experimentalData.Lp)
            if self.fftshiftSwitch:
                raise ValueError('ASP propagator works only with fftshiftSwitch = False!')

        elif self.propagator =='scaledASP':
            _,self.propagator.Q1,self.propagator.Q2 = scaledASP(np.squeeze(self.optimizable.probe[0, 0, 0, 0, :, :]),
                                                                self.experimentalData.zo, self.experimentalData.wavelength,
                                                                self.experimentalData.dxo,self.experimentalData.dxd)



    def setPositionOrder(self):
        if self.positionOrder == 'sequential':
            self.positionIndices = np.arange(self.experimentalData.numFrames)

        elif self.positionOrder == 'random':
            if self.optimizable.error.size == 0:
                self.positionIndices = np.arange(self.experimentalData.numFrames)
            else:
                if len(self.optimizable.error) < 2:
                    self.positionIndices = np.arange(self.experimentalData.numFrames)
                else:
                    self.positionIndices = np.arange(self.experimentalData.numFrames)
                    np.random.shuffle(self.positionIndices)
        else:
            raise ValueError('position order not properly set')



    def changeExperimentalData(self, experimentalData:ExperimentalData):
        self.experimentalData = experimentalData

    def changeOptimizable(self, optimizable: Optimizable):

        self.optimizable = optimizable


    def startReconstruction(self):
        raise NotImplementedError()


    def convert2single(self):
        """
        Convert the datasets to single precision. Matches: convert2single.m
        :return:
        """
        self.dtype_complex = np.complex64
        self.dtype_real = np.float32
        self._match_dtypes_complex()
        self._match_dtypes_real()

    def _match_dtypes_complex(self):
        raise NotImplementedError()

    def _match_dtypes_real(self):
        raise NotImplementedError()

    def detector2object(self):
        """
        Propagate the ESW to the object plane (in-place).

        Matches: detector2object.m
        :return:
        """
        if self.propagator == 'Fraunhofer':
            self.ifft2s()
        elif self.propagator == 'Fresnel':
            self.ifft2s()
            self.optimizable.esw = self.optimizable.esw * np.conj(self.propagator.quadraticPhase)
        elif self.propagator == 'ASP':
            self.optimizable.esw = ifft2c(fft2c(self.optimizable.esw) * np.conj(self.propagator.transferFunction))
        elif self.propagator == 'scaledASP':
            self.optimizable.esw = ifft2c(fft2c(self.optimizable.esw) * np.conj(self.propagator.Q2)
                                          ) * np.conj(self.propagator.Q1)
        else:
            raise Exception('Propagator is not properly set, choose from Fraunhofer, Fresnel, ASP and scaledASP')


    def exportOjb(self, extension='.mat'):
        """
        Export the object.

        If extension == '.mat', export to matlab file.
        If extension == '.png', export to a png file (with amplitude-phase)

        Matches: exportObj (except for the PNG)

        :return:
        """
        raise NotImplementedError()

    def fft2s(self):
        """
        fft2s.m
        :return:
        """
        if self.fftshiftSwitch:
            self.optimizable.ESW = np.fft.fft2(self.optimizable.esw, norm='ortho') #/ self.experimentalData.Np
        else:
            self.optimizable.ESW = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.optimizable.esw),norm='ortho')) #/ self.experimentalData.Np

    def getBeamWidth(self):
        """
        Matches getBeamWith.m
        :return:
        """
        raise NotImplementedError()


    def getErrorMetrics(self):
        """
        matches getErrorMetrics.m
        :return:
        """

        if not self.saveMemory:
            # Calculate mean error for all positions (make separate function for all of that)
            if self.FourierMaskSwitch:
                self.errorAtPos = np.sum(np.abs(self.detectorError) * self.W)
            else:
                self.errorAtPos = np.sum(np.abs(self.detectorError))

        self.errorAtPos /= (self.energyAtPos + 1)
        eAverage = np.sum(self.errorAtPos)

        # append to error vector (for plotting error as function of iteration)
        self.optimizable.error = np.append(self.optimizable.error, eAverage)



    def getRMSD(self, positionIndex):
        """
        Root mean square deviation between ptychogram and intensity estimate
        :param positionIndex:
        :return:
        """
        # TODO: change error for the 6D array
        # self.currentDetectorError = abs(self.optimizable.Imeasured-self.optimizable.Iestimated)
        self.currentDetectorError = np.sum(abs(self.optimizable.Imeasured-self.optimizable.Iestimated),axis=(0,1,2,3))
        if self.saveMemory:
            if self.FourierMaskSwitch and not self.CPSCswitch:
                self.errorAtPos[positionIndex] = np.sum(self.currentDetectorError*self.W)
            elif self.FourierMaskSwitch and self.CPSCswitch:
                raise NotImplementedError
            else:
                self.errorAtPos[positionIndex] = np.sum(self.currentDetectorError)
        else:
            self.detectorError[positionIndex] = self.currentDetectorError



    def ifft2s(self):
        """ Inverse FFT"""
        if self.fftshiftSwitch:
            self.optimizable.eswUpdate = np.fft.ifft2(self.optimizable.ESW, norm='ortho')# * self.experimentalData.Np
        else:
            self.optimizable.eswUpdate = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(self.optimizable.ESW),norm='ortho')) #* self.experimentalData.Np


    def intensityProjection(self, positionIndex):
        """ Compute the projected intensity.
            Barebones, need to implement other methods
        """
        self.object2detector()

        # zero division mitigator
        gimmel = 1e-10

        # Todo: Background mode, CPSCswitch, Fourier mask

        # these are amplitudes rather than intensities
        # self.optimizable.Iestimated = np.sum(np.abs(self.optimizable.ESW)**2, axis = 0)
        self.optimizable.Iestimated = np.sum(np.abs(self.optimizable.ESW) ** 2, axis= (0,1,2,3), keepdims=True)
        self.optimizable.Imeasured = self.experimentalData.ptychogram[positionIndex]

        self.getRMSD(positionIndex)

        # TODO: implement other update methods
        frac = np.sqrt(self.optimizable.Imeasured / (self.optimizable.Iestimated + gimmel))

        # update ESW
        self.optimizable.ESW = self.optimizable.ESW * frac
        self.detector2object()


    def object2detector(self):
        """
        Implements object2detector.m
        :return:
        """
        if self.propagator == 'Fraunhofer':
            self.fft2s()
        elif self.propagator == 'Fresnel':
            self.optimizable.esw = self.optimizable.esw * self.propagator.quadraticPhase
            self.fft2s()
        elif self.propagator == 'ASP':
            self.optimizable.esw = ifft2c( fft2c(self.optimizable.esw) * self.propagator.transferFunction)
        elif self.propagator == 'scaledASP':
            self.optimizable.esw = ifft2c(fft2c(self.optimizable.esw * self.propagator.Q1) * self.propagator.Q2)
        else:
            raise Exception('Propagator is not properly set, choose from Fraunhofer, Fresnel, ASP and scaledASP')

    def orthogonalization(self):
        """
        Perform orthogonalization
        :return:
        """
        if self.optimizable.npsm > 1:
            for id_l in range(self.optimizable.nlambda):
                for id_s in range(self.optimizable.nslice):
                    self.optimizable.probe[id_l,0,:,id_s,:,:], self.normalizedEigenvaluesProbe, self.MSPVprobe =\
                        orthogonalizeModes(self.optimizable.probe[id_l,0,:,id_s,:,:])
                    self.optimizable.purity = np.sqrt(np.sum(self.normalizedEigenvaluesProbe**2))
            # self.optimizable.probe, self.normalizedEigenvaluesProbe, self.MSPVprobe =\
            #     orthogonalizeModes(self.optimizable.probe)
            # self.optimizable.purity = np.sqrt(np.sum(self.normalizedEigenvaluesProbe**2))
        elif self.optimizable.nosm > 1:
            for id_l in range(self.optimizable.nlambda):
                for id_s in range(self.optimizable.nslice):
                    self.optimizable.object[id_l,:,0,id_s,:,:], self.normalizedEigenvaluesObject, self.MSPVobject =\
                        orthogonalizeModes(self.optimizable.object[id_l,:,0,id_s,:,:])
            # self.optimizable.object, self.normalizedEigenvaluesObject, self.MSPVobject = \
            #     orthogonalizeModes(self.optimizable.object)
        else:
            pass


    def initializeObject(self):
        """
        Initialize the object.
        :return:
        """
        self.optimizable.initialize_object()


    def showReconstruction(self, loop):
        """
        Show the reconstruction process.
        :param loop: the iteration number
        :return:
        """
        if self.experimentalData.operationMode == 'FPM':
            object_estimate = abs(fft2c(self.optimizable.object))
            probe_estimate = self.optimizable.probe
        else:
            # object_estimate = self.optimizable.object
            object_estimate = np.squeeze(self.optimizable.object)
            probe_estimate = np.squeeze(self.optimizable.probe)

        if loop == 0:
            self.monitor.initializeVisualisation()
        elif np.mod(loop, self.monitor.figureUpdateFrequency) == 0:
            # self.monitor.updatePlot(object_estimate[0,0,:,0,:,:])
            self.monitor.updatePlot(object_estimate=object_estimate,probe_estimate=probe_estimate)
            print('iteration:%i' %len(self.optimizable.error))
            print('runtime:')
            print('error:')
        # TODO: print info

    def applyConstraints(self, loop):
        """
        Apply constraints.
        :param loop: loop number
        :return:
        """


        if np.mod(loop, self.orthogonalizationFrequency) == 0:
            self.orthogonalization()


        # Todo: probePowerCorrectionSwitch, objectSmoothenessSwitch,
        #  probeSmoothenessSwitch, absObjectSwitch, comStabilizationSwitch, objectContrastSwitch

        # modulus enforced probe
        if self.probePowerCorrectionSwitch:
            raise NotImplementedError()

        if self.objectSmoothenessSwitch:
            raise NotImplementedError()

        # not specific to ePIE, -> baseReconstructor
        if self.probeSmoothenessSwitch:
            raise NotImplementedError()

        # not specific to ePIE
        if self.absObjectSwitch:
            raise NotImplementedError()

        if self.comStabilizationSwitch:
            raise NotImplementedError()

        if self.PSDestimationSwitch:
            raise NotImplementedError()

        if self.objectContrastSwitch:
            raise NotImplementedError()

    ## Python-specific things
    def showEndResult(self):
        import matplotlib.pyplot as plt
        initial_guess = ifft2c(self.optimizable.initialObject[0, :, :])
        recon = ifft2c(self.optimizable.object[0, :, :])
        plt.figure(10)
        plt.ioff()
        plt.subplot(221)
        plt.title('initial guess')
        plt.imshow(abs(initial_guess))
        plt.subplot(222)
        plt.title('amplitude')
        plt.imshow(abs(recon))
        plt.subplot(224)
        plt.title('phase')
        plt.imshow(np.angle(recon))
        plt.subplot(223)
        plt.title('probe phase')
        plt.imshow(np.angle(self.optimizable.probe[0, :, :]))
        plt.pause(10)
