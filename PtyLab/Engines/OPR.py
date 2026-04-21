import numpy as np
from matplotlib import pyplot as plt

try:
    import cupy as cp
except ImportError:
    # print('Cupy not available, will not be able to run GPU based computation')
    # Still define the name, we'll take care of it later but in this way it's still possible
    # to see that gPIE exists for example.
    cp = None

import logging
import sys
import warnings

import tqdm

from PtyLab.Engines.BaseEngine import BaseEngine
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Monitor.Monitor import Monitor
from PtyLab.Params.Params import Params

# fracPy imports
from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.utils.fsvd import rsvd
from PtyLab.utils.gpuUtils import asNumpyArray, getArrayModule, isGpuArray


class OPR(BaseEngine):
    def __init__(
        self,
        reconstruction: Reconstruction,
        experimentalData: ExperimentalData,
        params: Params,
        monitor: Monitor,
    ):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to ePIE reconstruction
        super().__init__(reconstruction, experimentalData, params, monitor)
        self.logger = logging.getLogger("ePIE")
        self.logger.info("Sucesfully created ePIE ePIE_engine")
        self.logger.info("Wavelength attribute: %s", self.reconstruction.wavelength)
        self.initializeReconstructionParams()

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the ePIE/OPR engine
        """
        self.alpha = self.params.OPR_alpha
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.numIterations = 50
        self.OPR_modes = self.params.OPR_modes
        self.n_subspace = self.params.OPR_subspace

    def reconstruct(self):
        self._prepareReconstruction()

        # OPR parameters
        Nmodes = self.OPR_modes.shape[0]
        Np = self.reconstruction.Np
        Nframes = self.experimentalData.numFrames
        mode_slice = self.OPR_modes
        n_subspace = self.n_subspace

        self.reconstruction.probe_stack = np.zeros(
            (Nmodes, Np, Np, Nframes), dtype=np.complex64
        )

        for i, mode in enumerate(self.OPR_modes):
            # fill the probe-stack with the initial guess of the probes
            initial_probe = asNumpyArray(self.reconstruction.probe[0, 0, mode, 0, :, :])
            self.reconstruction.probe_stack[i, :, :, :] = np.repeat(
                initial_probe[:, :, np.newaxis],
                Nframes,
                axis=2,
            )

        # actual reconstruction ePIE_engine
        self.pbar = tqdm.trange(
            self.numIterations, desc="OPR", file=sys.stdout, leave=True
        )
        for loop in self.pbar:
            self.it = loop
            # set position order
            self.setPositionOrder()
            for positionLoop, positionIndex in enumerate(self.positionIndices):
                # get object patch
                row, col = self.reconstruction.positions[positionIndex]
                sy = slice(row, row + self.reconstruction.Np)
                sx = slice(col, col + self.reconstruction.Np)
                # note that object patch has size of probe array
                objectPatch = self.reconstruction.object[..., sy, sx].copy()

                # Get dim reduced probe and move to GPU
                self.reconstruction.probe[0, 0, mode_slice, 0, :, :] = cp.array(
                    self.reconstruction.probe_stack[:, :, :, positionIndex]
                )

                # make exit surface wave
                self.reconstruction.esw = objectPatch * self.reconstruction.probe

                # propagate to camera, intensityProjection, propagate back to object
                self.intensityProjection(positionIndex)

                # difference term
                DELTA = self.reconstruction.eswUpdate - self.reconstruction.esw

                if loop % self.params.OPR_tv_freq == 0 and self.params.OPR_tv:
                    self.reconstruction.object[..., sy, sx] = self.objectPatchUpdate_TV(
                        objectPatch, DELTA
                    )
                else:
                    # object update
                    self.reconstruction.object[..., sy, sx] = self.objectPatchUpdate(
                        objectPatch, DELTA
                    )

                # probe update
                self.reconstruction.probe = self.probeUpdate(
                    objectPatch, DELTA, weight=1
                )

                # save first, dominant probe mode back to CPU
                self.reconstruction.probe_stack[:, :, :, positionIndex] = asNumpyArray(
                    self.reconstruction.probe[0, 0, mode_slice, 0, :, :]
                )

            # get error metric
            self.getErrorMetrics()

            if self.params.OPR_orthogonalize_modes:
                self.orthogonalizeIncoherentModes()

            self.reconstruction.probe_stack = self.orthogonalizeProbeStack(
                self.reconstruction.probe_stack, n_subspace
            )

            # apply Constraints
            self.applyConstraints(loop)

            # show reconstruction
            self.showReconstruction(loop)

        if self.params.gpuFlag:
            self.logger.info("switch to cpu")
            self._move_data_to_cpu()
            self.params.gpuFlag = 0

    def orthogonalizeIncoherentModes(self):
        """
        Function which cycles through the probe stack and orthogonalizes
        all incoherent modes of all postions
        """
        nFrames = self.experimentalData.numFrames
        n = self.reconstruction.Np
        nModes = self.reconstruction.probe_stack.shape[0]
        for pos in range(nFrames):
            # Move current position probes to GPU
            probe = cp.array(self.reconstruction.probe_stack[:, :, :, pos])
            probe = probe.reshape(nModes, n * n)

            U, s, Vh = self.svd(probe)

            modes = (s[:, None] * Vh).reshape(nModes, n, n)
            self.reconstruction.probe_stack[:, :, :, pos] = asNumpyArray(modes)

    def average(self, arr):
        """
        Calculates the average from neighboring values of a numpy array
        :param arr: 1-dimensional input array, which is used to
        calculate the average
        :return: 1-dimensionl array with the same shape as the input array
        """
        arr_start = arr[:-1]
        arr_end = arr[1:]
        arr_end = cp.append(arr_end, 0)
        arr_start = cp.append(0, arr_start)
        divider = cp.ones_like(arr) * 3
        divider[0] = 2
        divider[-1] = 2
        return (arr + arr_end + arr_start) / divider

    def svd(self, P):
        if isGpuArray(P):
            try:
                return cp.linalg.svd(P, full_matrices=False)
            except:
                print(
                    "Something is wrong with SVD on cuda. Probably an installation error"
                )
                raise
        A, v, At = np.linalg.svd(asNumpyArray(P), full_matrices=False)
        if isGpuArray(P):
            A = cp.array(A)
            v = cp.array(v)
            At = cp.array(At)
        return A, v, At

    def rsvd(self, P, n_dim):
        return rsvd(P, n_dim)
        # A, v, At = self.svd(P)
        # v[n_dim:] = 0
        # return A, v, At

    def orthogonalizeProbeStack(self, probe_stack, n_dim):
        """
        Takes the probe stack maps it by a truncated singular value decomposition in to
        a lower dimensional (n_dim) space.
        :param probe_stack: Probes of all positions
        :param n_dim: Dimension of the lower dimensional sub space
        :return: reduced probe stack
        """
        n = self.reconstruction.Np
        nFrames = self.experimentalData.numFrames

        for i, mode in enumerate(self.OPR_modes):
            # Move only the current mode to GPU
            mode_gpu = cp.array(probe_stack[i, :, :, :])
            reshaped_mode = mode_gpu.reshape(n * n, nFrames)

            if self.params.OPR_tsvd_type == "randomized":
                U, s, Vh = self.rsvd(reshaped_mode, n_dim)
            elif self.params.OPR_tsvd_type in ("exact", "numpy"):
                if self.params.OPR_tsvd_type == "numpy":
                    warnings.warn(
                        'OPR_tsvd_type="numpy" is deprecated; use "exact" instead.',
                        DeprecationWarning,
                        stacklevel=2,
                    )
                # Gram matrix approach: avoids cp.linalg.svd on (n²×nFrames) which
                # overflows CUSOLVER's 32-bit workspace buffer for large Np
                gram = reshaped_mode.T.conj() @ reshaped_mode  # (nFrames, nFrames)
                w, V = cp.linalg.eigh(gram)  # ascending eigenvalues
                idx = cp.argsort(w)[::-1][:n_dim]  # top n_dim, descending
                s = cp.sqrt(cp.maximum(w[idx], 0.0))  # (n_dim,)
                Vh = V[:, idx].T.conj()  # (n_dim, nFrames)
                U = reshaped_mode @ V[:, idx] / (s[None, :] + 1e-17)  # (n², n_dim)

            if self.params.OPR_neighbor_constraint:
                # Calculate the average of neigboring singular values
                content = cp.dot(cp.diag(s), Vh)
                for j in range(n_dim):
                    content[j] = self.average(content[j])

                res = self.alpha * mode_gpu + (1 - self.alpha) * cp.dot(
                    U, content
                ).reshape(n, n, nFrames)
            else:
                update = (U @ (s[:, None] * Vh)).reshape(n, n, nFrames)
                res = self.alpha * mode_gpu + (1 - self.alpha) * update

            probe_stack[i, :, :, :] = asNumpyArray(res)

        return probe_stack

    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        ePIE object update function
        :param objectPatch: Slice of the object array
        :param DELTA:
        :return: updated object patch
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)

        frac = self.reconstruction.probe.conj() / xp.max(
            xp.sum(xp.abs(self.reconstruction.probe) ** 2, axis=(0, 1, 2, 3))
        )
        return objectPatch + self.betaObject * xp.sum(
            frac * DELTA, axis=(0, 2, 3), keepdims=True
        )

    def probeUpdate(
        self, objectPatch: np.ndarray, DELTA: np.ndarray, weight: float, gimmel=0.1
    ):
        """
        Update the probe
        :param objectPatch: Slice of the object array
        :param DELTA:
        :return: updated probe
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        frac = objectPatch.conj() / (
            xp.max(xp.sum(xp.abs(objectPatch) ** 2, axis=(0, 1, 2, 3))) + gimmel
        )
        frac = frac * weight
        r = self.reconstruction.probe + self.betaProbe * xp.sum(
            frac * DELTA, axis=(0, 1, 3), keepdims=True
        )
        return r
