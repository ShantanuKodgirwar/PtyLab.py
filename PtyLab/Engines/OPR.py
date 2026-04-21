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

        stack_shape = (Nmodes, Np, Np, Nframes)
        device = self.params.OPR_probe_stack_device
        probe_stack_on_gpu = False
        if self.params.gpuFlag and device in ("auto", "gpu"):
            try:
                self.reconstruction.probe_stack = cp.zeros(stack_shape, dtype=cp.complex64)
                probe_stack_on_gpu = True
            except cp.cuda.memory.OutOfMemoryError:
                if device == "gpu":
                    raise
                self.reconstruction.probe_stack = np.zeros(stack_shape, dtype=np.complex64)
        else:
            self.reconstruction.probe_stack = np.zeros(stack_shape, dtype=np.complex64)
        self._probe_stack_on_gpu = probe_stack_on_gpu

        for i, mode in enumerate(self.OPR_modes):
            initial_probe = asNumpyArray(self.reconstruction.probe[0, 0, mode, 0, :, :])
            col = np.repeat(initial_probe[:, :, np.newaxis], Nframes, axis=2)
            if probe_stack_on_gpu:
                self.reconstruction.probe_stack[i] = cp.array(col)
            else:
                self.reconstruction.probe_stack[i] = col

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

                # load probe for this position
                if self._probe_stack_on_gpu:
                    self.reconstruction.probe[0, 0, mode_slice, 0, :, :] = (
                        self.reconstruction.probe_stack[:, :, :, positionIndex]
                    )
                else:
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

                # save probe for this position
                if self._probe_stack_on_gpu:
                    self.reconstruction.probe_stack[:, :, :, positionIndex] = (
                        self.reconstruction.probe[0, 0, mode_slice, 0, :, :]
                    )
                else:
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

        if self._probe_stack_on_gpu:
            self.reconstruction.probe_stack = asNumpyArray(self.reconstruction.probe_stack)
            self._probe_stack_on_gpu = False
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
        gpu = self._probe_stack_on_gpu
        for pos in range(nFrames):
            probe = (self.reconstruction.probe_stack[:, :, :, pos] if gpu
                     else cp.array(self.reconstruction.probe_stack[:, :, :, pos]))
            probe = probe.reshape(nModes, n * n)
            U, s, Vh = self.svd(probe)
            modes = (s[:, None] * Vh).reshape(nModes, n, n)
            if gpu:
                self.reconstruction.probe_stack[:, :, :, pos] = modes
            else:
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

    def _streaming_rsvd(self, mode_flat, n_dim, chunk_size):
        """Streaming randomized SVD; avoids loading n^2 x Nframes to GPU at once."""
        n2, nFrames = mode_flat.shape
        n_samples = 2 * n_dim
        sketch = (
            cp.random.normal(0, 1, (nFrames, n_samples))
            + 1j * cp.random.normal(0, 1, (nFrames, n_samples))
        ).astype(cp.complex64)
        Y = cp.zeros((n2, n_samples), dtype=cp.complex64)
        for start in range(0, n2, chunk_size):
            end = min(start + chunk_size, n2)
            chunk = cp.array(mode_flat[start:end])
            Y[start:end] = chunk @ sketch
            del chunk
        del sketch
        Q, _ = cp.linalg.qr(Y)
        del Y
        B = cp.zeros((n_samples, nFrames), dtype=cp.complex64)
        for start in range(0, n2, chunk_size):
            end = min(start + chunk_size, n2)
            chunk = cp.array(mode_flat[start:end])
            B += Q[start:end].T.conj() @ chunk
            del chunk
        U_hat, s, Vt = cp.linalg.svd(B, full_matrices=False)
        del B
        U = Q @ U_hat[:, :n_dim]
        del Q, U_hat
        return U, s[:n_dim], Vt[:n_dim]

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
        chunk_size = self.params.OPR_spatial_chunk_size

        for i, mode in enumerate(self.OPR_modes):
            # reshape to (n²,Nframes); for both numpy and cupy this is a view,
            # so writes to mode_flat update probe_stack[i] in-place
            mode_flat = probe_stack[i].reshape(n * n, nFrames)

            if self._probe_stack_on_gpu:
                # --- GPU fast path: mode_flat is already on GPU, no transfers needed ---
                if self.params.OPR_tsvd_type == "randomized":
                    U, s, Vh = self.rsvd(mode_flat, n_dim)
                    V_top = None
                else:
                    if self.params.OPR_tsvd_type == "numpy":
                        warnings.warn(
                            'OPR_tsvd_type="numpy" is deprecated; use "exact" instead.',
                            DeprecationWarning,
                            stacklevel=2,
                        )
                    gram = mode_flat.T.conj() @ mode_flat
                    w, V = cp.linalg.eigh(gram)
                    idx = cp.argsort(w)[::-1][:n_dim]
                    s = cp.sqrt(cp.maximum(w[idx], 0.0))
                    Vh = V[:, idx].T.conj()
                    V_top = V[:, idx]
                    U = None
                    del gram, V, w, idx

                if self.params.OPR_neighbor_constraint:
                    content = cp.dot(cp.diag(s), Vh)
                    for j in range(n_dim):
                        content[j] = self.average(content[j])
                    right = content
                else:
                    right = s[:, None] * Vh  # (n_dim, nFrames)

                if U is not None:  # randomized: U is (n²×r) on GPU
                    M_r = U @ right
                else:  # exact: recompute U via V_top
                    U_approx = mode_flat @ V_top / (s[None, :] + 1e-17)
                    M_r = U_approx @ right
                    del U_approx
                mode_flat[:] = self.alpha * mode_flat + (1 - self.alpha) * M_r

                del s, Vh, right, M_r
                if V_top is not None:
                    del V_top
                if U is not None:
                    del U
                continue

            # --- CPU path: chunked accumulation to avoid loading full mode to GPU ---
            if self.params.OPR_tsvd_type == "randomized":
                U, s, Vh = self._streaming_rsvd(mode_flat, n_dim, chunk_size)
                V_top = None  # use U[start:end] directly in update loop
            elif self.params.OPR_tsvd_type in ("exact", "numpy"):
                if self.params.OPR_tsvd_type == "numpy":
                    warnings.warn(
                        'OPR_tsvd_type="numpy" is deprecated; use "exact" instead.',
                        DeprecationWarning,
                        stacklevel=2,
                    )
                gram = cp.zeros((nFrames, nFrames), dtype=cp.complex64)
                for start in range(0, n * n, chunk_size):
                    end = min(start + chunk_size, n * n)
                    chunk = cp.array(mode_flat[start:end])
                    gram += chunk.T.conj() @ chunk
                    del chunk
                w, V = cp.linalg.eigh(gram)
                del gram
                idx = cp.argsort(w)[::-1][:n_dim]
                s = cp.sqrt(cp.maximum(w[idx], 0.0))
                Vh = V[:, idx].T.conj()
                V_top = V[:, idx]
                U = None
                del V, w, idx

            if self.params.OPR_neighbor_constraint:
                content = cp.dot(cp.diag(s), Vh)
                for j in range(n_dim):
                    content[j] = self.average(content[j])
                right = content
            else:
                right = s[:, None] * Vh  # (n_dim, nFrames)

            for start in range(0, n * n, chunk_size):
                end = min(start + chunk_size, n * n)
                chunk = cp.array(mode_flat[start:end])
                if U is not None:
                    U_chunk = U[start:end]
                else:
                    U_chunk = chunk @ V_top / (s[None, :] + 1e-17)
                mode_flat[start:end] = asNumpyArray(
                    self.alpha * chunk + (1 - self.alpha) * (U_chunk @ right)
                )
                del chunk

            del s, Vh, right
            if V_top is not None:
                del V_top
            if U is not None:
                del U

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
