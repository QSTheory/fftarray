import numpy as np
import pyfftw

from .tensor_lib import TensorLib, PrecisionSpec

pyfftw.interfaces.cache.enable()


class PyFFTWTensorLib(TensorLib):

    def __init__(self, precision: PrecisionSpec = "default"):
        TensorLib.__init__(self, precision=precision)
        # TODO Use the direct pyfftw interface?
        # Might need info about shape which would introduce a further internal API
        self.fftn = lambda values, precision: pyfftw.interfaces.numpy_fft.fftn(values)
        self.ifftn = lambda values, precision: pyfftw.interfaces.numpy_fft.ifftn(values)
        self.numpy_ufuncs = np
        self.array = np.array