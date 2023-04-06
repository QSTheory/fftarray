import warnings

import numpy as np

from .tensor_lib import TensorLib, PrecisionSpec


class NumpyTensorLib(TensorLib):

    def __init__(self, precision: PrecisionSpec = "default"):
        TensorLib.__init__(self, precision=precision)
        self.numpy_ufuncs = np
        self.array = np.array

    def fftn(self, values, precision: PrecisionSpec):
        transformed = np.fft.fftn(values)
        if precision == "fp32":
            warnings.warn('numpy.fft.fftn always computes in double precision. \
                Since precision was set to fp32 the result is automatically \
                truncated.')
            return transformed.astype(np.complex64)
        return transformed

    def ifftn(self, values, precision: PrecisionSpec):
        transformed = np.fft.ifftn(values)
        if precision == "fp32":
            warnings.warn('numpy.fft.ifftn always computes in double \
                precision. Since precision was set to fp32 the result is \
                automatically truncated.')
            return transformed.astype(np.complex64)
        return transformed
