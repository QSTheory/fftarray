from typing import Callable, Union
from types import ModuleType
import warnings

import numpy as np
from numpy.typing import NDArray

from .tensor_lib import TensorLib, PrecisionSpec


class NumpyTensorLib(TensorLib):

    def __init__(self, precision: PrecisionSpec = "default"):
        TensorLib.__init__(self, precision=precision)

    def fftn(self, values) -> Union[NDArray[np.complex64], NDArray[np.complex128]]:
        transformed = np.fft.fftn(values)
        if self.precision == "fp32":
            warnings.warn(
                "numpy.fft.fftn always computes in double precision. " +
                "Since precision was set to fp32 the result is automatically " +
                "truncated."
            )
            return transformed.astype(np.complex64)
        return transformed

    def ifftn(self, values) -> Union[NDArray[np.complex64], NDArray[np.complex128]]:
        transformed = np.fft.ifftn(values)
        if self.precision == "fp32":
            warnings.warn('numpy.fft.ifftn always computes in double precision. Since precision was set to fp32 the result is automatically truncated.')
            return transformed.astype(np.complex64)
        return transformed

    @property
    def numpy_ufuncs(self) -> ModuleType:
        return np

    @property
    def array(self) -> Callable[..., NDArray]:
        return np.array
