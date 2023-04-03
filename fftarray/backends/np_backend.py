from typing import Callable, Union
from types import ModuleType
import numpy as np
from numpy.typing import NDArray

from .tensor_lib import TensorLib, PrecisionSpec
import warnings


class NumpyTensorLib(TensorLib):
    def __init__(self, precision: PrecisionSpec = "default"):
        TensorLib.__init__(self, precision=precision)

    def fftn(self, values, precision: Union[int, str]) -> Union[NDArray[np.complex64], NDArray[np.complex128]]:
        transformed = np.fft.fftn(values)
        if precision == "fp32":
            warnings.warn('numpy.fft.fftn always computes in double precision. Since precision was set to fp32 the result is automatically truncated.')
            return transformed.astype(np.complex64)
        else:
            return transformed

    def ifftn(self, values, precision: PrecisionSpec) -> Union[NDArray[np.complex64], NDArray[np.complex128]]:
        transformed = np.fft.ifftn(values)
        if precision == "fp32":
            warnings.warn('numpy.fft.ifftn always computes in double precision. Since precision was set to fp32 the result is automatically truncated.')
            return transformed.astype(np.complex64)
        else:
            return transformed
        
    @property
    def numpy_ufuncs(self) -> ModuleType:
        return np

    @property
    def array(self) -> Callable[..., NDArray]:
        return np.array
