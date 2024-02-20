import numpy as np
from typing import Callable, Sequence
from types import ModuleType
from numpy.typing import NDArray, ArrayLike
import pyfftw

from .tensor_lib import TensorLib, PrecisionSpec

pyfftw.interfaces.cache.enable()


class PyFFTWTensorLib(TensorLib):

    def __init__(self, precision: PrecisionSpec = "default"):
        TensorLib.__init__(self, precision=precision)
        # TODO Use the direct pyfftw interface?
        # Might need info about shape which would introduce a further internal API

    def fftn(self, values: ArrayLike, axes: Sequence[int]) -> ArrayLike:
        return pyfftw.interfaces.numpy_fft.fftn(values, axes=axes)

    def ifftn(self, values: ArrayLike, axes: Sequence[int]) -> ArrayLike:
        return pyfftw.interfaces.numpy_fft.ifftn(values, axes=axes)

    @property
    def numpy_ufuncs(self) -> ModuleType:
        return np

    @property
    def array(self) -> Callable[..., NDArray]:
        return np.array

    @property
    def array_type(self):
        return np.ndarray
