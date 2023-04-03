import numpy as np
from typing import Any, Callable
from types import ModuleType
from numpy.typing import NDArray
import pyfftw

from .tensor_lib import TensorLib, PrecisionSpec

pyfftw.interfaces.cache.enable()

class PyFFTWTensorLib(TensorLib):
    def __init__(self, precision: PrecisionSpec = "default"):
        TensorLib.__init__(self, precision=precision)
        # TODO Use the direct pyfftw interface?
        # Might need info about shape which would introduce a further internal API

    def fftn(self, values: Any, *args, **kwargs) -> Any:
        return pyfftw.interfaces.numpy_fft.fftn(values)
    
    def ifftn(self, values: Any, *args, **kwargs) -> Any:
        return pyfftw.interfaces.numpy_fft.ifftn(values)
    
    @property
    def numpy_ufuncs(self) -> ModuleType:
        return np

    @property
    def array(self) -> Callable[..., NDArray]:
        return np.array