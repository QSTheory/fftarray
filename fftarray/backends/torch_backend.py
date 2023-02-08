import numpy as np
import torch
from .tensor_lib import TensorLib, PrecisionSpec
from functools import reduce, partial

class TorchUfuncs():
    def __getattribute__(self, key):
        if key == "iscomplexobj":
            return np.iscomplexobj
        elif key == "isrealobj":
            return np.isrealobj
        elif key == "transpose":
            return torch.permute
        elif key == "power":
            return torch.pow
        elif key == "conjugate":
            return lambda x: torch.conj(x).resolve_conj()
        elif key == "round":
            return lambda x: torch.round(torch.tensor(x))
        elif key == "floor":
            return lambda x: torch.floor(torch.tensor(x))
        elif key == "ceil":
            return lambda x: torch.ceil(torch.tensor(x))
        else:
            return getattr(torch, key)


class TorchTensorLib(TensorLib):
    def __init__(self, precision: PrecisionSpec = "default", device = None):
        TensorLib.__init__(self, precision=precision)
        self.numpy_ufuncs = TorchUfuncs()
        if device is None:
            self.array = torch.tensor
        else:
            self.array = partial(torch.tensor, device=device)
        self.fftn = lambda values, precision: torch.fft.fftn(values)
        self.ifftn = lambda values, precision: torch.fft.ifftn(values)

    @property
    def real_type(self):
        if self.precision == "fp32":
            return torch.float32
        elif self.precision == "fp64":
            return torch.float64
        elif self.precision == "default":
            return torch.float

        assert False, "Unreachable"

    @property
    def complex_type(self):
        if self.precision == "fp32":
            return torch.complex64
        elif self.precision == "fp64":
            return torch.complex128
        elif self.precision == "default":
            return torch.cfloat

        assert False, "Unreachable"

    def has_precision(self, x, target: PrecisionSpec) -> bool:
        if target == "default":
            assert x.dtype == np.float32 or x.dtype == np.float64 \
                or x.dtype == np.complex64 or x.dtype == np.complex128 \
                or x.dtype == torch.float32 or x.dtype == torch.float64 \
                or x.dtype == torch.complex64 or x.dtype == torch.complex128, \
                    "Only floating point types are allowed in FFTArrays."
            return True
        else:
            return self.precision_from_dtype(x.dtype) == target

    def precision_from_dtype(self, dtype) -> PrecisionSpec:
        if dtype == np.float64 or dtype == np.complex128 or dtype == torch.float64 or dtype == torch.complex128:
            return "fp64"
        elif dtype == np.float32 or dtype == np.complex64 or dtype == torch.float32 or dtype == torch.complex64:
            return "fp32"
        else:
            raise ValueError(f"Unsupported dtype {dtype}.")

    def reduce_multiply(self, values) -> float:
        """
        :meta private:
        """
        return reduce(lambda a,b: a*b, values)