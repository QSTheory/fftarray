from __future__ import annotations
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Iterable, TYPE_CHECKING
from types import ModuleType

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from ..fft_array import FFTDimension
    from ..lazy_state import LazyState


PrecisionSpec = Literal["default", "fp32", "fp64"]

@dataclass
class TensorLib(metaclass=ABCMeta):

    precision: PrecisionSpec

    @abstractmethod
    def fftn(self, values: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def ifftn(self, values: ArrayLike) -> ArrayLike:
        ...

    @property
    @abstractmethod
    def numpy_ufuncs(self) -> ModuleType:
        ...

    @property
    @abstractmethod
    def array(self) -> Callable[..., ArrayLike]:
        ...

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and self.precision == other.precision

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(precision={repr(self.precision)})"

    def get_values_lazy_factors_applied(
            self,
            values,
            dims: Iterable[FFTDimension],
            lazy_state: LazyState,
        ):
        """
            This function takes all dims so that it has more freedom to optimize the application over all dimensions.
            # TODO Get the aliasing and copy story for values straight.
        """

        scalar_phase: complex = 0.
        for dim_idx, dim in enumerate(dims):
            phase_factors = lazy_state.phase_factors_for_dim(dim.name)

            # TODO: This computation could be optimised by explicitly switching
            # the data-intensive loop over n with the loop over the powers.
            phases_to_apply = {}
            for i, factor in phase_factors.values.items():
                if factor != 0.:
                    if i == 0:
                        # (x**0 is one for all real x, therefore we can compute
                        # as a scalar scale.)
                        scalar_phase += factor
                    else:
                        phases_to_apply[i] = factor
            values = self.apply_phase_factors(
                values=values,
                dim_idx=dim_idx,
                factors=phases_to_apply
            )

        scale = lazy_state.scale * np.exp(1.j * scalar_phase)
        if scale != 1.0:
            values = self.apply_scale(values=values, scale=scale)
        return values

    def apply_scale(self, values, scale):
        # values is raw numpy and therefore we need to force scale possibly
        # down to a lower precision.
        # TODO Why not?
        # values *= self.as_array_with_precision(scale, dim.precision)
        # TODO dim is not necessarily defined here. All current callers do that
        # but there is no principal guarantuee.
        if np.imag(scale) == 0:
            scale = np.real(scale)
        scale = self.as_array(scale)
        # Vaslues is potentially aliased here, therefore this would be wrong.
        # if scale.dtype == values.dtype:
        #     values *= scale
        # else:
        values = values * scale
        return values

    def apply_phase_factors(
            self,
            values,
            dim_idx: int,
            factors: Dict[int, complex]
        ):
        if len(factors) == 0:
            return values
        factors_list = list(factors.items())

        def _get_phase_arr(factor: complex, n: int, i: int):
            indices = self.numpy_ufuncs.arange(
                0,
                values.shape[dim_idx],
                dtype=self.real_type,
            )
            return self.as_array(factor) * (indices**i)

        phase_arr = _get_phase_arr(
            n=values.shape[dim_idx],
            i=factors_list[0][0],
            factor=factors_list[0][1],
        )
        for i, factor in factors_list[1:]:
            phase_arr += _get_phase_arr(n=values.shape[dim_idx], i=i, factor=factor)

        # Ensure correct broadcasting
        extended_shape = np.ones(len(values.shape), dtype=int)
        extended_shape[dim_idx] = -1
        phase_arr = phase_arr.reshape(tuple(extended_shape))

        exponent = self.array(1.j, dtype=self.complex_type) * phase_arr
        # TODO This version does not implicitly upcast values from real to complex but would be faster
        # values *= self.numpy_ufuncs.exp(exponent)
        # TODO Could optimise exp for purely real and purely complex cases
        values = values * self.numpy_ufuncs.exp(exponent)
        return values


    def reduce_multiply(self, values) -> float:
        """

        :meta private:
        """
        return self.numpy_ufuncs.multiply.reduce(values)

    @property
    def real_type(self):
        if self.precision == "fp32":
            return np.float32
        if self.precision == "fp64":
            return np.float64
        if self.precision == "default":
            return float
        assert False, "Unreachable"

    @property
    def complex_type(self):
        if self.precision == "fp32":
            return np.complex64
        if self.precision == "fp64":
            return np.complex128
        if self.precision == "default":
            return complex
        assert False, "Unreachable"

    def as_array(self, x):
        if self.numpy_ufuncs.iscomplexobj(x):
            dtype = self.complex_type
        else:
            assert self.numpy_ufuncs.isrealobj(x)
            dtype = self.real_type
        return self.array(x, dtype = dtype)

    def has_precision(self, x, target: PrecisionSpec) -> bool:
        if target == "default":
            assert x.dtype in [np.float32, np.float64, np.complex64, np.complex128], \
                    "Only floating point types are allowed in FFTArrays."
            return True
        return self.precision_from_dtype(x.dtype) == target

    def precision_from_dtype(self, dtype) -> PrecisionSpec:
        if dtype in [np.float64, np.complex128]:
            return "fp64"
        if dtype in [np.float32, np.complex64]:
            return "fp32"
        raise ValueError(f"Unsupported dtype {dtype}.")


