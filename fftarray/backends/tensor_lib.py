from __future__ import annotations
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Callable, Literal, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
from types import ModuleType

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from ..fft_array import FFTDimension, Space


PrecisionSpec = Literal["default", "fp32", "fp64"]

@dataclass
class TensorLib(metaclass=ABCMeta):

    precision: PrecisionSpec

    @abstractmethod
    def fftn(self, values: ArrayLike, axes: Sequence[int]) -> ArrayLike:
        ...

    @abstractmethod
    def ifftn(self, values: ArrayLike, axes: Sequence[int]) -> ArrayLike:
        ...

    @property
    @abstractmethod
    def numpy_ufuncs(self) -> ModuleType:
        ...

    @property
    @abstractmethod
    def array(self) -> Callable[..., ArrayLike]:
        ...

    @property
    @abstractmethod
    def array_type(self):
        ...

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and self.precision == other.precision

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(precision={repr(self.precision)})"

    def get_transform_signs(
                self,
                input_factors_applied: Iterable[bool],
                target_factors_applied: Iterable[bool],
            ) -> Optional[List[Literal[-1, 1, None]]]:

        do_return_list = False
        signs: List[Literal[-1, 1, None]] = []
        for input_factor_applied, target_factor_applied in zip(input_factors_applied, target_factors_applied):
            # If both are the same, we do not need to do anything

            if input_factor_applied == target_factor_applied:
                signs.append(None)
            else:
                do_return_list = True
                # 1: Go from applied (external) to not applied (internal)
                # -1 internal to external
                sign: Literal[1, -1] = 1 if input_factor_applied else -1
                signs.append(sign)

        if do_return_list:
            return signs
        else:
            return None

    def get_values_with_lazy_factors(
            self,
            values,
            dims: Tuple[FFTDimension, ...],
            input_factors_applied: Iterable[bool],
            target_factors_applied: Iterable[bool],
            spaces: Iterable[Space],
        ):
        """
            This function takes all dims so that it has more freedom to optimize the application over all dimensions.
            # TODO Get the aliasing and copy story for values straight.
        """
        # TODO: Numpy would probably profit from in-place operations here.
        #       Just one copy in the beginning and then everything in-place.
        #       Does the python Array API allow us to do that generically?

        signs = self.get_transform_signs(
            input_factors_applied=input_factors_applied,
            target_factors_applied=target_factors_applied,
        )

        if signs is not None:
            values = self.apply_scale_phases(
                values=values,
                dims=dims,
                signs=signs,
                spaces=spaces,
            )

        return values

    def apply_scale_phases(
            self,
            values,
            dims: Tuple[FFTDimension, ...],
            signs: List[Literal[1,-1, None]],
            spaces: Iterable[Space],
        ):

        values = self.apply_scale(
            values=values,
            dims=dims,
            signs=signs,
            spaces=spaces,
        )

        values = self.apply_phases(
            values=values,
            dims=dims,
            signs=signs,
            spaces=spaces,
        )
        return values

    def apply_scale(
        self,
        values,
        dims: Iterable[FFTDimension],
        signs: List[Literal[-1, 1, None]],
        spaces: Iterable[Space],
    ):
        # Real-numbered scale
        scale: float = 1.
        do_apply = False
        for dim, sign, dim_space in zip(dims, signs, spaces):
            if sign is not None and dim_space == "freq":
                # TODO: Write as separate mul or div?
                scale = scale * (dim.d_freq*dim.n)**sign
                do_apply = True
        # We cannot test for float == 1. because that value might be dynamic under tracing.
        if do_apply:
            # as array to ensure correct precision.
            values = values * self.as_array(scale)
        return values

    def apply_phases(
                self,
                values,
                dims: Iterable[FFTDimension],
                signs: List[Literal[-1, 1, None]],
                spaces: Iterable[Space],
            ):

        per_idx_phase_factors = self.array(0., self.real_type)
        for dim_idx, (dim, sign, dim_space) in enumerate(zip(dims, signs, spaces)):
            # If both are the same, we do not need to do anything

            if sign is not None:
                # Create indices with correct shape
                indices = self.numpy_ufuncs.arange(0, dim.n, dtype=self.real_type)
                extended_shape = np.ones(len(values.shape), dtype=int)
                extended_shape[dim_idx] = -1
                indices = indices.reshape(tuple(extended_shape))

                if dim_space == "pos":
                    # x = indices * dim.d_pos + dim.pos_min
                    per_idx_values = -sign*2*np.pi*dim.freq_min*dim.d_pos*indices
                else:
                    # f = indices * dim.d_freq + dim.freq_min
                    per_idx_values = sign*2*np.pi*dim.pos_min*(
                        dim.freq_min + dim.d_freq*indices
                    )

                per_idx_phase_factors = per_idx_phase_factors + per_idx_values

        if len(per_idx_phase_factors.shape) > 0: # type: ignore
            # TODO: Figure out typing
            exponent = self.array(1.j, dtype=self.complex_type) * per_idx_phase_factors # type: ignore
            # TODO Could optimise exp into cos and sin because exponent is purely complex
            # Is that faster or more precise? Should we test that or just do it?
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


