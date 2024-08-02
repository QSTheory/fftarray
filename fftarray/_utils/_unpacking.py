from typing import (
    Tuple, List, Any, Union, TypeVar, Iterable, Dict, Hashable,
    Optional, Generic, TYPE_CHECKING
)
from numbers import Number
from functools import reduce
from dataclasses import dataclass
from ..backends.backend import Backend
from ..named_array import align_named_arrays

if TYPE_CHECKING:
    from ..fft_array import FFTDimension, FFTArray, Space



T = TypeVar("T")

#------------
# Helpers to reduce objects that should be same for all elements of a list
#------------
def reduce_equal(objects: Iterable[T], error_msg: str) -> T:
    """
        Reduce the Iterable to a single instance while checking the assumption that all objects are the same.
    """
    def join_equal(a, b):
        if a == b:
            return a
        raise ValueError(error_msg)
    return reduce(join_equal, objects)


class UniformValue(Generic[T]):
    """
        Allows the same reduction as "_reduce_equal" but when running through a loop.
    """
    is_set: bool
    value: Any

    def __init__(self)-> None:
        self.is_set = False

    @property
    def val(self) -> T:
        if self.is_set is False:
            raise ValueError("Value has never ben set.")
        else:
            return self.value

    @val.setter
    def val(self, value: T):
        self.set(value)

    def set(self, value: T):
        if self.is_set:
            if not self.value == value:
                raise ValueError("Did not set value equal to previously set value.")
        else:
            self.value = value
        self.is_set = True

    def get(self, *args: T) -> T:
        # Only first arg is valid and could be a default argument.
        # Need this complicated capture to check if an arg was provided.
        # None is a valid default after all
        assert len(args) < 2
        if self.is_set:
            return self.value

        if len(args) == 1:
            return args[0]

        raise ValueError("Value has never been set.")


def norm_param(val: Union[T, Iterable[T]], n: int, types) -> Tuple[T, ...]:
    """
       `val` has to be immutable.
    """
    if isinstance(val, types):
        return (val,)*n

    # TODO: Can we make this type check work?
    return tuple(val) # type: ignore

@dataclass
class UnpackedValues:
    # FFTDimensions in the order in which they appear in each non-scalar value.
    dims: Tuple["FFTDimension", ...]
    # Values without any dimensions, etc.
    values: List[Union[Number, Any]]
    # Shared backend between all values.
    backend: Backend
    # outer list: dim_idx, inner_list: op_idx, None: dim does not appear in operand
    factors_applied: List[List[bool]]
    # Space per dimension, must be homogeneous over all values
    space: List["Space"]
    # eager per dimension, must be homogeneous over all values
    eager: List[bool]


@dataclass
class UnpackedDimProperties:
    dim: UniformValue["FFTDimension"]
    factors_applied: List[bool]
    eager: UniformValue[bool]
    space: UniformValue["Space"]

    def __init__(self, n_operands: int):
        self.dim = UniformValue()
        # We broadcast the values with the phase factors applied
        # (Each element should have the same value just duplicated along the new dimension.)
        # If factors_applied is True we prevent multiplying the phase-factor of the new dimension
        # with the values.
        self.factors_applied = [True]*n_operands
        self.eager = UniformValue()
        self.space = UniformValue()

def unpack_fft_arrays(
        values: List[Union[Number, "FFTArray", Any]],
    ) -> UnpackedValues:
    """
        This handles all "alignment" of input values.
        Align dimensions, unify them, unpack all operands to a simple list of values.
    """
    dims: Dict[Hashable, UnpackedDimProperties] = {}
    arrays_to_align: List[Tuple[List[Hashable], Any]] = []
    array_indices = []
    unpacked_values: List[Optional[Union[Number, Any]]] = [None]*len(values)
    backend: UniformValue[Backend] = UniformValue()

    for op_idx, x in enumerate(values):
        if isinstance(x, Number):
            unpacked_values[op_idx] = x
        elif hasattr(x, "shape") and not isinstance(x, FFTArray):
            if x.shape == ():
                unpacked_values[op_idx] = x
            else:
                raise ValueError(
                    "Cannot multiply coordinate-less arrays with an FFTArray."
                )
        else:
            array_indices.append(op_idx)
            assert isinstance(x, FFTArray)

            backend.set(x.backend)
            # input_factors_applied = x._factors_applied
            # target_factors_applied = list(x._factors_applied)

            for dim_idx, fft_dim in enumerate(x._dims):
                if not fft_dim.name in dims:
                    dim_props = UnpackedDimProperties(len(values))
                    dims[fft_dim.name] = dim_props
                else:
                    dim_props = dims[fft_dim.name]

                try:
                    dim_props.dim.set(fft_dim)
                except ValueError:
                    raise ValueError(
                        "Tried to call ufunc on FFTArrays with " +
                        "different dimension of name " +
                        f"{fft_dim.name}."
                    )

                try:
                    dim_props.space.set(x._spaces[dim_idx])
                except ValueError:
                    raise ValueError(
                        "Tried to call ufunc on FFTArrays with " +
                        "different spaces in dimension of name " +
                        f"{fft_dim.name}." +
                        "They have to be explicitly converted " +
                        "into an identical space."
                    )

                try:
                    dim_props.eager.set(x._eager[dim_idx])
                except ValueError:
                    raise ValueError(
                        "Tried to call ufunc on FFTArrays with " +
                        "different eager settings in dimension of name " +
                        f"{fft_dim.name}."
                    )

                dim_props.factors_applied[op_idx] = x._factors_applied[dim_idx]

            elem_dim_names = [fft_dim.name for fft_dim in x._dims]
            arrays_to_align.append((elem_dim_names, x._values))


    # Broadcasting
    dim_names, aligned_arrs = align_named_arrays(arrays_to_align, backend=backend.get())
    for op_idx, arr in zip(array_indices, aligned_arrs):
        unpacked_values[op_idx] = arr

    dims_list = [dims[dim_name].dim.get() for dim_name in dim_names]
    space_list = [dims[dim_name].space.get() for dim_name in dim_names]
    eager_list = [dims[dim_name].eager.get() for dim_name in dim_names]
    factors_applied = [dims[dim_name].factors_applied for dim_name in dim_names]
    # TODO: Why is this necessary?
    # unpacked_values = [backend.get().as_array(x) for x in unpacked_values]

    for value in unpacked_values:
        assert value is not None

    return UnpackedValues(
        dims = tuple(dims_list),
        values = unpacked_values, # type: ignore
        space = space_list,
        factors_applied=factors_applied,
        eager=eager_list,
        backend = backend.get(),
    )
