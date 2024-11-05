from __future__ import annotations
from ast import Assert
from collections import abc
from typing import (
    Callable, Mapping, Optional, Union, List, Any, Tuple, Dict, Hashable,
    Literal, TypeVar, Iterable, Set, get_args, TYPE_CHECKING
)
from abc import ABCMeta
from copy import copy
from numbers import Number
from dataclasses import dataclass

import numpy as np

from .space import Space
from .fft_dimension import FFTDimension
from .named_array import get_axes_transpose, align_named_arrays
from .backends.backend import Backend
from .backends.numpy import NumpyBackend
from ._utils.uniform_value import UniformValue

from ._utils.ufuncs import binary_ufuncs, unary_ufunc
from ._utils.formatting import (
    fft_dim_table, fft_array_props_table, format_bytes,
)
# from .unpacking import UnpackedValues, unpack_fft_arrays
from ._utils.indexing import (
    LocFFTArrayIndexer, check_missing_dim_names,
    tuple_indexers_from_dict_or_tuple, tuple_indexers_from_mapping,
)

EllipsisType = TypeVar('EllipsisType')

T = TypeVar("T")

_DEFAULT_BACKEND: Backend = NumpyBackend("default")

def set_default_backend(backend: Backend) -> None:
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND= backend

def get_default_backend() -> Backend:
    return _DEFAULT_BACKEND

def default_backend(backend: Backend):
    return DefaultBackendContext(backend=backend)

class DefaultBackendContext:
    def __init__(self, backend: Backend):
        self.override = backend

    def __enter__(self):
        self.previous = get_default_backend()
        set_default_backend(self.override)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_default_backend(self.previous)

_DEFAULT_EAGER: bool = False

def set_default_eager(eager: bool) -> None:
    global _DEFAULT_EAGER
    _DEFAULT_EAGER= eager

def get_default_eager() -> bool:
    return _DEFAULT_EAGER

def default_eager(eager: bool):
    return DefaultEagerContext(eager=eager)

class DefaultEagerContext:
    def __init__(self, eager: bool):
        self.override = eager

    def __enter__(self):
        self.previous = get_default_eager()
        set_default_eager(self.override)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_default_eager(self.previous)

def norm_param(val: Union[T, Iterable[T]], n: int, types) -> Tuple[T, ...]:
    """
       `val` has to be immutable.
    """
    if isinstance(val, types):
        return (val,)*n

    # TODO: Can we make this type check work?
    return tuple(val) # type: ignore



def mul_transforms(unp_inp: UnpackedValues) -> Tuple[List[List[Literal[-1, 1, None]]], List[bool]]:
    factor_transforms: List[List[Literal[-1, 1, None]]] = [[None]*len(unp_inp.dims) for _ in range(2)]
    final_factors_applied: List[bool] = []
    # Element-wise multiplication is commutative with the multiplication of the phase factors.
    # So we can always directly multiply with the inner values and can delay up to one set of phase factors per dimension.

    # We only handle two operands.
    # If both have a phase factor we must remove it for one of the values.
    # Otherwise we can just take the raw values
    for dim_idx in range(len(unp_inp.dims)):
        fac_applied: Tuple[bool, bool] = (unp_inp.factors_applied[dim_idx][0], unp_inp.factors_applied[dim_idx][1])

        # If both are not applied we have to apply the factor once
        if fac_applied == (False, False):
            # We pick to always do it on the second one.
            # Divide requires this, multiply could also choose the first.
            factor_transforms[1][dim_idx] = -1

        # If both operands are applied, the final will be too, otherwise it will not.
        final_factors_applied.append(all(fac_applied))

    return factor_transforms, final_factors_applied

def add_transforms(unp_inp: UnpackedValues) -> Tuple[List[List[Literal[-1, 1, None]]], List[bool]]:
    factor_transforms: List[List[Literal[-1, 1, None]]] = [[None]*len(unp_inp.dims) for _ in range(2)]
    final_factors_applied: List[bool] = []

    for dim_idx in range(len(unp_inp.dims)):
        fac_applied = (unp_inp.factors_applied[dim_idx][0], unp_inp.factors_applied[dim_idx][1])
        if fac_applied[0] == fac_applied[1]:
            # Both factors still need to be applied => factor them out
            final_factors_applied.append(fac_applied[0])
        else:
            final_factors_applied.append(unp_inp.eager[dim_idx])

            # Same as the commented out code below.
            # Not sure if it is readable enough.

            # If the first operand is applied and eager, we convert the second one.
            # Same if it is not applied and lazy.
            # Otherwise we convert the first one.
            transformed_op_idx = int(fac_applied[0] == unp_inp.eager[dim_idx])
            # If we are eager we want to the applied state, so sign=-1
            # else we want to the internal state so we apply 1.
            factor_transforms[transformed_op_idx][dim_idx] = -1 if unp_inp.eager[dim_idx] else 1
    return factor_transforms, final_factors_applied

def default_transforms(unp_inp: UnpackedValues) -> Tuple[List[List[Literal[-1, 1, None]]], List[bool]]:
    factor_transforms: List[List[Literal[-1, 1, None]]] = [[None]*len(unp_inp.dims) for _ in range(2)]
    final_factors_applied: List[bool] = []

    # Define factor_transforms such that factors are applied for
    # both operators because there is no special case applicable
    for op_idx in [0,1]:
        if unp_inp.is_fftarray[op_idx]:
            res = unp_inp.backend.get_transform_signs(
                input_factors_applied=[unp_inp.factors_applied[dim_idx][op_idx] for dim_idx in range(len(unp_inp.dims))],
                target_factors_applied=[True]*len(unp_inp.dims),
            )
            if res is not None:
                factor_transforms[op_idx] = res

    final_factors_applied = [True]*len(unp_inp.dims)
    return factor_transforms, final_factors_applied


# Implementing NEP 13 https://numpy.org/neps/nep-0013-ufunc-overrides.html
# See also https://numpy.org/doc/stable/user/basics.dispatch.html
def two_inputs_func(x1, x2, unp_inp, op, get_transforms=default_transforms, kwargs={}):
    inputs = [x1, x2]
    factor_transforms, final_factors_applied = get_transforms(unp_inp)

    # Apply above defined scale and phase factors depending on the specific case
    for op_idx, signs_op in zip([0,1], factor_transforms):
        if isinstance(inputs[op_idx], FFTArray):
            unp_inp.values[op_idx] = unp_inp.backend.apply_scale_phases(
                values=unp_inp.values[op_idx],
                dims=unp_inp.dims,
                signs=signs_op,
                spaces=unp_inp.space,
            )

    values = op(*unp_inp.values, **kwargs)
    return FFTArray(
        values=values,
        space=unp_inp.space,
        dims=unp_inp.dims,
        eager=unp_inp.eager,
        factors_applied=final_factors_applied,
        backend=unp_inp.backend,
    )

def elementwise_two_operands(name: str, get_transforms=default_transforms) -> Callable[[Any, Any], FFTArray]:
    def fun(x1, x2) -> FFTArray:
        unp_inp: UnpackedValues = unpack_fft_arrays([x1, x2])
        return two_inputs_func(
            x1=x1,
            x2=x2,
            unp_inp=unp_inp,
            op=getattr(unp_inp.backend.numpy_ufuncs, name),
            get_transforms=get_transforms,
        )
    return fun

def elementwise_one_operand(name: str) -> Callable[[FFTArray], FFTArray]:
    def single_element_ufunc(inp: FFTArray) -> FFTArray:
        assert isinstance(inp, FFTArray)
        op = getattr(inp.backend.numpy_ufuncs, name)
        values = op(inp.values(space=inp.space))
        return FFTArray(
            values=values,
            space=inp.space,
            dims=inp.dims,
            eager=inp.eager,
            factors_applied=True,
            backend=inp.backend,
        )
    return single_element_ufunc


#------------------
# Special Implementations
#------------------

# This one is the only one with kwargs, so just done by hand.
def clip(x, *, min=None, max=None) -> FFTArray:
    assert isinstance(x, FFTArray)
    op = getattr(x.backend.numpy_ufuncs, "clip")
    values = op(x.values(space=x.space), min=min, max=max)
    return FFTArray(
        values=values,
        space=x.space,
        dims=x.dims,
        eager=x.eager,
        factors_applied=True,
        backend=x.backend,
    )

# These use special shortcuts in the phase application.
add = elementwise_two_operands("add", add_transforms)
subtract = elementwise_two_operands("subtract", add_transforms)
multiply = elementwise_two_operands("multiply", mul_transforms)
divide = elementwise_two_operands("divide", mul_transforms)


def abs(x):
    assert isinstance(x, FFTArray)
    # For abs the final result does not change if we apply the phases
    # to the values so we can simply ignore the phases.
    values = x.backend.numpy_ufuncs.abs(x._values)
    # The scale can be applied after abs which is more efficient in the case of a complex input
    signs: List[Literal[-1, 1, None]] | None = x.backend.get_transform_signs(
        # Can use input because with a single value no broadcasting happened.
        input_factors_applied=x._factors_applied,
        target_factors_applied=[True]*len(x._factors_applied),
    )
    if signs is not None:
        values = x.backend.apply_scale(
            values=values,
            dims=x.dims,
            signs=signs,
            spaces=x.space,
        )

    return FFTArray(
        values=values,
        space=x.space,
        dims=x.dims,
        eager=x.eager,
        factors_applied=True,
        backend=x.backend,
    )

#------------------
# Single operand element-wise functions
#------------------
acos = elementwise_one_operand("acos")
acosh = elementwise_one_operand("acosh")
asin = elementwise_one_operand("asin")
asinh = elementwise_one_operand("asinh")
atan = elementwise_one_operand("atan")
atanh = elementwise_one_operand("atanh")
bitwise_invert = elementwise_one_operand("bitwise_invert")
ceil = elementwise_one_operand("ceil")
conj = elementwise_one_operand("conj")
cos = elementwise_one_operand("cos")
cosh = elementwise_one_operand("cosh")
exp = elementwise_one_operand("exp")
expm1 = elementwise_one_operand("expm1")
floor = elementwise_one_operand("floor")
imag = elementwise_one_operand("imag")
isfinite = elementwise_one_operand("isfinite")
isinf = elementwise_one_operand("isinf")
isnan = elementwise_one_operand("isnan")
log = elementwise_one_operand("log")
log1p = elementwise_one_operand("log1p")
log2 = elementwise_one_operand("log2")
log10 = elementwise_one_operand("log10")
logical_not = elementwise_one_operand("logical_not")
negative = elementwise_one_operand("negative")
positive = elementwise_one_operand("positive")
real = elementwise_one_operand("real")
round = elementwise_one_operand("round")
sign = elementwise_one_operand("sign")
signbit = elementwise_one_operand("signbit")
sin = elementwise_one_operand("sin")
sinh = elementwise_one_operand("sinh")
square = elementwise_one_operand("square")
sqrt = elementwise_one_operand("sqrt")
tan = elementwise_one_operand("tan")
tanh = elementwise_one_operand("tanh")
trunc = elementwise_one_operand("trunc")

#------------------
# Two operand element-wise functions
#------------------
atan2 = elementwise_two_operands("atan2")
bitwise_and = elementwise_two_operands("bitwise_and")
bitwise_left_shift = elementwise_two_operands("bitwise_left_shift")
bitwise_or = elementwise_two_operands("bitwise_or")
bitwise_right_shift = elementwise_two_operands("bitwise_right_shift")
bitwise_xor = elementwise_two_operands("bitwise_xor")
copysign = elementwise_two_operands("copysign")
equal = elementwise_two_operands("equal")
floor_divide = elementwise_two_operands("floor_divide")
greater = elementwise_two_operands("greater")
greater_equal = elementwise_two_operands("greater_equal")
hypot = elementwise_two_operands("hypot")
less = elementwise_two_operands("less")
less_equal = elementwise_two_operands("less_equal")
logaddexp = elementwise_two_operands("logaddexp")
logical_and = elementwise_two_operands("logical_and")
logical_or = elementwise_two_operands("logical_or")
logical_xor = elementwise_two_operands("logical_xor")
maximum = elementwise_two_operands("maximum")
minimum = elementwise_two_operands("minimum")
not_equal = elementwise_two_operands("not_equal")
pow = elementwise_two_operands("pow")
remainder = elementwise_two_operands("remainder")



class FFTArray(metaclass=ABCMeta):
    """The base class of `PosArray` and `FreqArray` that implements all shared
    behavior.
    """

    # _dims are stored as a sequence and not by name because their oder needs
    # to match the order of dimensions in _values.
    _dims: Tuple[FFTDimension, ...]
    # Contains an array instance of _backend with _lazy_state not yet applied.
    _values: Any
    # Marks each dimension whether it is in position or frequency space
    _spaces: Tuple[Space, ...]
    # Marks each dimension whether the phase-factors should be applied directly after executing a fft or ifft
    _eager: Tuple[bool, ...]
    # Marks each dim whether its phase_factors still need to be applied
    _factors_applied: Tuple[bool, ...]
    # TODO: implement device [#18](https://github.com/QSTheory/fftarray/issues/18)
    # Contains the array backend, precision and device to be used for operations.
    _backend: Backend

    def __init__(
            self,
            values,
            dims: Iterable[FFTDimension],
            space: Union[Space, Iterable[Space]],
            backend: Optional[Backend] = None,
            eager: Optional[Union[bool, Iterable[bool]]] = None,
            factors_applied: Union[bool, Iterable[bool]] = True,
        ):
        """
        Construct a new instance of FFTArray from raw values.
        For normal usage it is recommended to construct
        new instances via the `fft_array()` function of FFTDimension
        since this ensures that the dimension parameters and the
        values match.

        TODO: Check that this does not copy?

        Parameters
        ----------
        values :
            The values to initialize the `FFTArray` with.
            For performance reasons they are assumed to not be aliased (or immutable)
            and therefore do not get copied under any circumstances.
            The type must fit with the specified backend.
        dims : Iterable[FFTDimension]
            The FFTDimensions for each dimension of the passed in values.
        space: Union[Space, Iterable[Space]]
            Specify the space of the coordinates and in which space the returned FFTArray is intialized.
        backend: Optional[Backend]
            The backend to use for the returned FFTArray.  `None` uses default `NumpyBackend("default")` which can be globally changed.
            The values are transformed into the appropiate type defined by the backend.
        eager: Union[bool, Iterable[bool]]
            The eager-mode to use for the returned FFTArray.  `None` uses default `False` which can be globally changed.
        factors_applied: Union[bool, Iterable[bool]]
            Whether the fft-factors are applied are already applied for the various dimensions.
            For external values this is usually `True` since `False` assumes the internal (and unstable)
            factors-format.

        Returns
        -------
        FFTArray
            The grid coordinates of the chosen space packed into an FFTArray with self as only dimension.

        See Also
        --------
        set_default_backend, get_default_backend
        set_default_eager, get_default_eager
        fft_array
        """

        if backend is None:
            backend = _DEFAULT_BACKEND

        if eager is None:
            eager = _DEFAULT_EAGER

        self._dims = tuple(dims)
        n_dims = len(self._dims)
        self._values = values
        self._spaces = norm_param(space, n_dims, str)
        self._eager = norm_param(eager, n_dims, bool)
        self._factors_applied = norm_param(factors_applied, n_dims, bool)
        self._backend = backend
        self._check_consistency()

    def __repr__(self: FFTArray) -> str:
        arg_str = ", ".join(
            [f"{name[1:] if name != '_spaces' else 'space'}={repr(value)}"
                for name, value in self.__dict__.items()]
        )
        return f"FFTArray({arg_str})"

    def __str__(self: FFTArray) -> str:
        bytes_str = format_bytes(self._values.nbytes)
        str_out = f"<fftarray.FFTArray ({bytes_str})>\n"
        str_out += f"Backend: {self.backend}\n"
        str_out += "Dimensions:\n"
        for i, dim in enumerate(self.dims):
            str_out += f" # {i}: {repr(dim.name)}\n"
        str_out += "\n" + fft_array_props_table(self) + "\n\n"
        for i, dim in enumerate(self.dims):
            str_out += fft_dim_table(dim, i==0, True, i) + "\n"
        str_out += f"\nvalues:\n{self.values(space=self.space)}"
        return str_out

    def __bool__(self: FFTArray):
        raise ValueError("The truth value of an array is ambiguous.")

    #--------------------
    # Operator Implementations
    #--------------------
    # Implement binary operations between FFTArray and also e.g. 1+wf and wf+1
    # This does intentionally not list all possible operators.
    __add__, __radd__ = binary_ufuncs(add)
    __sub__, __rsub__ = binary_ufuncs(subtract)
    __mul__, __rmul__ = binary_ufuncs(multiply)
    __truediv__, __rtruediv__ = binary_ufuncs(divide)
    __floordiv__, __rfloordiv__ = binary_ufuncs(floor_divide)
    __pow__, __rpow__ = binary_ufuncs(pow)

    # Implement comparison operators
    __gt__, _ = binary_ufuncs(greater)
    __ge__, _ = binary_ufuncs(greater_equal)
    __lt__, _ = binary_ufuncs(less)
    __le__, _ = binary_ufuncs(less_equal)
    __ne__, _ = binary_ufuncs(not_equal)
    __eq__, _ = binary_ufuncs(equal)


    # Implement unary operations
    __neg__ = unary_ufunc(negative)
    __pos__ = unary_ufunc(positive)
    __abs__ = unary_ufunc(abs)
    __invert__ = unary_ufunc(negative)

    #--------------------
    # Selection
    #--------------------

    def __getitem__(
            self,
            item: Union[
                int, slice, EllipsisType,
                Tuple[Union[int, slice, EllipsisType],...],
                Mapping[Hashable, Union[int, slice]],
            ]
        ) -> FFTArray:
        """This method is called when indexing an FFTArray instance by integer index,
            i.e., by using the index value via FFTArray[].
            It supports dimensional lookup via position and name.
            The indexing behaviour is mainly defined to match the one of
            xarray.DataArray with the major difference that we always keep
            all dimensions.
            The indexing is performed in the current state of the FFTArray,
            i.e., each dimension is indexed in its respective state (pos or freq).
            When the returned FFTArray object is effectively indexed,
            its internal state will always have all fft factors applied.

            Example usage:
            arr_2d = (
                x_dim.fft_array(space="pos", backend=some_backend)
                + y_dim.fft_array(space="pos", backend=some_backend)
            )
            Four ways of retrieving an FFTArray object
            with index 3 along x and first 5 values along y:

            arr_2d[{"x": 3, "y": slice(0, 5)}]
            arr_2d[3,:5]
            arr_2d[3][:,:5] # do not use, just for explaining functionality
            arr_2d[:,:5][3] # do not use, just for explaining functionality

        Parameters
        ----------
        item : Union[ int, slice, EllipsisType, Tuple[Union[int, slice, EllipsisType],...], Mapping[Hashable, Union[int, slice]], ]
            An indexer object with either dimension lookup method either
            via position or name. When using positional lookup, the order
            of the dimensions in the FFTArray object is applied (FFTArray.dims).
            Per dimension, each indexer can be supplied as an integer or a slice.
            Array-like indexers are not supported as in the general case,
            the resulting coordinates can not be supported with a valid FFTDimension.
        FFTArray
            A new FFTArray with the same dimensionality as this FFTArray,
            except each dimension and the FFTArray values are indexed.
            The resulting FFTArray still fully supports FFTs.
        """

        # Catch special case where effectively no indexing happens,
        # i.e., just return FFTArray object as is (without changing internal state)
        if item is Ellipsis:
            return self
        if isinstance(item, abc.Mapping) and len(item) == 0:
            return self

        # Handle two cases of supplying indexing information, either
        # via keyword args (Mapping) or via tuple using order of dims
        # Return full tuple of indexers as slice or int object
        tuple_indexers: Tuple[Union[int, slice], ...] = tuple_indexers_from_dict_or_tuple(
            indexers=item, # type: ignore
            dim_names=tuple(dim.name for dim in self.dims) # type: ignore
        )

        new_dims = []
        for index, orig_dim, space in zip(tuple_indexers, self._dims, self.space):
            if index == slice(None, None, None):
                # No selection, just keep the old dim.
                new_dims.append(orig_dim)
                continue
            if not isinstance(index, slice):
                index = slice(index, index+1, None)
            try:
                # We perform all index sanity checks in _dim_from_slice
                new_dims.append(orig_dim._dim_from_slice(index, space))
            # Do not specifically catch jax.errors.ConcretizationTypeError in order to not have to import jax here.
            except Exception as e:
                # This condition is fullfilled when the index is a traced object
                if "Trace" in str(index):
                    raise NotImplementedError(
                        f"FFTArray indexing does not support "
                        + "jitted indexers. Here, your index for "
                        + f"dimension {orig_dim.name} is a traced object"
                    ) from e
                else:
                    additional_msg = (
                        "An error occurred when evaluating the index "
                        + f"dimension {orig_dim.name}: "
                    )
                    orig_msg = str(e)
                    raise type(e)(additional_msg + orig_msg)


        selected_values = self.values(space=self.space).__getitem__(tuple_indexers)
        # Dimensions with the length 1 are dropped in numpy indexing.
        # We decided against this and keeping even dimensions of length 1.
        # So we have to reintroduce those dropped dimensions via reshape.
        selected_values = selected_values.reshape(tuple(dim.n for dim in new_dims))

        return FFTArray(
            values=selected_values,
            dims=new_dims,
            space=self.space,
            eager=self.eager,
            factors_applied=[True]*len(new_dims),
            backend=self.backend,
        )

    @property
    def loc(self):
        return LocFFTArrayIndexer(self)

    def isel(
            self,
            indexers: Optional[Dict[str, Union[int, slice]]] = None,
            missing_dims: Literal["raise", "warn", "ignore"] = 'raise',
            **indexers_kwargs: Union[int, slice],
        ) -> FFTArray:
        """
        Inspired by xarray.DataArray.isel
        """

        # Check for correct use of indexers (either via positional
        # indexers arg or via indexers_kwargs)
        if indexers is not None and indexers_kwargs:
            raise ValueError(
                "cannot specify both keyword arguments and "
                + "positional arguments to FFTArray.isel"
            )

        # Handle two ways of supplying indexers, either via positional
        # argument "indexers" or via keyword arguments for each dimension
        final_indexers: Dict[str, Union[int, slice]]
        if indexers is None:
            final_indexers = indexers_kwargs
        else:
            final_indexers = indexers

        if not isinstance(final_indexers, dict):
            raise ValueError(
                "indexers must be a dictionary or keyword arguments"
            )

        # handle case of empty indexers via supplying indexers={} or nothing at all
        if len(final_indexers) == 0:
            return self

        # Check for indexer names that are not present in FFTArray and
        # according to user choice, raise Error, throw warning or ignore
        check_missing_dim_names(
            indexer_names=final_indexers.keys(),
            dim_names=tuple(self.dims_dict.keys()),
            missing_dims=missing_dims
        )

        # Map indexers into full tuple of valid indexers, one entry per dimension
        tuple_indexers: Tuple[Union[int, slice], ...] = tuple_indexers_from_mapping(
            final_indexers, # type: ignore
            dim_names=[dim.name for dim in self.dims], # type: ignore
        )

        return self.__getitem__(tuple_indexers)

    def sel(
            self,
            indexers: Optional[Dict[str, Union[float, slice]]] = None,
            missing_dims: Literal["raise", "warn", "ignore"] = 'raise',
            method: Optional[Literal["nearest", "pad", "ffill", "backfill", "bfill"]] = None,
            **indexers_kwargs: Union[float, slice],
        ) -> FFTArray:
        """Inspired by xarray.DataArray.sel
        In comparison to itx xarray implementation, there is an add-on:
        - Implements missing_dims arg and accordingly raises errors
        """

        # Check for correct use of indexers (either via positional
        # indexers arg or via indexers_kwargs)
        if indexers is not None and indexers_kwargs:
            raise ValueError(
                "cannot specify both keyword arguments and "
                + "positional arguments to FFTArray.sel"
            )

        # Handle two ways of supplying indexers, either via positional
        # argument "indexers" or via keyword arguments for each dimension
        final_indexers: Dict[str, Union[float, slice]]
        if indexers is None:
            final_indexers = indexers_kwargs
        else:
            final_indexers = indexers

        if not isinstance(final_indexers, dict):
            raise ValueError(
                "indexers must be a dictionary or keyword arguments"
            )

        # handle case of empty indexers via supplying indexers={} or nothing at all
        if len(final_indexers) == 0:
            return self

        # Check for indexer names that are not present in FFTArray and
        # according to user choice, raise Error, throw warning or ignore
        check_missing_dim_names(
            indexer_names=final_indexers.keys(),
            dim_names=tuple(self.dims_dict.keys()),
            missing_dims=missing_dims
        )

        # As opposed to FFTArray.isel, here we have to find the appropriate
        # indices for the coordinate indexers by checking the respective
        # FFTDimension
        tuple_indexers_as_integer = []
        for dim, space in zip(self.dims, self.space):
            if dim.name in final_indexers:
                index = final_indexers[dim.name] # type: ignore
                try:
                    tuple_indexers_as_integer.append(
                        dim._index_from_coord(
                            coord=index,
                            space=space,
                            backend=self.backend,
                            method=method,
                        )
                    )
                except Exception as e:
                    # Here, we check for traced indexer values or
                    # traced FFTDimension and throw a helpful error message
                    # in addition to the original error raised when trying
                    # to map the coord to an index
                    if "Trace" in str(index):
                        raise NotImplementedError(
                            f"FFTArray indexing does not support "
                            + "jitted indexers. Here, your index for "
                            + f"dimension {dim.name} is a traced object"
                        ) from e
                    elif dim._dynamically_traced_coords and "Trace" in str(e):
                        raise NotImplementedError(
                            "dynamically_traced_coords of dimension "
                            + f"{dim.name} must be False to index "
                            + "by label/coordinate."
                        ) from e
                    else:
                        raise e
            else:
                tuple_indexers_as_integer.append(slice(None, None))

        # The rest can be handled by the integer indexing method as we
        # mapped the coordinates to the index representation above
        return self.__getitem__(tuple(tuple_indexers_as_integer))

    @property
    def dims_dict(self) -> Dict[Hashable, FFTDimension]:
        # TODO Ordered Mapping?
        return {dim.name: dim for dim in self._dims}

    @property
    def sizes(self) -> Dict[Hashable, int]:
        # TODO Ordered Mapping?
        return {dim.name: dim.n for dim in self._dims}

    @property
    def dims(self) -> Tuple[FFTDimension, ...]:
        return tuple(self._dims)

    @property
    def shape(self: FFTArray) -> Tuple[int, ...]:
        """..

        Returns
        -------
        Tuple[int, ...]
            Shape of the wavefunction's values.
        """
        return self._values.shape

    def values(self, space: Union[Space, Iterable[Space]]) -> Any:
        """
            Return the values with all lazy state applied.
            Does not mutate self.
            Therefore each call evaluates its lazy state again.
            Use `.into(factors_applied=True)` if you want to evaluate it once and reuse it multiple times.
        """
        fft_arr = self.into(space=space)
        return fft_arr._backend.get_values_with_lazy_factors(
            values=fft_arr._values,
            dims=fft_arr._dims,
            input_factors_applied=fft_arr._factors_applied,
            target_factors_applied=[True]*len(fft_arr._dims),
            spaces=fft_arr._spaces,
            ensure_copy=True,
        )

    def into(
            self,
            space: Optional[Union[Space, Iterable[Space]]] = None,
            backend: Optional[Backend] = None,
            eager: Optional[Union[bool, Iterable[bool]]] = None,
            factors_applied: Optional[Union[bool, Iterable[bool]]] = None,
        ) -> FFTArray:

        values = self._values
        dims = self._dims
        n_dims = len(dims)

        if space is None:
            space_norm = self._spaces
        else:
            space_norm = norm_param(space, n_dims, str)

        if eager is None:
            eager_norm = self._eager
        else:
            eager_norm = norm_param(eager, n_dims, bool)



        if backend is None:
            backend_norm = self._backend
        else:
            backend_norm = backend
            if backend_norm.numpy_ufuncs.iscomplexobj(values):
                values = backend_norm.array(values, dtype=backend_norm.complex_type)
            elif backend_norm.numpy_ufuncs.issubdtype(values.dtype, backend_norm.numpy_ufuncs.floating):
                values = backend_norm.array(values, dtype=backend_norm.real_type)
            else:
                values = backend_norm.array(values)


        needs_fft = [old != new for old, new in zip(self._spaces, space_norm)]
        current_factors_applied = list(self._factors_applied)
        if any(needs_fft):
            pre_fft_applied = [
                False if fft_needed else old_lazy
                for fft_needed, old_lazy in zip(needs_fft, self._factors_applied)
            ]
            values = backend_norm.get_values_with_lazy_factors(
                values=values,
                dims=dims,
                input_factors_applied=self._factors_applied,
                target_factors_applied=pre_fft_applied,
                spaces=self._spaces,
                ensure_copy=False,
            )
            fft_axes = []
            ifft_axes = []
            for dim_idx, (old_space, new_space) in enumerate(zip(self._spaces, space_norm)):
                if old_space != new_space:
                    if old_space == "pos":
                        fft_axes.append(dim_idx)
                    else:
                        ifft_axes.append(dim_idx)
                    current_factors_applied[dim_idx] = False

            if len(fft_axes) > 0:
                values = backend_norm.fftn(values, axes=fft_axes)

            if len(ifft_axes) > 0:
                values = backend_norm.ifftn(values, axes=ifft_axes)


        if factors_applied is None:
            factors_norm_list = []
            for is_eager, fft_needed, is_applied in zip(eager_norm, needs_fft, self._factors_applied):
                if fft_needed:
                    factors_norm_list.append(is_eager)
                else:
                    # We did not do a fft, so just take whatever it was before
                    factors_norm_list.append(is_applied)
            factors_norm = tuple(factors_norm_list)
        else:
            factors_norm = norm_param(factors_applied, n_dims, bool)

        # Bring values into the target form respective lazy state
        values = backend_norm.get_values_with_lazy_factors(
            values=values,
            dims=dims,
            input_factors_applied=current_factors_applied,
            target_factors_applied=factors_norm,
            spaces=space_norm,
            ensure_copy=False,
        )

        return FFTArray(
            dims=dims,
            values=values,
            space=space_norm,
            eager=eager_norm,
            factors_applied=factors_norm,
            backend=backend_norm,
        )

    @property
    def backend(self) -> Backend:
        return self._backend

    # @property
    # def is_eager(self) -> bool:
    #     return self._lazy_state is None

    # def evaluate_lazy_state(self: TFFTArray) -> TFFTArray:
    #     """
    #         Return the same object from view of the public API.
    #         But if the `values`-accessor is used multiple times this improives performance.
    #     """
    #     if self.is_eager:
    #         return self
    #     return self._set_lazy_state(LazyState())

    def transpose(self: FFTArray, *dims: Hashable) -> FFTArray:
        """
            Transpose with dimension names.
        """
        new_dim_names = list(dims)
        old_dim_names = [dim.name for dim in self._dims]
        if len(new_dim_names) == 0:
            new_dim_names = copy(old_dim_names)
            new_dim_names.reverse()
        else:
            assert len(new_dim_names) == len(self._dims)

        axes_transpose = get_axes_transpose(old_dim_names, new_dim_names)
        transposed_values = self._backend.numpy_ufuncs.transpose(self._values, tuple(axes_transpose))

        transposed_arr = FFTArray(
            values=transposed_values,
            dims=[self._dims[idx] for idx in axes_transpose],
            space=[self._spaces[idx] for idx in axes_transpose],
            eager=[self._eager[idx] for idx in axes_transpose],
            factors_applied=[self._factors_applied[idx] for idx in axes_transpose],
            backend=self.backend,
        )
        return transposed_arr

    # def _set_backend(self: TFFTArray, backend: Optional[Backend] = None) -> TFFTArray:
    #     """
    #         Set backend if it is not None.
    #         Collects just a bit of code from all `pos_array` and `freq_array` implementations.
    #     """
    #     res = self
    #     if backend:
    #         res = res.with_backend(backend)
    #     return res

    #--------------------
    # Interface to implement
    #--------------------
    # @abstractmethod
    # def pos_array(self, backend: Optional[Backend] = None) -> PosArray:
    #     ...

    # @abstractmethod
    # def freq_array(self, backend: Optional[Backend] = None) -> FreqArray:
    #     ...

    @property
    def space(self) -> Tuple[Space, ...]:
        """
            Enables automatically and easily detecting in which space a generic FFTArray curently is.
        """
        return self._spaces

    @property
    def eager(self) -> Tuple[bool, ...]:
        """
            If eager is False, the phase factors are not directly applied after an FFT.
            Otherwise they are always left as is and eager does not have any impact on the behavior of this class.
        """
        return self._eager


    #--------------------
    # Default implementations that may be overriden if there are performance benefits
    #--------------------
    # @property
    # def pos_abs(self) -> PosArray:
    #     return np.abs(self.pos_array) # type: ignore

    # @property
    # def pos_sq(self) -> PosArray:
    #     return self.pos_abs**2

    # @property
    # def freq_abs(self) -> FreqArray:
    #     return np.abs(self.freq_array) # type: ignore

    # @property
    # def freq_sq(self) -> FreqArray:
    #     return self.freq_abs**2

    #--------------------
    # Helpers for the implementation
    #--------------------
    @property
    def d_freq(self) -> float:
        """..

        Returns
        -------
        float
            The product of the `d_freq` of all active dimensions.
        """
        return self._backend.reduce_multiply(
            self._backend.array([fft_dim.d_freq for fft_dim in self._dims])
        )

    @property
    def d_pos(self) -> float:
        """..

        Returns
        -------
        float
            The product of the `d_pos` of all active dimensions.
        """
        return self._backend.reduce_multiply(
            self._backend.array([fft_dim.d_pos for fft_dim in self._dims])
        )

    def np_array(self: FFTArray, space: Union[Space, Iterable[Space]], dtype = None):
        """..

        Returns
        -------
        NDArray
            The values of this FFTArray in the specified space as a bare numpy array.
        """

        values = self.into(backend=NumpyBackend(self.backend.precision)).values(space=space)
        return np.array(values, dtype=dtype)

    def _check_consistency(self) -> None:
        """
            Check some invariants of FFTArray.
        """
        if not isinstance(self._values, self._backend.array_type):
            raise ValueError(
                f"Passed in values of type '{type(self._values)}' "
                + f"which is not the array type '{self._backend.array_type}'"
                + f" of the backend '{self._backend}'."
            )
        assert len(self._dims) == len(self._values.shape)
        assert len(self._spaces) == len(self._values.shape)
        assert len(self._eager) == len(self._values.shape)
        assert len(self._factors_applied) == len(self._values.shape)

        dim_names: Set[Hashable] = set()
        for n, dim in zip(self._values.shape, self._dims):
            assert dim.n == n, \
                "Passed in inconsistent n from FFTDimension and values."
            assert dim.name not in dim_names, \
                f"Passed in FFTDimension of name {dim.name} twice!"
            dim_names.add(dim.name)

        assert all([dim_space in get_args(Space) for dim_space in self._spaces])
        assert all([isinstance(dim_eager, bool) for dim_eager in self._eager])
        assert all([isinstance(factor_applied, bool) for factor_applied in self._factors_applied])



@dataclass
class UnpackedValues:
    # FFTDimensions in the order in which they appear in each non-scalar value.
    dims: Tuple[FFTDimension, ...]
    # Values without any dimensions, etc.
    values: List[Union[Number, Any]]
    # whether the input was a FFTArray
    is_fftarray: List[bool]
    # Shared backend between all values.
    backend: Backend
    # outer list: dim_idx, inner_list: op_idx, None: dim does not appear in operand
    factors_applied: List[List[bool]]
    # Space per dimension, must be homogeneous over all values
    space: List[Space]
    # eager per dimension, must be homogeneous over all values
    eager: List[bool]


@dataclass
class UnpackedDimProperties:
    dim: UniformValue[FFTDimension]
    factors_applied: List[bool]
    eager: UniformValue[bool]
    space: UniformValue[Space]

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
        values: List[Union[Number, FFTArray, Any]],
    ) -> UnpackedValues:
    """
        This handles all "alignment" of input values.
        Align dimensions, unify them, unpack all operands to a simple list of values.
    """
    dims: Dict[Hashable, UnpackedDimProperties] = {}
    arrays_to_align: List[Tuple[List[Hashable], Any]] = []
    array_indices = []
    unpacked_values: List[Optional[Union[Number, Any]]] = [None]*len(values)
    is_fftarray: List[bool] = [isinstance(x, FFTArray) for x in values]
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
        is_fftarray = is_fftarray,
        space = space_list,
        factors_applied=factors_applied,
        eager=eager_list,
        backend = backend.get(),
    )


