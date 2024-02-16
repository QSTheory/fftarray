from __future__ import annotations
from typing import (
    Mapping, Optional, Union, List, Any, Tuple, Dict, Hashable,
    Literal, TypeVar, Iterable, Set, Generic, get_args
)
from abc import ABCMeta
from copy import copy
from numbers import Number
from dataclasses import dataclass
import warnings

import numpy as np
import numpy.lib.mixins

from .named_array import align_named_arrays, get_axes_transpose
from .backends.tensor_lib import TensorLib
from .backends.np_backend import NumpyTensorLib
from .helpers import reduce_equal, UniformValue

# TODO: instead: T_FFTArray = TypeVar("T_FFTArray", bound="FFTArray")
T = TypeVar("T")

Space = Literal["pos", "freq"]

#-------------
# Helper functions to support type inference on binary and unary functions in FFTArray
#-------------
def _binary_ufuncs(op):
    def fun(self: FFTArray, other) -> FFTArray:
        return op(self, other)
    def fun_ref(self: FFTArray, other) -> FFTArray:
        return op(other, self)
    return fun, fun_ref

def _unary_ufunc(op):
    def fun(self: FFTArray) -> FFTArray:
        return op(self)
    return fun

class LocFFTArrayIndexer(Generic[T]):
    """
        `FFTArray.loc` allows indexing by dim index but by coordinate position.
        In order to support the indexing operator on a property
        we need this indexable helper class to be returned by the property `loc`.
    """
    _arr: FFTArray

    def __init__(self, arr: FFTArray) -> None:
        self._arr = arr

    def __getitem__(self, item) -> FFTArray:
        if isinstance(item, slice):
            assert item == slice(None, None, None)
            return self._arr.values
        slices = []
        for dim, dim_item, space in zip(self._arr.dims, item, self._arr._space):
            if isinstance(dim_item, slice):
                slices.append(dim_item)
            else:
                # TODO: check for length!=2 tuples (?why?) instead check step==1 or convert to Tuple[float, float]
                slices.append(dim._index_from_coord(dim_item, method=None, space=space, tlib=self._arr.tlib))
        return self._arr.__getitem__(tuple(slices))

def _norm_param(val: Union[T, Iterable[T]], n: int, types) -> Tuple[T, ...]:
    """
       `val` has to be immutable.
    """
    if isinstance(val, types):
        return (val,)*n

    # TODO: Can we make this type check work?
    return tuple(val) # type: ignore

# Think about if we want to inherit a more specific abc class like abc.Mapping
class FFTArray(metaclass=ABCMeta):
    """
        The base class of `PosArray` and `FreqArray` that implements all shared behavior.
    """

    # _dims are stored as a sequence and not by name because their oder needs
    # to match the order of dimensions in _values.
    _dims: Tuple[FFTDimension, ...]
    # Contains an array instance of _tlib with _lazy_state not yet applied.
    _values: Any
    # Marks each dimension whether it is in position or frequency space
    _space: Tuple[Space, ...]
    # Marks each dimension whether the phase-factors should be applied directly after executing a fft or ifft
    _eager: Tuple[bool, ...]
    # Marks each dim whether its phase_factors still need to be applied
    _factors_applied: Tuple[bool, ...]
    # TODO: implement device [#18](https://github.com/QSTheory/fftarray/issues/18)
    # Contains the array backend, precision and device to be used for operations.
    _tlib: TensorLib

    def __init__(
            self,
            values,
            dims: Iterable[FFTDimension],
            space: Union[Space, Iterable[Space]],
            eager: Union[bool, Iterable[bool]],
            factors_applied: Union[bool, Iterable[bool]],
            tlib: TensorLib,
        ):
        """
            This constructor is not meant for normal usage.
            Construct new values via the `fft_array()` function of FFTDimension.
        """
        self._dims = tuple(dims)
        n_dims = len(self._dims)
        self._values = values
        self._space = _norm_param(space, n_dims, str)
        self._eager = _norm_param(eager, n_dims, bool)
        self._factors_applied = _norm_param(factors_applied, n_dims, bool)
        self._tlib = tlib
        self._check_consistency()

    #--------------------
    # Numpy Interfaces
    #--------------------

    # Support numpy ufuncs like np.sin, np.cos, etc.
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return _array_ufunc(self, ufunc, method, inputs, kwargs)

    # Support numpy array protocol.
    # Many libraries use this to coerce special types to plain numpy array e.g.
    # via np.array(fftarray)
    def __array__(self):
        return np.array(self.values)

    # Implement binary operations between FFTArray and also e.g. 1+wf and wf+1
    # This does intentionally not list all posiible operators.
    __add__, __radd__ = _binary_ufuncs(np.add)
    __sub__, __rsub__ = _binary_ufuncs(np.subtract)
    __mul__, __rmul__ = _binary_ufuncs(np.multiply)
    __truediv__, __rtruediv__ = _binary_ufuncs(np.true_divide)
    __floordiv__, __rfloordiv__ = _binary_ufuncs(np.floor_divide)
    __pow__, __rpow__ = _binary_ufuncs(np.power)

    # Implement unary operations
    __neg__ = _unary_ufunc(np.negative)
    __pos__ = _unary_ufunc(np.positive)
    __abs__ = _unary_ufunc(np.absolute)
    __invert__ = _unary_ufunc(np.invert)

    #--------------------
    # Selection
    #--------------------

    def __getitem__(
            self,
            item: Union[int, slice, ellipsis, Tuple[Union[int, slice, ellipsis],...]]
        ) -> FFTArray:
        # Parse item to tuple of different dimensions where for each dimension
        # there can be different indexers: either int or slice or no appearance
        # If the length of the item as a tuple is smaller than the
        # length(FFTArray.dims), then add slice(None, None, None) for missing dimensions.
        tuple_indexers: Tuple[Union[int, slice]]
        if not isinstance(item, tuple):
            if item is Ellipsis:
                return self
            tuple_indexers = (item,)
        else:
            tuple_indexers = item

        if tuple_indexers.count(Ellipsis) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")

        n_dims = len(self.dims)
        if len(tuple_indexers) < n_dims:
            if len(tuple_indexers) == 1:
                tuple_indexers += (slice(None, None, None),) * (n_dims-1)
            else:
                if tuple_indexers[0] is Ellipsis:
                    missing_dim_indexers = n_dims - len(tuple_indexers) + 1
                    tuple_indexers = (
                        (slice(None, None, None),) * missing_dim_indexers
                        + tuple_indexers
                    )
                elif tuple_indexers[-1] is Ellipsis:
                    missing_dim_indexers = n_dims - len(tuple_indexers) + 1
                    tuple_indexers += (slice(None, None, None),) * missing_dim_indexers
                else:
                    missing_dim_indexers = n_dims - len(tuple_indexers)
                    tuple_indexers += (slice(None, None, None),) * missing_dim_indexers

        new_dims = []
        for index, orig_dim, space in zip(tuple_indexers, self._dims, self._space):
            if index == slice(None, None, None):
                # No selection, just keep the old dim.
                new_dims.append(orig_dim)
                continue
            if not isinstance(index, slice):
                index = slice(index, index+1, None)
            # We perform all index sanity checks in _dim_from_slice
            new_dims.append(orig_dim._dim_from_slice(index, space))

        selected_values = self.values.__getitem__(item)
        # Dimensions with the length 1 are dropped in numpy indexing.
        # We decided against this and keeping even dimensions of length 1.
        # So we have to reintroduce those dropped dimensions via reshape.
        selected_values = selected_values.reshape(tuple(dim.n for dim in new_dims))

        # TODO: Implement test to verify that I correctly changed new FFTArray
        # factors_applied to True because we evaluate self.values here
        return FFTArray(
            values=selected_values,
            dims=new_dims,
            space=self._space,
            eager=self._eager,
            factors_applied=[True]*len(new_dims),
            tlib=self.tlib,
        )

    @property
    def loc(self):
        return LocFFTArrayIndexer(self)

    def isel(
            self,
            indexers: Optional[Mapping[Hashable, Union[int, slice]]] = None,
            missing_dims: Literal["raise", "warn", "ignore"] = 'raise',
            **indexers_kwargs: Any,
        ) -> FFTArray:
        """
        Inspired by xarray.DataArray.isel
        """

        if missing_dims not in ["raise", "warn", "ignore"]:
            raise ValueError(
                f"missing_dims={missing_dims} is not valid, it has to be "
                + "one of the following: 'raise', 'warn', 'ignore'"
            )

        # Check for correct use of indexers (either via positional
        # indexers arg or via indexers_kwargs)
        if indexers is not None and indexers_kwargs:
            raise ValueError(
                "cannot specify both keyword arguments and "
                + "positional arguments to FFTArray.isel"
            )
        if indexers is None:
            indexers = indexers_kwargs

        invalid_indexers = [indexer for indexer in indexers if indexer not in self.dims_dict]

        if len(invalid_indexers) > 0:
            if missing_dims == "raise":
                raise ValueError(
                    f"Dimensions {invalid_indexers} do not exist. "
                    + f"Expected one or more of {tuple(self.dims_dict)}"
                )
            elif missing_dims == "warn":
                warnings.warn(
                    f"Dimensions {invalid_indexers} do not exist. "
                    + "These selections will be ignored"
                )

        slices = []
        for dim in self.dims:
            if dim.name in indexers:
                index = indexers[dim.name]
                if not isinstance(index, (slice, int)):
                    raise KeyError(
                        "Using FFTArray.isel, the index for each dimension "
                        + "has to be given as 'slice' or 'int'. "
                        + f"Your input for {dim.name}={index} of type "
                        + f"{type(index)} is not valid"
                    )
                slices.append(indexers[dim.name])
            else:
                slices.append(slice(None, None, None))

        return self.__getitem__(tuple(slices))

    def sel(self, method: Optional[Literal["nearest"]] = None, **kwargs):
        """
            Supports in addition to its xarray-counterpart tuples for ranges.
        """
        #TODO: check that kwargs are float or slice
        slices = []
        for dim, space in zip(self.dims, self._space):
            if dim.name in kwargs:
                slices.append(
                    # mypy error: kwargs is of type "Dict[str, Any]" but
                    # dim.name is of type "Hashable". However, the if condition
                    # already makes sure that dim.name is a key of kwargs (so
                    # also of type "str")
                    # TODO: check slice step == 1
                    dim._index_from_coord(
                        kwargs[dim.name], # type: ignore
                        method=method,
                        space=space,
                        tlib=self.tlib,
                    )
                )
            else:
                slices.append(slice(None, None, None))
        return self.__getitem__(tuple(slices))

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

    @property
    def values(self) -> Any:
        """
            Return the values with all lazy state applied.
            Does not mutate self.
            Therefore each call evaluates its lazy state again.
            Use `evaluate_lazy_state` if you want to evaluate it once and reuse it multiple times.
        """
        # TODO Ensure defensive copy here for the Numpy Backend?
        return self._tlib.get_values_with_lazy_factors(
            values=self._values,
            dims=self._dims,
            input_factors_applied=self._factors_applied,
            target_factors_applied=[True]*len(self._dims),
            space=self._space,
        )

    def into(
            self,
            space: Optional[Union[Space, Iterable[Space]]] = None,
            eager: Optional[Union[bool, Iterable[bool]]] = None,
            factors_applied: Optional[Union[bool, Iterable[bool]]] = None,
            tlib: Optional[TensorLib] = None,
        ) -> FFTArray:

        values = self._values
        dims = self._dims
        n_dims = len(dims)

        if space is None:
            space_norm = self._space
        else:
            space_norm = _norm_param(space, n_dims, str)

        if eager is None:
            eager_norm = self._eager
        else:
            eager_norm = _norm_param(eager, n_dims, bool)



        if tlib is None:
            tlib_norm = self._tlib
        else:
            tlib_norm = tlib
            if tlib_norm.numpy_ufuncs.iscomplexobj(self._values):
                values = tlib_norm.array(values, dtype=tlib_norm.complex_type)
            else:
                values = tlib_norm.array(values, dtype=tlib_norm.real_type)


        needs_fft = [old != new for old, new in zip(self._space, space_norm)]
        current_factors_applied = list(self._factors_applied)
        if any(needs_fft):
            pre_fft_applied = [
                False if fft_needed else old_lazy
                for fft_needed, old_lazy in zip(needs_fft, self._factors_applied)
            ]
            values = tlib_norm.get_values_with_lazy_factors(
                values=values,
                dims=dims,
                input_factors_applied=self._factors_applied,
                target_factors_applied=pre_fft_applied,
                space=self._space,
            )
            fft_axes = []
            ifft_axes = []
            for dim_idx, (old_space, new_space) in enumerate(zip(self._space, space_norm)):
                if old_space != new_space:
                    if old_space == "pos":
                        fft_axes.append(dim_idx)
                    else:
                        ifft_axes.append(dim_idx)
                    current_factors_applied[dim_idx] = False

            if len(fft_axes) > 0:
                values = tlib_norm.fftn(values, axes=fft_axes)

            if len(ifft_axes) > 0:
                values = tlib_norm.ifftn(values, axes=ifft_axes)


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
            factors_norm = _norm_param(factors_applied, n_dims, bool)

        # Bring values into the target form respective lazy state
        values = tlib_norm.get_values_with_lazy_factors(
            values=values,
            dims=dims,
            input_factors_applied=current_factors_applied,
            target_factors_applied=factors_norm,
            space=space_norm,
        )

        return FFTArray(
            dims=dims,
            values=values,
            space=space_norm,
            eager=eager_norm,
            factors_applied=factors_norm,
            tlib=tlib_norm,
        )

    @property
    def tlib(self) -> TensorLib:
        return self._tlib

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
        transposed_values = self._tlib.numpy_ufuncs.transpose(self._values, tuple(axes_transpose))

        transposed_arr = FFTArray(
            values=transposed_values,
            dims=[self._dims[idx] for idx in axes_transpose],
            space=[self._space[idx] for idx in axes_transpose],
            eager=[self._eager[idx] for idx in axes_transpose],
            factors_applied=[self._factors_applied[idx] for idx in axes_transpose],
            tlib=self.tlib,
        )
        return transposed_arr

    # def _set_tlib(self: TFFTArray, tlib: Optional[TensorLib] = None) -> TFFTArray:
    #     """
    #         Set tlib if it is not None.
    #         Collects just a bit of code from all `pos_array` and `freq_array` implementations.
    #     """
    #     res = self
    #     if tlib:
    #         res = res.with_tlib(tlib)
    #     return res

    #--------------------
    # Interface to implement
    #--------------------
    # @abstractmethod
    # def pos_array(self, tlib: Optional[TensorLib] = None) -> PosArray:
    #     ...

    # @abstractmethod
    # def freq_array(self, tlib: Optional[TensorLib] = None) -> FreqArray:
    #     ...

    @property
    def space(self) -> Tuple[Space, ...]:
        """
            Enables automatically and easily detecting in which space a generic FFTArray curently is.
        """
        return self._space

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
        return self._tlib.reduce_multiply(
            self._tlib.array([fft_dim.d_freq for fft_dim in self._dims])
        )

    @property
    def d_pos(self) -> float:
        """..

        Returns
        -------
        float
            The product of the `d_pos` of all active dimensions.
        """
        return self._tlib.reduce_multiply(
            self._tlib.array([fft_dim.d_pos for fft_dim in self._dims])
        )

    def _check_consistency(self) -> None:
        """
            Check some invariants of FFTArray.
        """
        # TODO: Implement new invariants
        assert len(self._dims) == len(self._values.shape)
        assert len(self._space) == len(self._values.shape)
        assert len(self._eager) == len(self._values.shape)
        assert len(self._factors_applied) == len(self._values.shape)

        dim_names: Set[Hashable] = set()
        for n, dim in zip(self._values.shape, self._dims):
            assert dim.n == n, \
                "Passed in inconsistent n from FFTDimension and values."
            assert dim.name not in dim_names, \
                f"Passed in FFTDimension of name {dim.name} twice!"
            dim_names.add(dim.name)

        assert all([dim_space in get_args(Space) for dim_space in self._space])
        assert all([isinstance(dim_eager, bool) for dim_eager in self._eager])
        assert all([isinstance(factor_applied, bool) for factor_applied in self._factors_applied])


# Implementing NEP 13 https://numpy.org/neps/nep-0013-ufunc-overrides.html
# See also https://numpy.org/doc/stable/user/basics.dispatch.html
def _array_ufunc(self: FFTArray, ufunc, method, inputs, kwargs):
    """Override NumPy ufuncs, per NEP-13."""
    if method != "__call__":
        return NotImplemented

    if "out" in kwargs:
        return NotImplemented

    # For now only unary and binary ufuncs
    if len(inputs) > 2:
        return NotImplemented

    # These special functions have shortcuts for the evaluation of lazy state.
    # Therefore the lazy_state is not applied on unpacking.
    # if ufunc == np.abs or ufunc == np.multiply :
    #     unpacked_inputs = _unpack_fft_arrays(inputs, True)
    # else:
    # Apply all lazy state because we have no shortcuts.
    unpacked_inputs = _unpack_fft_arrays(inputs)

    # Returning NotImplemented gives other operands a chance to see if they support interacting with us.
    # Not really necessary here.
    if not all(isinstance(x, (Number, FFTArray)) or hasattr(x, "__array__") for x in unpacked_inputs.values):
        return NotImplemented

    # Look up the actual ufunc
    try:
        tensor_lib_ufunc = getattr(unpacked_inputs.tlib.numpy_ufuncs, ufunc.__name__)
    except:
        return NotImplemented

    # Element-wise multiplication is commutative with the multiplication of the phase factor.
    # So we can always directly multiply with the inner values and keep the outer wrapper class as is
    # We can also shortcut the multiplication of a scalar into the lazy_scale. (Commented out at the moment)
    # if ufunc == np.abs and unpacked_inputs.lazy_state is not None:
    #     # For abs we can drop the phases without ever applying them
    #     # since they would have evaluated to one.
    #     values = tensor_lib_ufunc(*unpacked_inputs.values, **kwargs)
    #     # TODO: Apply scale
    #     assert False
    #     return FFTArray(
    #         values=values,
    #         space=unpacked_inputs.space,
    #         dims=unpacked_inputs.dims,
    #         eager=unpacked_inputs.eager,
    #         factors_applied=True,
    #     )
    # if ufunc == np.multiply and unpacked_inputs.lazy_state is not None:
    #     values = tensor_lib_ufunc(*unpacked_inputs.values, **kwargs)
    #     return _pack_values(
    #         values,
    #         unpacked_inputs.space,
    #         unpacked_inputs.dims,
    #         unpacked_inputs.lazy_state,
    #     )
    # Further ops: conj, both lazy factors in addition, would need different unpacking...?
    assert all(unpacked_inputs.factors_applied)
    values = tensor_lib_ufunc(*unpacked_inputs.values, **kwargs)
    return FFTArray(
        values=values,
        space=unpacked_inputs.space,
        dims=unpacked_inputs.dims,
        eager=unpacked_inputs.eager,
        factors_applied=unpacked_inputs.factors_applied,
        tlib=unpacked_inputs.tlib,
    )

# def _pack_values(
#         values,
#         space: Space,
#         dims: List[FFTDimension],
#         tlib: TensorLib,
#         lazy_state: Optional[LazyState],
#     ) -> FFTArray:
#     """
#         Finish up a value after a ufunc.
#         Internally it is more parametrized and this puts it back together.
#     """
#     assert tlib == _get_tensor_lib(dims)
#     assert tlib.has_precision(values, tlib.precision)
#     # eager does not matter here because it is overwritten with a concrete lazy_state
#     return FFTArray(
#         values=values,
#         dims=dims,

#     )
#     if space == "pos":
#         arr: FFTArray = PosArray(values, dims=dims, eager=True)
#     else:
#         assert space == "freq"
#         arr = FreqArray(values, dims=dims, eager=True)
#     arr._lazy_state = lazy_state
#     return arr




@dataclass
class UnpackedValues:
    # FFTDimensions in the order in which they appear in each non-scalar value.
    dims: List[FFTDimension]
    # Values without any dimensions, etc.
    values: List[Union[Number, Any]]
    # Space nper dimension in which all values were
    space: List[Space]
    # True => all factors applied, False factors missing in exactly one of the values.
    factors_applied: List[bool]
    # Fails if not homogeneous in all values.
    eager: List[bool]
    # Shared tensor-lib between all values.
    tlib: TensorLib

@dataclass
class UnpackedDimProperties:
    dim: UniformValue[FFTDimension]
    # factors_applied: bool
    eager: UniformValue[bool]
    space: UniformValue[Space]

    def __init__(self):
        self.dim = UniformValue()
        # self.input_factors_applied = True
        # self.target_factors_applied = True
        self.eager = UniformValue()
        self.space = UniformValue()

def _unpack_fft_arrays(
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
    tlib: UniformValue[TensorLib] = UniformValue()

    for idx, x in enumerate(values):
        if isinstance(x, Number):
            unpacked_values[idx] = x
        elif hasattr(x, "shape") and not isinstance(x, FFTArray):
            if x.shape == ():
                unpacked_values[idx] = x
            else:
                raise ValueError(
                    "Cannot multiply coordinate-less arrays with a FFTArray."
                )
        else:
            array_indices.append(idx)
            assert isinstance(x, FFTArray)

            tlib.set(x.tlib)
            # input_factors_applied = x._factors_applied
            # target_factors_applied = list(x._factors_applied)

            for dim_idx, fft_dim in enumerate(x._dims):
                if not fft_dim.name in dims:
                    dim_props = UnpackedDimProperties()
                    dims[fft_dim.name] = dim_props
                else:
                    dim_props = dims[fft_dim.name]

                try:
                    dim_props.dim.set(fft_dim)
                except ValueError:
                    raise ValueError(
                        "Tried to call ufunc on two FFTArrays with " +
                        "different dimension of name " +
                        f"{fft_dim.name}."
                    )

                try:
                    dim_props.space.set(x._space[dim_idx])
                except ValueError:
                    raise ValueError(
                        "Tried to call ufunc on two FFTArrays with " +
                        "different spaces in dimension of name " +
                        f"{fft_dim.name}." +
                        "One of them has to be explicitly converted " +
                        "into the other space to ensure the correct space."
                    )

                try:
                    dim_props.eager.set(x._eager[dim_idx])
                except ValueError:
                    raise ValueError(
                        "Tried to call ufunc on two FFTArrays with " +
                        "different eager settings in dimension of name " +
                        f"{fft_dim.name}."
                    )

                # if not keep_lazy this will just be overwritten later.
                # if dim_props.factors_applied:
                #     dim_props.factors_applied = x._factors_applied[dim_idx]
                # else:
                #     # need to apply this factor
                #     target_factors_applied[dim_idx] = True

            # if not keep_lazy:
            #     target_factors_applied = [True]*len(target_factors_applied)

            # Very cheap, if nothing changes.
            raw_arr = tlib.get().get_values_with_lazy_factors(
                values = x._values,
                dims=x._dims,
                input_factors_applied=x._factors_applied,
                # TODO: This needs more thought, do it in a later refactor.
                target_factors_applied=[True]*len(x._factors_applied),
                space=x._space,
            )

            elem_dim_names = [fft_dim.name for fft_dim in x._dims]
            arrays_to_align.append((elem_dim_names, raw_arr))


    # Broadcasting
    dim_names, aligned_arrs = align_named_arrays(arrays_to_align, tlib=tlib.get())
    for idx, arr in zip(array_indices, aligned_arrs):
        unpacked_values[idx] = arr

    dims_list = [dims[dim_name].dim.get() for dim_name in dim_names]
    space_list = [dims[dim_name].space.get() for dim_name in dim_names]
    eager_list = [dims[dim_name].eager.get() for dim_name in dim_names]
    unpacked_values = [tlib.get().as_array(x) for x in unpacked_values]

    for value in unpacked_values:
        assert not value is None

    return UnpackedValues(
        dims = dims_list,
        values = unpacked_values, # type: ignore
        space = space_list,
        factors_applied=[True]*len(dims_list),
        eager=eager_list,
        tlib = tlib.get(),
    )


@dataclass
class FFTDimension:
    """Properties of an FFTWave grid for one dimension.

    This class encapsulates all the properties of the position and frequency
    coordinate grids for one dimension.

    Note that properties associated with the position grid are denoted by `pos`,
    whereas the frequency grid properties are denoted with `freq`.

    It takes care that the spacing lines up according to the mathematics of the
    FFT. The mathematics of the Discrete Fourier Transform automatically
    determine the third component of ``freq_extent``, ``pos_extent`` and
    resolution if the other two are set. The fact that resolution has to be a
    positive integer therefore also quantizes the ratio of the extent and
    sample spacing in both position and frequency space.

    **Parameters**::

        loose_params: Union[str, List[str]] = [] # List of loose grid parameters (parameters that can be improved by the constraint solver).

        n:           Union[int, Literal["power_of_two", "even"]] = "power_of_two" # The number of position and frequency grid points.
                                                                                  # Instead of supplying an integer, one of the rounding modes "even" or "power_of_two" can be chosen.
        d_pos:       Optional[float] = None # The distance between two neighboring position grid points.
        d_freq:      Optional[float] = None # The distance between two neighboring frequency grid points.
        pos_min:     Optional[float] = None # The smallest position grid point.
        pos_max:     Optional[float] = None # The largest position grid point.
        pos_middle:  Optional[float] = None # The middle of the position grid.
        pos_extent:  Optional[float] = None # The length of the position grid.
        freq_min:    Optional[float] = None # The smallest frequency grid point.
        freq_max:    Optional[float] = None # The largest frequency grid point.
        freq_extent: Optional[float] = None # The length of the frequency grid.
        freq_middle: Optional[float] = None # The offset of the frequency grid.

    **Implementation details**

    The grid in both spaces (position and frequency) goes from min to max
    including both points. Therefore ``d_pos = (pos_max-pos_min)/(n-1)``. The
    grid always consists of an even number of points. Therefore, the number of
    samples n has to be an even integer. The frequencies in frequency space can
    be acquired via ``numpy.fft.fftfreq(n, d_pos)``. These frequencies are
    spatial frequencies in the unit cycles/m. The wavelength lambda is the space
    equivalent of T for time signals. => ``lambda = 1/numpy.fft.fftfreq(n,
    d_pos)`` According to DeBroglie we have ``lambda = h/p`` => ``p = h *
    numpy.fft.fftfreq(n, d_pos)``

    The pos_middle is the sample on the right hand side of the exact center of
    the grid.

    **Examples**::

        n = 4
                        pos_middle
             pos_min           pos_max
                |-----|-----|-----|
        index:  0     1     2     3
                 d_pos d_pos d_pos

        n = 5
                        pos_middle
             pos_min                 pos_max
                |-----|-----|-----|-----|
        index:  0     1     2     3     4
                 d_pos d_pos d_pos d_pos

    The freq_middle is the sample on the right hand side of the exact center of
    the grid.

    **Examples**::

        n = 4
                          freq_middle
             freq_min             freq_max
                |------|------|------|
        index:  0      1      2      3
                 d_freq d_freq d_freq

        n = 6

             freq_min           freq_middle     freq_max
                |------|------|------|------|------|
        index:  0      1      2      3      4      5
                 d_freq d_freq d_freq d_freq d_freq

    .. highlight:: none

    These are the symbolic definitions of all the different names (for even ``n``)::

        pos_extent = pos_max - pos_min
        pos_middle = 0.5 * (pos_min + pos_max + d_pos)
        d_pos = pos_extent/(n-1)

        freq_extent = freq_max - freq_min
        freq_middle = 0.5 * (freq_max + freq_min + d_freq)
        d_freq = freq_extent/(n-1)

        d_freq * d_pos * n = 2*pi

    For odd ``n`` the definitions for ``pos_middle`` and ``freq_middle`` change to ensure that
    they and the minimum and maximum position and frequency are actually sampled and not in between two samples.::

        pos_middle = 0.5 * (pos_min + pos_max)
        freq_middle = 0.5 * (freq_max + freq_min)

    For performance reasons it is recommended to have ``n`` be a power of two.

    Individual array coordinates::

        pos = np.arange(0, n) * d_pos + pos_min
        freq = np.fft.fftfreq(n = n, d = d_pos) + freq_middle

    .. highlight:: none

    These arrays fulfill the following properties::

        np.max(pos) = pos_max
        np.min(pos) = pos_min
        np.max(freq) = freq_max
        np.min(freq) = freq_min

        pos[1]-pos[0] = d_pos (if n >= 2)
    """

    _pos_min: float
    _freq_min: float
    _d_pos: float
    _n: int
    _name: Hashable

    def __init__(
            self,
            name: str,
            n: int,
            d_pos: float,
            pos_min: float,
            freq_min: float,
        ):
        self._name = name
        self._n = n
        self._d_pos = d_pos
        self._pos_min = pos_min
        self._freq_min = freq_min

    def __repr__(self: FFTDimension) -> str:
        arg_str = ", ".join([f"{name[1:]}={repr(value)}" for name, value in self.__dict__.items()])
        return f"FFTDimension({arg_str})"

    def __str__(self: FFTDimension) -> str:
        properties = (
            f"\t Number of grid points n: {self.n} \n\t " +
            f"Position space: min={self.pos_min}, middle={self.pos_middle}, " +
            f"max={self.pos_max}, extent={self.pos_extent}, d_pos={self.d_pos} \n\t " +
            f"Frequency space: min={self.freq_min}, middle={self.freq_middle}, " +
            f"max={self.freq_max}, extent={self.freq_extent}, d_freq={self.d_freq}"
        )
        return (
            f"FFTDimension with name '{self.name}' and the " +
            f"following properties:\n{properties}"
        )

    @property
    def n(self: FFTDimension) -> int:
        """..

        Returns
        -------
        float
            The number of grid points.
        """
        return self._n

    @property
    def name(self: FFTDimension) -> Hashable:
        """..

        Returns
        -------
        float
            The name of his FFTDimension.
        """
        return self._name

    # ---------------------------- Position Space ---------------------------- #

    @property
    def d_pos(self: FFTDimension) -> float:
        """..

        Returns
        -------
        float
            The distance between two neighboring position grid points.
        """
        return self._d_pos

    @property
    def pos_min(self: FFTDimension) -> float:
        """..

        Returns
        -------
        float
            The smallest position grid point.
        """
        return self._pos_min

    @property
    def pos_max(self: FFTDimension) -> float:
        """..

        Returns
        -------
        float
            The largest position grid point.
        """
        return (self.n - 1) * self.d_pos + self.pos_min

    @property
    def pos_middle(self: FFTDimension) -> float:
        """..

        Returns
        -------
        float
            The middle of the position grid.
            If n is even, it is defined as the (n/2+1)'th position grid point.
        """
        return self.pos_min + self.n//2 * self.d_pos

    @property
    def pos_extent(self: FFTDimension) -> float:
        """..

        Returns
        -------
        float
            The length of the position grid.
            It is defined as `pos_max - pos_min`.
        """
        return (self.n - 1) * self.d_pos

    def _dim_from_slice(
            self,
            range: slice,
            space: Space,
        ) -> FFTDimension:
        """
            Get a new FFTDimension for a interval selection in a given space.
            Does not support steps!=1.

            Indexing behaviour is the same as for a numpy array with the
            difference that we raise an IndexError if the resulting size
            is not at least 1. We require n>=1 to create a valid FFTDimension.
        """
        if not(range.step is None or range.step == 1):
            raise IndexError(
                f"You can't index using {range} but only " +
                f"slice({range.start}, {range.stop}) with implicit index step 1. " +
                "Substepping requires reducing the respective other space " +
                "which is not well defined due to the arbitrary choice of " +
                "which part of the space to keep (constant min, middle or max?). "
            )

        # TODO: this does not work for jitted function call, use of JaxTensorLib
        # make this maybe a TensorLib method.
        def _remap_index_check_int(
                index: int,
                dim_n: int,
                index_kind: Literal["start", "stop"],
            ) -> int:
            # TODO: this version yields analog indexing to xarray and numpy
            # but does not work with jitted functions, should the behaviour depend
            # on the tensorlib? however, there is no tlib associated with a FFTDimension
            if index is None:
                if index_kind == "start":
                    return 0
                else:
                    return dim_n
            try:
                # TODO: This might need a tlib method instead to check for correct type
                # abs_sq.tlib.numpy_ufuncs.issubdtype(abs_sq.values.dtype, abs_sq.tlib.numpy_ufuncs.floating)
                index = index.item()
            except:
                ...
            if not isinstance(index, int):
                raise IndexError("only integers, slices (`:`), ellipsis (`...`) are valid indices.")
            if index < -dim_n:
                return 0
            if index < 0:
                return index + dim_n
            if index >= dim_n:
                return dim_n
            return index

        start = _remap_index_check_int(range.start, self.n, index_kind="start")
        end = _remap_index_check_int(range.stop, self.n, index_kind="stop")

        n = end - start
        if n < 1:
            raise IndexError(
                f"Your indexing {range} is not valid. To create a valid "
                + "FFTDimension, the stop index must be bigger than the start "
                + "index in order to keep at least one sample (n>=1)."
            )

        return self._dim_from_start_and_n(start=start, n=n, space=space)

    def _dim_from_start_and_n(
            self,
            start: int,
            n: int,
            space: Space,
        ) -> FFTDimension:
        # TODO: do we still require to skip FFTDimension.__init__? I Think not
        # because we moved the constraint solver out of the class init.
        new = self.__class__.__new__(self.__class__)
        new._name = self.name
        new._n = n

        if space == "pos":
            new._pos_min = self.pos_min + start*self.d_pos
            new._freq_min = self.freq_min
            new._d_pos = self.d_pos
        elif space == "freq":
            new._pos_min = self.pos_min
            new._freq_min = self.freq_min + start*self.d_freq
            new._d_pos = 1./(self.d_freq*n)
        else:
            assert False, "Unreachable"
        return new

    def _index_from_coord(
            self,
            coord: Union[float, Tuple[float, float]],
            space: Space,
            tlib: TensorLib,
            method: Optional[Literal["nearest", "pad", "ffill", "backfill", "bfill"]] = None,
        ):
        """
            Compute index from given coordinate `x` which can be float or Tuple[float, float].
            In case of tuple input, return slice object

            min maps 2.5 to index 3 (bfill, backfill), max currently maps 2.5 to index 2 (pad, ffill)
        """
        if isinstance(coord, tuple):
            if method is not None:
                raise NotImplementedError(
                    f"cannot use method: `{method}` if the coord argument "
                    + f"is not scalar, here: {coord}."
                )
            if coord[0] is None:
                coord_start = getattr(self, f"{space}_min")
            else:
                coord_start = coord[0]
            if coord[1] is None:
                coord_stop = getattr(self, f"{space}_max")
            else:
                coord_stop = coord[1]
            idx_min = self._index_from_coord(coord_start, method="bfill", space=space, tlib=tlib)
            idx_max = self._index_from_coord(coord_stop, method="ffill", space=space, tlib=tlib)
            return slice(
                tlib.array(idx_min, dtype=int),
                tlib.array(idx_max+1, dtype=int) # slice stop is open interval, add 1
            )

        if space == "pos":
            raw_idx = (coord - self.pos_min) / self.d_pos
        else:
            raw_idx = (coord - self.freq_min) / self.d_freq

        clamped_index = tlib.numpy_ufuncs.min(tlib.array([
            tlib.numpy_ufuncs.max(tlib.array([0, raw_idx])),
            self.n - 1
        ]))

        try:

            if method is None:
                # NOTE: This is not jittable, the main problem is that this would
                # have to raise an Error sometimes which is not supported (I think).
                # For if in general, one could implement a tlib.cond method.
                if (
                    tlib.numpy_ufuncs.round(raw_idx) != raw_idx or
                    not tlib.numpy_ufuncs.array_equal(clamped_index, raw_idx)
                ):
                    raise KeyError(
                        f"No exact index found for {coord} in {space}-space of dim " +
                        f'"{self.name}". Try the keyword argument ' +
                        'method="nearest".'
                    )
                final_idx = raw_idx
            elif  method == "nearest":
                # The combination of floor and +0.5 prevents the "ties to even" rounding of floating point numbers.
                final_idx = tlib.numpy_ufuncs.floor(clamped_index + 0.5)
            elif method in ["bfill", "backfill"]:
                final_idx = tlib.numpy_ufuncs.ceil(clamped_index)
                if raw_idx > self.n - 1:
                    raise KeyError(
                        f"Coord {coord} not found with method '{method}', "
                        + "you could try one of the following instead: "
                        + "'ffill', 'pad' or 'nearest'."
                    )
            elif method in ["ffill", "pad"]:
                final_idx = tlib.numpy_ufuncs.floor(clamped_index)
                if raw_idx < 0:
                    raise KeyError(
                        f"Coord {coord} not found with method '{method}', "
                        + "you could try one of the following instead: "
                        + "'bfill', 'backfill' or 'nearest'."
                    )
            else:
                raise ValueError(f"Specified unsupported look-up method `{method}`.")

        except Exception as e:
            # TODO: think about implementing test for this and maybe custom FFTArrayErrorType
            # maybe later, we'll see that we don't need the special nearest handling
            if type(e).__name__ == "TracerBoolConversionError":
                raise Exception(
                    "You can only use FFTDimension._index_from_coord within "
                    + f"a jitted function with method `nearest`, not `{method}`."
                ) from e
            else:
                raise e

        return tlib.array(final_idx, dtype=int)

    # ---------------------------- Frequency Space --------------------------- #

    @property
    def d_freq(self: FFTDimension) -> float:
        """..

        Returns
        -------
        float
            The distance between frequency grid points.
        """
        return 1./(self.n*self.d_pos)

    @property
    def freq_min(self: FFTDimension) -> float:
        """..

        Returns
        -------
        float
            The smallest frequency grid point.

        """
        return self._freq_min

    @property
    def freq_middle(self: FFTDimension) -> float:
        """..

        Returns
        -------
        float
            The middle of the frequency grid.
            If n is even, it is defined as the (n/2+1)'th frequency grid point.
        """
        return self.freq_min + self.n//2 * self.d_freq

    @property
    def freq_max(self: FFTDimension) -> float:
        """..

        Returns
        -------
        float
            The largest frequency grid point.
        """
        return (self.n - 1) * self.d_freq + self.freq_min

    @property
    def freq_extent(self: FFTDimension) -> float:
        """..

        Returns
        -------
        float
            The length of the frequency grid.
            It is defined as `freq_max - freq_min`.
        """
        return (self.n - 1) * self.d_freq

    def _raw_coord_array(
                self: FFTDimension,
                tlib: TensorLib,
                space: Space,
            ):

        indices = tlib.numpy_ufuncs.arange(
            0,
            self.n,
            dtype = tlib.real_type,
        )

        if space == "pos":
            return indices * self.d_pos + self.pos_min
        elif space == "freq":
            return indices * self.d_freq + self.freq_min
        else:
            raise ValueError(f"space has to be either 'pos' or 'freq', not {space}.")

    def fft_array(
            self: FFTDimension,
            tlib: TensorLib,
            space: Space,
            eager: bool = False,
        ) -> FFTArray:
        """..

        Returns
        -------
        FFTArray
            The grid coordinates of the chosen space packed into an FFTArray with self as only dimension.
        """

        values = self._raw_coord_array(
            tlib=tlib,
            space=space,
        )

        return FFTArray(
            values=values,
            dims=[self],
            eager=eager,
            factors_applied=True,
            space=space,
            tlib=tlib,
        )

    def np_array(self: FFTDimension, space: Space):
        return self._raw_coord_array(tlib=NumpyTensorLib(), space=space)
