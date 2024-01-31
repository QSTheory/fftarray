from __future__ import annotations
from typing import (
    Optional, Union, List, Any, Tuple, Dict, Hashable,
    Literal, TypeVar, Iterable, Set, Generic
)
from abc import ABCMeta, abstractmethod
from numpy.typing import ArrayLike, NDArray
from copy import copy, deepcopy
from numbers import Number
from dataclasses import dataclass

import numpy as np
import numpy.lib.mixins

from .named_array import align_named_arrays, transpose_array
from .fft_constraint_solver import _z3_constraint_solver
from .lazy_state import PhaseFactors, LazyState, get_lazy_state_to_apply
from .backends.tensor_lib import TensorLib
from .backends.np_backend import NumpyTensorLib
from .helpers import reduce_equal, UniformValue

TFFTArray = TypeVar("TFFTArray", bound="FFTArray")
T = TypeVar("T")

Space = Literal["pos", "freq"]

def _get_tensor_lib(dims: Iterable[FFTDimension]) -> TensorLib:
    return reduce_equal(
        map(lambda dim: dim._default_tlib, dims),
        "Tried to join arrays with different tensor-libs or precision settings."
    )

#-------------
# Helper functions to support type inference on binary and unary functions in FFTArray
#-------------
def _binary_ufuncs(op):
    def fun(self: TFFTArray, other) -> TFFTArray:
        return op(self, other)
    def fun_ref(self: TFFTArray, other) -> TFFTArray:
        return op(other, self)
    return fun, fun_ref

def _unary_ufunc(op):
    def fun(self: TFFTArray) -> TFFTArray:
        return op(self)
    return fun

class LocFFTArrayIndexer(Generic[T]):
    """
        `wf.loc` allows indexing by dim index but by coordinate position.
        In order to support the indexing operator on a property
        we need this indexable helper class to be returned by the property `loc`.
    """
    _arr: FFTArray

    def __init__(self, arr: FFTArray) -> None:
        self._arr = arr

    def __getitem__(self: LocFFTArrayIndexer, item) -> FFTArray:
        if isinstance(item, slice):
            assert item == slice(None, None, None)
            return self._arr.values
        slices = []
        for dim, dim_item in zip(self._arr.dims, item):
            if isinstance(dim_item, slice):
                slices.append(dim_item)
            else:
                slices.append(dim._index_from_coord(dim_item, method=None, space=self._arr.space))
        return self._arr.__getitem__(tuple(slices))


class FFTArray(metaclass=ABCMeta):
    """
        The base class of `PosArray` and `FreqArray` that implements all shared behavior.
    """

    # _dims are stored as a sequence and not by name because their oder needs
    # to match the order of dimensions in _values.
    _dims: Tuple[FFTDimension, ...]
    # Contains an array instance of _tlib with _lazy_state not yet applied.
    _values: Any
    # Contains all lazy computations.
    # `None` signals that eager evaluation should be used.
    _lazy_state: Optional[LazyState]
    # Contains the array backend, precision and device to be used for operations.
    _tlib: TensorLib

    def __init__(
            self,
            values: ArrayLike,
            dims: Iterable[FFTDimension],
            eager: bool,
        ):
        """
            This constructor should only be used internally.
            This class initself cannot even do ffts.
            Construct new values via the `pos_array()` and `freq_array()` functions of FFTDimension.
        """
        self._dims = tuple(dims)
        self._values = values
        self._lazy_state = None if eager else LazyState()
        self._tlib = _get_tensor_lib(self._dims)
        self._check_consistency()

    #--------------------
    # Numpy Interfaces
    #--------------------

    # Support numpy ufuncs like np.sin, np.cos, etc.
    def __array_ufunc__(self: TFFTArray, ufunc, method, *inputs, **kwargs) -> TFFTArray:
        return _array_ufunc(self, ufunc, method, inputs, kwargs)

    # Support numpy array protocol.
    # Many libraries use this to coerce special types to plain numpy array e.g.
    # via np.array(fftarray)
    def __array__(self: TFFTArray) -> NDArray:
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

    def __getitem__(self: TFFTArray, item) -> TFFTArray:
        new_dims = []
        if isinstance(item, slice):
            item = [item]
        for index, dimension in zip(item, self._dims):
            if not isinstance(index, slice):
                new_dim = dimension._dim_from_start_and_n(
                    start=index,
                    n=1,
                    space=self.space,
                )
                index = slice(index, index+1, None)
            elif index == slice(None, None, None):
                # No selection, just keep the old dim.
                new_dim = dimension
            else:
                new_dim = dimension._dim_from_slice(index, self.space)

            new_dims.append(new_dim)

        selected_values = self.values.__getitem__(item)
        # Dimensions with the length 1 are dropped in numpy indexing.
        # We decided against this and keeping even dimensions of length 1.
        # So we have to reintroduce those dropped dimensions via reshape.
        selected_values = selected_values.reshape(tuple(dim.n for dim in new_dims))

        return self.__class__(
            values = selected_values,
            dims = new_dims,
            eager = self._lazy_state is None
        )

    @property
    def loc(self: TFFTArray) -> LocFFTArrayIndexer:
        return LocFFTArrayIndexer(self)

    def isel(self: TFFTArray, **kwargs) -> TFFTArray:
        slices = []
        for dim in self.dims:
            if dim.name in kwargs:
                slices.append(kwargs[dim.name])
            else:
                slices.append(slice(None, None, None))
        return self.__getitem__(tuple(slices))

    def sel(self: TFFTArray, method: Optional[Literal["nearest"]] = None, **kwargs) -> TFFTArray:
        """
            Supports in addition to its xarray-counterpart tuples for ranges.
        """
        slices = []
        for dim in self.dims:
            if dim.name in kwargs:
                slices.append(
                    # mypy error: kwargs is of type "Dict[str, Any]" but
                    # dim.name is of type "Hashable". However, the if condition
                    # already makes sure that dim.name is a key of kwargs (so
                    # also of type "str")
                    dim._index_from_coord(
                        kwargs[dim.name], # type: ignore
                        method=method,
                        space=self.space,
                    )
                )
            else:
                slices.append(slice(None, None, None))
        return self.__getitem__(tuple(slices))

    #--------------------
    # Shared Interface
    #--------------------
    @property
    def tlib(self: TFFTArray) -> TensorLib:
        return self._tlib

    def with_tlib(self: TFFTArray, tlib: TensorLib) -> TFFTArray:
        new_dims = [dim.set_default_tlib(tlib) for dim in self._dims]
        if tlib.numpy_ufuncs.iscomplexobj(self._values):
            new_arr = tlib.array(self._values, dtype=tlib.complex_type)
        else:
            new_arr = tlib.array(self._values, dtype=tlib.real_type)
        new_arr = self.__class__(
            values=new_arr,
            dims=new_dims,
            eager=self.is_eager,
        )
        new_arr._lazy_state = self._lazy_state
        return new_arr

    def set_eager(self: TFFTArray, eager: bool) -> TFFTArray:
        """
            Forces evaluation of current lazy state
        """
        if self.is_eager == eager:
            return self
        return self.__class__(values=self.values, dims=self._dims, eager=eager)

    @property
    def dims_dict(self: TFFTArray) -> Dict[Hashable, FFTDimension]:
        # TODO Ordered Mapping?
        return {dim.name: dim for dim in self._dims}

    @property
    def sizes(self: TFFTArray) -> Dict[Hashable, int]:
        # TODO Ordered Mapping?
        return {dim.name: dim.n for dim in self._dims}

    @property
    def dims(self: TFFTArray) -> Tuple[FFTDimension, ...]:
        return tuple(self._dims)

    @property
    def shape(self: TFFTArray) -> Tuple[int, ...]:
        """..

        Returns
        -------
        Tuple[int, ...]
            Shape of the wavefunction's values.
        """
        return self._values.shape

    @property
    def values(self: TFFTArray) -> ArrayLike:
        """
            Return the values with all lazy state applied.
            Does not mutate self.
            Therefore each call evaluates its lazy state again.
            Use `evaluate_lazy_state` if you want to evaluate it once and reuse it multiple times.
        """
        if self._lazy_state is None:
            # TODO There is no defensive copy here for the Numpy Backend
            return self._values
        return self._tlib.get_values_lazy_factors_applied(
            self._values,
            self._dims,
            self._lazy_state,
        )

    def _set_lazy_state(self: TFFTArray, target_state: LazyState) -> TFFTArray:
        """
            Modifies the values such that the internal lazy state matches the given target state.
            Used for making the input math the output of scan-loops.
        """
        assert self._lazy_state is not None, \
            "Cannot call _set_lazy_state on an eager FFTArray."
        to_apply = get_lazy_state_to_apply(self._lazy_state, target_state)
        # Use the raw _values since we apply only the delta between existing and target.
        new_values = self._tlib.get_values_lazy_factors_applied(
            self._values,
            self._dims,
            to_apply,
        )
        # Relies on immutability
        new_arr = self.__class__(
            values=new_values,
            dims=self.dims,
            eager=self.is_eager,
        )
        new_arr._lazy_state = deepcopy(target_state)
        return new_arr

    def add_phase_factor(
            self: TFFTArray,
            dim_name: Hashable,
            factor_name: str,
            phase_factors: PhaseFactors
        ) -> TFFTArray:
        """
            Add a phase factor lazily or eagerly (depending on the setting).
        """
        # Relies on immutability of values and dims
        # TODO values is only immutable in the jax-backend.
        new_arr = self.__class__(
            values=self._values,
            dims=self._dims,
            eager=self.is_eager,
        )
        if self._lazy_state is None:
            dim_idx = {dim.name: idx for idx, dim in enumerate(self._dims)}[dim_name]
            new_arr._values = new_arr.tlib.apply_phase_factors(
                values=self._values,
                dim_idx=dim_idx,
                factors=phase_factors.values,
            )
        else:
            new_arr._lazy_state = self._lazy_state.add_phase_factor(
                dim=dim_name,
                factor_name=factor_name,
                phase_factors=phase_factors
            )
        return new_arr

    def add_scale(self: TFFTArray, scale: complex) -> TFFTArray:
        """
            Add a scale lazily or eagerly (depending on the setting).
        """
        # Relies on immutability of values and dims
        new_arr = self.__class__(
            values=self._values,
            dims=self._dims,
            eager=self.is_eager,
        )
        if self._lazy_state is None:
            new_arr._values = self.tlib.apply_scale(new_arr.values, scale=scale)
        else:
            new_arr._lazy_state = self._lazy_state.add_scale(scale=scale)
        return new_arr

    @property
    def is_eager(self: TFFTArray) -> bool:
        return self._lazy_state is None

    def evaluate_lazy_state(self: TFFTArray) -> TFFTArray:
        """
            Return the same object from view of the public API.
            But if the `values`-accessor is used multiple times this improives performance.
        """
        if self.is_eager:
            return self
        return self._set_lazy_state(LazyState())

    def transpose(self: TFFTArray, *dims: Hashable) -> TFFTArray:
        """
            Transpose with dimension names.
        """
        new_dim_names = list(dims)
        old_dim_names = [dim.name for dim in self._dims]
        dims_dict = self.dims_dict
        if len(new_dim_names) == 0:
            new_dim_names = copy(old_dim_names)
            new_dim_names.reverse()
        else:
            assert len(new_dim_names) == len(self._dims)

        transposed_values = transpose_array(
            self._values,
            tlib = self.tlib,
            old_dims = old_dim_names,
            new_dims = new_dim_names,
        )
        new_dims = [dims_dict[name] for name in new_dim_names]
        transposed_arr = self.__class__(
            values=transposed_values,
            dims=new_dims,
            eager=self.is_eager,
        )
        transposed_arr._lazy_state = self._lazy_state
        return transposed_arr

    def _set_tlib(self: TFFTArray, tlib: Optional[TensorLib] = None) -> TFFTArray:
        """
            Set tlib if it is not None.
            Collects just a bit of code from all `pos_array` and `freq_array` implementations.
        """
        res = self
        if tlib:
            res = res.with_tlib(tlib)
        return res

    #--------------------
    # Interface to implement
    #--------------------
    @abstractmethod
    def pos_array(
            self: TFFTArray,
            tlib: Optional[TensorLib] = None
        ) -> PosArray:
        ...

    @abstractmethod
    def freq_array(
            self: TFFTArray,
            tlib: Optional[TensorLib] = None
        ) -> FreqArray:
        ...

    @property
    @abstractmethod
    def space(self: TFFTArray) -> Space:
        """
            Enables automatically and easily detecting in which space a generic FFTArray curently is.
        """
        ...

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
    def d_freq(self: TFFTArray) -> float:
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
    def d_pos(self: TFFTArray) -> float:
        """..

        Returns
        -------
        float
            The product of the `d_pos` of all active dimensions.
        """
        return self._tlib.reduce_multiply(
            self._tlib.array([fft_dim.d_pos for fft_dim in self._dims])
        )

    def _check_consistency(self: TFFTArray) -> None:
        """
            Check some invariants of FFTArray.
        """
        assert len(self._dims) == len(self._values.shape)
        dim_names: Set[Hashable] = set()
        for n, dim in zip(self._values.shape, self._dims):
            assert dim.n == n, \
                "Passed in inconsistent n from FFTDimension and values."
            assert dim.name not in dim_names, \
                f"Passed in FFTDimension of name {dim.name} twice!"
            dim_names.add(dim.name)

        if self._lazy_state is not None:
            for dim, phase_factors in zip(self._dims, self._lazy_state._phases_per_dim):
                assert isinstance(phase_factors, dict)
                assert dim.name in self._lazy_state._phases_per_dim


class PosArray(FFTArray):
    def pos_array(
            self: PosArray,
            tlib: Optional[TensorLib] = None,
        ) -> PosArray:
        return self._set_tlib(tlib)

    def freq_array(
            self: PosArray,
            tlib: Optional[TensorLib] = None,
        ) -> FreqArray:
        res_pos = self
        for dim in self._dims:
            res_pos = res_pos.add_phase_factor(
                dim.name,
                "fft_shift_pos",
                PhaseFactors({1: -2*np.pi*dim.freq_min*dim.d_pos}),
            )

        res_freq = FreqArray(
            dims=self.dims,
            values=self._tlib.fftn(res_pos.values),
            eager=self.is_eager
        )

        for dim in self._dims:
            res_freq = res_freq.add_phase_factor(
                dim.name,
                "fft_phase_freq",
                PhaseFactors({
                    0: -2*np.pi*dim.pos_min*dim.freq_min,
                    1: -2*np.pi*dim.pos_min*dim.d_freq,
                }),
            )

        res_freq = res_freq.add_scale(_freq_scale_factor(self._dims))

        return res_freq._set_tlib(tlib)

    @property
    def space(self: PosArray) -> Literal["pos"]:
        return "pos"


class FreqArray(FFTArray):
    def pos_array(
            self: FreqArray,
            tlib: Optional[TensorLib] = None,
        ) -> PosArray:
        res_freq = self
        for dim in self._dims:
            res_freq = res_freq.add_phase_factor(
                dim.name,
                "fft_phase_freq",
                PhaseFactors({
                    0: 2*np.pi*dim.pos_min*dim.freq_min,
                    1: 2*np.pi*dim.pos_min*dim.d_freq,
                }),
            )

        # The scale is here intentionally before the call to ifftn in order to symmetrically cancel
        # with the forward direction.
        res_freq = res_freq.add_scale(1./_freq_scale_factor(self._dims))

        res_pos = PosArray(
            dims=self.dims,
            # TODO: Generic backend selection
            values=self._tlib.ifftn(res_freq.values),
            eager=self.is_eager
        )
        for dim in self._dims:
            res_pos = res_pos.add_phase_factor(
                dim.name,
                "fft_shift_pos",
                PhaseFactors({1: 2*np.pi*dim.freq_min*dim.d_pos}),
            )

        return res_pos._set_tlib(tlib)

    def freq_array(
            self: FreqArray,
            tlib: Optional[TensorLib] = None,
        ) -> FreqArray:
        return self._set_tlib(tlib)

    @property
    def space(self: FreqArray) -> Literal["freq"]:
        return "freq"


def _freq_scale_factor(dims: Iterable[FFTDimension]) -> float:
    """
        Returns the product of $\delta x = 1/(\delta f * N)$ for all dimensions.
    """
    scale_factor = 1.
    for dim in dims:
        scale_factor *= (1./(dim.d_freq * dim.n))
    return scale_factor


def _pack_values(
            values: ArrayLike,
            space: Space,
            dims: List[FFTDimension],
            lazy_state: Optional[LazyState],
        ) -> FFTArray:
    """
        Finish up a value after a ufunc.
        Internally it is more parametrized and this puts it back together.
    """
    tlib = _get_tensor_lib(dims)
    assert tlib.has_precision(values, tlib.precision)
    # eager does not matter here because it is overwritten with a concrete lazy_state
    if space == "pos":
        arr: FFTArray = PosArray(values, dims=dims, eager=True)
    else:
        assert space == "freq"
        arr = FreqArray(values, dims=dims, eager=True)
    arr._lazy_state = lazy_state
    return arr


# Implementing NEP 13 https://numpy.org/neps/nep-0013-ufunc-overrides.html
# See also https://numpy.org/doc/stable/user/basics.dispatch.html
def _array_ufunc(self, ufunc, method, inputs, kwargs):
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
    if ufunc == np.abs or ufunc == np.multiply :
        unpacked_inputs = _unpack_fft_arrays(inputs, True)
    else:
        # Apply all lazy state because we have no shortcuts.
        unpacked_inputs = _unpack_fft_arrays(inputs, False)

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
    if ufunc == np.abs and unpacked_inputs.lazy_state is not None:
        values = tensor_lib_ufunc(*unpacked_inputs.values, **kwargs)
        return _pack_values(
            values,
            unpacked_inputs.space,
            unpacked_inputs.dims,
            # For abs we need to drop the phases without ever applying them
            # since they would have evaluated to one.
            LazyState(scale=unpacked_inputs.lazy_state._scale),
        )
    if ufunc == np.multiply and unpacked_inputs.lazy_state is not None:
        # Would be cool but creates a long weird chain of problems with jax tracing.
        # It is also in general problematic because under tracing it moves a computation from run- to trace-time.
        # We would definitely need a safeguard here that we do not make `scale` an abstract tracer.
        # if any(unpacked_inputs.is_scalar):
        #     # Can currently only be one index since there are only two operands
        #     # and one of them has to be a FFTArray.
        #     for is_scalar, inp in zip(unpacked_inputs.is_scalar, unpacked_inputs.values):
        #         if is_scalar:
        #             unpacked_inputs.lazy_state.scale *= inp
        #         else:
        #             values = inp
        # else:
        values = tensor_lib_ufunc(*unpacked_inputs.values, **kwargs)
        return _pack_values(
            values,
            unpacked_inputs.space,
            unpacked_inputs.dims,
            unpacked_inputs.lazy_state,
        )
    # conj?
    assert (
        unpacked_inputs.lazy_state is None
        or unpacked_inputs.lazy_state == LazyState()
    )
    values = tensor_lib_ufunc(*unpacked_inputs.values, **kwargs)
    return _pack_values(
        values,
        unpacked_inputs.space,
        unpacked_inputs.dims,
        unpacked_inputs.lazy_state,
    )


@dataclass
class UnpackedValues:

    # FFTDimensions in the order in which they appear in each non-scalar value.
    dims: List[FFTDimension]
    # Values without any dimensions, etc.
    values: List[Union[Number, Any]]
    # Flag for each value in values whether it is a scalar.
    # Currently unused
    is_scalar: List[bool]
    # Space in whcih all values were
    space: Space
    # LazyState not yet applied to the values in total.
    # Makes only sense when either only having a single FFTArray or the operation between them is multiply.
    lazy_state: Optional[LazyState]
    # Shared tensor-lib between all values.
    tlib: TensorLib


def _unpack_fft_arrays(
        values: List[Union[Number, FFTArray, Any]],
        keep_lazy: bool,
    ) -> UnpackedValues:
    """
        This handles all "alignment" of input values.
        Align dimensions, unify them, unpack all operands to a simple list of values.
        May collect all lazy_state or just apply it before storing the values as plain values.
    """
    dims: Dict[Hashable, FFTDimension] = {}
    arrays_to_align: List[Tuple[List[Hashable], Any]] = []
    array_indices = []
    unpacked_values: List[Optional[Union[Number, Any]]] = []
    is_scalar: List[bool] = []
    space: Optional[Space] = None
    lazy_state = LazyState()
    is_eager: UniformValue[bool] = UniformValue()

    for idx, x in enumerate(values):
        if isinstance(x, Number):
            unpacked_values.append(x)
            is_scalar.append(True)
        elif hasattr(x, "shape") and not isinstance(x, FFTArray):
            if x.shape == ():
                unpacked_values.append(x)
                is_scalar.append(True)
            else:
                raise ValueError(
                    "Cannot multiply coordinate-less arrays with a FFTArray."
                )
        else:
            unpacked_values.append(None)
            is_scalar.append(False)
            array_indices.append(idx)
            assert isinstance(x, FFTArray)
            if x._lazy_state is None:
                is_eager.val = True
            else:
                is_eager.val = False
            for fft_dim in x._dims:
                if fft_dim.name in dims:
                    if fft_dim != dims[fft_dim.name]:
                        raise ValueError(
                            "Tried to call ufunc on two FFTArrays with " +
                            "different coordinates in dimension " +
                            f"{fft_dim.name}."
                        )
                else:
                    dims[fft_dim.name] = fft_dim

            if space is None:
                space = x.space
            else:
                if space != x.space:
                    raise ValueError(
                        "Tried to call ufunc on two FFTArrays in different " +
                        "spaces. One of them has to be explicitly converted " +
                        "into the other space to ensure the correct space."
                    )

            elem_dim_names = [fft_dim.name for fft_dim in x._dims]
            if keep_lazy and x._lazy_state is not None:
                lazy_state = lazy_state + x._lazy_state
                raw_arr = x._values
            else:
                raw_arr = x.values

            arrays_to_align.append((elem_dim_names, raw_arr))

    tlib = _get_tensor_lib(dims.values())

    # Broadcasting
    dim_names, aligned_arrs = align_named_arrays(arrays_to_align, tlib=tlib)
    for idx, arr in zip(array_indices, aligned_arrs):
        unpacked_values[idx] = arr

    dims_list = [dims[dim_name] for dim_name in dim_names]

    for i in range(len(unpacked_values)):
        unpacked_values[i] = tlib.as_array(unpacked_values[i])

    assert space is not None
    for value in unpacked_values:
        assert not value is None

    ret_lazy = None if is_eager.val else lazy_state

    return UnpackedValues(
        dims = dims_list,
        values = unpacked_values, # type: ignore
        is_scalar = is_scalar,
        space = space,
        lazy_state = ret_lazy,
        tlib = tlib,
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
    _d_freq: float
    _n: int
    _name: Hashable
    _default_tlib: TensorLib
    _default_eager: bool

    def __init__(
            self,
            name: str,
            *,
            n: Union[int, Literal["power_of_two", "even"]] = "power_of_two",
            d_pos: Optional[float] = None,
            d_freq: Optional[float] = None,
            pos_min: Optional[float] = None,
            pos_max: Optional[float] = None,
            pos_middle: Optional[float] = None,
            pos_extent: Optional[float] = None,
            freq_min: Optional[float] = None,
            freq_max: Optional[float] = None,
            freq_extent: Optional[float] = None,
            freq_middle: Optional[float] = None,
            loose_params: Optional[Union[str, List[str]]] = None,
            default_tlib: TensorLib = NumpyTensorLib(),
            default_eager: bool = False,
        ):
        self._name = name
        self._default_tlib = default_tlib
        self._default_eager = default_eager

        if isinstance(loose_params, str):
            loose_params = [loose_params]
        elif loose_params is None:
            loose_params = []

        params = _z3_constraint_solver(
            constraints=dict(
                n = n,
                d_pos = d_pos,
                d_freq = d_freq,
                pos_min = pos_min,
                pos_max = pos_max,
                pos_middle = pos_middle,
                pos_extent = pos_extent,
                freq_min = freq_min,
                freq_max = freq_max,
                freq_extent = freq_extent,
                freq_middle = freq_middle
            ),
            loose_params=loose_params,
            make_suggestions=True
        )

        self._pos_min = params["pos_min"]
        self._freq_min = params["freq_min"]
        self._d_pos = params["d_pos"]
        self._d_freq = params["d_freq"]
        self._n = int(params["n"])

    def __repr__(self: FFTDimension) -> str:
        arg_str = ", ".join([f"{name[1:]}={repr(value)}" for name, value in self.__dict__.items() if name != "_d_freq"])
        return f"FFTDimension({arg_str})"

    def __str__(self: FFTDimension) -> str:
        evaluated = 'eagerly' if self._default_eager else 'lazily'
        properties = (
            f"\t Number of grid points n: {self.n} \n\t " +
            f"Position space: min={self.pos_min}, middle={self.pos_middle}, " +
            f"max={self.pos_max}, extent={self.pos_extent}, d_pos={self.d_pos} \n\t " +
            f"Frequency space: min={self.freq_min}, middle={self.freq_middle}, " +
            f"max={self.freq_max}, extent={self.freq_extent}, d_freq={self.d_freq}"
        )
        return (
            f"FFTDimension with name '{self.name}' on backend " +
            f"'{self.default_tlib}' evaluated {evaluated} with the " +
            f"following properties:\n{properties}"
        )

    def set_default_tlib(self, tlib: TensorLib) -> FFTDimension:
        dim = copy(self)
        dim._default_tlib = tlib
        return dim

    @property
    def default_tlib(self) -> TensorLib:
        return self._default_tlib

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

    def pos_array(
            self: FFTDimension,
            tlib: Optional[TensorLib] = None
        ) -> PosArray:
        """..

        Returns
        -------
        jax.numpy.ndarray
            The position grid coordinates.
        """
        dim = self
        if tlib is not None:
            dim = dim.set_default_tlib(tlib)

        indices = dim.default_tlib.numpy_ufuncs.arange(
            0,
            dim.n,
            dtype = dim.default_tlib.real_type,
        )

        return PosArray(
            values = indices * dim.d_pos + dim.pos_min,
            dims = [dim],
            eager = self._default_eager,
        )

    def _dim_from_slice(self, range: slice, space: Space) -> FFTDimension:
        """
            Get a new FFTDimension for a interval selection in a given space.
            Does not support steps!=1.
        """
        if not(range.step is None or range.step == 1):
            raise ValueError(
                "Substepping is not supported because it is not well defined " +
                "how to cut frequency space with an arbitrary offset."
            )
        start = range.start
        if range.stop is None:
            stop = start+1
        else:
            stop = range.stop
        n = stop - start
        assert n >= 1
        return self._dim_from_start_and_n(start=start, n=n, space=space)

    def _dim_from_start_and_n(
            self,
            start: int,
            n: int,
            space: Space,
        ) -> FFTDimension:
        new = self.__class__.__new__(self.__class__)
        new._name = self.name
        new._default_tlib = self._default_tlib
        new._default_eager = self._default_eager
        new._n = n

        # d_freq * d_pos * n == 1,
        if space == "pos":
            new._pos_min = self.pos_min + start*self.d_pos
            new._freq_min = self.freq_min
            new._d_pos = self.d_pos
            new._d_freq = 1./(self.d_pos*n)
        elif space == "freq":
            new._pos_min = self.pos_min
            new._freq_min = self.freq_min + start*self.d_freq
            new._d_pos = 1./(self.d_freq*n)
            new._d_freq = self.d_freq
        else:
            assert False, "Unreachable"
        return new


    def _index_from_coord(
            self,
            x,
            method: Optional[Literal["nearest", "min", "max"]],
            space: Space,
        ):
        """
            Compute index from given coordinate `x`.
        """
        if isinstance(x, tuple):
            sel_min, sel_max = x
            idx_min = self._index_from_coord(sel_min, method="min", space=space)
            idx_max = self._index_from_coord(sel_max, method="max", space=space)
            # The max is an open intervel, therefore add one.
            return slice(idx_min, idx_max+1)

        if space == "pos":
            raw_idx = (x - self.pos_min) / self.d_pos
        else:
            raw_idx = (x - self.freq_min) / self.d_freq


        if method is None:
            # TODO This is not jittable.
            # Either fix it or document it.
            # Would probably need a tlib if...
            if self.default_tlib.numpy_ufuncs.round(raw_idx) != raw_idx:
                raise KeyError(
                    f"No exact index found for {x} in {space}-space of dim " +
                    f'"{self.name}". Try the keyword argument ' +
                    'method="nearest".'
                )
            idx = raw_idx
        elif method in ["nearest", "min", "max"]:
            # Clamp index into valid range
            raw_idx = self.default_tlib.numpy_ufuncs.max(self.default_tlib.array([0, raw_idx]))
            raw_idx = self.default_tlib.numpy_ufuncs.min(self.default_tlib.array([self.n, raw_idx]))
            if  method == "nearest":
                # The combination of floor and +0.5 prevents the "ties to even" rounding of floating point numbers.
                # We only need one branch since our indices are always positive.
                idx = self.default_tlib.numpy_ufuncs.floor(raw_idx + 0.5)
            elif method == "min":
                idx = self.default_tlib.numpy_ufuncs.ceil(raw_idx)
            elif method == "max":
                idx = self.default_tlib.numpy_ufuncs.floor(raw_idx)
        else:
            raise ValueError("Specified unsupported look-up method.")

        return self.default_tlib.array(idx, dtype=int)

    # ---------------------------- Frequency Space --------------------------- #

    @property
    def d_freq(self: FFTDimension) -> float:
        """..

        Returns
        -------
        float
            The distance between frequency grid points.
        """
        return self._d_freq

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

    def freq_array(
            self: FFTDimension,
            tlib: Optional[TensorLib] = None,
        ) -> FreqArray:
        """..

        Returns
        -------
        jax.numpy.ndarray
            The frequency grid coordinates.
        """
        dim = self
        if tlib is not None:
            dim = dim.set_default_tlib(tlib)

        indices = dim.default_tlib.numpy_ufuncs.arange(
            0,
            dim.n,
            dtype = dim.default_tlib.real_type,
        )

        return FreqArray(
            values = indices * dim.d_freq + dim.freq_min,
            dims = [dim],
            eager = self._default_eager,
        )
