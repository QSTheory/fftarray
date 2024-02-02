from __future__ import annotations
from typing import (
    Optional, Union, List, Any, Tuple, Dict, Hashable,
    Literal, TypeVar, Iterable, Set, Generic
)
from abc import ABCMeta
from copy import copy
from numbers import Number
from dataclasses import dataclass

import numpy as np
import numpy.lib.mixins

from .named_array import align_named_arrays, get_axes_transpose
from .fft_constraint_solver import _z3_constraint_solver
from .backends.tensor_lib import TensorLib
from .backends.np_backend import NumpyTensorLib
from .helpers import reduce_equal, UniformValue

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
        `wf.loc` allows indexing by dim index but by coordinate position.
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
                slices.append(dim._index_from_coord(dim_item, method=None, space=space))
        return self._arr.__getitem__(tuple(slices))

def _norm_param(val: Union[T, Iterable[T]], types) -> Tuple[T, ...]:
    if isinstance(val, types):
        return tuple([val])

    # TODO: Can we make this type check work?
    return tuple(val) # type: ignore

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
    # Contains the array backend, precision and device to be used for operations.
    _tlib: TensorLib

    def __init__(
            self,
            values,
            dims: Iterable[FFTDimension],
            space: Union[Space, Iterable[Space]],
            eager: Union[bool, Iterable[bool]],
            factors_applied: Union[bool, Iterable[bool]],
        ):
        """
            This constructor is not meant for normal usage.
            Construct new values via the `fft_array()` function of FFTDimension.
        """
        self._dims = tuple(dims)
        self._values = values
        self._space = _norm_param(space, str)
        self._eager = _norm_param(eager, bool)
        self._factors_applied = _norm_param(factors_applied, bool)
        self._tlib = _get_tensor_lib(self._dims)
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

    def __getitem__(self, item):
        new_dims = []

        if isinstance(item, slice):
            item = [item]
        for index, dimension, space in zip(item, self._dims, self._space):
            if not isinstance(index, slice):
                new_dim = dimension._dim_from_start_and_n(
                    start=index,
                    n=1,
                    space=space,
                )
                index = slice(index, index+1, None)
            elif index == slice(None, None, None):
                # No selection, just keep the old dim.
                new_dim = dimension
            else:
                new_dim = dimension._dim_from_slice(index, space)

            new_dims.append(new_dim)

        selected_values = self.values.__getitem__(item)
        # Dimensions with the length 1 are dropped in numpy indexing.
        # We decided against this and keeping even dimensions of length 1.
        # So we have to reintroduce those dropped dimensions via reshape.
        selected_values = selected_values.reshape(tuple(dim.n for dim in new_dims))

        return FFTArray(
            values=selected_values,
            dims=new_dims,
            space=self._space,
            eager=self._eager,
            factors_applied=self._factors_applied,
        )

    @property
    def loc(self):
        return LocFFTArrayIndexer(self)

    def isel(self, **kwargs):
        slices = []
        for dim in self.dims:
            if dim.name in kwargs:
                slices.append(kwargs[dim.name])
            else:
                slices.append(slice(None, None, None))
        return self.__getitem__(tuple(slices))

    def sel(self, method: Optional[Literal["nearest"]] = None, **kwargs):
        """
            Supports in addition to its xarray-counterpart tuples for ranges.
        """
        slices = []
        for dim, space in zip(self.dims, self._space):
            if dim.name in kwargs:
                slices.append(
                    # mypy error: kwargs is of type "Dict[str, Any]" but
                    # dim.name is of type "Hashable". However, the if condition
                    # already makes sure that dim.name is a key of kwargs (so
                    # also of type "str")
                    dim._index_from_coord(
                        kwargs[dim.name], # type: ignore
                        method=method,
                        space=space,
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
            target_factors_applied=tuple([True]*len(self._dims)),
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

        if space is None:
            space_norm = self._space
        else:
            space_norm = _norm_param(space, str)

        if eager is None:
            eager_norm = self._eager
        else:
            eager_norm = _norm_param(eager, bool)
            dims = tuple([dim.set_default_eager(eager) for dim, eager in zip(dims, eager_norm)])



        if tlib is None:
            tlib_norm = self._tlib
        else:
            tlib_norm = tlib
            dims = tuple([dim.set_default_tlib(tlib) for dim in dims])
            if tlib.numpy_ufuncs.iscomplexobj(self._values):
                values = tlib.array(values, dtype=tlib.complex_type)
            else:
                values = tlib.array(values, dtype=tlib.real_type)


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
            factors_norm = _norm_param(factors_applied, bool)

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
            factors_applied=[self._factors_applied[idx] for idx in axes_transpose]
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
        dim_names: Set[Hashable] = set()
        for n, dim in zip(self._values.shape, self._dims):
            assert dim.n == n, \
                "Passed in inconsistent n from FFTDimension and values."
            assert dim.name not in dim_names, \
                f"Passed in FFTDimension of name {dim.name} twice!"
            dim_names.add(dim.name)

        # if self._lazy_state is not None:
        #     for dim, phase_factors in zip(self._dims, self._lazy_state._phases_per_dim):
        #         assert isinstance(phase_factors, dict)
        #         assert dim.name in self._lazy_state._phases_per_dim


# class PosArray(FFTArray):
#     def pos_array(self, tlib: Optional[TensorLib] = None) -> PosArray:
#         return self._set_tlib(tlib)

#     def freq_array(self, tlib: Optional[TensorLib] = None) -> FreqArray:
#         res_pos = self
#         for dim in self._dims:
#             res_pos = res_pos.add_phase_factor(
#                 dim.name,
#                 "fft_shift_pos",
#                 PhaseFactors({1: -2*np.pi*dim.freq_min*dim.d_pos}),
#             )

#         res_freq = FreqArray(
#             dims=self.dims,
#             values=self._tlib.fftn(res_pos.values),
#             eager=self.is_eager
#         )

#         for dim in self._dims:
#             res_freq = res_freq.add_phase_factor(
#                 dim.name,
#                 "fft_phase_freq",
#                 PhaseFactors({
#                     0: -2*np.pi*dim.pos_min*dim.freq_min,
#                     1: -2*np.pi*dim.pos_min*dim.d_freq,
#                 }),
#             )

#         res_freq = res_freq.add_scale(_freq_scale_factor(self._dims))

#         return res_freq._set_tlib(tlib)

#     @property
#     def space(self) -> Literal["pos"]:
#         return "pos"


# class FreqArray(FFTArray):
#     def pos_array(self, tlib: Optional[TensorLib] = None) -> PosArray:
#         res_freq = self
#         for dim in self._dims:
#             res_freq = res_freq.add_phase_factor(
#                 dim.name,
#                 "fft_phase_freq",
#                 PhaseFactors({
#                     0: 2*np.pi*dim.pos_min*dim.freq_min,
#                     1: 2*np.pi*dim.pos_min*dim.d_freq,
#                 }),
#             )

#         # The scale is here intentionally before the call to ifftn in order to symmetrically cancel
#         # with the forward direction.
#         res_freq = res_freq.add_scale(1./_freq_scale_factor(self._dims))

#         res_pos = PosArray(
#             dims=self.dims,
#             # TODO: Generic backend selection
#             values=self._tlib.ifftn(res_freq.values),
#             eager=self.is_eager
#         )
#         for dim in self._dims:
#             res_pos = res_pos.add_phase_factor(
#                 dim.name,
#                 "fft_shift_pos",
#                 PhaseFactors({1: 2*np.pi*dim.freq_min*dim.d_pos}),
#             )

#         return res_pos._set_tlib(tlib)

#     def freq_array(self, tlib: Optional[TensorLib] = None) -> FreqArray:
#         return self._set_tlib(tlib)

#     @property
#     def space(self) -> Literal["freq"]:
#         return "freq"


# def _freq_scale_factor(dims: Iterable[FFTDimension]) -> float:
#     """
#         Returns the product of $\delta x = 1/(\delta f * N)$ for all dimensions.
#     """
#     scale_factor = 1.
#     for dim in dims:
#         scale_factor *= (1./(dim.d_freq * dim.n))
#     return scale_factor



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

    def set_default_eager(self, eager: bool) -> FFTDimension:
        dim = copy(self)
        dim._default_eager = eager
        return dim

    @property
    def default_eager(self) -> bool:
        return self._default_eager

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
        ) -> FFTArray:
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

        return FFTArray(
            values = indices * dim.d_pos + dim.pos_min,
            dims = [dim],
            eager = self._default_eager,
            factors_applied=True,
            space="pos",
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
        ) -> FFTArray:
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

        return FFTArray(
            values = indices * dim.d_freq + dim.freq_min,
            dims = [dim],
            eager = self._default_eager,
            factors_applied=True,
            space="freq",
        )
