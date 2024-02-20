from __future__ import annotations
from typing import (
    Optional, Union, List, Any, Tuple, Dict, Hashable,
    Literal, TypeVar, Iterable, Set, Generic, get_args
)
from abc import ABCMeta
from copy import copy
from numbers import Number
from dataclasses import dataclass

import numpy as np
import numpy.lib.mixins

from .named_array import align_named_arrays, get_axes_transpose
from .backends.tensor_lib import TensorLib
from .backends.np_backend import NumpyTensorLib
from .helpers import UniformValue, format_bytes, format_n, truncate_str

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
        for dim, dim_item, space in zip(self._arr.dims, item, self._arr._spaces):
            if isinstance(dim_item, slice):
                slices.append(dim_item)
            else:
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

def _fft_dim_table(
        dim: FFTDimension,
        include_header=True,
        include_dim_name=False,
        dim_index: Optional[int] = None,
    ) -> str:
    """Constructs a table for FFTDimension.__str__ and FFTArrar.__str__
    containting the grid parameters for each space.
    """
    str_out = ""
    headers = ["space", "d", "min", "middle", "max", "extent"]
    if include_dim_name:
        headers.insert(0, "dimension")
    if include_header:
        if dim_index is not None:
            # handled separately to give it a smaller width
            str_out += "| # "
        for header in headers:
            # give space smaller width to stay below 80 characters per line
            str_out += f"|{header:^7}" if header == "space" else f"|{header:^10}"
        str_out += "|\n" + int(dim_index is not None)*"+---"
        for header in headers:
            str_out += "+" + (7*"-" if header == "space" else 10*"-")
        str_out += "+\n"
    dim_prop_headers = headers[int(include_dim_name)+1:]
    for k, space in enumerate(get_args(Space)):
        if dim_index is not None:
            str_out += f"|{dim_index:^3}" if k==0 else f"|{'':^3}"
        if include_dim_name:
            dim_name = str(dim.name)
            if len(dim_name) > 10:
                if k == 0:
                    str_out += f"|{dim_name[:10]}"
                else:
                    str_out += f"|{truncate_str(dim_name[10:], 10)}"
            else:
                str_out += f"|{dim_name:^10}" if k==0 else f"|{'':^10}"
        str_out += f"|{space:^7}|"
        for header in dim_prop_headers:
            attr = f"d_{space}" if header == "d" else f"{space}_{header}"
            nmbr = getattr(dim, attr)
            frmt_nmbr = f"{nmbr:.2e}" if abs(nmbr)>1e3 or abs(nmbr)<1e-2 else f"{nmbr:.2f}"
            str_out += f"{frmt_nmbr:^10}|"
        str_out += "\n"
    return str_out[:-1]

def _fft_array_props_table(fftarr: FFTArray) -> str:
    """Constructs a table for FFTArray.__str__ containing the FFTArray
    properties (space, n, eager, factors_applied) per dimension
    """
    str_out = "| # "
    headers = ["dimension", "space", "n", "eager", "factors_applied"]
    for header in headers:
        # give space smaller width to stay below 80 characters per line
        str_out += f"|{header:^7}" if header == "space" else f"|{header:^10}"
    str_out += "|\n+---"
    for header in headers:
        str_out += "+" + (10 + 5*int(header=='factors_applied') - 3*int(header=="space"))*"-"
    str_out += "+\n"
    for i, dim in enumerate(fftarr.dims):
        str_out += f"|{i:^3}"
        str_out += f"|{truncate_str(str(dim.name), 10):^10}"
        str_out += f"|{(fftarr.space[i]):^7}"
        str_out += f"|{format_n(dim.n):^10}"
        str_out += f"|{repr(fftarr.eager[i]):^10}"
        str_out += f"|{repr(fftarr._factors_applied[i]):^15}"
        str_out += "|\n"
    return str_out[:-1]

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
    _spaces: Tuple[Space, ...]
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
        self._spaces = _norm_param(space, n_dims, str)
        self._eager = _norm_param(eager, n_dims, bool)
        self._factors_applied = _norm_param(factors_applied, n_dims, bool)
        self._tlib = tlib
        self._check_consistency()

    def __repr__(self: FFTArray) -> str:
        arg_str = ", ".join(
            [f"{name[1:] if name != '_spaces' else 'space'}={repr(value)}"
                for name, value in self.__dict__.items()]
        )
        return f"FFTArray({arg_str})"

    def __str__(self: FFTArray) -> str:
        bytes_str = format_bytes(self._values.nbytes)
        title = f" FFTArray ({bytes_str}) "
        str_out = f"{title:-^80}\n"
        str_out += f"TensorLib: {self.tlib}\n"
        str_out += "Dimensions:\n"
        for i, dim in enumerate(self.dims):
            str_out += f" # {i}: {repr(dim.name)}\n"
        str_out += "\n" + _fft_array_props_table(self) + "\n\n"
        for i, dim in enumerate(self.dims):
            str_out += _fft_dim_table(dim, i==0, True, i) + "\n"
        str_out += f"\nvalues:\n{self.values}"
        return str_out

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

    def __getitem__(self, item) -> FFTArray:
        new_dims = []

        if isinstance(item, slice):
            item = [item]
        for index, dimension, space in zip(item, self._dims, self._spaces):
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
            space=self._spaces,
            eager=self._eager,
            factors_applied=self._factors_applied,
            tlib=self.tlib,
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
        for dim, space in zip(self.dims, self._spaces):
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
            spaces=self._spaces,
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
            space_norm = self._spaces
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
            if tlib_norm.numpy_ufuncs.iscomplexobj(values):
                values = tlib_norm.array(values, dtype=tlib_norm.complex_type)
            elif tlib_norm.numpy_ufuncs.issubdtype(values.dtype, tlib_norm.numpy_ufuncs.floating):
                values = tlib_norm.array(values, dtype=tlib_norm.real_type)
            else:
                values = tlib_norm.array(values)


        needs_fft = [old != new for old, new in zip(self._spaces, space_norm)]
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
                spaces=self._spaces,
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
            spaces=space_norm,
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
            space=[self._spaces[idx] for idx in axes_transpose],
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
        if not isinstance(self._values, self._tlib.array_type):
            raise ValueError(
                f"Passed in values of type '{type(self._values)}' "
                + f"which is not the array type '{self._tlib.array_type}'"
                + f" of the tensor-lib '{self._tlib}'."
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

    # Split out the single-element case because we can then skip the whole unpacking.
    if len(inputs) == 1:
        inp = inputs[0]
        assert isinstance(inp, FFTArray)
        return _single_element_ufunc(ufunc=ufunc, inp=inp, kwargs=kwargs)

    unp_inp: UnpackedValues = _unpack_fft_arrays(inputs)

    # Returning NotImplemented gives other operands a chance to see if they support interacting with us.
    # Not really necessary here.
    if not all(isinstance(x, (Number, FFTArray)) or hasattr(x, "__array__") for x in unp_inp.values):
        return NotImplemented

    # Look up the actual ufunc
    try:
        tensor_lib_ufunc = getattr(unp_inp.tlib.numpy_ufuncs, ufunc.__name__)
    except:
        return NotImplemented

    factor_transforms: List[List[Literal[-1, 1, None]]] = [[None]*len(unp_inp.dims) for _ in range(2)]
    final_factors_applied: List[bool] = []

    if (ufunc == np.multiply or ufunc == np.divide) and len(inputs) == 2:
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

    elif (ufunc == np.add or ufunc == np.subtract) and len(inputs) == 2:
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

    else:
        # Define factor_transforms such that factors are applied for
        # both operators because there is no special case applicable
        for op_idx in [0,1]:
            if isinstance(inputs[op_idx], FFTArray):
                res = unp_inp.tlib.get_transform_signs(
                    input_factors_applied=[unp_inp.factors_applied[dim_idx][op_idx] for dim_idx in range(len(unp_inp.dims))],
                    target_factors_applied=[True]*len(unp_inp.dims),
                )
                if res is not None:
                    factor_transforms[op_idx] = res

        final_factors_applied = [True]*len(unp_inp.dims)

    # Apply above defined scale and phase factors depending on the specific case
    for op_idx, signs_op in zip([0,1], factor_transforms):
        if isinstance(inputs[op_idx], FFTArray):
            unp_inp.values[op_idx] = unp_inp.tlib.apply_scale_phases(
                values=unp_inp.values[op_idx],
                dims=unp_inp.dims,
                signs=signs_op,
                spaces=unp_inp.space,
            )

    values = tensor_lib_ufunc(*unp_inp.values, **kwargs)
    return FFTArray(
        values=values,
        space=unp_inp.space,
        dims=unp_inp.dims,
        eager=unp_inp.eager,
        factors_applied=final_factors_applied,
        tlib=unp_inp.tlib,
    )

def _single_element_ufunc(ufunc, inp: FFTArray, kwargs):
    try:
        tensor_lib_ufunc = getattr(inp.tlib.numpy_ufuncs, ufunc.__name__)
    except:
        return NotImplemented

    if ufunc == np.abs:
        # For abs the final result does not change if we apply the phases
        # to the values so we can simply ignore the phases.
        values = tensor_lib_ufunc(inp._values, **kwargs)
        # The scale can be applied after abs which is more efficient in the case of a complex input
        signs: List[Literal[-1, 1, None]] | None = inp.tlib.get_transform_signs(
            # Can use input because with a single value no broadcasting happened.
            input_factors_applied=inp._factors_applied,
            target_factors_applied=[True]*len(inp._factors_applied),
        )
        if signs is not None:
            values = inp.tlib.apply_scale(
                values=values,
                dims=inp.dims,
                signs=signs,
                spaces=inp.space,
            )

        return FFTArray(
            values=values,
            space=inp.space,
            dims=inp.dims,
            eager=inp.eager,
            factors_applied=True,
            tlib=inp.tlib,
        )

    # Fallback if no special case applies
    values = tensor_lib_ufunc(inp.values, **kwargs)
    return FFTArray(
        values=values,
        space=inp.space,
        dims=inp.dims,
        eager=inp.eager,
        factors_applied=True,
        tlib=inp.tlib,
    )

@dataclass
class UnpackedValues:
    # FFTDimensions in the order in which they appear in each non-scalar value.
    dims: Tuple[FFTDimension, ...]
    # Values without any dimensions, etc.
    values: List[Union[Number, Any]]
    # Shared tensor-lib between all values.
    tlib: TensorLib
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

            tlib.set(x.tlib)
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
    dim_names, aligned_arrs = align_named_arrays(arrays_to_align, tlib=tlib.get())
    for op_idx, arr in zip(array_indices, aligned_arrs):
        unpacked_values[op_idx] = arr

    dims_list = [dims[dim_name].dim.get() for dim_name in dim_names]
    space_list = [dims[dim_name].space.get() for dim_name in dim_names]
    eager_list = [dims[dim_name].eager.get() for dim_name in dim_names]
    factors_applied = [dims[dim_name].factors_applied for dim_name in dim_names]
    # TODO: Why is this necessary?
    # unpacked_values = [tlib.get().as_array(x) for x in unpacked_values]

    for value in unpacked_values:
        assert value is not None

    return UnpackedValues(
        dims = tuple(dims_list),
        values = unpacked_values, # type: ignore
        space = space_list,
        factors_applied=factors_applied,
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
        arg_str = ", ".join(
            [f"{name[1:]}={repr(value)}"
                for name, value in self.__dict__.items()]
        )
        return f"FFTDimension({arg_str})"

    def __str__(self: FFTDimension) -> str:
        n_str = format_n(self.n)
        str_out = f"FFTDimension: name={repr(self.name)}, n={n_str}\n"
        str_out += _fft_dim_table(self)
        return str_out

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
        new._n = n

        # d_freq * d_pos * n == 1,
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
            x,
            method: Optional[Literal["nearest", "min", "max"]],
            space: Space,
            tlib: TensorLib,
        ):
        """
            Compute index from given coordinate `x`.
        """
        if isinstance(x, tuple):
            sel_min, sel_max = x
            idx_min = self._index_from_coord(sel_min, method="min", space=space, tlib=tlib)
            idx_max = self._index_from_coord(sel_max, method="max", space=space, tlib=tlib)
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
            if tlib.numpy_ufuncs.round(raw_idx) != raw_idx:
                raise KeyError(
                    f"No exact index found for {x} in {space}-space of dim " +
                    f'"{self.name}". Try the keyword argument ' +
                    'method="nearest".'
                )
            idx = raw_idx
        elif method in ["nearest", "min", "max"]:
            # Clamp index into valid range
            raw_idx = tlib.numpy_ufuncs.max(tlib.array([0, raw_idx]))
            raw_idx = tlib.numpy_ufuncs.min(tlib.array([self.n, raw_idx]))
            if  method == "nearest":
                # The combination of floor and +0.5 prevents the "ties to even" rounding of floating point numbers.
                # We only need one branch since our indices are always positive.
                idx = tlib.numpy_ufuncs.floor(raw_idx + 0.5)
            elif method == "min":
                idx = tlib.numpy_ufuncs.ceil(raw_idx)
            elif method == "max":
                idx = tlib.numpy_ufuncs.floor(raw_idx)
        else:
            raise ValueError("Specified unsupported look-up method.")

        return tlib.array(idx, dtype=int)

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
