from __future__ import annotations
from collections import abc
from typing import (
    Mapping, Optional, Union, List, Any, Tuple, Dict, Hashable,
    Literal, TypeVar, Iterable, Set, get_args
)
from abc import ABCMeta
from copy import copy
from numbers import Number
from dataclasses import dataclass

import numpy as np

from .named_array import align_named_arrays, get_axes_transpose
from .backends.tensor_lib import TensorLib
from .backends.np_backend import NumpyTensorLib

from ._utils._ufuncs import binary_ufuncs, unary_ufunc
from ._utils._formatting import (
    fft_dim_table, fft_array_props_table, format_bytes, format_n
)
from ._utils._unpacking import UniformValue, norm_param
from ._utils._indexing import (
    LocFFTArrayIndexer, check_substepping, check_missing_dim_names,
    tuple_indexers_from_dict_or_tuple, tuple_indexers_from_mapping,
    remap_index_check_int
)

EllipsisType = TypeVar('EllipsisType')
Space = Literal["pos", "freq"]


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
        self._spaces = norm_param(space, n_dims, str)
        self._eager = norm_param(eager, n_dims, bool)
        self._factors_applied = norm_param(factors_applied, n_dims, bool)
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
        str_out = f"<fftarray.FFTArray ({bytes_str})>\n"
        str_out += f"TensorLib: {self.tlib}\n"
        str_out += "Dimensions:\n"
        for i, dim in enumerate(self.dims):
            str_out += f" # {i}: {repr(dim.name)}\n"
        str_out += "\n" + fft_array_props_table(self) + "\n\n"
        for i, dim in enumerate(self.dims):
            str_out += fft_dim_table(dim, i==0, True, i) + "\n"
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
    # This does intentionally not list all possible operators.
    __add__, __radd__ = binary_ufuncs(np.add)
    __sub__, __rsub__ = binary_ufuncs(np.subtract)
    __mul__, __rmul__ = binary_ufuncs(np.multiply)
    __truediv__, __rtruediv__ = binary_ufuncs(np.true_divide)
    __floordiv__, __rfloordiv__ = binary_ufuncs(np.floor_divide)
    __pow__, __rpow__ = binary_ufuncs(np.power)

    # Implement comparison operators
    __gt__, _ = binary_ufuncs(np.greater)
    __ge__, _ = binary_ufuncs(np.greater_equal)
    __lt__, _ = binary_ufuncs(np.less)
    __le__, _ = binary_ufuncs(np.less_equal)
    __ne__, _ = binary_ufuncs(np.not_equal)
    __eq__, _ = binary_ufuncs(np.equal)


    # Implement unary operations
    __neg__ = unary_ufunc(np.negative)
    __pos__ = unary_ufunc(np.positive)
    __abs__ = unary_ufunc(np.absolute)
    __invert__ = unary_ufunc(np.invert)

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
                x_dim.fft_array(space="pos", tlib=some_tlib)
                + y_dim.fft_array(space="pos", tlib=some_tlib)
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


        selected_values = self.values.__getitem__(tuple_indexers)
        # Dimensions with the length 1 are dropped in numpy indexing.
        # We decided against this and keeping even dimensions of length 1.
        # So we have to reintroduce those dropped dimensions via reshape.
        selected_values = selected_values.reshape(tuple(dim.n for dim in new_dims))

        return FFTArray(
            values=selected_values,
            dims=new_dims,
            space=self._spaces,
            eager=self._eager,
            factors_applied=[True]*len(new_dims),
            tlib=self.tlib,
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
        """
            Inspired by xarray.DataArray.sel
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
                            tlib=self.tlib,
                            method=method,
                        )
                    )
                except Exception as e:
                    # Here, we check for traced indexer values and throw
                    # a helpful error message in addition to the original
                    # error raised when trying to map the coord to an index
                    if "Trace" in str(index):
                        raise NotImplementedError(
                            f"FFTArray indexing does not support "
                            + "jitted indexers. Here, your index for "
                            + f" dimension {dim.name} is a traced object"
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
            space_norm = norm_param(space, n_dims, str)

        if eager is None:
            eager_norm = self._eager
        else:
            eager_norm = norm_param(eager, n_dims, bool)



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
            factors_norm = norm_param(factors_applied, n_dims, bool)

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
        str_out = f"<fftarray.FFTDimension (name={repr(self.name)})>\n"
        str_out += f"n={n_str}\n"
        str_out += fft_dim_table(self)
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

        # Catch invalid slice objects with range.step != 1
        check_substepping(range)

        start = remap_index_check_int(range.start, self.n, index_kind="start")
        end = remap_index_check_int(range.stop, self.n, index_kind="stop")

        n = end - start
        # Check validity of slice object which has to
        # yield at least one dimension value
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
        """Returns new FFTDimension instance starting at a specific value
        in either pos or freq space and setting variable dimension length.
        """

        if space == "pos":
            pos_min = self.pos_min + start*self.d_pos
            freq_min = self.freq_min
            d_pos = self.d_pos
        elif space == "freq":
            pos_min = self.pos_min
            freq_min = self.freq_min + start*self.d_freq
            d_pos = 1./(self.d_freq*n)
        else:
            assert False, "Unreachable"

        return FFTDimension(
            name=self.name, # type: ignore
            n=n,
            pos_min=pos_min,
            freq_min=freq_min,
            d_pos=d_pos,
        )

    def _index_from_coord(
            self,
            coord: Union[float, slice],
            space: Space,
            tlib: TensorLib,
            method: Optional[Literal["nearest", "pad", "ffill", "backfill", "bfill"]] = None,
        ) -> Union[int, slice]:
        """Compute index from given coordinate which can be float or slice.
        In case of slice input, find the dimension indices which are
        including the selected coordinates, and return appropriate slice object.

        Short explanation what "pad", "ffill", "backfill", "bfill" do:
            bfill, backfill: maps 2.5 to next valid index 3
            pad, ffill: maps 2.5 to previous valid index 2
        """
        # The first part handles coords supplied as slice object whereas
        # it prepares those and distributes the actual work to the second
        # part of this function which handles scalar objects
        if isinstance(coord, slice):
            check_substepping(coord)
            if method is not None:
                # This catches slices supplied to FFTArray.sel or isel with
                # a method != None (e.g. nearest) which is not supported
                raise NotImplementedError(
                    f"cannot use method: `{method}` if the coord argument "
                    + f"is not scalar, here: {coord}."
                )
            # Handle slice objects with start or end being None whereas
            # we substitute those with the FFTDimension bounds
            if coord.start is None:
                coord_start = getattr(self, f"{space}_min")
            else:
                coord_start = coord.start
            if coord.stop is None:
                coord_stop = getattr(self, f"{space}_max")
            else:
                coord_stop = coord.stop

            # Use the scalar part of this function with the methods bfill and ffill
            # to yield indices to include the respective coordinates
            idx_min: int = self._index_from_coord(coord_start, method="bfill", space=space, tlib=tlib) # type: ignore
            idx_max: int = self._index_from_coord(coord_stop, method="ffill", space=space, tlib=tlib) # type: ignore
            return slice(
                idx_min,
                idx_max + 1 # as slice.stop is non-inclusive, add 1
            )
        else:
            # Calculate the float index regarding the FFTDimension as
            # an infinite grid
            if space == "pos":
                raw_idx = (coord - self.pos_min) / self.d_pos
            else:
                raw_idx = (coord - self.freq_min) / self.d_freq

            # Clamp float index to the valid range of 0 to n-1
            clamped_index = min(
                max(0, raw_idx),
                self.n - 1
            )

            # Handle different methods case by case here
            if method is None:
                # We round the raw float indices here and check whether they
                # match their rounded int-like value, if not we throw a KeyError
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
                # We only need one branch since our indices are always positive.
                final_idx = tlib.numpy_ufuncs.floor(clamped_index + 0.5)
            elif method in ["bfill", "backfill"]:
                # We propagate towards the next highest index and then check
                # its validity by checking if it's smaller or equal than
                # the dimension length n
                final_idx = np.ceil(clamped_index)
                if raw_idx > self.n - 1:
                    raise KeyError(
                        f"Coord {coord} not found with method '{method}', "
                        + "you could try one of the following instead: "
                        + "'ffill', 'pad' or 'nearest'."
                    )
            elif method in ["ffill", "pad"]:
                # We propagate back to the next smalles index and then check
                # its validity by checking if it's at least 0
                final_idx = np.floor(clamped_index)
                if raw_idx < 0:
                    raise KeyError(
                        f"Coord {coord} not found with method '{method}', "
                        + "you could try one of the following instead: "
                        + "'bfill', 'backfill' or 'nearest'."
                    )
            else:
                raise ValueError(f"Specified unsupported look-up method `{method}`.")

            # Transform index to integer here. We can do this because we
            # ensured validity in the cases above, especially for method = None
            return int(final_idx)

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
