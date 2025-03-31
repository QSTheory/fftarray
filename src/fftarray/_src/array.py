from __future__ import annotations
from collections import abc
from typing import (
    Mapping, Optional, Union, List, Any, Tuple, Dict,
    Literal, TypeVar, Iterable, Set, get_args, Callable,
)
from numbers import Number
from dataclasses import dataclass
import textwrap

import numpy as np
import numpy.typing as npt
import array_api_compat

from .space import Space
from .dimension import Dimension
from .named_array import align_named_arrays
from .uniform_value import UniformValue

from .formatting import dim_table, format_bytes, format_n
from .indexing import (
    LocArrayIndexer, check_missing_dim_names,
    tuple_indexers_from_dict_or_tuple, tuple_indexers_from_mapping,
)
from .helpers import norm_param, norm_space
from .op_lazy_luts import (
    TwoOperandTransforms,
    default_transforms_lut,
    add_transforms_lut,
    mul_transforms_lut,
    div_transforms_lut,
)
from .transform_application import do_fft, get_transform_signs, apply_lazy, complex_type

EllipsisType = TypeVar('EllipsisType')



def abs(x: Array, /) -> Array:
    """Implements abs with a special shortcut to statically eliminate the phase part
        of potential phase factors.

    Args:
        x (Array): input array

    Returns:
        Array: x with elementwise abs applied.
    """
    assert isinstance(x, Array)
    # For abs the final result does not change if we apply the phases
    # to the values so we can simply ignore the phases.
    values = x.xp.abs(x._values)
    # The scale can be applied after abs which is more efficient in the case of a complex input
    signs = get_transform_signs(
        # Can use input because with a single value no broadcasting happened.
        input_factors_applied=x._factors_applied,
        target_factors_applied=[True]*len(x._factors_applied),
    )
    if signs is not None:
        values = apply_lazy(
            values=values,
            dims=x.dims,
            signs=signs,
            spaces=x.spaces,
            xp=x.xp,
            scale_only=True,
        )

    return Array(
        values=values,
        spaces=x.spaces,
        dims=x.dims,
        eager=x.eager,
        factors_applied=(True,)*len(x.dims),
        xp=x.xp,
    )

def two_inputs_func(
            unp_inp: UnpackedValues,
            op,
            transforms_lut: TwoOperandTransforms=default_transforms_lut,
        ) -> Array:
    # For a zero-dimensional Array, the factors do not need to be applied and we
    # can shortcut this (also would not work as eager and factors_applied are
    # empty).
    if len(unp_inp.dims) > 0:
        # Compute look-up indices by interpreting the three bits as a binary number.
        lut_indices = np.array(unp_inp.eager)*4 + np.dot(unp_inp.factors_applied,[2,1])
        # Compute the required phase applications for each operand per dimension
        # by applying the rules encoded in the look-up table.
        factor_transforms = transforms_lut.factor_application_signs[lut_indices]
        # Compute the resulting factors_applied per dimension
        # by applying the rules encoded in the look-up table.
        final_factors_applied = transforms_lut.final_factor_state[lut_indices]

        # Apply above defined scale and phase factors depending on the specific case
        for op_idx in [0,1]:
            signs = factor_transforms[:,op_idx]
            if not np.all(signs==0):
                sub_values: Any = unp_inp.values[op_idx]
                if isinstance(sub_values, Number):
                    # This function is only called if at least one of the operands is an Array.
                    # But the transform-signs LUTs may also coerce an operand from factors_applied=True
                    # to factors_applied=False.
                    # And this operand may be a scalar which is not supported by apply_lazy.
                    # So this check transforms the scalar into a complex array with the same number of dimensions
                    # as the other operand.
                    # The other operand has to be complex because it can be the only source for even
                    # introducing factors_applied=False into this operation.
                    # This can only happen here because functions like pow are valid for mixed inputs
                    # and for example turning 2 into 2. changes the actual result.
                    # For addition and subtraction (which are the LUTs causing this path to be hit)
                    # this upcast here is fine, but that has to be determined for each LUT separately.

                    other_values: Any = unp_inp.values[not op_idx]
                    dtype = other_values.dtype
                    shape = (1,)*len(other_values.shape)
                    sub_values = unp_inp.xp.full(shape, sub_values, dtype=dtype)
                elif sub_values.shape == ():
                    # sub_values is 0d Array, shape needs to be taken from
                    # other_values to match the unp_inp.
                    # other_values is an Array since otherwise no phase factors
                    # would need to be applied.
                    other_values = unp_inp.values[not op_idx]
                    dtype = complex_type(unp_inp.xp, sub_values.dtype)
                    shape = (1,)*len(other_values.shape)
                    sub_values = unp_inp.xp.full(shape, sub_values, dtype=dtype)
                else:
                    dtype = complex_type(unp_inp.xp, sub_values.dtype)
                    sub_values = unp_inp.xp.asarray(sub_values, dtype=dtype, copy=True)

                unp_inp.values[op_idx] = apply_lazy(
                    xp=unp_inp.xp,
                    values=sub_values,
                    dims=unp_inp.dims,
                    signs=signs.tolist(),
                    spaces=unp_inp.spaces,
                    scale_only=False,
                )
    else:
        final_factors_applied = np.array([])

    values = op(*unp_inp.values)
    return Array(
        values=values,
        spaces=unp_inp.spaces,
        dims=unp_inp.dims,
        eager=unp_inp.eager,
        factors_applied=tuple(final_factors_applied.tolist()),
        xp=unp_inp.xp,
    )

def elementwise_two_operands(
        name: str,
        transforms_lut: TwoOperandTransforms = default_transforms_lut,
        swap_operands: bool = False,
    ): # This type makes problem for the dunder methods -> Callable[[Any, Any], Array]:
    """Helper function handling elementwise functions for two operands.

    :meta private:
    """

    if swap_operands:
        def fun(x1, x2, /) -> Array:
            unp_inp: UnpackedValues = unpack_arrays([x2, x1])
            op_norm = getattr(unp_inp.xp, name)
            return two_inputs_func(
                unp_inp=unp_inp,
                op=op_norm,
                transforms_lut=transforms_lut,
            )
    else:
        def fun(x1, x2, /) -> Array:
            unp_inp: UnpackedValues = unpack_arrays([x1, x2])
            op_norm = getattr(unp_inp.xp, name)
            return two_inputs_func(
                unp_inp=unp_inp,
                op=op_norm,
                transforms_lut=transforms_lut,
            )

    fun.__doc__ = textwrap.dedent(
        f"""..

        Wrapper around the underlying element-wise function ``{name}`` from the Python Array API standard.
        See https://data-apis.org/array-api/latest/API_specification/generated/array_api.{name}.html
        """
    )
    return fun


def elementwise_one_operand(
        name: str,
    ) -> Callable[[Array], Array]:
    """Helper function handling elementwise functions for a single operand.

    :meta private:
    """

    def single_element_func(x: Array, /) -> Array:
        assert isinstance(x, Array)
        op_norm = getattr(x.xp, name)

        values = op_norm(x.values(x.spaces))
        return Array(
            values=values,
            spaces=x.spaces,
            dims=x.dims,
            eager=x.eager,
            factors_applied=(True,)*len(x.dims),
            xp=x.xp,
        )
    single_element_func.__doc__ = textwrap.dedent(
        f"""..

        Wrapper around the underlying element-wise function ``{name}`` from the Python Array API standard.
        See https://data-apis.org/array-api/latest/API_specification/generated/array_api.{name}.html
        """
    )

    return single_element_func


class Array:
    """The Array class facilitates the generalized Discrete Fourier transform
    and all scientific operations on its values while retaining coordinate grid
    information.

    Parameters
    ----------
    values : Any
        The values array with respect to the dimensions.
    dims : Tuple[Dimension, ...]
        Dimensions defining the coordinates of the values.
    spaces : Tuple[Space, ...]
        Space of the given values per Dimension.
    eager : Tuple[bool, ...]
        Whether the factors should be applied to the internal values after
        performing a space transition.
    factors_applied : Tuple[bool, ...]
        Whether the lazy factors have been applied to the given values.
    xp : Any
        The array API namespace.

    Examples
    --------
    First, we initialize the dimensions using `fa.dim_from_constraints`:

    >>> import fftarray as fa
    >>> x_dim: fa.Dimension = fa.dim_from_constraints("x", pos_min=-1, pos_max=1., n=64, freq_middle=0.)
    >>> y_dim: fa.Dimension = fa.dim_from_constraints("y", pos_min=-10, pos_max=10., n=128, freq_middle=0.)

    Based on these, the coordinates can be retrieved in form of 1D Arrays. These
    can be combined to a 2D Array via automatic broadcasting:

    >>> x: fa.Array = fa.coords_from_dim(x_dim, "pos")
    >>> y: fa.Array = fa.coords_from_dim(y_dim, "pos")
    >>> xy = x**2 * fa.sin(y)

    The space (as well as other attributes such as xp, eager and
    factors_applied) via respective `into` functions.

    >>> xy_freq: fa.Array = xy.into_space("freq")

    Integrate the Array from x=-0.2 to x=0.2 by indexing the given region before
    intgrating it via `fa.integrate`:

    >>> res = fa.integrate(xy.sel(x=slice(-0.2,0.2)))

    See Also
    --------
    array
    coords_from_dim
    full
    """

    # _dims are stored as a sequence and not by name because their oder needs
    # to match the order of dimensions in _values.
    _dims: Tuple[Dimension, ...]
    # Contains an array instance of _xp.
    _values: Any
    # Marks each dimension whether it is in position or frequency space
    _spaces: Tuple[Space, ...]
    # Marks each dimension whether the phase-factors should be applied directly after executing a fft or ifft
    _eager: Tuple[bool, ...]
    # Marks each dim whether its phase_factors still need to be applied
    _factors_applied: Tuple[bool, ...]
    # Contains the array library of the values.
    # This library must be Array API compatible.
    # When using a non-compatible base library, it should be
    # wrapped by array_api_compat.array_namespace.
    _xp: Any

    # TODO: implement device [#18](https://github.com/QSTheory/fftarray/issues/18)


    def __init__(
            self,
            values,
            dims: Tuple[Dimension, ...],
            spaces: Tuple[Space, ...],
            eager: Tuple[bool, ...],
            factors_applied: Tuple[bool, ...],
            xp,
        ):

        self._dims = dims
        self._values = values
        self._spaces = spaces
        self._eager = eager
        self._factors_applied = factors_applied
        self._xp = xp


    def __repr__(self: Array) -> str:
        arg_str = ", ".join(
            [f"{name[1:] if name != '_spaces' else 'space'}={repr(value)}"
                for name, value in self.__dict__.items()]
        )
        return f"Array({arg_str})"

    def __str__(self: Array) -> str:
        shape_str = ""
        for i, dim in enumerate(self.dims):
            shape_str += f"{dim.name}: {format_n(dim.n)}"
            if i < len(self.dims)-1:
                shape_str += ", "
        str_out = f"<fftarray.Array ({shape_str})>"
        # Array API does not guarantuee existence of this attribute.
        if hasattr(self._values, "nbytes"):
            bytes_str = format_bytes(self._values.nbytes)
            str_out += f" Size: {bytes_str}"
        str_out += "\n"
        for i, (dim, space) in enumerate(zip(self.dims, self.spaces, strict=True)):
            str_out += dim_table(
                dim=dim,
                include_header=(i==0),
                include_dim_name=True,
                spaces=(space,),
            ) + "\n"
        str_out += f"Values<{self._xp.__name__}>:\n"
        str_out += f"{self.values(self.spaces)}"
        return str_out

    def __bool__(self: Array):
        raise ValueError("The truth value of an array is ambiguous.")

    #--------------------
    # Operator Implementations
    #--------------------
    # Implement binary operations between Array and also Scalars e.g. 1+wf and wf+1
    # We need to map directly to the dunder methods (as opposed to just reusing xp.add, etc...)
    # in order to ensure the correct promotion rules, since those differ for scalars in the
    # current (v2023.12) edition of the Array API standard.

    # Arithemtic Operators
    __pos__ = elementwise_one_operand("positive")
    __neg__ = elementwise_one_operand("negative")
    __add__ = elementwise_two_operands(
        name="add",
        transforms_lut=add_transforms_lut,
    )
    __radd__ = elementwise_two_operands(
        name="add",
        transforms_lut=add_transforms_lut,
        swap_operands=True,
    )
    __sub__ = elementwise_two_operands(
        name="subtract",
        transforms_lut=add_transforms_lut,
    )
    __rsub__ = elementwise_two_operands(
        name="subtract",
        transforms_lut=add_transforms_lut,
        swap_operands=True,
    )
    __mul__ = elementwise_two_operands(
        name="multiply",
        transforms_lut=mul_transforms_lut,
    )
    __rmul__ = elementwise_two_operands(
        name="multiply",
        transforms_lut=mul_transforms_lut,
        swap_operands=True,
    )
    __truediv__ = elementwise_two_operands(
        name="divide",
        transforms_lut=div_transforms_lut,
    )
    __rtruediv__ = elementwise_two_operands(
        name="divide",
        transforms_lut=div_transforms_lut,
        swap_operands=True,
    )
    # floor div should only support real inputs, just always apply all phase factors
    __floordiv__ = elementwise_two_operands(
        name="floor_divide",
        transforms_lut=default_transforms_lut,
    )
    __rfloordiv__ = elementwise_two_operands(
        name="floor_divide",
        transforms_lut=default_transforms_lut,
        swap_operands=True,
    )
    __mod__ = elementwise_two_operands("remainder")
    __rmod__ = elementwise_two_operands("remainder", swap_operands=True)
    __pow__ = elementwise_two_operands("pow")
    __rpow__ = elementwise_two_operands("pow", swap_operands=True)

    # Bitwise Operators
    __invert__ = elementwise_one_operand("bitwise_invert")
    __and__ = elementwise_two_operands("bitwise_and")
    __rand__ = elementwise_two_operands("bitwise_and", swap_operands=True)
    __or__ = elementwise_two_operands("bitwise_or")
    __ror__ = elementwise_two_operands("bitwise_or", swap_operands=True)
    __xor__ = elementwise_two_operands("bitwise_xor")
    __rxor__ = elementwise_two_operands("bitwise_xor", swap_operands=True)
    __lshift__ = elementwise_two_operands("bitwise_left_shift")
    __rlshift__ = elementwise_two_operands("bitwise_left_shift", swap_operands=True)
    __rshift__ = elementwise_two_operands("bitwise_right_shift")
    __rrshift__ = elementwise_two_operands("bitwise_right_shift", swap_operands=True)

    # Comparison Operators
    __lt__ = elementwise_two_operands("less")
    __le__ = elementwise_two_operands("less_equal")
    __gt__ = elementwise_two_operands("greater")
    __ge__ = elementwise_two_operands("greater_equal")
    __eq__ = elementwise_two_operands("equal")
    __ne__ = elementwise_two_operands("not_equal")

    # Other Operators
    __abs__ = abs

    #--------------------
    # Selection
    #--------------------

    def __getitem__(
            self,
            item: Union[
                int, slice, EllipsisType,
                Tuple[Union[int, slice, EllipsisType],...],
                Mapping[str, Union[int, slice]],
            ]
        ) -> Array:
        """This method is called when indexing an Array instance by integer index,
            i.e., by using the index value via Array[].
            It supports dimensional lookup via position and name.
            The indexing behaviour is mainly defined to match the one of
            xarray.DataArray with the major difference that we always keep
            all dimensions.
            The indexing is performed in the current state of the Array,
            i.e., each dimension is indexed in its respective state (pos or freq).
            When the returned Array object is effectively indexed,
            its internal state will always have all fft factors applied.

            Example usage:
            arr_2d = (
                coords_from_dim(x_dim, "pos")
                + coords_from_dim(y_dim, "pos")
            )
            Four ways of retrieving an Array object
            with index 3 along x and first 5 values along y:

            arr_2d[{"x": 3, "y": slice(0, 5)}]
            arr_2d[3,:5]
            arr_2d[3][:,:5] # do not use, just for explaining functionality
            arr_2d[:,:5][3] # do not use, just for explaining functionality

        Parameters
        ----------
        item : Union[ int, slice, EllipsisType, Tuple[Union[int, slice, EllipsisType],...], Mapping[str, Union[int, slice]], ]
            An indexer object with either dimension lookup method either
            via position or name. When using positional lookup, the order
            of the dimensions in the Array object is applied (Array.dims).
            Per dimension, each indexer can be supplied as an integer or a slice.
            Array-like indexers are not supported as in the general case,
            the resulting coordinates can not be supported with a valid Dimension.
        Array
            A new Array with the same dimensionality as this Array,
            except each dimension and the Array values are indexed.
            The resulting Array still fully supports FFTs.
        """

        # Catch special case where effectively no indexing happens,
        # i.e., just return Array object as is (without changing internal state)
        if item is Ellipsis:
            return self
        if isinstance(item, abc.Mapping) and len(item) == 0:
            return self

        # Handle two cases of supplying indexing information, either
        # via keyword args (Mapping) or via tuple using order of dims
        # Return full tuple of indexers as slice or int object
        tuple_indexers: Tuple[Union[int, slice], ...] = tuple_indexers_from_dict_or_tuple(
            indexers=item, # type: ignore
            dim_names=tuple(dim.name for dim in self.dims)
        )

        new_dims = []
        for index, orig_dim, space in zip(tuple_indexers, self._dims, self.spaces, strict=True):
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
                        "Array indexing does not support "
                        + "jitted indexers. Here, your index for "
                        + f"dimension {orig_dim.name} is a traced object."
                    ) from e
                else:
                    raise type(e)(
                        "An error occurred when evaluating the index "
                        + f"dimension {orig_dim.name}."
                    ) from e


        selected_values = self.values(self.spaces).__getitem__(tuple_indexers)
        # Dimensions with the length 1 are dropped in numpy indexing.
        # We decided against this and keeping even dimensions of length 1.
        # So we have to reintroduce those dropped dimensions via reshape.
        selected_values = self._xp.reshape(selected_values, tuple(dim.n for dim in new_dims))

        return Array(
            values=selected_values,
            dims=tuple(new_dims),
            spaces=self.spaces,
            eager=self.eager,
            factors_applied=(True,)*len(new_dims),
            xp=self._xp,
        )

    @property
    def loc(self):
        """Attribute for location based indexing like xarray."""
        return LocArrayIndexer(self)

    def isel(
            self,
            indexers: Optional[Mapping[str, Union[int, slice]]] = None,
            missing_dims: Literal["raise", "warn", "ignore"] = 'raise',
            **indexers_kwargs: Union[int, slice],
        ) -> Array:
        """Select specific values along respective dimension names by indices.
        Inspired by `xarray.DataArray.isel`.

        Parameters
        ----------
        indexers : Optional[Mapping[str, Union[int, slice]]], optional
            Mappping from Dimension name to the index which can be an int or a
            slice, by default None
        missing_dims : Literal[&quot;raise&quot;, &quot;warn&quot;, &quot;ignore&quot;], optional
            Behavior, if selected Dimensions are not part of the Array, by
            default 'raise'

        Returns
        -------
        Array
            Indexed Array.
        """

        # Check for correct use of indexers (either via positional
        # indexers arg or via indexers_kwargs)
        if indexers is not None and indexers_kwargs:
            raise ValueError(
                "Cannot specify both keyword arguments and "
                + "positional arguments to Array.isel"
            )

        # Handle two ways of supplying indexers, either via positional
        # argument "indexers" or via keyword arguments for each dimension
        final_indexers: Mapping[str, Union[int, slice]]
        if indexers is None:
            final_indexers = indexers_kwargs
        else:
            final_indexers = indexers

        if not isinstance(final_indexers, dict):
            raise ValueError(
                "Indexers must be a dictionary or keyword arguments"
            )

        # handle case of empty indexers via supplying indexers={} or nothing at all
        if len(final_indexers) == 0:
            return self

        # Check for indexer names that are not present in Array and
        # according to user choice, raise Error, throw warning or ignore
        check_missing_dim_names(
            indexer_names=final_indexers.keys(),
            dim_names=tuple(self.dims_dict.keys()),
            missing_dims=missing_dims
        )

        # Map indexers into full tuple of valid indexers, one entry per dimension
        tuple_indexers: Tuple[Union[int, slice], ...] = tuple_indexers_from_mapping(
            final_indexers, # type: ignore
            dim_names=[dim.name for dim in self.dims],
        )

        return self.__getitem__(tuple_indexers)

    def sel(
            self,
            indexers: Optional[Mapping[str, Union[float, slice]]] = None,
            missing_dims: Literal["raise", "warn", "ignore"] = 'raise',
            method: Optional[Literal["nearest", "pad", "ffill", "backfill", "bfill"]] = None,
            **indexers_kwargs: Union[float, slice],
        ) -> Array:
        """Select specific values along respective dimension names by
        label/coordinate.
        Inspired by xarray.DataArray.sel

        Parameters
        ----------
        indexers : Optional[Mapping[str, Union[float, slice]]], optional
            Mappping from Dimension name to the coordinate/label which can be a
            float or a slice, by default None
        missing_dims : Literal[&quot;raise&quot;, &quot;warn&quot;, &quot;ignore&quot;], optional
            Behavior, if selected Dimensions are not part of the Array, by
            default 'raise'
        method : Optional[Literal[&quot;nearest&quot;, &quot;pad&quot;, &quot;ffill&quot;, &quot;backfill&quot;, &quot;bfill&quot;]], optional
            Method to use for inexact matches

            - None (default): only exact matches
            - pad / ffill: propagate last valid index value forward
            - backfill / bfill: propagate next valid index value backward
            - nearest: use nearest valid index value

        Returns
        -------
        Array
            Indexed Array.

        Notes
        -----
        In comparison to its xarray implementation, there is an add-on:

        - Implements missing_dims arg and accordingly raises errors

        """

        # Check for correct use of indexers (either via positional
        # indexers arg or via indexers_kwargs)
        if indexers is not None and indexers_kwargs:
            raise ValueError(
                "cannot specify both keyword arguments and "
                + "positional arguments to Array.sel"
            )

        # Handle two ways of supplying indexers, either via positional
        # argument "indexers" or via keyword arguments for each dimension
        final_indexers: Mapping[str, Union[float, slice]]
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

        # Check for indexer names that are not present in Array and
        # according to user choice, raise Error, throw warning or ignore
        check_missing_dim_names(
            indexer_names=final_indexers.keys(),
            dim_names=tuple(self.dims_dict.keys()),
            missing_dims=missing_dims
        )

        # As opposed to Array.isel, here we have to find the appropriate
        # indices for the coordinate indexers by checking the respective
        # Dimension
        tuple_indexers_as_integer = []
        for dim, space in zip(self.dims, self.spaces, strict=True):
            if dim.name in final_indexers:
                index = final_indexers[dim.name]
                try:
                    tuple_indexers_as_integer.append(
                        dim.index_from_coord(
                            coord=index,
                            space=space,
                            method=method,
                        )
                    )
                except Exception as e:
                    # Here, we check for traced indexer values or
                    # traced Dimension and throw a helpful error message
                    # in addition to the original error raised when trying
                    # to map the coord to an index
                    if "Trace" in str(index):
                        raise NotImplementedError(
                            "Array indexing does not support "
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
    def dims_dict(self) -> Dict[str, Dimension]:
        """Dictionary mapping each dimension name to the Array's Dimension."""
        # TODO Ordered Mapping?
        return {dim.name: dim for dim in self._dims}

    @property
    def sizes(self) -> Dict[str, int]:
        """Dictionary mapping each dimension name to the number of grid points.
        """
        # TODO Ordered Mapping?
        return {dim.name: dim.n for dim in self._dims}

    @property
    def dims(self) -> Tuple[Dimension, ...]:
        """Tuple of the Array's Dimensions."""
        return tuple(self._dims)

    @property
    def shape(self: Array) -> Tuple[int, ...]:
        """Shape of the Array's values."""
        return self._values.shape

    def values(
            self: Array,
            space: Union[Space, Iterable[Space]],
            /,
            *,
            xp: Optional[Any] = None,
            dtype: Optional[Any] = None,
        ) -> Any:
        """The raw values with all lazy fft factors applied.

        Parameters
        ----------
        space : Union[Space, Iterable[Space]]
            The space in which the values should be returned. Can be specified
            per dimension.
        xp : Optional[Any], optional
            The array API namespace to use for the returned values. If it is
            None, the array namespace from self is used.
        dtype : Optional[Any], optional
            The dtype to use for the returned values. If it is None, the dtype
            of self is used.

        Returns
        -------
        Any
            The Array's values.
        """
        if xp is None:
            xp = self.xp
        else:
            xp = array_api_compat.array_namespace(xp.asarray(1))
        space_norm: Tuple[Space, ...] = norm_space(space, len(self.dims))
        if space_norm != self.spaces or not all(self._factors_applied):
            # Setting eager before-hand allows copy-elision without the move option.
            arr = self.into_eager(True).into_space(space).into_factors_applied(True)
            return xp.asarray(arr._values, dtype=dtype)
        return xp.asarray(self._values, dtype=dtype, copy=True)

    @property
    def xp(self):
        """The array API namespace driving the mathematical operations."""
        return self._xp

    def into_xp(self, xp, /) -> Array:
        """Change the array API namespace.

        Parameters
        ----------
        xp : Any
            The new array API namespace.

        Returns
        -------
        Array
            A new Array with the new array API namespace.
        """
        # Since Array is immutable, this does not necessarily need to copy.
        values = xp.asarray(self._values, copy=None)
        return Array(
            dims=self._dims,
            values=values,
            spaces=self._spaces,
            eager=self._eager,
            factors_applied=self._factors_applied,
            xp=array_api_compat.array_namespace(values),
        )

    @property
    def dtype(self):
        """The dtype of the Array's values."""
        return self._values.dtype

    def into_dtype(self, dtype, /) -> Array:
        """Change the values' dtype.

        Parameters
        ----------
        dtype : Any
            The new dtype.

        Returns
        -------
        Array
            A new Array whose values have the new dtype.
        """
        # Hard-code this special case (which also exists in numpy)
        # in order to give an Array API compatible way to upcast
        # to complex without explicitly handling precision in user code.
        if dtype == "complex":
            dtype = complex_type(xp=self.xp, dtype=self._values.dtype)

        if not all(self._factors_applied) and not self.xp.isdtype(dtype, "complex floating"):
            raise ValueError(
                "If any `factors_applied' is False, the values have to be of dtype 'complex floating'"
                + " since the not applied phase factor implies a complex value."
            )

        # Since Array is immutable, this does not need to copy.
        # copy=False never raises an error but just avoids the copy if possible.
        values = self._xp.astype(self._values, dtype, copy=False)
        return Array(
            dims=self._dims,
            values=values,
            spaces=self._spaces,
            eager=self._eager,
            factors_applied=self._factors_applied,
            xp=self._xp,
        )

    @property
    def factors_applied(self) -> Tuple[bool, ...]:
        """Whether the lazy fft factors have been applied to the internal
        values.
        """
        return self._factors_applied

    def into_factors_applied(self, factors_applied: Union[bool, Iterable[bool]], /) -> Array:
        """Change the factors_applied state.
        Note that the values of the returned Array will be of dtype complex, no
        matter if the factors_applied state changed.

        Parameters
        ----------
        factors_applied : Union[bool, Iterable[bool]]
            The new factors_applied state.

        Returns
        -------
        Array
            A new Array whose values correspond to the new factors_applied
            state.
        """
        factors_applied_norm = norm_param(factors_applied, len(self._dims), bool)

        signs = get_transform_signs(
            input_factors_applied=self._factors_applied,
            target_factors_applied=factors_applied_norm,
        )

        if not self.xp.isdtype(self.dtype, ("real floating", "complex floating")):
            raise ValueError(f"'into_factors_applied' requires an Array with a float or complex dtype, but got passed array of type '{self.dtype}'")

        values = self.xp.astype(self._values, complex_type(self.xp, self._values.dtype), copy=True)

        if signs is not None:
            values = apply_lazy(
                xp=self.xp,
                values=values,
                dims=self.dims,
                signs=signs,
                spaces=self.spaces,
                scale_only=False,
            )


        return Array(
            dims=self._dims,
            values=values,
            spaces=self._spaces,
            eager=self._eager,
            factors_applied=factors_applied_norm,
            xp=self._xp,
        )


    @property
    def eager(self) -> Tuple[bool, ...]:
        """Whether the space transformation should be lazy or not.

        - `eager=False`: The fft factors are not directly applied after a space
          transformation but only when needed, i.e., lazily.
        - `eager=True`: The fft factors are always directly applied after a
          space transformation.
        """
        return self._eager

    def into_eager(self, eager: Union[bool, Iterable[bool]], /) -> Array:
        """Change the eager state.

        Parameters
        ----------
        eager : Union[bool, Iterable[bool]]
            New eager state.

        Returns
        -------
        Array
            A new Array with the new eager settings.
        """
        eager_norm = norm_param(eager, len(self.dims), bool)

        # Can just reuse everything since all attributes are immutable.
        return Array(
            dims=self._dims,
            values=self._values,
            spaces=self.spaces,
            eager=eager_norm,
            factors_applied=self.factors_applied,
            xp=self.xp,
        )

    @property
    def spaces(self) -> Tuple[Space, ...]:
        """The spaces in which the values currently are in, per dimension."""
        return self._spaces

    def into_space(self, space: Union[Space, Iterable[Space]], /) -> Array:
        """Transform the space(s) of the Array's values.

        Parameters
        ----------
        space : Union[Space, Iterable[Space]]
            The new space(s).

        Returns
        -------
        Array
            A new Array whose values are in the new space.

        Raises
        ------
        ValueError
            If the Array does not have a float dtype.
        """

        values = self._values
        are_values_owned = False

        if not self.xp.isdtype(values.dtype, ("real floating", "complex floating")):
            raise ValueError(f"'into' requires an Array with a float dtype, but got passed array of type '{values.dtype}'")

        # At this point we need to do either an FFT and/or apply phase factors
        # both require complex numbers.
        if not self.xp.isdtype(values.dtype, 'complex floating'):
            values = self.xp.astype(values, complex_type(self.xp, values.dtype), copy=True)
            are_values_owned = True

        dims = self._dims
        n_dims = len(dims)
        space_after = norm_space(space, n_dims)

        needs_fft = [old != new for old, new in zip(self._spaces, space_after, strict=True)]
        if not any(needs_fft):
            factors_applied_after = self._factors_applied
        else:
            factors_after_list = []
            for is_eager, fft_needed, is_applied in zip(self.eager, needs_fft, self._factors_applied, strict=True):
                if fft_needed:
                    factors_after_list.append(is_eager)
                else:
                    # We did not do a fft, so just take whatever it was before
                    factors_after_list.append(is_applied)
            factors_applied_after = tuple(factors_after_list)

            values, are_values_owned = do_fft(
                values=values,
                dims=self._dims,
                space_before=self._spaces,
                space_after=space_after,
                xp=self.xp,
                factors_applied_before=self._factors_applied,
                factors_applied_after=factors_applied_after,
                needs_fft=needs_fft,
                are_values_owned=are_values_owned,
            )

        return Array(
            dims=dims,
            values=values,
            spaces=space_after,
            eager=self.eager,
            factors_applied=factors_applied_after,
            xp=self.xp,
        )

    def _check_consistency(self) -> None:
        """Check some invariants of Array."""

        assert isinstance(self._dims, tuple)
        assert isinstance(self._spaces, tuple)
        assert isinstance(self._eager, tuple)
        assert isinstance(self._factors_applied, tuple)

        assert len(self._dims) == len(self._values.shape)
        assert len(self._spaces) == len(self._values.shape)
        assert len(self._eager) == len(self._values.shape)
        assert len(self._factors_applied) == len(self._values.shape)

        dim_names: Set[str] = set()
        for n, dim in zip(self._values.shape, self._dims, strict=True):
            assert dim.n == n, \
                "Passed in inconsistent n from Dimension and values."
            assert dim.name not in dim_names, \
                f"Passed in Dimension of name {dim.name} twice!"
            dim_names.add(dim.name)

        assert all([isinstance(dim, Dimension) for dim in self._dims])
        assert all([dim_space in get_args(Space) for dim_space in self._spaces])
        assert all([isinstance(dim_eager, bool) for dim_eager in self._eager])
        assert all([isinstance(factor_applied, bool) for factor_applied in self._factors_applied])

        # Check that the Array API namespace is properly wrapped.
        assert self._xp == array_api_compat.array_namespace(self._xp.asarray(0))

        # Check that values are of the stored Array API namespace.
        assert self._xp == array_api_compat.array_namespace(self._values)

        if not all(self._factors_applied):
            assert self.xp.isdtype(self._values.dtype, 'complex floating')



@dataclass
class UnpackedValues:
    # Dimensions in the order in which they appear in each non-scalar value.
    dims: Tuple[Dimension, ...]
    # Values without any dimensions, etc.
    values: List[Union[Number, Any]]
    # Shared array namespace between all values.
    xp: Any
    # dim 0: dim_idx, dim 1: op_idx
    factors_applied: npt.NDArray[np.bool_]
    # Space per dimension, must be homogeneous over all values
    spaces: Tuple[Space, ...]
    # eager per dimension, must be homogeneous over all values
    eager: Tuple[bool, ...]


@dataclass
class UnpackedDimProperties:
    dim: UniformValue[Dimension]
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

def unpack_arrays(
        values: List[Union[Number, Array, Any]],
    ) -> UnpackedValues:
    """
        This handles all "alignment" of input values.
        Align dimensions, unify them, unpack all operands to a simple list of values.
    """
    dims: Dict[str, UnpackedDimProperties] = {}
    arrays_to_align: List[Tuple[List[str], Any]] = []
    array_indices = []
    unpacked_values: List[Optional[Union[Number, Any]]] = [None]*len(values)
    xp: UniformValue[Any] = UniformValue()

    for op_idx, x in enumerate(values):
        if isinstance(x, Number):
            unpacked_values[op_idx] = x
        elif hasattr(x, "shape") and not isinstance(x, Array):
            if x.shape == ():
                unpacked_values[op_idx] = x
            else:
                raise ValueError(
                    "Cannot combine coordinate-less arrays with an Array."
                )
        else:
            array_indices.append(op_idx)
            assert isinstance(x, Array)

            xp.set(x.xp)
            # input_factors_applied = x._factors_applied
            # target_factors_applied = list(x._factors_applied)

            for dim_idx, dim in enumerate(x._dims):
                if dim.name not in dims:
                    dim_props = UnpackedDimProperties(len(values))
                    dims[dim.name] = dim_props
                else:
                    dim_props = dims[dim.name]

                try:
                    dim_props.dim.set(dim)
                except ValueError:
                    raise ValueError(
                        "Tried to combine Arrays with " +
                        "different dimension of name " +
                        f"{dim.name}."
                    ) from None

                try:
                    dim_props.space.set(x._spaces[dim_idx])
                except ValueError:
                    raise ValueError(
                        "Tried to call ufunc on Arrays with " +
                        "different spaces in dimension of name " +
                        f"{dim.name}. " +
                        "They have to be explicitly converted " +
                        "into an identical space."
                    ) from None

                try:
                    dim_props.eager.set(x._eager[dim_idx])
                except ValueError:
                    raise ValueError(
                        "Tried to call ufunc on Arrays with " +
                        "different eager settings in dimension of name " +
                        f"{dim.name}."
                    ) from None

                dim_props.factors_applied[op_idx] = x._factors_applied[dim_idx]

            elem_dim_names = [dim.name for dim in x._dims]
            arrays_to_align.append((elem_dim_names, x._values))


    # Broadcasting
    dim_names, aligned_arrs = align_named_arrays(arrays_to_align, xp=xp.get())
    for op_idx, arr in zip(array_indices, aligned_arrs, strict=True):
        unpacked_values[op_idx] = arr

    dims_list = [dims[dim_name].dim.get() for dim_name in dim_names]
    space_list = [dims[dim_name].space.get() for dim_name in dim_names]
    eager_list = [dims[dim_name].eager.get() for dim_name in dim_names]
    factors_applied: npt.NDArray[np.bool_] = np.array([dims[dim_name].factors_applied for dim_name in dim_names])

    for value in unpacked_values:
        assert value is not None

    assert xp.get() == array_api_compat.array_namespace(xp.get().asarray(0))

    return UnpackedValues(
        dims = tuple(dims_list),
        values = unpacked_values, # type: ignore
        spaces = tuple(space_list),
        factors_applied=factors_applied,
        eager=tuple(eager_list),
        xp = xp.get(),
    )


