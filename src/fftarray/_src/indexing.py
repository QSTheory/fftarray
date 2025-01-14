from typing import (
    Dict, List, Optional, Tuple, TypeVar, Union, Iterable,
    Literal, Generic, TYPE_CHECKING
)
import warnings

if TYPE_CHECKING:
    from .array import Array

# there is no EllipsisType unfortunately, but this helps the reader at least
EllipsisType = TypeVar('EllipsisType')
T = TypeVar("T")

class LocArrayIndexer(Generic[T]):
    """
        `Array.loc` allows indexing by coordinate position.
        It supports both positional and name look up (via dict).
        In order to support the indexing operator `__getitem__` for coordinate (label)
        instead of integer (index) look up (e.g. `arr.loc[1:3]`), we need this indexable helper
        class to be returned by the property `loc`.
    """
    _arr: "Array"

    def __init__(self, arr: "Array") -> None:
        self._arr = arr

    def __getitem__(
            self,
            item: Union[
                int, slice, EllipsisType,
                Tuple[Union[int, slice, EllipsisType],...],
                Dict[str, Union[int, slice]],
            ]
        ) -> "Array":
        """This method is called when indexing an Array instance by label,
            i.e., by using the coordinate value via Array.loc[].
            It supports dimensional lookup via position and name.
            The indexing behaviour is mainly defined to match the one of
            `xarray.DataArray` with the major difference that we always keep
            all dimensions.
            The indexing is performed in the current state of the Array,
            i.e., each dimension is indexed in its respective state (pos or freq).
            When the indexing actually changes the returned Array by removing some values,
            its internal state will always have all fft factors applied.

            Example usage:
            arr_2d = (
                fa.coords_from_dim(x_dim, "pos")
                + fa.coords_from_dim(y_dim, "pos")
            )
            Four ways of retrieving an Array object
            with coordinate 3 along x and coordinates values
            between the dimension min (either pos_min or freq_min)
            and 5 along y:

            arr_2d.loc[{"x": 3, "y": slice(None, 5)}]
            arr_2d.loc[3,:5]
            arr_2d.loc[3][:,:5] # don't use, just for explaining functionality
            arr_2d.loc[:,:5][3] # don't use, vjust for explaining functionality

        Parameters
        ----------
        item : Union[ int, slice, EllipsisType, Tuple[Union[int, slice, EllipsisType],...], Mapping[str, Union[int, slice]], ]
            An indexer object with dimension lookup method either
            via position or name. When using positional lookup, the order
            of the dimensions in the Array object is used (Array.dims).
            Per dimension, each indexer can be supplied as an integer or a slice.
            Array-like indexers are not supported as in the general case,
            the resulting coordinates cannot be expressed as a valid Dimension.
        Array
            A new Array with the same dimensions as this Array,
            except each dimension and the Array values are indexed.
            The resulting Array still fully supports FFTs.
        """

        if item is Ellipsis:
            return self._arr

        if isinstance(item, dict):
            return self._arr.sel(item) # type: ignore

        if not isinstance(item, tuple):
            item = (item,)

        tuple_indexers = parse_tuple_indexer_to_dims(
            item,
            n_dims=len(self._arr.dims)
        )

        integer_indexers: List[Union[int, slice]] = []
        for index, dim, space in zip(
            tuple_indexers, self._arr.dims, self._arr.spaces, strict=True
        ):
            integer_indexers.append(
                dim.index_from_coord(
                    coord=index,
                    space=space,
                    method=None,
                )
            )

        return self._arr.__getitem__(
            tuple(integer_indexers)
        )

def parse_tuple_indexer_to_dims(
    tuple_indexers: Tuple[Union[float, slice, EllipsisType], ...],
    n_dims: int,
) -> Tuple[Union[float, slice], ...]:
    """Return full tuple of indexers matching the length given by n_dims.
    Handles special case of Ellipsis as one of the indexers in which case
    it fills up the missing dimensions in the place of the Ellipsis.
    The missing dimensions are filled up with slice(None, None),
    i.e., no indexing along that dimension.
    """

    if tuple_indexers.count(Ellipsis) > 1:
        raise IndexError("positional indexing only supports a single ellipsis ('...')")

    if len(tuple_indexers) > n_dims:
        raise IndexError(
            "too many indices for Array: Array is "
            f"{n_dims}-dimensional, but {len(tuple_indexers)} were indexed."
        )

    # Case of length 1 tuple_indexers with only Ellipsis is already
    # handled before in Array indexing logic, therefore ignored here
    full_tuple_indexers: Tuple[Union[float, slice], ...]

    # An ellipsis in positional indexing can be used to fill up all missing
    # dimensions in the place where the ellipsis is put.
    if Ellipsis in tuple_indexers:
        index_ellipsis = tuple_indexers.index(Ellipsis)
        # The number of missing dimensions (without counting the ellipsis)
        missing_dim_indexers = n_dims - len(tuple_indexers) + 1
        # Replace the ellipsis with slice(None, None) whereas we keep
        # the indexers before and after the ellipsis
        full_tuple_indexers = (
                tuple_indexers[:index_ellipsis] # type: ignore
                + (slice(None, None),) * missing_dim_indexers
                + tuple_indexers[index_ellipsis+1:]
        )
    else:
        # Just fill up all non-mentioned dimensions with slice(None, None)
        full_tuple_indexers = (
            tuple_indexers # type: ignore
            + (slice(None, None),) * (n_dims-len(tuple_indexers))
        )

    return full_tuple_indexers

def check_missing_dim_names(
    indexer_names: Iterable[str],
    dim_names: Tuple[str, ...],
    missing_dims: Literal["raise", "warn", "ignore"],
) -> None:
    """Check for indexers with a dimension name that does not appear in the Array.
    Depending on the choice of missing_dims,
    either raises a ValueError, throws a warning or just ignores.
    These three different choices for how to handle missing dimensions are
    inspired by xarray and can be set by the user on calling Array.sel or isel.
    Other invalidities are handled elsewhere.
    """

    if missing_dims not in ["raise", "warn", "ignore"]:
        raise ValueError(
            f"missing_dims={missing_dims} is not valid, it has to be "
            + "one of the following: 'raise', 'warn', 'ignore'"
        )

    # Check for indexer names that don't exist in the indexed Array
    invalid_indexers = [indexer for indexer in indexer_names if indexer not in dim_names]

    if len(invalid_indexers) > 0:
        if missing_dims == "raise":
            raise ValueError(
                f"Dimensions {invalid_indexers} do not exist. "
                + f"Expected one or more of {dim_names}"
            )
        elif missing_dims == "warn":
            warnings.warn(
                f"Dimensions {invalid_indexers} do not exist. "
                + "These selections will be ignored",
                stacklevel=2,
            )

def tuple_indexers_from_mapping(
    indexers: Dict[str, Union[int, slice]],
    dim_names: Iterable[str],
) -> Tuple[Union[int, slice], ...]:
    """Return full tuple of indexers matching the order given by dim_names.
    In case of missing indexers for a specific dimension, fill up with
    slice(None, None), i.e., no indexing along that dimension.
    """

    tuple_indexers: List[Union[int, slice]] = []
    for dim_name in dim_names:
        if dim_name in indexers:
            tuple_indexers.append(indexers[dim_name])
        else:
            tuple_indexers.append(slice(None, None))
    return tuple(tuple_indexers)

def tuple_indexers_from_dict_or_tuple(
    indexers: Union[
        int, slice,
        Tuple[Union[int, slice, EllipsisType], ...],
        Dict[str, Union[int, slice]],
    ],
    dim_names: Tuple[str, ...],
) -> Tuple[Union[int, slice], ...]:
    """Take indexers in either dict or tuple format and sort these
    either by name in the order of the supplied dim_names (dict case)
    or fill up to full tuple of indexers matching the length of supplied
    dim_names. Also handles special cases with Ellipsis as part of the
    positional indexing.
    Raises ValueError or IndexError in case of invalid indexers.
    """

    full_tuple_indexers: Tuple[Union[int, slice], ...]

    if isinstance(indexers, dict):
        # Here, we check for invalid indexers and always throw a ValueError if
        # we find some. This case applies when indexing via [] or .loc[]
        invalid_indexers = [indexer for indexer in indexers if indexer not in dim_names]
        if len(invalid_indexers) > 0:
            raise ValueError(
                f"Dimensions {invalid_indexers} do not exist. "
                + f"Expected one or more of {dim_names}"
            )
        # Fill up indexers to match dimensions of indexed Array
        full_tuple_indexers = tuple_indexers_from_mapping(
            indexers,
            dim_names=dim_names,
        )
    else:
        # Handle case of positional indexing where we fill up the indexers
        # to match the dimensionality of the indexed Array
        tuple_indexers: Tuple[Union[int, slice, EllipsisType], ...]
        if not isinstance(indexers, tuple):
            tuple_indexers = (indexers,)
        else:
            tuple_indexers = indexers

        full_tuple_indexers = parse_tuple_indexer_to_dims(
            tuple_indexers, # type: ignore
            len(dim_names)
        )

    return full_tuple_indexers

def check_substepping(_slice: slice):
    """We do not support substepping, i.e., a slicing step != 1.
    For explanation why not, see error message below.
    We also do not support negative step -1 which would invert the order
    of the array, which does not make sense for us with Dimension.
    """
    if not(_slice.step is None or _slice.step == 1):
        raise IndexError(
            f"You can't index using {_slice} but only " +
            f"slice({_slice.start}, {_slice.stop}) with implicit index step 1. " +
            "Substepping requires reducing the respective other space " +
            "which is not well defined due to the arbitrary choice of " +
            "which part of the space to keep (constant min, middle or max?). "
        )

def remap_index_check_int(
    index: Optional[int],
    dim_n: int,
    index_kind: Literal["start", "stop"],
) -> int:
    # Special support for case of slice(None, None)
    if index is None:
        if index_kind == "start":
            return 0
        else:
            return dim_n
    # Catch invalid index objects, here everything that is not an integer
    if not isinstance(index, int):
        raise IndexError("only integers, slices (`:`), ellipsis (`...`) are valid indices.")
    # Special case of slice objects smaller than length of dimension (assume 8)
    # then slice(-20,None) would be mapped to slice(0, None) here
    if index < -dim_n:
        return 0
    # Handle case of negative indices that start with -1 at
    # last index and end with -dim_n for first value.
    if index < 0:
        return index + dim_n
    # Special case of slice objects bigger than length of dimension,
    # e.g., slice(None,20) would be mapped to slice(None, 8) with
    # dimension length 8.
    if index >= dim_n:
        return dim_n
    return index
