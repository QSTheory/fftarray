from typing import (
    Dict, List, Tuple, TypeVar, Union, Iterable,
    Hashable, Literal, Literal
)
import warnings

# there is no EllipsisType unfortunately, but this helps the reader at least
EllipsisType = TypeVar('EllipsisType')

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
            "too many indices for FFTArray: FFTArray is "
            f"{n_dims}-dimensional, but {len(tuple_indexers)} were indexed."
        )

    # Case of length 1 tuple_indexers with only Ellipsis is already
    # handled before in FFTArray indexing logic, therefore ignored here
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

def check_invalid_indexers(
    indexer_names: Iterable[Hashable],
    dim_names: Tuple[Hashable, ...],
    missing_dims: Literal["raise", "warn", "ignore"],
) -> None:
    """Check for invalid indexers and depending on the choice of missing_dims,
    either raise a ValueError, throw a warning or just ignore.
    This method handles invalid indexers in the sense that these are not
    matching any FFTArray dimension name.
    The three different choices for how to handle missing dimensions are
    inspired by xarray and can be set by the user on calling FFTArray.sel or isel.
    Other invalidities are handled elsewhere.
    """

    if missing_dims not in ["raise", "warn", "ignore"]:
        raise ValueError(
            f"missing_dims={missing_dims} is not valid, it has to be "
            + "one of the following: 'raise', 'warn', 'ignore'"
        )

    # Check for indexer names that don't exist in the indexed FFTArray
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
                + "These selections will be ignored"
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
        # Fill up indexers to match dimensions of indexed FFTArray
        full_tuple_indexers = tuple_indexers_from_mapping(
            indexers,
            dim_names=dim_names,
        )
    else:
        # Handle case of positional indexing where we fill up the indexers
        # to match the dimensionality of the indexed FFTArray
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
    of the array, which does not make sense for us with FFTDimension.
    """
    if not(_slice.step is None or _slice.step == 1):
        raise IndexError(
            f"You can't index using {_slice} but only " +
            f"slice({_slice.start}, {_slice.stop}) with implicit index step 1. " +
            "Substepping requires reducing the respective other space " +
            "which is not well defined due to the arbitrary choice of " +
            "which part of the space to keep (constant min, middle or max?). "
        )
