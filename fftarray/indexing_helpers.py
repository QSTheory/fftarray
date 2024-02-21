from typing import (
    Dict, List, Tuple, TypeVar, Union, Iterable,
    Hashable, Literal, Literal
)
import warnings

EllipsisType = TypeVar('EllipsisType')

def parse_tuple_indexer_to_dims(
    tuple_indexers: Tuple[Union[float, slice, EllipsisType], ...],
    n_dims: int,
) -> Tuple[Union[float, slice], ...]:

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

    if Ellipsis in tuple_indexers:
        index_ellipsis = tuple_indexers.index(Ellipsis)
        missing_dim_indexers = n_dims - len(tuple_indexers) + 1
        full_tuple_indexers = (
                tuple_indexers[:index_ellipsis] # type: ignore
                + (slice(None, None, None),) * missing_dim_indexers
                + tuple_indexers[index_ellipsis+1:]
        )
    else:
        full_tuple_indexers = (
            tuple_indexers # type: ignore
            + (slice(None, None, None),) * (n_dims-len(tuple_indexers))
        )

    return full_tuple_indexers

def check_invalid_indexers(
    indexer_names: Iterable[Hashable],
    dim_names: Tuple[Hashable, ...],
    missing_dims: Literal["raise", "warn", "ignore"],
) -> None:

    if missing_dims not in ["raise", "warn", "ignore"]:
        raise ValueError(
            f"missing_dims={missing_dims} is not valid, it has to be "
            + "one of the following: 'raise', 'warn', 'ignore'"
        )

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
    """
    Return complete tuple of indexers (slice or float).
    """

    tuple_indexers: List[Union[int, slice]] = []
    for dim_name in dim_names:
        if dim_name in indexers:
            tuple_indexers.append(indexers[dim_name])
        else:
            tuple_indexers.append(slice(None, None, None))
    return tuple(tuple_indexers)

def tuple_indexers_from_dict_or_tuple(
    indexers: Union[
        int, slice,
        Tuple[Union[int, slice, EllipsisType], ...],
        Dict[str, Union[int, slice]],
    ],
    dim_names: Tuple[str, ...],
) -> Tuple[Union[int, slice], ...]:

    full_tuple_indexers: Tuple[Union[int, slice], ...]

    if isinstance(indexers, dict):
        invalid_indexers = [indexer for indexer in indexers if indexer not in dim_names]
        if len(invalid_indexers) > 0:
            raise ValueError(
                f"Dimensions {invalid_indexers} do not exist. "
                + f"Expected one or more of {dim_names}"
            )
        full_tuple_indexers = tuple_indexers_from_mapping(
            indexers,
            dim_names=dim_names,
        )
    else:
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
    if not(_slice.step is None or _slice.step == 1):
        raise IndexError(
            f"You can't index using {_slice} but only " +
            f"slice({_slice.start}, {_slice.stop}) with implicit index step 1. " +
            "Substepping requires reducing the respective other space " +
            "which is not well defined due to the arbitrary choice of " +
            "which part of the space to keep (constant min, middle or max?). "
        )
