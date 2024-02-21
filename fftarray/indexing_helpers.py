from typing import (
    Tuple, TypeVar, Union, Optional, Iterable,
    Mapping, Hashable, Literal, Literal
)
from collections import abc
import warnings

EllipsisType = TypeVar('EllipsisType', bound=type(Ellipsis))

def parse_tuple_indexer_to_dims(
    tuple_indexers: Tuple[Union[float, slice, EllipsisType]],
    n_dims: int,
) -> Tuple[Union[float, slice]]:

    if tuple_indexers.count(Ellipsis) > 1:
        raise IndexError("positional indexing only supports a single ellipsis ('...')")

    if len(tuple_indexers) > n_dims:
        raise IndexError(
            "too many indices for FFTArray: FFTArray is "
            f"{n_dims}-dimensional, but {len(tuple_indexers)} were indexed."
        )

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

    return tuple_indexers

def check_invalid_indexers(
    indexers: Optional[Mapping[Hashable, Union[int, slice]]],
    dim_names: Tuple[Hashable],
    missing_dims: Literal["raise", "warn", "ignore"],
) -> None:

    if missing_dims not in ["raise", "warn", "ignore"]:
        raise ValueError(
            f"missing_dims={missing_dims} is not valid, it has to be "
            + "one of the following: 'raise', 'warn', 'ignore'"
        )

    invalid_indexers = [indexer for indexer in indexers if indexer not in dim_names]

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
    indexers: Mapping[Hashable, Union[float, slice]],
    dim_names: Iterable[Hashable],
) -> Tuple[Union[int, slice]]:
    """
    Return complete tuple of indexers (slice or float).
    """

    tuple_indexers = []
    for dim_name in dim_names:
        if dim_name in indexers:
            tuple_indexers.append(indexers[dim_name])
        else:
            tuple_indexers.append(slice(None, None, None))
    return tuple(tuple_indexers)

def tuple_indexers_from_dict_or_tuple(
    indexers: Union[
        int, slice, Tuple[Union[int, slice, EllipsisType],...],
        Mapping[Hashable, Union[int, slice]],
    ],
    dim_names: Tuple[Hashable],
) -> Tuple[Union[int, slice]]:

    tuple_indexers: Tuple[Union[int, slice]]

    if isinstance(indexers, abc.Mapping):
        invalid_indexers = [indexer for indexer in indexers if indexer not in dim_names]
        if len(invalid_indexers) > 0:
            raise ValueError(
                f"Dimensions {invalid_indexers} do not exist. "
                + f"Expected one or more of {dim_names}"
            )
        tuple_indexers = tuple_indexers_from_mapping(
            indexers,
            dim_names=dim_names,
        )
    else:
        if not isinstance(indexers, tuple):
            tuple_indexers = (indexers,)
        else:
            tuple_indexers = indexers

        tuple_indexers = parse_tuple_indexer_to_dims(
            tuple_indexers,
            len(dim_names)
        )

    return tuple_indexers

def check_substepping(_slice: slice):
    if not(_slice.step is None or _slice.step == 1):
        raise IndexError(
            f"You can't index using {_slice} but only " +
            f"slice({_slice.start}, {_slice.stop}) with implicit index step 1. " +
            "Substepping requires reducing the respective other space " +
            "which is not well defined due to the arbitrary choice of " +
            "which part of the space to keep (constant min, middle or max?). "
        )
