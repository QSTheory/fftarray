from collections import abc
from typing import Hashable, Iterable, Literal, Mapping, Optional, Tuple, TypeVar, Generic, Any, List, Union
from functools import reduce
import warnings


T = TypeVar("T")

#------------
# Helpers to reduce objects that should be same for all elements of a list
#------------
def reduce_equal(objects: Iterable[T], error_msg: str) -> T:
    """
        Reduce the Iterable to a single instance while checking the assumption that all objects are the same.
    """
    def join_equal(a, b):
        if a == b:
            return a
        raise ValueError(error_msg)
    return reduce(join_equal, objects)


class UniformValue(Generic[T]):
    """
        Allows the same reduction as "_reduce_equal" but when running through a loop.
    """
    is_set: bool
    value: Any

    def __init__(self)-> None:
        self.is_set = False

    @property
    def val(self) -> T:
        if self.is_set is False:
            raise ValueError("Value has never ben set.")
        else:
            return self.value

    @val.setter
    def val(self, value: T):
        self.set(value)

    def set(self, value: T):
        if self.is_set:
            if not self.value == value:
                raise ValueError("Did not set value equal to previously set value.")
        else:
            self.value = value
        self.is_set = True

    def get(self, *args: T) -> T:
        # Only first arg is valid and could be a default argument.
        # Need this complicated capture to check if an arg was provided.
        # None is a valid default after all
        assert len(args) < 2
        if self.is_set:
            return self.value

        if len(args) == 1:
            return args[0]

        raise ValueError("Value has never been set.")

EllipsisType = TypeVar('EllipsisType', bound=type(Ellipsis))

def parse_tuple_indexer_to_dims(
    tuple_indexers: Tuple[Union[float, slice, EllipsisType]],
    n_dims: int,
) -> Tuple[Union[float, slice]]:

    # TODO: implement test
    if tuple_indexers.count(Ellipsis) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")

    # TODO: implement test
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
    indexers: Mapping[Hashable, Union[int, slice]],
    dim_names: Iterable[Hashable],
    user_call_method: str,
) -> Tuple[Union[int, slice]]:
    """
    Return complete tuple of valid indexers (slice or int).
    """

    # TODO: test if allowed_types with int leads to problem wit jax
    # types but I think not because we use jax.Array[0].item() or so
    allowed_types = (slice, int)

    tuple_indexers = []
    for dim_name in dim_names:
        if dim_name in indexers:
            index = indexers[dim_name]
            if not isinstance(index, allowed_types):
                raise KeyError(
                    f"Using FFTArray{user_call_method}, the index for "
                    + "each dimension has to be given as one of the "
                    + f"following: {allowed_types}."
                    + f"Your input for {dim_name}={index} of type "
                    + f"{type(index)} is not valid"
                )
            tuple_indexers.append(indexers[dim_name])
        else:
            tuple_indexers.append(slice(None, None, None))
    return tuple(tuple_indexers)

def tuple_indexers_from_dict_or_kwargs(
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
        # TODO: the following is maybe a weakness of the call logic, we
        # use user_call=[] and int allowed_types here although this method
        # is called differently in other methods (like sel or isel)
        # BUG: the below user_call_method is wrong and resulting error
        # message needs to be rethought, could also be FFTArray.loc[]
        tuple_indexers = tuple_indexers_from_mapping(
            indexers,
            dim_names=dim_names,
            user_call_method='[]',
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

# TODO: This method does not feel right here, we need an FFTDimension
# Either put this in fft_array.py again or include type checks in
# FFTDimension._index_from_coord maybe
def coord_as_integer(
    coord: Union[slice, float],
    dim,
    space,
    tlib,
    method,
):

    if isinstance(coord, slice):
        if not(coord.step is None or coord.step == 1):
            raise IndexError(
                f"You can't index using {coord} but only " +
                f"slice({coord.start}, {coord.stop}) with implicit index step 1. " +
                "Substepping requires reducing the respective other space " +
                "which is not well defined due to the arbitrary choice of " +
                "which part of the space to keep (constant min, middle or max?). "
            )
        indexer = (coord.start, coord.stop)
    else:
        indexer = coord

    return dim._index_from_coord(
        coord=indexer,
        space=space,
        tlib=tlib,
        method=method
    )
