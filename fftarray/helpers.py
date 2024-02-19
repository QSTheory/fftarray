from typing import Iterable, Literal, Tuple, TypeVar, Generic, Any, List, Union
from functools import reduce


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
    tuple_indexers: Tuple[Union[int, slice, EllipsisType]],
    n_dims: int,
) -> Tuple[Union[int, slice]]:

    if tuple_indexers.count(Ellipsis) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")

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
