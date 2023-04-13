from typing import Iterable, TypeVar, Generic, Any
from functools import reduce


T = TypeVar("T")

#------------
# Helpers to reduce objects that should be same for all elements of a list
#------------
def reduce_equal(objects: Iterable[T], error_msg: str) -> T:
    """Reduce the Iterable to a single instance while checking the assumption
    that all objects are the same.
    """
    def join_equal(a, b):
        if a == b:
            return a
        raise ValueError(error_msg)
    return reduce(join_equal, objects)


class UniformValue(Generic[T]):
    """The idea of this class is that we often have the pattern that some
    property needs to be the same (but in principal arbitrary) value
    or None in all elements of a list.
    Since it could be None we cannot just take the first element of this list
    to get the value that all should adhere to.
    Therefore this class takes on the value first set
    and then checks on all consecutive sets that the value remained the same.

    In essence this allows the same reduction as "_reduce_equal" but when
    running through a loop.
    """
    
    is_set: bool
    value: Any

    def __init__(self)-> None:
        self.is_set = False

    @property
    def val(self) -> T:
        if self.is_set is False:
            raise ValueError("Value has never ben set.")
        return self.value

    @val.setter
    def val(self, value: T):
        if self.is_set:
            assert self.value == value
        self.value = value
        self.is_set = True
        