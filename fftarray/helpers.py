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
        else:
            raise ValueError(error_msg)
    return reduce(join_equal, objects)

class UniformValue(Generic[T]):
    """Allows the same reduction as "_reduce_equal" but when running through a 
    loop.
    """
    is_set: bool
    value: Any

    def __init__(self)-> None:
        self.is_set = False

    @property
    def val(self) -> T:
        if not self.is_set:
            raise ValueError("Value has never ben set.")
        else:
            return self.value

    @val.setter
    def val(self, value: T):
        if self.is_set:
            assert self.value == value
        else:
            self.value = value
        self.is_set = True