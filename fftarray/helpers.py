from typing import Iterable, TypeVar, Generic, Any, List
import numpy as np
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


def format_bytes(bytes) -> str:
    """Converts bytes to KiB, MiB, GiB and TiB."""
    step_unit = 1024
    for x in ["bytes", "KiB", "MiB", "GiB"]:
        if bytes < step_unit:
            return f"{bytes:3.1f} {x}"
        bytes /= step_unit
    return f"{bytes:3.1f} TiB"

def format_n(n: int) -> str:
    """Get string representation of an integer.
    Returns 2^m if n is powert of two (m=log_2(n)).
    Uses scientific notation if n is larger than 1e6.
    """
    if (n & (n-1) == 0) and n != 0:
        # n is power of 2
        return f"2^{int(np.log2(n))}"
    if n >= 10000:
        # scientific notation
        return f"{n:.2e}"
    return f"{n:n}"

def truncate_str(string: str, width: int) -> str:
    """Truncates string that is longer than width."""
    if len(string) > width:
        string = string[:width-3] + '...'
    return string
