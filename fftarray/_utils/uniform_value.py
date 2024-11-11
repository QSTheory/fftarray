from typing import Any, TypeVar, Generic

T = TypeVar("T")

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


