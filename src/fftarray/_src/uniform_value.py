from typing import Any, TypeVar, Generic

T = TypeVar("T")

class UniformValue(Generic[T]):
    """
        Allows to check that a set of values is equal while running through a loop.
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
        """Get the set value or the optionally passed in default.
        The default can only be the very first positional argument, but `None` is a valid default.

        Returns
        -------
        T
            The collected value or the default.

        Raises
        ------
        ValueError
            Raises a ValueError if the value has never been set.
        """
        # Only first arg is valid and could be a default argument.
        # Need this complicated capture to check if an arg was provided.
        # None is a valid default after all
        assert len(args) < 2
        if self.is_set:
            return self.value

        if len(args) == 1:
            return args[0]

        raise ValueError("Value has never been set.")


