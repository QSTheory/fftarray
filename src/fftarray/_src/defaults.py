import numpy as np

from .compat_namespace import get_compat_namespace
_DEFAULT_XP = get_compat_namespace(np)

def set_default_xp(xp) -> None:
    """Set the default value for the ``xp`` argument of array creation functions.
    If this is set within a context manager, it gets reset by the context manager
    on leaving to the value it had before the context manager was entered.

    Parameters
    ----------
    xp:
        Array API namespace to set as default.
        It gets automatically wrapped with ``array-api-compat``
        if necessary.
    """

    global _DEFAULT_XP
    # We want the wrapped namespace everywhere by default.
    # If the array library fully supports the Python Array API
    # this becomes the default namespace.
    _DEFAULT_XP= get_compat_namespace(xp)

def get_default_xp():
    """
    Returns
    -------
    Any
        Current default Array API namespace in the array creation functions.
    """
    return _DEFAULT_XP


class DefaultArrayNamespaceContext:
    """Context manager for overriding the default Array API namespace locally.

    On exit it automatically sets the default back to the value that was set when the context was entered.
    Any changes to the global default are then overridden.
    """
    def __init__(self, xp):
        self.override = xp

    def __enter__(self):
        self.previous = get_default_xp()
        set_default_xp(self.override)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_default_xp(self.previous)

def default_xp(xp) -> DefaultArrayNamespaceContext:
    """Create a context manager to override the default Array namespace locally."""
    return DefaultArrayNamespaceContext(xp=xp)


_DEFAULT_EAGER: bool = False

def set_default_eager(eager: bool) -> None:
    """Set the default value for the ``eager`` argument of array creation functions.
    If this is set within a context manager, it gets reset by the context manager
    on leaving to the value it had before the context manager was entered.

    Parameters
    ----------
    eager:
        Value to set as default for ``eager``.
    """
    global _DEFAULT_EAGER
    _DEFAULT_EAGER= eager

def get_default_eager() -> bool:
    """
    Returns
    -------
    bool
        Current default value for ``eager`` in the array creation functions.
    """
    return _DEFAULT_EAGER

class DefaultEagerContext:
    """Context manager for overriding the default value for ``eager`` locally.

    On exit it automatically sets the default back to the value that was set when the context was entered.
    Any changes to the global default are then overridden.
    """

    def __init__(self, eager: bool):
        self.override = eager

    def __enter__(self):
        self.previous = get_default_eager()
        set_default_eager(self.override)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_default_eager(self.previous)

def default_eager(eager: bool) -> DefaultEagerContext:
    """Create a context manager to override the default values for ``eager`` locally."""
    return DefaultEagerContext(eager=eager)

