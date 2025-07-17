Setting defaults
================

FFTArray has two defaults to influence the :mod:`Array creation functions <fftarray._src.creation_functions>`.
They can either be set globally or overridden locally with a context manager.
Both defaults are always last priority in all creation functions.
If their value is directly given or can be inferred from the input parameters those take precedence.

Eager
#####

.. autofunction:: fftarray.get_default_eager
.. autofunction:: fftarray.set_default_eager
.. autofunction:: fftarray.default_eager
.. autoclass:: fftarray._src.defaults.DefaultEagerContext

Array Namespace
###############

.. autofunction:: fftarray.get_default_xp
.. autofunction:: fftarray.set_default_xp
.. autofunction:: fftarray.default_xp
.. autoclass:: fftarray._src.defaults.DefaultArrayNamespaceContext
