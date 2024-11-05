"""
.. rubric:: The objects inside the tables can be imported directly from :py:mod:`fftarray`:

Provided by :py:mod:`fftarray.fft_array`:

.. currentmodule:: fftarray.fft_array

.. autosummary::
   :nosignatures:

   FFTArray
   FFTDimension

Provided by :py:mod:`fftarray.tools`:

.. currentmodule:: fftarray.tools

.. autosummary::
   :nosignatures:

   shift_frequency
   shift_position

Example:

.. code-block:: python

   >>> from fftarray import FFTArray, FFTDimension

"""
from .fft_dimension import FFTDimension
from .fft_array import FFTArray, Space
from ._utils.defaults import (
    set_default_backend, get_default_backend, set_default_eager,
    get_default_eager, default_backend, default_eager
)
from .tools import shift_frequency, shift_position
