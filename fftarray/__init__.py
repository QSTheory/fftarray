"""
.. rubric:: The objects inside the tables can be imported directly from :py:mod:`fftarray`:

Provided by :py:mod:`fftarray.fft_array`:

.. currentmodule:: fftarray.fft_array

.. autosummary::
   :nosignatures:

   FFTDimension

Provided by :py:mod:`fftarray.tools`:

.. currentmodule:: fftarray.tools

.. autosummary::
   :nosignatures:

   shift_frequency
   shift_position

Example:

.. code-block:: python

   >>> from fftarray import FFTDimension

"""
from .fft_array import FFTDimension, FFTArray
from .tools import shift_frequency, shift_position
from .fft_constraint_solver import round_up_to_next_power_of_two
