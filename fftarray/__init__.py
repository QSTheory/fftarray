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

from ._utils.defaults import (
    set_default_backend, get_default_backend, set_default_eager,
    get_default_eager, default_backend, default_eager
)
from .space import Space
from .fft_dimension import FFTDimension, dim
from .fft_array import FFTArray

from .creation_functions import array_from_dim

from .tools import shift_frequency, shift_position


from typing import Optional, Literal, Union, List
try:
   from .constraint_solver import get_fft_dim_from_constraints as dim_from_constraints
except ModuleNotFoundError:
   def dim_from_constraints(
         name: str,
         *,
         n: Union[int, Literal["power_of_two", "even"]] = "power_of_two",
         d_pos: Optional[float] = None,
         d_freq: Optional[float] = None,
         pos_min: Optional[float] = None,
         pos_max: Optional[float] = None,
         pos_middle: Optional[float] = None,
         pos_extent: Optional[float] = None,
         freq_min: Optional[float] = None,
         freq_max: Optional[float] = None,
         freq_extent: Optional[float] = None,
         freq_middle: Optional[float] = None,
         loose_params: Optional[Union[str, List[str]]] = None,
         dynamically_traced_coords: bool = True,
    ) -> FFTDimension:
      raise ModuleNotFoundError("You need to install `fftarray[helpers]` to use the constraint solver.")




