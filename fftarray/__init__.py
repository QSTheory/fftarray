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
    set_default_xp, get_default_xp, default_xp,
    set_default_dtype_name, get_default_dtype_name, default_dtype_name,
    set_default_eager, get_default_eager, default_eager,
)

from .space import Space
from .fft_dimension import FFTDimension, dim
from .fft_array import FFTArray

from .creation_functions import (
   array,
   array_from_dim,
   coords_array,
   full,
)
from .statistical_functions import sum

from .tools import shift_frequency, shift_position

from .jax_pytrees import jax_register_pytree_nodes

from .elementwise_functions import (
   abs,
   acos,
   acosh,
   add,
   asin,
   asinh,
   atan,
   atan2,
   atanh,
   bitwise_and,
   bitwise_left_shift,
   bitwise_invert,
   bitwise_or,
   bitwise_right_shift,
   bitwise_xor,
   ceil,
   clip,
   conj,
   copysign,
   cos,
   cosh,
   divide,
   equal,
   exp,
   expm1,
   floor,
   floor_divide,
   greater,
   greater_equal,
   hypot,
   imag,
   isfinite,
   isinf,
   isnan,
   less,
   less_equal,
   log,
   log1p,
   log2,
   log10,
   logaddexp,
   logical_and,
   logical_not,
   logical_or,
   logical_xor,
   maximum,
   minimum,
   multiply,
   negative,
   not_equal,
   positive,
   pow,
   real,
   remainder,
   round,
   sign,
   signbit,
   sin,
   sinh,
   square,
   sqrt,
   subtract,
   tan,
   tanh,
   trunc,
)

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




