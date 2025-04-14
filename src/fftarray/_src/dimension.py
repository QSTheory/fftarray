from __future__ import annotations
from typing import Optional, Union, Literal, Any
from typing_extensions import assert_never
from dataclasses import dataclass

import numpy as np
from .helpers import norm_xp
from .formatting import dim_table, format_n
from .indexing import check_substepping, remap_index_check_int

from .space import Space


def dim(
        name: str,
        n: int,
        d_pos: float,
        pos_min: float,
        freq_min: float,
        *,
        dynamically_traced_coords: bool = True,
    ) -> Dimension:
    """Initialize a :class:`Dimension`.

    Parameters
    ----------
    name : str
        Name of the dimension.
    n : int
        Number of grid points.
    d_pos : float
        Distance between two neighboring position grid points.
    pos_min : float
        Smallest position grid point.
    freq_min : float
        Smallest frequency grid point
    dynamically_traced_coords : bool, optional
        Only relevant for use with JAX tracing. Whether the coordinate values
        should be dynamically traced such that the grid can be altered inside a
        jitted function, by default True

    Returns
    -------
    Dimension
        Initialized Dimension.

    See Also
    --------
    Dimension
    """

    return Dimension(
        name=name,
        n=n,
        d_pos=d_pos,
        pos_min=pos_min,
        freq_min=freq_min,
        dynamically_traced_coords=dynamically_traced_coords,
    )

@dataclass
class Dimension:
    """Properties of an Array grid for one dimension.

    This class encapsulates all the properties of the position and frequency
    coordinate grids for one dimension.

    Note that properties associated with the position grid are denoted by `pos`,
    whereas the frequency grid properties are denoted with `freq`.

    A minimal set of parameters that uniquely define the grid is selected to
    initialize the Dimension. Based on these, all other grid properties are
    computed while taking into account the constraints of the discrete Fourier
    Transform. The complete set of constraints is given by::

        pos_extent = pos_max - pos_min pos_middle = 0.5 * (pos_min + pos_max +
        d_pos) d_pos = pos_extent/(n-1)

        freq_extent = freq_max - freq_min freq_middle = 0.5 * (freq_max +
        freq_min + d_freq) d_freq = freq_extent/(n-1)

        d_freq * d_pos * n = 2*pi

    Parameters
    ----------
    name : str
        Name of the dimension.
    n : int
        Number of grid points.
    d_pos : float
        Distance between two neighboring position grid points.
    pos_min : float
        Smallest position grid point.
    freq_min : float
        Smallest frequency grid point
    dynamically_traced_coords : bool
        Only relevant for use with JAX tracing. Whether the coordinate values
        should be dynamically traced such that the grid can be altered inside a
        jitted function

    Notes
    -----
    **Implementation details**

    The grid in both spaces (position and frequency) goes from min to max
    including both points. Therefore ``d_pos = (pos_max-pos_min)/(n-1)``. The
    grid always consists of an even number of points. Therefore, the number of
    samples n has to be an even integer. The frequencies in frequency space can
    be acquired via ``numpy.fft.fftfreq(n, d_pos)``. These frequencies are
    spatial frequencies in the unit cycles/m. The wavelength lambda is the space
    equivalent of T for time signals. => ``lambda = 1/numpy.fft.fftfreq(n,
    d_pos)`` According to DeBroglie we have ``lambda = h/p`` => ``p = h *
    numpy.fft.fftfreq(n, d_pos)``

    The pos_middle is the sample on the right hand side of the exact center of
    the grid.

    **Examples**::

        n = 4
                        pos_middle
             pos_min           pos_max
                |-----|-----|-----|
        index:  0     1     2     3
                 d_pos d_pos d_pos

        n = 5
                        pos_middle
             pos_min                 pos_max
                |-----|-----|-----|-----|
        index:  0     1     2     3     4
                 d_pos d_pos d_pos d_pos

    The freq_middle is the sample on the right hand side of the exact center of
    the grid.

    **Examples**::

        n = 4
                          freq_middle
             freq_min             freq_max
                |------|------|------|
        index:  0      1      2      3
                 d_freq d_freq d_freq

        n = 6

             freq_min           freq_middle     freq_max
                |------|------|------|------|------|
        index:  0      1      2      3      4      5
                 d_freq d_freq d_freq d_freq d_freq

    .. highlight:: none

    These are the symbolic definitions of all the different names (for even
    ``n``)::

        pos_extent = pos_max - pos_min
        pos_middle = 0.5 * (pos_min + pos_max + d_pos)
        d_pos = pos_extent/(n-1)

        freq_extent = freq_max - freq_min
        freq_middle = 0.5 * (freq_max + freq_min + d_freq)
        d_freq = freq_extent/(n-1)

        d_freq * d_pos * n = 2*pi

    For odd ``n`` the definitions for ``pos_middle`` and ``freq_middle`` change
    to ensure that they and the minimum and maximum position and frequency are
    actually sampled and not in between two samples.::

        pos_middle = 0.5 * (pos_min + pos_max)
        freq_middle = 0.5 * (freq_max + freq_min)

    For performance reasons it is recommended to have ``n`` be a power of two.

    Individual array coordinates::

        pos = np.arange(0, n) * d_pos + pos_min
        freq = np.fft.fftfreq(n = n, d = d_pos) + freq_middle

    .. highlight:: none

    These arrays fulfill the following properties::

        np.max(pos) = pos_max
        np.min(pos) = pos_min
        np.max(freq) = freq_max
        np.min(freq) = freq_min

        pos[1]-pos[0] = d_pos (if n >= 2)

    See Also
    --------
    fftarray.dim
    """

    _pos_min: float
    _freq_min: float
    _d_pos: float
    _n: int
    _name: str
    _dynamically_traced_coords: bool

    def __init__(
            self,
            name: str,
            n: int,
            d_pos: float,
            pos_min: float,
            freq_min: float,
            dynamically_traced_coords: bool,
        ):
        self._name = name
        self._n = n
        self._d_pos = d_pos
        self._pos_min = pos_min
        self._freq_min = freq_min
        self._dynamically_traced_coords = dynamically_traced_coords

    def __repr__(self: Dimension) -> str:
        arg_str = ", ".join(
            [f"{name[1:]}={repr(value)}"
                for name, value in self.__dict__.items()]
        )
        return f"Dimension({arg_str})"

    def __str__(self: Dimension) -> str:
        n_str = format_n(self.n)
        str_out = f"<fftarray.Dimension (name={repr(self.name)})>\n"
        str_out += f"n={n_str}\n"
        str_out += dim_table(self)
        return str_out

    @property
    def n(self: Dimension) -> int:
        """..

        Returns
        -------
        float
            The number of grid points.
        """
        return self._n

    @property
    def name(self: Dimension) -> str:
        """..

        Returns
        -------
        float
            The name of his Dimension.
        """
        return self._name

    # ---------------------------- Position Space ---------------------------- #

    @property
    def d_pos(self: Dimension) -> float:
        """..

        Returns
        -------
        float
            The distance between two neighboring position grid points.
        """
        return self._d_pos

    @property
    def pos_min(self: Dimension) -> float:
        """..

        Returns
        -------
        float
            The smallest position grid point.
        """
        return self._pos_min

    @property
    def pos_max(self: Dimension) -> float:
        """..

        Returns
        -------
        float
            The largest position grid point.
        """
        return (self.n - 1) * self.d_pos + self.pos_min

    @property
    def pos_middle(self: Dimension) -> float:
        """..

        Returns
        -------
        float
            The middle of the position grid.
            If n is even, it is defined as the (n/2+1)'th position grid point.
        """
        return self.pos_min + self.n//2 * self.d_pos

    @property
    def pos_extent(self: Dimension) -> float:
        """..

        Returns
        -------
        float
            The length of the position grid.
            It is defined as `pos_max - pos_min`.
        """
        return (self.n - 1) * self.d_pos

    def _dim_from_slice(
            self,
            range: slice,
            space: Space,
        ) -> Dimension:
        """
            Get a new Dimension for a interval selection in a given space.
            Does not support steps!=1.

            Indexing behaviour is the same as for a numpy array with the
            difference that we raise an IndexError if the resulting size
            is not at least 1. We require n>=1 to create a valid Dimension.
        """

        # Catch invalid slice objects with range.step != 1
        check_substepping(range)

        start = remap_index_check_int(range.start, self.n, index_kind="start")
        end = remap_index_check_int(range.stop, self.n, index_kind="stop")

        n = end - start
        # Check validity of slice object which has to
        # yield at least one dimension value
        if n < 1:
            raise IndexError(
                f"Your indexing {range} is not valid. To create a valid "
                + "Dimension, the stop index must be bigger than the start "
                + "index in order to keep at least one sample (n>=1)."
            )

        return self._dim_from_start_and_n(start=start, n=n, space=space)

    def _dim_from_start_and_n(
            self,
            start: int,
            n: int,
            space: Space,
        ) -> Dimension:
        """Returns new Dimension instance starting at a specific value
        in either pos or freq space and setting variable dimension length.
        """

        match space:
            case "pos":
                pos_min = self.pos_min + start*self.d_pos
                freq_min = self.freq_min
                d_pos = self.d_pos
            case "freq":
                pos_min = self.pos_min
                freq_min = self.freq_min + start*self.d_freq
                d_pos = 1./(self.d_freq*n)
            case _:
                assert_never(space)

        return Dimension(
            name=self.name,
            n=n,
            pos_min=pos_min,
            freq_min=freq_min,
            d_pos=d_pos,
            dynamically_traced_coords=self._dynamically_traced_coords,
        )

    def index_from_coord(
            self,
            coord: Union[float, slice],
            space: Space,
            *,
            method: Optional[Literal["nearest", "pad", "ffill", "backfill", "bfill"]] = None,
        ) -> Union[int, slice]:
        """
        For the Dimension, retrieve the index corresponding to a given
        coordinate in a specified space.

        Parameters
        ----------
        coord : Union[float, slice]
            The coordinate or range of coordinates for which to find the index.
            If a slice is provided, the function will handle it accordingly.
        space : Space
            The space in which the coordinate is defined. It can be either "pos" or "freq".
        method : Optional[Literal["nearest", "pad", "ffill", "backfill", "bfill"]], optional
            The method to use for finding the index when the exact coordinate is not found.
            Supported methods are:
            - "nearest": Find the index representing the nearest coordinate.
            - "pad" or "ffill": Forward fill to the next smallest index.
            - "backfill" or "bfill": Backward fill to the next highest index.
            Default is None, which requires an exact match.

        Returns
        -------
        Union[int, slice]
            The index or range of indices corresponding to the given coordinate(s).
            If a float coord is provided, the function returns an integer index.
            If a slice object is provided, the function returns a slice object.

        Raises
        ------
        NotImplementedError
            If a method is provided for a slice object.
        KeyError
            If no exact index is found and method is None, or
            if the coordinate is out of bounds for the specified method.
        ValueError
            If an unsupported method is specified.
        """
        # The first part handles coords supplied as slice object whereas
        # it prepares those and distributes the actual work to the second
        # part of this function which handles scalar objects
        if isinstance(coord, slice):
            check_substepping(coord)
            if method is not None:
                # This catches slices supplied to Array.sel or isel with
                # a method != None (e.g. nearest) which is not supported
                raise NotImplementedError(
                    f"Retrieving the index from coord with method `{method}` "
                    f"is only supported for scalars, not slice objects: {coord}."
                )
            # Handle slice objects with start or end being None whereas
            # we substitute those with the Dimension bounds
            if coord.start is None:
                coord_start = getattr(self, f"{space}_min")
            else:
                coord_start = coord.start
            if coord.stop is None:
                coord_stop = getattr(self, f"{space}_max")
            else:
                coord_stop = coord.stop

            # Use the scalar part of this function with the methods bfill and ffill
            # to yield indices to include the respective coordinates
            idx_min: int = self.index_from_coord(coord_start, method="bfill", space=space) # type: ignore
            idx_max: int = self.index_from_coord(coord_stop, method="ffill", space=space) # type: ignore
            return slice(
                idx_min,
                idx_max + 1 # as slice.stop is non-inclusive, add 1
            )
        else:
            # Calculate the float index regarding the Dimension as
            # an infinite grid
            if space == "pos":
                raw_idx = (coord - self.pos_min) / self.d_pos
            else:
                raw_idx = (coord - self.freq_min) / self.d_freq

            # Clamp float index to the valid range of 0 to n-1
            clamped_index = min(
                max(0, raw_idx),
                self.n - 1
            )

            # Handle different methods case by case here
            if method is None:
                # We round the raw float indices here and check whether they
                # match their rounded int-like value, if not we throw a KeyError
                if (round(raw_idx) != raw_idx or clamped_index != raw_idx):
                    raise KeyError(
                        f"No exact index found for {coord} in {space}-space of dim " +
                        f'"{self.name}". Try the keyword argument ' +
                        'method="nearest".'
                    )
                final_idx = raw_idx
            elif  method == "nearest":
                # The combination of floor and +0.5 prevents the "ties to even" rounding of floating point numbers.
                # We only need one branch since our indices are always positive.
                final_idx = np.floor(clamped_index + 0.5)
            elif method in ["bfill", "backfill"]:
                # We propagate towards the next highest index and then check
                # its validity by checking if it's smaller or equal than
                # the dimension length n
                final_idx = np.ceil(clamped_index)
                if raw_idx > self.n - 1:
                    raise KeyError(
                        f"Coord {coord} not found with method '{method}', "
                        + "you could try one of the following instead: "
                        + "'ffill', 'pad' or 'nearest'."
                    )
            elif method in ["ffill", "pad"]:
                # We propagate back to the next smalles index and then check
                # its validity by checking if it's at least 0
                final_idx = np.floor(clamped_index)
                if raw_idx < 0:
                    raise KeyError(
                        f"Coord {coord} not found with method '{method}', "
                        + "you could try one of the following instead: "
                        + "'bfill', 'backfill' or 'nearest'."
                    )
            else:
                raise ValueError(f"Specified unsupported look-up method `{method}`.")

            # Transform index to integer here. We can do this because we
            # ensured validity in the cases above, especially for method = None
            return int(final_idx)

    # ---------------------------- Frequency Space --------------------------- #

    @property
    def d_freq(self: Dimension) -> float:
        """..

        Returns
        -------
        float
            The distance between frequency grid points.
        """
        return 1./(self.n*self.d_pos)

    @property
    def freq_min(self: Dimension) -> float:
        """..

        Returns
        -------
        float
            The smallest frequency grid point.

        """
        return self._freq_min

    @property
    def freq_middle(self: Dimension) -> float:
        """..

        Returns
        -------
        float
            The middle of the frequency grid.
            If n is even, it is defined as the (n/2+1)'th frequency grid point.
        """
        return self.freq_min + self.n//2 * self.d_freq

    @property
    def freq_max(self: Dimension) -> float:
        """..

        Returns
        -------
        float
            The largest frequency grid point.
        """
        return (self.n - 1) * self.d_freq + self.freq_min

    @property
    def freq_extent(self: Dimension) -> float:
        """..

        Returns
        -------
        float
            The length of the frequency grid.
            It is defined as `freq_max - freq_min`.
        """
        return (self.n - 1) * self.d_freq

    def values(
            self: Dimension,
            space: Space,
            /,
            *,
            xp: Optional[Any] = None,
            dtype: Optional[Any] = None,
            device: Optional[Any] = None,
        ):
        """Returns the Dimension values for the respective space.

        Parameters
        ----------
        self : Dimension
            Dimension providing the parameters.
        space : Space
            The space for which the values are returned.
        xp : Optional[Any], optional
            The Array API namespace to use for the returned values. If it is
            None, the default namespace from ``get_default_xp()`` is used.
        dtype : Optional[Any], optional
            The dtype to use for the returned values. If it is None, the
            default real floating point dtype of the determined ``xp`` is used.

        Returns
        -------
        Any
            The Dimension's values.
        """

        xp = norm_xp(xp_arg=xp)

        if dtype is not None and not xp.isdtype(dtype, "real floating"):
            raise ValueError(
                "Coordinates can only have a real-valued floating point dtype. " \
                f"Passed in {dtype}."
            )

        indices = xp.arange(
            0.,
            self.n,
            dtype=dtype,
            device=device,
        )

        match space:
            case "pos":
                # The explicit asarray call is necessary for numpy < 2.0 due to
                # its upcasting rules regarding multiplication with scalars.
                return xp.asarray(indices * self.d_pos + self.pos_min, dtype=dtype)
            case "freq":
                return xp.asarray(indices * self.d_freq + self.freq_min, dtype=dtype)
            case _:
                assert_never(space)

