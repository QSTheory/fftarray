from typing import Tuple, Any, List

from .fft_array import FFTArray, Space
from .fft_dimension import FFTDimension

def fftarray_flatten(
    arr: FFTArray
) -> Tuple[
        Tuple[Any, Tuple[FFTDimension, ...]],
        Tuple[
            Tuple[Space, ...],
            Tuple[bool, ...],
            Tuple[bool, ...],
            Any # Array namespace
        ]
    ]:
    children = (arr._values, arr._dims)
    aux_data = (arr._spaces, arr._eager, arr._factors_applied, arr._xp)
    return (children, aux_data)

def fftarray_unflatten(aux_data, children) -> FFTArray:
    (values, dims) = children
    (spaces, eager, factors_applied, xp) = aux_data
    # We explicitly do not want to call the constructor here.
    # The consistency check fails (needlessly) for PyTreeArrays and other special "tricks".
    self = FFTArray.__new__(FFTArray)
    self._values = values
    self._dims = dims
    self._spaces = spaces
    self._eager = eager
    self._factors_applied = factors_applied
    self._xp = xp
    return self

def fft_dimension_flatten(v: FFTDimension) -> Tuple[List[Any], List[Any]]:
        """The `flatten_func` used by `jax.tree_util.register_pytree_node` to
        flatten an FFTDimension.

        :meta private:

        Parameters
        ----------
        v : FFTDimension
            The FFTDimension to flatten.

        Returns
        -------
        Tuple[List[Any], List[Any]]
            The flatted FFTDimension. Contains ``children`` and ``aux_data``.

        See Also
        --------
        jax.tree_util.register_pytree_node
        """
        children: List[Any]
        aux_data: List[Any]
        if v._dynamically_traced_coords:
            # dynamically traced, write _pos_min, _freq_min and _d_pos into children
            children = [
                v._pos_min,
                v._freq_min,
                v._d_pos,
            ]
            aux_data = [
                v._name,
                v._n,
                v._dynamically_traced_coords,
            ]
            return (children, aux_data)
        # static, write everything into aux_data
        children = []
        aux_data = [
            v._name,
            v._n,
            v._pos_min,
            v._freq_min,
            v._d_pos,
            v._dynamically_traced_coords,
        ]
        return (children, aux_data)


def fft_dimension_unflatten(aux_data, children) -> FFTDimension:
    """The `unflatten_func` used by `jax.tree_util.register_pytree_node` to
    unflatten an FFTDimension.

    :meta private:

    Parameters
    ----------
    aux_data : list
        Auxiliary data.
    children : list
        Flattened children.

    Returns
    -------
    FFTDimension
        The unflattened FFTDimension.

    See Also
    --------
    jax.tree_util.register_pytree_node
    """
    # the last element of aux_data is the dynamically_traced_coords flag
    if aux_data[-1]:
        # dynamically traced, _pos_min, _freq_min, _d_pos in children
        fftdim = FFTDimension.__new__(FFTDimension)
        fftdim._name = aux_data[0]
        fftdim._n = aux_data[1]
        fftdim._pos_min = children[0]
        fftdim._freq_min = children[1]
        fftdim._d_pos = children[2]
        fftdim._dynamically_traced_coords = aux_data[2]
        return fftdim
    # static, everything in aux_data
    fftdim = FFTDimension.__new__(FFTDimension)
    fftdim._name = aux_data[0]
    fftdim._n = aux_data[1]
    fftdim._pos_min = aux_data[2]
    fftdim._freq_min = aux_data[3]
    fftdim._d_pos = aux_data[4]
    fftdim._dynamically_traced_coords = aux_data[5]
    return fftdim


def jax_register_pytree_nodes():
    from jax.tree_util import register_pytree_node
    register_pytree_node(
        FFTArray,
        fftarray_flatten,
        fftarray_unflatten,
    )

    register_pytree_node(
        FFTDimension,
        fft_dimension_flatten,
        fft_dimension_unflatten,
    )
