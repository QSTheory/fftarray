from typing import Tuple, Any, List

from .array import Array, Space
from .dimension import Dimension

def array_flatten(
    arr: Array
) -> Tuple[
        Tuple[Any, Tuple[Dimension, ...]],
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

def array_unflatten(aux_data, children) -> Array:
    (values, dims) = children
    (spaces, eager, factors_applied, xp) = aux_data
    # We explicitly do not want to call the constructor here.
    # The consistency check fails (needlessly) for PyTreeArrays and other special "tricks".
    self = Array.__new__(Array)
    self._values = values
    self._dims = dims
    self._spaces = spaces
    self._eager = eager
    self._factors_applied = factors_applied
    self._xp = xp
    return self

def dimension_flatten(v: Dimension) -> Tuple[List[Any], List[Any]]:
        """The `flatten_func` used by `jax.tree_util.register_pytree_node` to
        flatten a Dimension.

        :meta private:

        Parameters
        ----------
        v : Dimension
            The Dimension to flatten.

        Returns
        -------
        Tuple[List[Any], List[Any]]
            The flatted Dimension. Contains ``children`` and ``aux_data``.

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


def dimension_unflatten(aux_data, children) -> Dimension:
    """The `unflatten_func` used by `jax.tree_util.register_pytree_node` to
    unflatten a Dimension.

    :meta private:

    Parameters
    ----------
    aux_data : list
        Auxiliary data.
    children : list
        Flattened children.

    Returns
    -------
    Dimension
        The unflattened Dimension.

    See Also
    --------
    jax.tree_util.register_pytree_node
    """
    # the last element of aux_data is the dynamically_traced_coords flag
    if aux_data[-1]:
        # dynamically traced, _pos_min, _freq_min, _d_pos in children
        dim = Dimension.__new__(Dimension)
        dim._name = aux_data[0]
        dim._n = aux_data[1]
        dim._pos_min = children[0]
        dim._freq_min = children[1]
        dim._d_pos = children[2]
        dim._dynamically_traced_coords = aux_data[2]
        return dim
    # static, everything in aux_data
    dim = Dimension.__new__(Dimension)
    dim._name = aux_data[0]
    dim._n = aux_data[1]
    dim._pos_min = aux_data[2]
    dim._freq_min = aux_data[3]
    dim._d_pos = aux_data[4]
    dim._dynamically_traced_coords = aux_data[5]
    return dim


def jax_register_pytree_nodes():
    from jax.tree_util import register_pytree_node
    register_pytree_node(
        Array,
        array_flatten,
        array_unflatten,
    )

    register_pytree_node(
        Dimension,
        dimension_flatten,
        dimension_unflatten,
    )
