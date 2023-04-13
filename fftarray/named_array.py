from typing import Sequence, Tuple, Hashable, Any, List, Dict
from dataclasses import dataclass

#-------------------
# TODO This is copied from abstraction but then quite significantly modified
#-------------------
def align_named_arrays(
        arrays: Sequence[Tuple[Sequence[Hashable], Any]],
        tlib
    ) -> Tuple[Sequence[Hashable], List[Any]]:
    """The arrays may have longer shapes than there are named dims.
    Those are always kept as the last dims.
    Reorders and expands dimensions so that all arrays have the same dim-names
    and shapes.

    Unnamed shapes may differ!
    This allows aligning all named dimensions of differently typed trees.

    Returns the new dim-names and the list of aligned arrays.
    """
    target_shape: Dict[Hashable, int] = {}
    for dims, arr in arrays:
        for i, dim in enumerate(dims):
            if dim in target_shape:
                assert target_shape[dim] == arr.shape[i], \
                    "Cannot align arrays with different lengths "+ \
                    f"({target_shape[dim]}, {arr.shape[i]}) in the same dim {dim}"
            else:
                target_shape[dim] = arr.shape[i]

    target_indices = {name: i for i, name in enumerate(target_shape.keys())}
    aligned_arrays = []
    for dims, arr in arrays:
        dim_names = [*dims]
        for target_dim, target_length in target_shape.items():
            if not target_dim in dims:
                arr = arr.reshape(-1, *arr.shape)
                dim_names.insert(0, target_dim)
        # TODO the list conversion of keys should not be necessary but is needed for mypy
        arr = transpose_array(
            arr,
            old_dims=dim_names,
            new_dims=list(target_shape.keys()),
            tlib=tlib
        )
        aligned_arrays.append(arr)
    return list(target_indices.keys()), aligned_arrays


@dataclass
class FillDim:

    index: int

    def __hash__(self):
        return hash(self.index)


def transpose_array(
        array: Any,
        tlib,
        old_dims: Sequence[Hashable],
        new_dims: Sequence[Hashable]
    ) -> Any:
    """`old_dims` and `new_dims` must be a transpose of one another.
    They may be shorter than array.shape. The last dims are left untouched.
    """
    assert len(old_dims) == len(new_dims)
    dim_index_lut = {dim: i for i, dim in enumerate(old_dims)}

    # Allow for unnamed axes at the end.
    # Used for transposing PyTreeArray-leafs.
    for i in range(len(old_dims), len(array.shape)):
        dim_index_lut[FillDim(i)] = i

    axes_transpose = [dim_index_lut[target_dim] for target_dim in new_dims]
    axes_transpose = [*axes_transpose, *range(len(axes_transpose), len(array.shape))]
    array = tlib.numpy_ufuncs.transpose(array, tuple(axes_transpose))
    return array
