from typing import Iterable, List, Literal, Union, get_args

import fftarray as fa

DTYPE_NAME = Literal[
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
    "complex64",
    "complex128",
]
dtypes_names_all = get_args(DTYPE_NAME)


def get_dims(n: int) -> List[fa.FFTDimension]:
    return [
        fa.dim(str(i), n=4+i, d_pos=1.*(i+1.), pos_min=0., freq_min=0.)
        for i in range(n)
    ]

def get_arr_from_dims(
        xp,
        dims: Iterable[fa.FFTDimension],
        spaces: Union[fa.Space, Iterable[fa.Space]] = "pos",
        dtype_name: DTYPE_NAME = "float64",
    ):
    dims = list(dims)
    if isinstance(spaces, str):
        spaces_norm: Iterable[fa.Space] = [spaces]*len(dims)
    else:
        spaces_norm = spaces
    arr = fa.array(
        values=xp.asarray(
            1.,
            dtype=getattr(xp, dtype_name)
        ),
        dims=[],
        space=[],
    )
    for dim, space in zip(dims, spaces_norm):
        arr += fa.array_from_dim(dim=dim, space=space, xp=xp)
    return arr
