from typing import Iterable, List, Union

import fftarray as fa



def get_dims(n: int) -> List[fa.FFTDimension]:
    return [
        fa.dim(str(i), n=4+i, d_pos=1.*(i+1.), pos_min=0., freq_min=0.)
        for i in range(n)
    ]

def get_arr_from_dims(
        xp,
        dims: Iterable[fa.FFTDimension],
        spaces: Union[fa.Space, Iterable[fa.Space]] = "pos",
    ):
    dims = list(dims)
    if isinstance(spaces, str):
        spaces_norm: Iterable[fa.Space] = [spaces]*len(dims)
    else:
        spaces_norm = spaces
    arr = fa.array(values=xp.asarray(1.), dims=[], space=[])
    for dim, space in zip(dims, spaces_norm):
        arr += fa.array_from_dim(dim=dim, space=space, xp=xp)
    return arr
