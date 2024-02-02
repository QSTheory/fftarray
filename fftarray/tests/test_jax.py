import numpy as np

from fftarray.backends.jax_backend import JaxTensorLib, make_matched_input, fft_array_scan
from fftarray import FFTDimension

def test_input_matching() -> None:
    x_dim = FFTDimension("x", n=4, d_pos=1, pos_min=0., freq_min=0., default_tlib=JaxTensorLib())

    def f(carry, x):
        carry["a"] = carry["a"].freq_array()
        carry["c"][0] = carry["a"].pos_array()
        return carry, (1, x)

    init = {"a": x_dim.pos_array(), "b": x_dim.freq_array(), "c": [x_dim.freq_array(), x_dim.pos_array()]}
    matched_input_ref = {"a": x_dim.freq_array(), "b": x_dim.freq_array(), "c": [x_dim.pos_array(), x_dim.pos_array()]}
    matched_input = make_matched_input(f, init, 45)
    assert type(matched_input["a"]) == type(matched_input_ref["a"])
    assert type(matched_input["b"]) == type(matched_input_ref["b"])
    assert type(matched_input["c"][0]) == type(matched_input_ref["c"][0]) # type: ignore
    assert type(matched_input["c"][1]) == type(matched_input_ref["c"][1]) # type: ignore


def test_fft_array_scan() -> None:
    state = 0
    def inner_fun(state, i):
        return state+i, None

    def outer_fun(state, i):
        state, _ = fft_array_scan(f=inner_fun, init=state, xs=np.array([5,6]))
        return state + i, None

    state, _ = fft_array_scan(f=outer_fun, init=state, xs=np.array([1,2]))
