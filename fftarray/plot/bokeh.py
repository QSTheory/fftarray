import numpy as np
from bokeh.plotting import figure, row

from ..fft_array import FFTArray

def plt_fft(arr: FFTArray):
    if len(arr.dims) == 1:
        dim = arr.dims[0]
        p_pos = figure(width=400, height=400, x_axis_label = f"{dim.name} pos coordinate")
        p_pos.line(np.array(dim.pos_array()), np.real(arr.pos_array().values), line_width=2, color = "navy", legend_label="real")
        p_pos.line(np.array(dim.pos_array()), np.imag(arr.pos_array().values), line_width=2, color = "firebrick", legend_label="imag")

        p_freq = figure(width=400, height=400, x_axis_label = f"{dim.name} freq coordinate")
        p_freq.line(np.array(dim.freq_array()), np.real(arr.freq_array().values), line_width=2, color = "navy", legend_label="real")
        p_freq.line(np.array(dim.freq_array()), np.imag(arr.freq_array().values), line_width=2, color = "firebrick", legend_label="imag")

        return row([p_pos, p_freq]) # type: ignore
    else:
        raise NotImplementedError