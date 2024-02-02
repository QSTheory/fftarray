from typing import Optional

import numpy as np
from bokeh.plotting import figure, row, show

from fftarray import FFTArray

def plt_fftarray(
        arr: FFTArray,
        data_name: Optional[str] = None,
        show_plot: bool = True,
    ):
    if len(arr.dims) == 1:
        dim = arr.dims[0]
        p_pos = figure(width=450, height=400, x_axis_label = f"{dim.name} pos coordinate", min_border=50)
        p_pos.line(np.array(dim.fft_array(space="pos")), np.real(arr.into(space="pos").values), line_width=2, color = "navy", legend_label="real")
        p_pos.line(np.array(dim.fft_array(space="pos")), np.imag(arr.into(space="pos").values), line_width=2, color = "firebrick", legend_label="imag")
        p_pos.title.text = f"{data_name or 'FFTArray values'} shown in position space" # type: ignore

        p_freq = figure(width=450, height=400, x_axis_label = f"{dim.name} freq coordinate", min_border=50)
        p_freq.line(np.array(dim.fft_array(space="freq")), np.real(arr.into(space="freq").values), line_width=2, color = "navy", legend_label="real")
        p_freq.line(np.array(dim.fft_array(space="freq")), np.imag(arr.into(space="freq").values), line_width=2, color = "firebrick", legend_label="imag")
        p_freq.title.text = f"{data_name or 'FFTArray values'} shown in frequency space" # type: ignore

        plot = row([p_pos, p_freq], sizing_mode="stretch_width") # type: ignore
    else:
        raise NotImplementedError

    if show_plot:
        show(plot)
    else:
        return plot