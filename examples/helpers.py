from typing import List, Optional

import numpy as np
from bokeh.plotting import figure, row, column, show
from bokeh.palettes import Turbo256
from bokeh.models import LinearColorMapper

from fftarray import FFTArray, Space

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
    elif len(arr.dims) == 2:
        row_plots = []
        spaces: List[Space] = ["pos", "freq"]
        for space in spaces:
            # Dimension properties
            dim_names = [dim.name for dim in arr.dims]
            dim_1_coord_values, dim_2_coord_values = tuple(np.array(dim.fft_array(space=space)) for dim in arr.dims)

            x_range = (np.min(dim_1_coord_values), np.max(dim_1_coord_values))
            y_range = (np.min(dim_2_coord_values), np.max(dim_2_coord_values))

            fig_props = dict(
                width=450, height=400, min_border=50,
                x_range=x_range,
                y_range=y_range,
                x_axis_label = f"{dim_names[0]} {space} coordinate",
                y_axis_label = f"{dim_names[1]} {space} coordinate",
            )

            # FFTArray values
            values_in_space = np.array(arr.into(space=space))
            values_imag_part = values_in_space.imag
            values_real_part = values_in_space.real

            color_map_min = min(np.min(values_imag_part), np.min(values_real_part))
            color_map_high = max(np.max(values_imag_part), np.max(values_real_part))

            if color_map_min == color_map_high:
                color_map_min = color_map_min
                color_map_high = color_map_min + 1

            color_mapper = LinearColorMapper(
                palette="Turbo256",
                low=color_map_min,
                high=color_map_high,
            )

            image_props = dict(
                color_mapper=color_mapper,
                dh=y_range[1]-y_range[0],
                dw=x_range[1]-x_range[0],
                x=x_range[0],
                y=y_range[0]
            )

            # Create bokeh density plots (real and imaginary part)
            fig_real_part = figure(
                **fig_props
            )
            fig_imag_part = figure(
                **fig_props
            )

            image_real_part = fig_real_part.image(
                image=[values_real_part],
                **image_props
            )

            fig_imag_part.image(
                image=[values_imag_part],
                **image_props
            )
            colorbar = image_real_part.construct_color_bar()

            fig_real_part.add_layout(colorbar, "right")
            fig_imag_part.add_layout(colorbar, "right")

            fig_real_part.title.text = f"Real part of {data_name or 'FFTArray values'} shown in position space" # type: ignore
            fig_imag_part.title.text = f"Imaginary part of {data_name or 'FFTArray values'} shown in position space" # type: ignore

            row_plots.append(column(fig_real_part, fig_imag_part))

        plot = row(row_plots, sizing_mode="stretch_width") # type: ignore
    else:
        raise NotImplementedError

    if show_plot:
        show(plot)
    else:
        return plot