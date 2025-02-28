from typing import List, Optional, Any
from typing_extensions import assert_never

import numpy as np
from bokeh.plotting import figure, row, column, show
from bokeh.models import LinearColorMapper, PrintfTickFormatter

import fftarray as fa

def plt_array(
        arr: fa.Array,
        data_name: Optional[str] = None,
        show_plot: bool = True,
    ):
    if len(arr.dims) == 1:
        dim = arr.dims[0]
        p_pos = figure(width=450, height=400, x_axis_label = f"{dim.name} pos coordinate", min_border=50)
        pos_values = arr.values("pos")
        p_pos.line(dim.np_array("pos"), np.real(pos_values), line_width=2, color = "navy", legend_label="real")
        p_pos.line(dim.np_array("pos"), np.imag(pos_values), line_width=2, color = "firebrick", legend_label="imag")
        p_pos.title.text = f"{data_name or 'Array values'} shown in position space" # type: ignore
        p_pos.xaxis[0].formatter = PrintfTickFormatter(format="%.1e")
        p_pos.yaxis[0].formatter = PrintfTickFormatter(format="%.1e")

        p_freq = figure(width=450, height=400, x_axis_label = f"{dim.name} freq coordinate", min_border=50)
        freq_values = arr.values("freq")
        p_freq.line(dim.np_array("freq"), np.real(freq_values), line_width=2, color = "navy", legend_label="real")
        p_freq.line(dim.np_array("freq"), np.imag(freq_values), line_width=2, color = "firebrick", legend_label="imag")
        p_freq.title.text = f"{data_name or 'Array values'} shown in frequency space" # type: ignore
        p_freq.xaxis[0].formatter = PrintfTickFormatter(format="%.1e")
        p_freq.yaxis[0].formatter = PrintfTickFormatter(format="%.1e")

        plot = row([p_pos, p_freq], sizing_mode="stretch_width") # type: ignore
    elif len(arr.dims) == 2:
        row_plots = []
        spaces: List[fa.Space] = ["pos", "freq"]
        for space in spaces:
            # Dimension properties
            dim_names = [dim.name for dim in arr.dims]

            fig_props = dict(
                width=450, height=400, min_border=50,
                x_range=tuple(getattr(arr.dims[0], f"{space}_{prop}") for prop in ["min", "max"]),
                y_range=tuple(getattr(arr.dims[1], f"{space}_{prop}") for prop in ["min", "max"]),
                x_axis_label = f"{dim_names[0]} {space} coordinate",
                y_axis_label = f"{dim_names[1]} {space} coordinate",
            )

            # Array values
            values_in_space = arr.np_array(space)
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
                dw=getattr(arr.dims[0], f"{space}_extent"),
                dh=getattr(arr.dims[1], f"{space}_extent"),
                x=getattr(arr.dims[0], f"{space}_min"),
                y=getattr(arr.dims[1], f"{space}_min"),
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
            colorbar.formatter = PrintfTickFormatter(format="%.1e")

            for fig in [fig_real_part, fig_imag_part]:

                fig.add_layout(colorbar, "right")
                fig.xaxis[0].formatter = PrintfTickFormatter(format="%.1e")
                fig.yaxis[0].formatter = PrintfTickFormatter(format="%.1e")

            space_name = "position" if space == "pos" else "frequency"
            fig_real_part.title.text = f"Real part of {data_name or 'Array values'} shown in {space_name} space" # type: ignore
            fig_imag_part.title.text = f"Imaginary part of {data_name or 'Array values'} shown in {space_name} space" # type: ignore

            row_plots.append(column(fig_real_part, fig_imag_part))

        plot = row(row_plots, sizing_mode="stretch_width") # type: ignore
    else:
        raise NotImplementedError

    if show_plot:
        show(plot)
    else:
        return plot

def plt_array_values_space_time(
        pos_values: Any,
        freq_values: Any,
        pos_grid: Any,
        freq_grid: Any,
        time: Any,
        pos_unit: str = "m",
        freq_unit: str = "1/m",
    ):
    """Plot the one-dimensional values in space-time as a image.
    """
    plots = []
    for space, values, grid in [["pos", pos_values, pos_grid], ["freq", freq_values, freq_grid]]:
        color_mapper = LinearColorMapper(palette="Viridis256", low=np.min(values), high=np.max(values))
        match space:
            case "pos":
                unit = pos_unit
                variable = "x"
            case "freq":
                unit = freq_unit
                variable = "f"
            case _:
                assert_never(space)

        plot = figure(
            x_axis_label = "time [s]",
            y_axis_label = f"{space} coordinate [{unit}]",
            x_range=(float(time[0]), float(time[-1])),
            y_range=(float(grid[0]), float(grid[-1]))
        )
        r = plot.image(
            image=[np.transpose(values)],
            x = time[0],
            y = grid[0],
            dw = time[-1] - time[0],
            dh = grid[-1] - grid[0],
            color_mapper=color_mapper
        )
        color_bar = r.construct_color_bar(padding=1)
        color_bar.formatter = PrintfTickFormatter(format="%.1e")
        plot.add_layout(color_bar, "right")
        plot.xaxis[0].formatter = PrintfTickFormatter(format="%.1e")
        plot.yaxis[0].formatter = PrintfTickFormatter(format="%.1e")
        plot.title.text = fr"$$|\Psi({variable})|^2$$ in {space} space" # type: ignore
        plots.append(plot)

    row_plot = row(plots, sizing_mode="stretch_width")
    show(row_plot)
