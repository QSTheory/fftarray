from typing import List, Optional, Any, Tuple
from typing_extensions import assert_never

import numpy as np
from bokeh.plotting import figure, row, column, show
from bokeh.models import LinearColorMapper, PrintfTickFormatter

import fftarray as fa


# global plot parameters for bokeh
plt_width       = 370
plt_height      = 260
plt_line_width  = 2
plt_border      =  50
plt_color1      = "navy"
plt_color2      = "firebrick"
plt_color3      = "limegreen"
x_range1        =   (-15,15)


def plt_array(
        arr: fa.Array,
        data_name: Optional[str] = None,
        show_plot: bool = True,
    ):
    """ Plot the real and the imaginary part of a given Array with dim<=2 both in position and frequency space using global plot parameters.

    Parameters
    ----------
    arr : fa.Array
            The Array to be plotted.
    data_name : str, optional
                The title of the plot. Defaults to 'Array values' if no title is given.
    show_plot : bool, optional
                Boolean, defines whether figures are shown upon return or not.

    Returns
    -------
    if (show_plot):
        show(Bokeh row plot)
            Rendered plot of the real and the imaginary part of ``arr`` both in position and in frequency space using global plot parameters.
    else:
        [Bokeh row plot]
             List of non-rendered figures of the real and the imaginary part of ``arr`` both in position and in frequency space using global plot parameters.

    Raises
    ------
    NotImplementedError
        If not ``len(arr.dims) ==1 .OR. len(arr.dims) ==2``.

    Used In
    --------
    Gaussians.ipynb
    """
    if len(arr.dims) == 1:
        dim = arr.dims[0]
        p_pos = figure(width=plt_width, height=plt_height, x_axis_label = f"{dim.name} pos coordinate", min_border=plt_border)
        pos_values = arr.values("pos", xp=np)
        p_pos.line(dim.values("pos", xp=np), np.real(pos_values), line_width=plt_line_width, color = plt_color1, legend_label="real")
        p_pos.line(dim.values("pos", xp=np), np.imag(pos_values), line_width=plt_line_width, color = plt_color2, legend_label="imag")
        p_pos.title.text = f"{data_name or 'Array values'} shown in position space" # type: ignore
        p_pos.xaxis[0].formatter = PrintfTickFormatter(format="%.1e")
        p_pos.yaxis[0].formatter = PrintfTickFormatter(format="%.1e")

        p_freq = figure(width=plt_width, height=plt_height, x_axis_label = f"{dim.name} freq coordinate", min_border=plt_border)
        freq_values = arr.values("freq", xp=np)
        p_freq.line(dim.values("freq", xp=np), np.real(freq_values), line_width=plt_line_width, color = plt_color1, legend_label="real")
        p_freq.line(dim.values("freq", xp=np), np.imag(freq_values), line_width=plt_line_width, color = plt_color2, legend_label="imag")
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
                width=plt_width, height=plt_height, min_border=plt_border,
                x_range=tuple(getattr(arr.dims[0], f"{space}_{prop}") for prop in ["min", "max"]),
                y_range=tuple(getattr(arr.dims[1], f"{space}_{prop}") for prop in ["min", "max"]),
                x_axis_label = f"{dim_names[0]} {space} coordinate",
                y_axis_label = f"{dim_names[1]} {space} coordinate",
            )

            # Array values
            values_in_space = arr.values(space, xp=np)
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
        pos_range: Optional[Tuple[float, float]] = None,
        freq_range: Optional[Tuple[float, float]] = None,
    ):
    """Plot the one-dimensional values in space-time as an image.
    """
    plots = []
    for space, values, grid in [["pos", pos_values, pos_grid], ["freq", freq_values, freq_grid]]:
        color_mapper = LinearColorMapper(palette="Viridis256", low=np.min(values), high=np.max(values))
        match space:
            case "pos":
                unit = pos_unit
                variable = "x"
                if pos_range is None:
                    plt_range = (float(grid[0]), float(grid[-1]))
                else:
                    plt_range = pos_range
            case "freq":
                unit = freq_unit
                variable = "f"
                if freq_range is None:
                    plt_range = (float(grid[0]), float(grid[-1]))
                else:
                    plt_range = freq_range
            case _:
                assert_never(space)

        plot = figure(
            x_axis_label = "time [s]",
            y_axis_label = f"{space} coordinate [{unit}]",
            x_range=(float(time[0]), float(time[-1])),
            y_range=plt_range,
            width=plt_width,
            height=plt_height,
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

def plt_deriv_sampling(
        plt_title: str,
        arr1: fa.Array,
        arr2: fa.Array,
        arr3: fa.Array,
        show_plot: bool = True,
    ):
    """ Plot the given real-valued Arrays ``g(x), g'(x), g''(x)`` in position space using global plot parameters.

    Parameters
    ----------
    plt_title : str
                The title of the plot.
    arr1 : fa.Array
            The real-valued Arrays defined as ``g(x)`` in Derivatie.ipynb.
    arr2 : fa.Array
            The real-valued Arrays defined as ``g'(x)`` in Derivatie.ipynb.
    arr3 : fa.Array
            The real-valued Arrays defined as ``g''(x)`` in Derivatie.ipynb.
    show_plot : bool, optional
                Boolean that defines whether the plot is supposed to be displayed upon return or not.

    Returns
    -------
    if (show_plot):
        show(Bokeh row plot)
            Rendered plot of ``g(x), g'(x), g''(x)`` in position space using global plot parameters.
    else:
        [Bokeh row plot]
             List of non-rendered figures for ``g(x), g'(x), g''(x)`` in position space using global plot parameters.

    Used In
    --------
    Derivative.ipynb
    """
    # check compatibility of dimensions
    assert len(arr1.dims) == 1
    assert len(arr2.dims) == 1
    assert arr1.dims[0] == arr2.dims[0]== arr3.dims[0]

    dim = arr1.dims[0] # save Dimension information for plot labels
    plots = []
    p=figure(
        title=f"{plt_title} comparison",
        width=plt_width,
        height=plt_height,
        x_axis_label = f"{dim.name} pos coordinate",
        x_range=(x_range1),
    )
    p.line(
        x=dim.values("pos", xp=np),
        y=arr1.values("pos", xp=np).real,
        legend_label="g(x)",
        color=plt_color1,
        line_width=plt_line_width,
        line_dash="solid",
    )
    p.line(
        x=dim.values("pos", xp=np),
        y=arr2.values("pos", xp=np).real,
        legend_label="g'(x)",
        color=plt_color2,
        line_width=plt_line_width,
        line_dash="dashed"
    )
    p.line(
        x=dim.values("pos", xp=np),
        y=arr3.values("pos", xp=np),
        legend_label="g''(x)",
        color=plt_color3,
        line_width=plt_line_width,
        line_dash="dotted"
    )
    p.legend.click_policy="hide"
    plots.append(p)

    figs = row(plots)

    if show_plot:
        show(figs)
    else:
        return figs

def plt_deriv_comparison(
        plt_title: str,
        arr1: fa.Array,
        name1: str,
        arr2: fa.Array,
        name2: str,
        show_plot: bool = True,
    ):
    """Plot the real parts of the given Arrays ``arr1, arr2`` (figure #1: Comparison) and their residuals (figure #2: Residuals) in position space using global plot parameters.

    Parameters
    ----------
    plt_title : str
                The title of the plot.
    arr1 : fa.Array
            Array, the real part of which is to be compared to the one of ``arr2``.
    name1 : str
            Legend label of ``arr1``.
    arr2 : fa.Array
            Array, the real part of which is to be compared to the one of ``arr1``.
    name2 : str
            Legend label of ``arr2``.
    show_plot : bool, optional
                Boolean that defines whether the plot is supposed to be displayed upon return or not.

    Returns
    -------
    if (show_plot):
        show(Bokeh row plot)
            Two rendered figures in position space using global plot parameters: Comparison (figure #1) and Residuals (figure #2).
    else:
        [Bokeh row plot]
            List of two non-rendered figures in position space using global plot parameters: Comparison (figure #1) and Residuals (figure #2).

    Used In
    --------
    Derivative.ipynb
    """
    assert len(arr1.dims) == 1 # check compatibility of dimensions
    assert len(arr2.dims) == 1
    assert arr1.dims[0] == arr2.dims[0]

    dim = arr1.dims[0] # save Dimension information for plot labels
    plots = []
    p=figure(
        title=f"{plt_title} Comparison",
        width=plt_width,
        height=plt_height,
        x_axis_label = f"{dim.name} pos coordinate",
        x_range=(x_range1),
    )
    p.line(
        x=dim.values("pos", xp=np),
        y=arr1.values("pos", xp=np).real,
        legend_label=f"{name1}",
        color=plt_color1,
        line_width=plt_line_width,
        line_dash="solid",
    )
    p.line(
        x=dim.values("pos", xp=np),
        y=arr2.values("pos", xp=np).real,
        legend_label=f"{name2}",
        color=plt_color2,
        line_width=plt_line_width,
        line_dash="dashed"
    )
    p.legend.click_policy="hide"
    plots.append(p)

    # Plot residuals
    p=figure(
        title=f"{plt_title} Residuals",
        width=plt_width,
        height=plt_height,
        x_axis_label = f"{dim.name} pos coordinate",
        x_range=(x_range1),
    )
    p.line(
        x=dim.values("pos", xp=np),
        y=arr1.values("pos", xp=np).real-arr2.values("pos", xp=np).real,
        legend_label=f"{name1}-{name2}",
        color=plt_color1,
        line_width=plt_line_width,
        line_dash="solid",
    )
    plots.append(p)

    figs = row(plots)

    if show_plot:
        show(figs)
    else:
        return figs