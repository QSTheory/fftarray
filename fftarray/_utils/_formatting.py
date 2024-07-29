from typing import Optional, get_args, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..fft_array import FFTArray, FFTDimension

def format_bytes(bytes) -> str:
    """Converts bytes to KiB, MiB, GiB and TiB."""
    step_unit = 1024
    for x in ["bytes", "KiB", "MiB", "GiB"]:
        if bytes < step_unit:
            return f"{bytes:3.1f} {x}"
        bytes /= step_unit
    return f"{bytes:3.1f} TiB"

def format_n(n: int) -> str:
    """Get string representation of an integer.
    Returns 2^m if n is powert of two (m=log_2(n)).
    Uses scientific notation if n is larger than 1e6.
    """
    if (n & (n-1) == 0) and n != 0:
        # n is power of 2
        return f"2^{int(np.log2(n))}"
    if n >= 10000:
        # scientific notation
        return f"{n:.2e}"
    return f"{n:n}"

def truncate_str(string: str, width: int) -> str:
    """Truncates string that is longer than width."""
    if len(string) > width:
        string = string[:width-3] + '...'
    return string

def fft_dim_table(
        dim: "FFTDimension",
        include_header=True,
        include_dim_name=False,
        dim_index: Optional[int] = None,
    ) -> str:
    """Constructs a table for FFTDimension.__str__ and FFTArrar.__str__
    containting the grid parameters for each space.
    """
    str_out = ""
    headers = ["space", "d", "min", "middle", "max", "extent"]
    if include_dim_name:
        headers.insert(0, "dimension")
    if include_header:
        if dim_index is not None:
            # handled separately to give it a smaller width
            str_out += "| # "
        for header in headers:
            # give space smaller width to stay below 80 characters per line
            str_out += f"|{header:^7}" if header == "space" else f"|{header:^10}"
        str_out += "|\n" + int(dim_index is not None)*"+---"
        for header in headers:
            str_out += "+" + (7*"-" if header == "space" else 10*"-")
        str_out += "+\n"
    dim_prop_headers = headers[int(include_dim_name)+1:]
    for k, space in enumerate(["pos", "freq"]):
        if dim_index is not None:
            str_out += f"|{dim_index:^3}" if k==0 else f"|{'':^3}"
        if include_dim_name:
            dim_name = str(dim.name)
            if len(dim_name) > 10:
                if k == 0:
                    str_out += f"|{dim_name[:10]}"
                else:
                    str_out += f"|{truncate_str(dim_name[10:], 10)}"
            else:
                str_out += f"|{dim_name:^10}" if k==0 else f"|{'':^10}"
        str_out += f"|{space:^7}|"
        for header in dim_prop_headers:
            attr = f"d_{space}" if header == "d" else f"{space}_{header}"
            nmbr = getattr(dim, attr)
            frmt_nmbr = f"{nmbr:.2e}" if abs(nmbr)>1e3 or abs(nmbr)<1e-2 else f"{nmbr:.2f}"
            str_out += f"{frmt_nmbr:^10}|"
        str_out += "\n"
    return str_out[:-1]

def fft_array_props_table(fftarr: "FFTArray") -> str:
    """Constructs a table for FFTArray.__str__ containing the FFTArray
    properties (space, n, eager, factors_applied) per dimension
    """
    str_out = "| # "
    headers = ["dimension", "space", "n", "eager", "factors_applied"]
    for header in headers:
        # give space smaller width to stay below 80 characters per line
        str_out += f"|{header:^7}" if header == "space" else f"|{header:^10}"
    str_out += "|\n+---"
    for header in headers:
        str_out += "+" + (10 + 5*int(header=='factors_applied') - 3*int(header=="space"))*"-"
    str_out += "+\n"
    for i, dim in enumerate(fftarr.dims):
        str_out += f"|{i:^3}"
        str_out += f"|{truncate_str(str(dim.name), 10):^10}"
        str_out += f"|{(fftarr.space[i]):^7}"
        str_out += f"|{format_n(dim.n):^10}"
        str_out += f"|{repr(fftarr.eager[i]):^10}"
        str_out += f"|{repr(fftarr._factors_applied[i]):^15}"
        str_out += "|\n"
    return str_out[:-1]
