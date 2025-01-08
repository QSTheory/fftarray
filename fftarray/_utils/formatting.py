from typing import Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..dimension import Dimension
    from ..array import Space

def format_bytes(bytes) -> str:
    """Converts bytes to KiB, MiB, GiB and TiB."""
    step_unit = 1024
    if bytes < 1024:
        return f"{bytes} bytes"

    bytes /= step_unit
    for x in ["KiB", "MiB", "GiB"]:
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
        dim: "Dimension",
        include_header=True,
        include_dim_name=False,
        spaces: Tuple["Space", ...] = ("pos", "freq"),
    ) -> str:
    """Constructs a table for Dimension.__str__ and Array.__str__
    containing the grid parameters for each space.
    """
    str_out = ""
    headers = ["space", "d", "min", "middle", "max", "extent"]
    if include_dim_name:
        headers.insert(0, "dimension")
    if include_header:
        for header in headers:
            # give space smaller width to stay below 80 characters per line
            str_out += f"|{header:^7}" if header == "space" else f"|{header:^10}"
        str_out += "|\n"
        for header in headers:
            str_out += "+" + (7*"-" if header == "space" else 10*"-")
        str_out += "+\n"
    dim_prop_headers = headers[int(include_dim_name)+1:]
    if include_dim_name:
        # dim name column only shown when printing an Array
        # in this case the Dimension properties are only shown in the current space
        assert len(spaces) == 1
        dim_name = dim.name
        if len(dim_name) > 10:
            str_out += f"|{truncate_str(dim_name, 10)}"
        else:
            str_out += f"|{dim_name:^10}"
    for space in spaces:
        str_out += f"|{space:^7}|"
        for header in dim_prop_headers:
            attr = f"d_{space}" if header == "d" else f"{space}_{header}"
            nmbr = getattr(dim, attr)
            frmt_nmbr = f"{nmbr:.2e}" if abs(nmbr)>1e3 or abs(nmbr)<1e-2 else f"{nmbr:.2f}"
            str_out += f"{frmt_nmbr:^10}|"
        str_out += "\n"
    return str_out[:-1]
