import os
import inspect
import fftarray as fa

RST_FILE_TEMPLATE = """{name}
{underline}

.. currentmodule:: fftarray

.. autoclass:: {name}

    .. rubric:: Methods

    .. autosummary::
        :toctree: generated/

        {methods}

    .. rubric:: Attributes

    .. autosummary::
        :toctree: generated/

        {attributes}
"""
for cls, name in [[fa.Array, "Array"], [fa.Dimension, "Dimension"]]:

    members = inspect.getmembers(cls)

    methods = [name for name, obj in members if inspect.isroutine(obj) and not name.startswith('_')]

    attributes = [name for name, obj in members if not inspect.isroutine(obj) and not name.startswith('_')]

    methods_str = f"\t{name}.{methods[0]}\n"
    for m in methods[1:]:
        methods_str += f"\t\t\t{name}.{m}\n"

    attributes_str = f"\t{name}.{attributes[0]}\n"
    for a in attributes[1:]:
        attributes_str += f"\t\t\t{name}.{a}\n"

    cls_rst_file = RST_FILE_TEMPLATE.format(
        name=name,
        underline="="*len(name),
        methods=methods_str,
        attributes=attributes_str
    )

    with open(f"source/api/{name.lower()}.rst", "w") as index_file:
        index_file.write(cls_rst_file)