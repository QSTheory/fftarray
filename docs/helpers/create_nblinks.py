import os
import glob

# source/examples.rst includes all .rst files inside source/examples/
EXAMPLES_DIR = "source/examples/"

# create source/examples if it does not exist yet
os.system(f"mkdir -p {EXAMPLES_DIR}")

# get all jupyter notebooks inside ../examples
examples = glob.glob("../examples/*.ipynb")
example_filenames = sorted([e.split("/")[-1].split(".")[0] for e in examples])

for example_filename in example_filenames:
    out_file_name = example_filename + ".nblink"
    with open(EXAMPLES_DIR+out_file_name, "w") as nblink_file:
        nblink_file.write('{\n\t"path": "../../../examples/'+example_filename+'.ipynb",\n\t"extra-media": ["../../../examples/helpers.py"]\n}')

INDEX_FILE_HEADER = """Examples
========

.. toctree::
	:maxdepth: 1

.. nbgallery::
   :titlesonly:

"""

with open(EXAMPLES_DIR + "index.rst", "w") as index_file:
    index_file_content = INDEX_FILE_HEADER
    for example_filename in example_filenames:
        index_file_content += f"   {example_filename}\n"
    index_file.write(index_file_content)
