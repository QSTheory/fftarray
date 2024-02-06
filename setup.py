import setuptools
import subprocess

version = (
    subprocess.run(["git", "tag", "--points-at", "HEAD"], stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()
)
if version == "":
    version = (
        subprocess.run(["git", "describe", "--tags", "--abbrev=0"], stdout=subprocess.PIPE)
        .stdout.decode("utf-8")
        .strip()
    )

assert "." in version

setuptools.setup(
    name="fftarray",
    version=version,
    author="Stefan Seckmeyer, Gabriel Müller, Christian Struckmann",
    author_email="",
    description="A library to manage and effiently implement all complexities around the complex Fast Fourier Transform.",
    long_description="",
    long_description_content_type="text/plain",
    url="",
    packages=["fftarray"],
    include_package_data=True,
    package_data={"fftarray": ["py.typed"]},
    zip_safe=False,
    classifiers=[],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "jax>=0.4.2",
        "jaxlib",
        "z3-solver",
    ],
    extras_require={
        "dev": [
            "numpy>=1.24",
            "xarray",
            "mypy>=0.910",
            "pytest",
            "hypothesis",
            "sphinx>=4.2",
            "sphinx_rtd_theme",
            "myst_parser",
            "mistune==0.8.4",
            "m2r2",
            "ipython",
            "nbformat",
            "bokeh",
            "pyFFTW",
        ],
        "examples": [
            "bokeh",
            "ipython",
        ],
    }
)
