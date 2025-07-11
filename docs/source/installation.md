# Installation

The required dependencies of FFTArray are kept small to ensure compatibility with many different environments.
For most use cases we recommend installing the optional constraint solver with the `z3` option:
```shell
pip install fftarray[z3]
```

Any array library besides NumPy like for example [JAX](https://github.com/jax-ml/jax?tab=readme-ov-file#installation) should be installed following their respective documentation.
Since each of them have different approaches on how to handle for example GPU support on different operating systems we do not recommend installing them via the optional dependency groups of FFTArray.
