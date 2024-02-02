import numpy as np

from fftarray.backends.jax_backend import JaxTensorLib
from fftarray import FFTDimension

from jax import config
config.update("jax_enable_x64", True)
