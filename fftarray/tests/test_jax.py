import numpy as np

from fftarray.backends.jax_backend import JaxTensorLib
from fftarray import FFTDimension

from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random

def test_jax_fp64():
    x = random.uniform(random.PRNGKey(0), (1000,), dtype=jnp.float64)
    assert x.dtype == jnp.float64
