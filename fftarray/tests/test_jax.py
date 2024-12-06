from jax import config
import jax.numpy as jnp
from jax import random
config.update("jax_enable_x64", True)


def test_jax_fp64():
    x = random.uniform(random.PRNGKey(0), (10,), dtype=jnp.float64)
    assert x.dtype == jnp.float64
