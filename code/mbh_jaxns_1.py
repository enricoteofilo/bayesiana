import os
from pathlib import Path
# Force JAX to ignore TPU/GPU backends in this environment.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
jnp = jax.numpy
jsp = jax.scipy
random = jax.random
grad = jax.grad
jit = jax.jit
vmap = jax.vmap
from jaxns import Prior, Model, NestedSampler
import tensorflow_probability.substrates.jax as tfp

tfpd = tfp.distributions

DEBUG = True


def import_bh_data(fname: str) -> dict:

    structured = np.genfromtxt(
        fname,
        names=True,
        dtype=None,
        encoding="utf-8",
        delimiter=",",
    )

    if DEBUG:
        print("Loading file with `np.genfromtxt`:\n", structured)
        print("Data types:", structured.dtype)
        print("Column names:", structured.dtype.names)

    columns = {}
    for name in structured.dtype.names:
        key = name.lstrip("#")
        values = structured[name]
        if np.issubdtype(values.dtype, np.number):
            columns[key] = jnp.asarray(values, dtype=jnp.float64)
        else:
            columns[key] = values.tolist()

    return columns


if __name__ == "__main__":
    bh_data = import_bh_data("data/bh_table_1.txt")
    print(f"Loaded columns: {list(bh_data.keys())}")
