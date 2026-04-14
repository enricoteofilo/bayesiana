import os
from dataclasses import dataclass
from functools import partial

# Keep computations in FP64 and on CPU for reproducibility.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
from quadax import quadcc

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

@partial(jax.jit, static_argnames=("xmin","xmax","ymin","ymax",))
def x_bounds_scalars(a,b,xmin,xmax,ymin,ymax):
    safe_a = jnp.where(a == 0.0, 1.0, a)
    # When a > 0
    low_pos = jnp.maximum(xmin, (ymin - b) / safe_a)
    high_pos = jnp.minimum(xmax, (ymax - b) / safe_a)
    # When a < 0
    low_neg = jnp.maximum(xmin, (ymax - b) / safe_a)
    high_neg = jnp.minimum(xmax, (ymin - b) / safe_a)
    # When a = 0
    low_zero = xmin
    high_zero = xmax
    # Estimates the bounds distinguishing among the cases
    low = jnp.where(a > 0.0, low_pos, jnp.where(a < 0.0, low_neg, low_zero))
    high = jnp.where(a > 0.0, high_pos, jnp.where(a < 0.0, high_neg, high_zero))
    return low, high

@partial(jax.jit, static_argnames=("xmin","xmax","ymin","ymax",))
def Lx(a,b,xmin,xmax,ymin,ymax):
    low, high = x_bounds_scalars(a,b,xmin,xmax,ymin,ymax)
    return jnp.maximum(0.0, high - low)

@partial(jax.jit, static_argnames=("xmin","xmax","ymin","ymax",))
def unnorm_prob_x_given_ab(x,a,b,xmin,xmax,ymin,ymax):
    return 0
    


    