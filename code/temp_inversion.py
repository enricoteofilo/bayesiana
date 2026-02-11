import os, sys
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

def function_pair(callable_1, callable_2, x, y, a, b, A_target, B_target):
    return jnp.array([callable_1(x, y, a, b)-A_target, callable_2(x, y, a, b)-B_target])

def newton_solver(callable_1, callable_2, guess_tuple, a, b, A_target, B_target, max_iter=1000, tol=sys.float_info.epsilon):
    return 0
