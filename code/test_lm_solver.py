import os, sys
from pathlib import Path
# Force JAX to ignore TPU/GPU backends in this environment.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import numpy as np
import scipy.optimize as sp_opt
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
jnp = jax.numpy
jsp = jax.scipy
random = jax.random
grad = jax.grad
jit = jax.jit
vmap = jax.vmap
import pickle
import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions
#from quadax import quadgk, quadcc
import optimistix as optx

@jit
def test_function(variables, args):
    x, y, z = variables
    a, b, c = args
    u = jnp.sin(a*x) + jnp.cos(b*y) + jnp.exp(c*z)
    v = jnp.tanh(x) + jnp.sinh(y) + jnp.sqrt(a**2 + b**2 + c**2)*z
    w = x**2 + y**2 + z**2 - a*b*c
    return u,v,w

if __name__ == "__main__":
    a, b, c = 1.0, 2.0, 3.0
    x_true, y_true, z_true = 10.0, 5.0, 0.0
    u_meas, v_meas, w_meas = test_function((x_true, y_true, z_true), (a, b, c))
    print(f"Measured values: u={u_meas}, v={v_meas}, w={w_meas}")
    
    @jit
    def noneq_system(variables, args):
        return test_function(variables, args) - jnp.array([u_meas, v_meas, w_meas])
    
    initial_guess = (1.0, 1.0, 1.0)
    solver = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8)



