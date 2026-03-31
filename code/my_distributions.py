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
from quadax import quadgk, quadcc

_TWO_OVER_PI = 2.0 / jnp.pi
_SQRT_TWO_OVER_PI = jnp.sqrt(2.0 / jnp.pi)
_LOG_2PI = jnp.log(2.0 * jnp.pi)
_LOG_2 = jnp.log(2.0)
DEBUG = False

@jit
def normal_logpdf(x, mean=0.0, sigma=1.0):
    inv_sigma = 1.0 / sigma
    return -jnp.log(sigma) - 0.5*(((x - mean)*inv_sigma)**2+jnp.log(2.0*jnp.pi))

@jit
def skewnormal_logpdf(x, loc=0.0, scale=1.0, shape=0.0):
    """Logarithm of the probability density function of the skew normal distribution."""
    inv_scale = 1.0 / scale
    t = (x - loc) * inv_scale
    normal_log_term = -jnp.log(scale) - 0.5*(t*t + _LOG_2PI)
    return normal_log_term + jsp.special.log_ndtr(shape * t) + _LOG_2

@jit
def skewnormal_cdf(x, loc=0.0, scale=1.0, shape=0.0, name=None):
    """Cumulative distribution function of the SkewNormal distribution."""
    inv_scale = 1.0 / scale
    t =(x - loc) * inv_scale
    return jsp.stats.norm.cdf(t, loc=0.0, scale=1.0) -2*tfp.math.owens_t(t,shape, name=name)

@jit
def lognormal_logpdf(x, loc=0.0, scale=1.0):
    inv_scale = 1.0 / scale
    logx = jnp.log(x)
    z = (logx - loc) * inv_scale
    return -logx + -jnp.log(scale) - 0.5*(z*z + _LOG_2PI)

@jit
def logskewnormal_logpdf(x, loc=0.0, scale=1.0, shape=0.0, name=None):
    logx = jnp.log(x)
    return skewnormal_logpdf(logx, loc, scale, shape, name=name) - logx

@jit
def logskewnormal_cdf(x, loc=0.0, scale=1.0, shape=0.0, name=None):
    return skewnormal_cdf(jnp.log(x), loc, scale, shape, name=name)

