"""
--- CORRELATION MODELS DEFINITIONS ---
This files contains the definition of the different tested models for the 
correlation between :math:`M_{SMBH}` and :math:`\sigma_{GC}`.

The models are `jax`-optimized and compatible with 
`JAXNS` nested sampling package.
"""
import os
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

_TWO_OVER_PI = 2.0 / jnp.pi
_SQRT_TWO_OVER_PI = jnp.sqrt(2.0 / jnp.pi)
_LOG_2PI = jnp.log(2.0 * jnp.pi)
_LOG_2 = jnp.log(2.0)
DEBUG = False

@jit
def linear_correlation(log_sigma_gc, a, b):
    """
    Defines a model for a correlation between :math:`M_{SMBH}` and 
    :math:`\sigma_{GC}` which is linear in logarithmic space

    .. math:: \log_{10}(M_{SMBH}/M_\odot) = b + a \log_{10}\left(\sigma_{GC}\,[\rm{km/s}]\right)


    Note: the logarithm is in base 10, not the natural logarithm!
    This is to be consistent with the literature on the :math:`M_{SMBH}-\sigma_{GC}` correlation.
    """
    return b + a * log_sigma_gc

@jit
def quadratic_correlation(log_sigma_gc, a, b, c):
    """
    Defines a model for a correlation between :math:`M_{SMBH}` and 
    :math:`\sigma_{GC}` which is quadratic in logarithmic space.

    .. math:: \log_{10}(M_{SMBH}/M_\odot) = b + a \log_{10}\left(\sigma_{GC}\,[\rm{km/s}]\right) +
                c \left(\log_{10}\left(\sigma_{GC}\,[\rm{km/s}]\right)\right)^2

    Note: the logarithm is in base 10, not the natural logarithm!
    This is to be consistent with the literature on the :math:`M_{SMBH}-\sigma_{GC}` correlation.

    Note: In this quadratic model the symmetry axis of the parabola is fixed to be 
    vertical. This breaks the rotational symmetry that is used to define the uninformative
    prior for the linear case, defined in `uninformative_prior_linear.py`, but also introduces an
    additional parameter :math:`c`
    
    An additional symmettry has to be exploited when building the uninformative prior.
    In `uninformative_prior_quadratic.py` the uninformative prior is built by exploiting the symmetry 
    under arbitrary translation in the 2-dimensional logarithmic space plane and the symmetry under 
    rescaling of the distance between the focus and the vertex of the parabola.
    """
    return c + b * log_sigma_gc + a * log_sigma_gc**2