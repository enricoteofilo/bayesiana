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

def import_bh_data(fname: str) -> dict:
    """
    Function to import the $M_{BH}$, $\sigma_{GC}$ and corresponding 
    uncertainties from the .txt file.
    The output is a dictionary with keys given by the header.
    """
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

    dict = {}
    for name in structured.dtype.names:
        key = name.lstrip("#")
        values = structured[name]
        if np.issubdtype(values.dtype, np.number):
            dict[key] = jnp.asarray(values, dtype=jnp.float64)
        else:
            dict[key] = values.tolist()

    return dict

def save_nested_sampler_results(results, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_nested_sampler_results(input_path: str):
    with Path(input_path).open("rb") as f:
        return pickle.load(f)

@jit
def skewnormal_logpdf(x, mean=0.0, sigma=1.0, shape=0.0):
    shape_squared = shape * shape
    delta = shape / jnp.sqrt(1.0 + shape_squared)
    scale = sigma / jnp.sqrt(1.0 - _TWO_OVER_PI * delta * delta)
    loc = mean - scale * delta * _SQRT_TWO_OVER_PI
    inv_scale = 1.0 / scale
    z = (x - loc) * inv_scale
    normal_log_term = -jnp.log(scale) - 0.5*(z*z + _LOG_2PI)
    return normal_log_term + jnp.log(jsp.stats.norm.cdf(shape * z)) + _LOG_2

@jit
def logskewnormal_logpdf(x, mean=0.0, sigma=1.0, shape=0.0):
    shape2 = shape * shape
    inv_sqrt_1p_shape2 = jax.lax.rsqrt(1.0 + shape2)
    delta = shape * inv_sqrt_1p_shape2
    delta2 = delta * delta

    inv_scale_norm = jax.lax.rsqrt(1.0 - _TWO_OVER_PI * delta2)
    scale = sigma * inv_scale_norm
    loc = mean - scale * delta * _SQRT_TWO_OVER_PI

    x_safe = jnp.maximum(x, jnp.finfo(jnp.asarray(x).dtype).tiny)
    inv_scale = 1.0 / scale
    logx = jnp.log(x_safe)
    z = (logx - loc) * inv_scale
    normal_log_term = -jnp.log(scale) - 0.5*(z*z + _LOG_2PI)
    out = normal_log_term - logx + _LOG_2 + jsp.special.log_ndtr(shape * z)
    return jnp.where(x > 0.0, out, -jnp.inf)

@jit
def logskewnormal_logpdf_faster(x, mean=0.0, sigma=1.0, shape=0.0):
    shape_squared = shape * shape
    delta = shape / jnp.sqrt(1.0 + shape_squared)
    scale = sigma / jnp.sqrt(1.0 - _TWO_OVER_PI * delta * delta)
    loc = mean - scale * delta * _SQRT_TWO_OVER_PI
    inv_scale = 1.0 / scale
    logx = jnp.log(x)
    z = (logx - loc) * inv_scale
    normal_log_term = -jnp.log(scale) - 0.5*(z*z + _LOG_2PI)
    return normal_log_term + jnp.log(jsp.stats.norm.cdf(shape * z)) + _LOG_2 - logx

@jit
def skewnormal_cdf(x, mean=0.0, sigma=1.0, shape=0.0, name=None):
    shape_squared = shape * shape
    delta = shape / jnp.sqrt(1.0 + shape_squared)
    scale = sigma / jnp.sqrt(1.0 - _TWO_OVER_PI * delta * delta)
    loc = mean - scale * delta * _SQRT_TWO_OVER_PI
    inv_scale = 1.0 / scale
    z = (x - loc) * inv_scale
    return jsp.stats.norm.cdf(z, loc=0.0, scale=1.0)-2*tfp.math.owens_t(z,shape, name=name)

@jit
def logskewnormal_pdf(x, mean=0.0, sigma=1.0, shape=0.0):
    return jnp.exp(logskewnormal_logpdf(x, mean, sigma, shape))

@jit
def skewnormal_pdf(x, mean=0.0, sigma=1.0, shape=0.0):
    return jnp.exp(skewnormal_logpdf(x, mean, sigma, shape))

@jit
def logskewnormal_mean(mean=0.0, sigma=1.0, shape=0.0, epsabs=sys.float_info.epsilon, epsrel=sys.float_info.epsilon):
    shape2 = shape * shape
    delta = shape / jnp.sqrt(1.0 + shape2)
    scale = sigma / jnp.sqrt(1.0 - _TWO_OVER_PI * delta * delta)
    loc = mean - scale * delta * _SQRT_TWO_OVER_PI
    y = 2.0 * jnp.exp(loc + 0.5 * scale * scale) * jsp.stats.norm.cdf(delta * scale)
    info = jnp.array(0, dtype=jnp.int32)
    return (y, info)

@jit
def logskewnormal_mean_numerical(mean=0.0, sigma=1.0, shape=0.0, epsabs=sys.float_info.epsilon, epsrel=sys.float_info.epsilon):

    def integrand(t):
        return t * logskewnormal_pdf(t, mean=mean, sigma=sigma, shape=shape)

    y, info = quadcc(integrand, [0.0, jnp.inf], epsabs=epsabs, epsrel=epsrel)
    return (y, info)

def kbn_cumsum(increments):
    """
    Kahan-Babuška-Neumaier summation algorithm for improved numerical stability 
    when summing a sequence of floating-point numbers.

    See https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    
    """
    # Indexing the input array
    n = len(increments)
    # Prepare the accumulator
    summation = np.zeros(n, dtype=np.float64)
    temp_sum = 0.0
    # Running compensation for lost low-order bits
    # is zero the first time
    comp = 0.0
    for i in range(1,n):
        t = temp_sum + increments[i]
        if np.abs(temp_sum) >= np.abs(increments[i]):
            # If former `sum` is bigger, low-order digits of `input[i]` 
            # are lost and can be evaluated by
            comp += (temp_sum - t) + increments[i]
        else:
            # Else the low-order digits lost are from `sum`
            comp += (increments[i] - t) + temp_sum
        temp_sum = t
        summation[i] = temp_sum + comp
    return summation
    
@jit
def kbn_cumsum_jax(increments):
    """
    Kahan-Babuška-Neumaier summation algorithm for improved numerical stability 
    when summing a sequence of floating-point numbers.
    Uses `jax.lax.scan` to allow for JIT compilation and automatic differentiation.


    See https://en.wikipedia.org/wiki/Kahan_summation_algorithm

    For non-JAX version, see :func:`kbn_cumsum`.
    """
    def iterative_block(carry, increment):
        temp_sum, comp = carry
        t = temp_sum + increment
        comp += jnp.where(jnp.abs(temp_sum) >= jnp.abs(increment),
                         (temp_sum - t) + increment,
                         (increment - t) + temp_sum
                         )
        temp_sum = t
        return (temp_sum, comp), temp_sum + comp
    # Indexing the input array
    init = (0.0, 0.0)
    # Executing the iterative block sequentially over the increments
    # array using `jax.lax.scan`
    _, summation = jax.lax.scan(iterative_block, init, increments[1:])
    return jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), summation])

