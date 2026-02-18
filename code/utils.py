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
import pickle
import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions

_TWO_OVER_PI = 2.0 / jnp.pi
_SQRT_TWO_OVER_PI = jnp.sqrt(2.0 / jnp.pi)
_LOG_2PI = jnp.log(2.0 * jnp.pi)
_LOG_2 = jnp.log(2.0)

def function_pair(callable_1, callable_2, x, y, a, b, A_target, B_target):
    """
    Function to cast the problem of finding solutions to the non linear system
    to finding the roots of a 2-components vector function of 2 variables.
    
    :param callable_1: Description
    :param callable_2: Description
    :param x: Description
    :param y: Description
    :param a: Description
    :param b: Description
    :param A_target: Description
    :param B_target: Description
    """
    return jnp.array([callable_1(x, y, a, b)-A_target, callable_2(x, y, a, b)-B_target])

def newton_solver(callable_1, callable_2, guess_tuple, a, b, A_target, B_target, max_iter=1000, tol=sys.float_info.epsilon, damping=0.0):
    """
    A simple implementation of the Newton-Raphson method to solve a system of two non-linear equations.
    """
    # Wrapper of the input functions pair to ensure the jacobian is computed 
    # with respect to the correct variables (x,y) and not the other args
    def F(v):
        x, y = v #unpacks the function inputs
        return function_pair(callable_1, callable_2, x, y, a, b, A_target, B_target)

    # Helper function for one Newton solver iteration. We exploit `jax`
    # autodifferentiation to accelerate the Jacobian computation.
    def one_step(state):
        v, _, iters = state
        J = jax.jacobian(F)(v)  # 2x2 Jacobian w.r.t. (x, y)
        # Evaluating the function at the current guess. If zero within the tolerance,
        # the system is considered solved.
        Fv = F(v)
        # The Newton step: the vector function is linearized around the current guess 
        # and we compute the displacement such that the residuals of the linearized 
        # system are zero. A damping term can be added to improve numerical stability.
        delta = jnp.linalg.solve(J + damping * jnp.eye(2), Fv)
        # We update the guess with the estimated displacement and return the new guess 
        # and the norm of the displacement.
        v_next = v - delta
        resid_next = jnp.linalg.norm(F(v_next))
        return v_next, resid_next, iters + 1

    # Early-exit loop: stop when residual is within tolerance or max_iter is hit
    def cond(state):
        _, resid_norm, iters = state
        return jnp.logical_and(resid_norm > tol, iters < max_iter)

    init_state = (guess_tuple, jnp.linalg.norm(F(guess_tuple)), 0)
    v_final, _, _ = jax.lax.while_loop(cond, one_step, init_state)
    return v_final

def save_nested_sampler_results(results, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_nested_sampler_results(input_path: str):
    with Path(input_path).open("rb") as f:
        return pickle.load(f)

@jit
def normal_logpdf(x, mean=0.0, sigma=1.0):
    inv_sigma = 1.0 / sigma
    return -jnp.log(sigma) - 0.5*(((x - mean)*inv_sigma)**2+jnp.log(2.0*jnp.pi))

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
def lognormal_logpdf(x, mean=0.0, sigma=1.0):
    inv_sigma = 1.0 / sigma
    logx = jnp.log(x)
    return -jnp.log(sigma) - 0.5*(((logx - mean)*inv_sigma)**2+_LOG_2PI) - logx

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
