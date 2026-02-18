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


def logskewnormal_system_residuals(mean, sigma, shape, measured_mean, M, deltaM_low, deltaM_high, quantile=0.32):
    """
    Residual vector for the system:
      1) measured_mean = E[X]
      2) quantile = P(M-deltaM_low < X <= M)
      3) quantile = P(M < X <= M+deltaM_high)
    where X follows a log-skew-normal law parameterized by (mean, sigma, shape)
    in log-space.
    """
    if M <= 0.0:
        raise ValueError("M must be strictly positive.")
    if M - deltaM_low <= 0.0:
        raise ValueError("M - deltaM_low must be strictly positive.")
    if deltaM_high <= 0.0:
        raise ValueError("deltaM_high must be strictly positive.")
    if sigma <= 0.0:
        raise ValueError("sigma must be strictly positive.")

    mean_model = logskewnormal_mean(mean=mean, sigma=sigma, shape=shape)[0]
    logM = jnp.log(M)
    logM_low = jnp.log(M - deltaM_low)
    logM_high = jnp.log(M + deltaM_high)

    lower_mass = skewnormal_cdf(logM, mean=mean, sigma=sigma, shape=shape) - \
                 skewnormal_cdf(logM_low, mean=mean, sigma=sigma, shape=shape)
    upper_mass = skewnormal_cdf(logM_high, mean=mean, sigma=sigma, shape=shape) - \
                 skewnormal_cdf(logM, mean=mean, sigma=sigma, shape=shape)

    return jnp.array([
        mean_model - measured_mean,
        lower_mass - quantile,
        upper_mass - quantile,
    ])


def solve_logskewnormal_from_mean_and_bounds(measured_mean, M, deltaM_low, deltaM_high, quantile=0.32,
                                             initial_guess=None, max_iter=80, tol=1e-10,
                                             fd_eps=1e-6, damping=0.0):
    """
    Solve for (mean, sigma, shape) of a log-skew-normal distribution from:
      - measured mean of X,
      - probability mass in [M-deltaM_low, M],
      - probability mass in [M, M+deltaM_high].

    Uses a damped Newton method with finite-difference Jacobian in the unconstrained
    variables (mean, log_sigma, shape) to enforce sigma > 0.
    """
    if initial_guess is None:
        initial_guess = jnp.array([jnp.log(M), 0.5, 0.0], dtype=jnp.float64)

    if float(initial_guess[1]) <= 0.0:
        raise ValueError("initial_guess[1] (sigma) must be strictly positive.")

    theta0 = np.array([
        float(initial_guess[0]),
        np.log(float(initial_guess[1])),
        float(initial_guess[2]),
    ], dtype=float)

    def residual_from_theta(theta_vec):
        mean = float(theta_vec[0])
        sigma = float(np.exp(theta_vec[1]))
        shape = float(theta_vec[2])
        residuals = logskewnormal_system_residuals(
            mean=mean,
            sigma=sigma,
            shape=shape,
            measured_mean=measured_mean,
            M=M,
            deltaM_low=deltaM_low,
            deltaM_high=deltaM_high,
            quantile=quantile,
        )
        residuals = np.asarray(residuals, dtype=float)
        if np.all(np.isfinite(residuals)):
            return residuals
        return np.array([1e12, 1e12, 1e12], dtype=float)

    result = sp_opt.least_squares(
        fun=residual_from_theta,
        x0=theta0,
        bounds=([-np.inf, np.log(1e-12), -50.0], [np.inf, np.log(1e12), 50.0]),
        method="trf",
        diff_step=fd_eps,
        ftol=tol,
        xtol=tol,
        gtol=tol,
        max_nfev=max_iter,
    )

    mean_sol = float(result.x[0])
    sigma_sol = float(np.exp(result.x[1]))
    shape_sol = float(result.x[2])

    info = {
        "converged": bool(result.success),
        "iterations": int(result.nfev),
        "residual_norm": float(np.linalg.norm(result.fun)),
        "residual_vector": result.fun,
        "message": result.message,
    }
    return jnp.array([mean_sol, sigma_sol, shape_sol]), info



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

@jit
def logskewnormal_pdf(x, mean=0.0, sigma=1.0, shape=0.0):
    return jnp.exp(logskewnormal_logpdf(x, mean, sigma, shape))

@jit
def skewnormal_pdf(x, mean=0.0, sigma=1.0, shape=0.0):
    return jnp.exp(skewnormal_logpdf(x, mean, sigma, shape))

@jit
def logskewnormal_mean(mean=0.0, sigma=1.0, shape=0.0, epsabs=sys.float_info.epsilon, epsrel=sys.float_info.epsilon):

    def integrand(t):
        return t * logskewnormal_pdf(t, mean=mean, sigma=sigma, shape=shape)

    y, info = quadcc(integrand, [0.0, jnp.inf], epsabs=epsabs, epsrel=epsrel)
    return (y, info)
