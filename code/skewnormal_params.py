import os, sys
from pathlib import Path
from functools import partial
import timeit
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
import optimistix as optx
from my_distributions import skewnormal_logpdf, logskewnormal_logpdf, skewnormal_cdf, logskewnormal_cdf
from utils import import_bh_data

DEBUG = False
MACHINE_EPSILON = sys.float_info.epsilon
lm_solver = optx.LevenbergMarquardt(rtol=MACHINE_EPSILON, atol=MACHINE_EPSILON)
dogleg_solver = optx.Dogleg(rtol=MACHINE_EPSILON, atol=MACHINE_EPSILON)

def logskewnormal_initial_guess(log_mode, log_left_edge, log_right_edge):
    """Function to estimate the initial guess for the parameters of the 
    LogSkewNormal distribution, given the log of the mode and the (in 
    principal asymmetric) uncertainty intervals."""
    return jnp.array([log_mode, 0.5*(log_right_edge-log_left_edge), 0.0])

def logskewnormal_mode_condition(params, args):
    """Defines the function that has to be zero at the mode of the LogSkewNormal"""
    loc, scale, shape = params
    log_mode = args
    inv_scale = 1.0 / scale
    t = (log_mode - loc) * inv_scale
    return shape * jnp.exp(-0.5 * (shape*t)**2) / jnp.sqrt(2 * jnp.pi) - scale * jsp.stats.norm.cdf(shape * t) + t * jsp.stats.norm.cdf(shape * t)

def logskewnormal_nonlin_system_mode_quantiles(params, args):
    loc, scale, shape = params
    log_mode, log_left_edge, log_right_edge = args
    mode_condition = logskewnormal_mode_condition(params, log_mode)
    left_edge_condition = skewnormal_cdf(log_left_edge, loc=loc, scale=scale, shape=shape) - 0.5 + 0.5 * jsp.special.erf(1/jnp.sqrt(2))
    right_edge_condition = skewnormal_cdf(log_right_edge, loc=loc, scale=scale, shape=shape) - 0.5 - 0.5 * jsp.special.erf(1/jnp.sqrt(2))
    return jnp.array([mode_condition, left_edge_condition, right_edge_condition])

def one_root_find(system, args, solver, initial_guess, max_steps=10000, throw=False):
    solution = optx.root_find(system, solver, initial_guess, args=args, max_steps=max_steps, throw=throw)
    residuals = system(solution.value, args)
    return solution.value, residuals, jnp.sum(residuals**2)

def one_least_squares(system, args, solver, initial_guess, max_steps=10000, throw=False):
    solution = optx.least_squares(system, solver, initial_guess, args=args, max_steps=max_steps, throw=throw)
    residuals = system(solution.value, args)
    return solution.value, residuals, jnp.sum(residuals**2)

def solve_logskewnormal_mode_quantiles(log_mode, log_left_edge, log_right_edge):
    args = (log_mode, log_left_edge, log_right_edge)
    initvals = logskewnormal_initial_guess(log_mode, log_left_edge, log_right_edge)
    try:
        solution, residuals, resid_sq_sum = one_root_find(
        logskewnormal_nonlin_system_mode_quantiles, args, dogleg_solver, initvals, max_steps=50000, throw=False)
    except:
        solution, residuals, resid_sq_sum = one_least_squares(
        logskewnormal_nonlin_system_mode_quantiles, args, dogleg_solver, initvals, max_steps=50000, throw=False)
    return solution, residuals, resid_sq_sum

batch_solver = jit(vmap(solve_logskewnormal_mode_quantiles, in_axes=(0, 0, 0)))

if __name__ == "__main__":
    bh_data = import_bh_data("./data/bh_table_1.txt")
    print(f"Loaded columns: {list(bh_data.keys())}")
    if DEBUG:
        for key in bh_data.keys():
            try:
                print(f"{key} type: {bh_data[key][1].dtype}")
            except:
                print(f"{key} type: {type(bh_data[key][1])}")

    print(bh_data["sigma_gc"])
    print(bh_data["M"])
    M_log_mode = jnp.log(bh_data["M"])
    M_log_left_edge = jnp.log(bh_data["M"]-bh_data["dM_low"])
    M_log_right_edge = jnp.log(bh_data["M"]+bh_data["dM_high"])
    sigma_gc_log_mode = jnp.log(bh_data["sigma_gc"])
    sigma_gc_log_left_edge = jnp.log(bh_data["sigma_gc"]-bh_data["sigma_gc_low"])
    sigma_gc_log_right_edge = jnp.log(bh_data["sigma_gc"]+bh_data["sigma_gc_high"])

    params, residuals, minimized_squares = batch_solver(M_log_mode, M_log_left_edge, M_log_right_edge)
    print(f"Estimated parameters: {params}")
    print(f"Residuals: {residuals}")
    print(f"Residuals sum square root normalized: {jnp.sqrt(minimized_squares/3)/MACHINE_EPSILON}")

    sigma_gc_params, sigma_gc_residuals, sigma_gc_minimized_squares = batch_solver(sigma_gc_log_mode, sigma_gc_log_left_edge, sigma_gc_log_right_edge)
    print(f"Estimated parameters for sigma_gc: {sigma_gc_params}")
    print(f"Residuals for sigma_gc: {sigma_gc_residuals}")
    print(f"Residuals sum square root normalized for sigma_gc: {jnp.sqrt(sigma_gc_minimized_squares/3)/MACHINE_EPSILON}")      





