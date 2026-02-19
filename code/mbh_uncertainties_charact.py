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
import matplotlib.pyplot as plt
from utils import save_nested_sampler_results, load_nested_sampler_results, import_bh_data
from utils import logskewnormal_logpdf, skewnormal_cdf, logskewnormal_mean
import optimistix as optx

DEBUG = True

@jit
def logskewnormal_system_residuals(mean, sigma, shape, x, deltax_low, deltax_high, quantile_low=0.34, quantile_high=0.34):
    mean_from_model = logskewnormal_mean(mean=mean, sigma=sigma, shape=shape)[0]
    logx = jnp.log(x)
    logx_high = jnp.log(x + deltax_high)
    logx_low = jnp.log(x - deltax_low)
    low_prob_mass = skewnormal_cdf(logx, mean=mean, sigma=sigma, shape=shape)-skewnormal_cdf(logx_low, mean=mean, sigma=sigma, shape=shape)
    up_prob_mass = skewnormal_cdf(logx_high, mean=mean, sigma=sigma, shape=shape)-skewnormal_cdf(logx, mean=mean, sigma=sigma, shape=shape)
    
    return jnp.array([
        mean_from_model - x,
        low_prob_mass - quantile_low,
        up_prob_mass - quantile_high,
    ])

def logskewnormal_params_system(pdf_params, args):
    mean, sigma, shape = pdf_params
    x, deltax_low, deltax_high, quantile_low, quantile_high = args
    out_1 = logskewnormal_mean(mean=mean, sigma=sigma, shape=shape)[0] - x
    out_2 = skewnormal_cdf(jnp.log(x), mean=mean, sigma=sigma, shape=shape)-skewnormal_cdf(jnp.log(x - deltax_low), mean=mean, sigma=sigma, shape=shape) - quantile_low
    out_3 = skewnormal_cdf(jnp.log(x + deltax_high), mean=mean, sigma=sigma, shape=shape)-skewnormal_cdf(jnp.log(x), mean=mean, sigma=sigma, shape=shape) - quantile_high
    return out_1, out_2, out_3


def logskewnormal_params_system_non_destructive(pdf_params, args):
    mean, sigma, shape = pdf_params
    x, deltax_low, deltax_high, quantile_low, quantile_high, central_mode = args
    logx = jnp.log(x)
    cdf_at_x = skewnormal_cdf(logx, mean=mean, sigma=sigma, shape=shape)

    if central_mode == "mean_x":
        out_1 = logskewnormal_mean(mean=mean, sigma=sigma, shape=shape)[0] - x
    elif central_mode == "mean_logx":
        out_1 = mean - logx
    elif central_mode == "median_x":
        out_1 = cdf_at_x - 0.5
    else:
        raise ValueError(f"Unknown central_mode '{central_mode}'. Use 'mean_x', 'mean_logx', or 'median_x'.")

    out_2 = cdf_at_x - skewnormal_cdf(jnp.log(x - deltax_low), mean=mean, sigma=sigma, shape=shape) - quantile_low
    out_3 = skewnormal_cdf(jnp.log(x + deltax_high), mean=mean, sigma=sigma, shape=shape) - cdf_at_x - quantile_high
    return out_1, out_2, out_3


def fit_logskewnormal_non_destructive(
    solver,
    x,
    deltax_low,
    deltax_high,
    quantile_low=0.34,
    quantile_high=0.34,
    central_mode="mean_x",
    max_steps=int(jnp.power(2, 18)),
):
    initial_guess = (
        np.log(x),
        0.5 * (np.log(x + deltax_high) - np.log(x - deltax_low)),
        0.0,
    )
    return optx.least_squares(
        logskewnormal_params_system_non_destructive,
        solver,
        initial_guess,
        args=(x, deltax_low, deltax_high, quantile_low, quantile_high, central_mode),
        max_steps=max_steps,
        throw=True,
    )


def fit_logskewnormal_non_destructive_machine_epsilon(
    x,
    deltax_low,
    deltax_high,
    quantile_low=0.34,
    quantile_high=0.34,
    central_mode="mean_logx",
    tol=sys.float_info.epsilon,
):
    if x <= deltax_low:
        raise ValueError(f"Invalid bounds: x={x} must be > deltax_low={deltax_low}.")

    logx = float(np.log(x))
    logx_low = float(np.log(x - deltax_low))
    logx_high = float(np.log(x + deltax_high))

    def residual_2d(z):
        log_sigma, shape = z
        sigma = np.exp(log_sigma)
        cdf_at_x = float(skewnormal_cdf(logx, mean=logx, sigma=sigma, shape=shape))
        cdf_low = float(skewnormal_cdf(logx_low, mean=logx, sigma=sigma, shape=shape))
        cdf_high = float(skewnormal_cdf(logx_high, mean=logx, sigma=sigma, shape=shape))
        return np.array([
            cdf_at_x - cdf_low - quantile_low,
            cdf_high - cdf_at_x - quantile_high,
        ], dtype=float)

    def residual_3d(theta):
        mean, log_sigma, shape = theta
        sigma = np.exp(log_sigma)
        cdf_at_x = float(skewnormal_cdf(logx, mean=mean, sigma=sigma, shape=shape))
        cdf_low = float(skewnormal_cdf(logx_low, mean=mean, sigma=sigma, shape=shape))
        cdf_high = float(skewnormal_cdf(logx_high, mean=mean, sigma=sigma, shape=shape))
        if central_mode == "mean_x":
            out_1 = float(logskewnormal_mean(mean=mean, sigma=sigma, shape=shape)[0] - x)
        elif central_mode == "median_x":
            out_1 = cdf_at_x - 0.5
        else:
            raise ValueError(f"Unsupported central_mode '{central_mode}' for 3D root.")
        return np.array([
            out_1,
            cdf_at_x - cdf_low - quantile_low,
            cdf_high - cdf_at_x - quantile_high,
        ], dtype=float)

    sigma0 = max(0.5 * (logx_high - logx_low), 1e-8)
    log_sigma0 = np.log(sigma0)

    best_norm = np.inf
    best_params = None
    best_residual = None

    if central_mode == "mean_logx":
        guesses = [
            np.array([log_sigma0, 0.0], dtype=float),
            np.array([log_sigma0, 1.0], dtype=float),
            np.array([log_sigma0, -1.0], dtype=float),
            np.array([log_sigma0 + 0.5, 2.0], dtype=float),
            np.array([log_sigma0 + 0.5, -2.0], dtype=float),
            np.array([log_sigma0 - 0.5, 3.0], dtype=float),
            np.array([log_sigma0 - 0.5, -3.0], dtype=float),
        ]
        for guess in guesses:
            sol = sp_opt.root(residual_2d, guess, method="hybr", tol=tol)
            res = residual_2d(sol.x)
            inf_norm = float(np.max(np.abs(res)))
            if inf_norm < best_norm:
                best_norm = inf_norm
                best_params = jnp.array([logx, np.exp(sol.x[0]), sol.x[1]], dtype=jnp.float64)
                best_residual = jnp.array([0.0, res[0], res[1]], dtype=jnp.float64)
    else:
        guesses = [
            np.array([logx, log_sigma0, 0.0], dtype=float),
            np.array([logx, log_sigma0, 2.0], dtype=float),
            np.array([logx, log_sigma0, -2.0], dtype=float),
            np.array([logx + 0.2, log_sigma0 + 0.5, 1.0], dtype=float),
            np.array([logx - 0.2, log_sigma0 - 0.5, -1.0], dtype=float),
        ]
        for guess in guesses:
            sol = sp_opt.root(residual_3d, guess, method="hybr", tol=tol)
            res = residual_3d(sol.x)
            inf_norm = float(np.max(np.abs(res)))
            if inf_norm < best_norm:
                best_norm = inf_norm
                best_params = jnp.array([sol.x[0], np.exp(sol.x[1]), sol.x[2]], dtype=jnp.float64)
                best_residual = jnp.array(res, dtype=jnp.float64)

    if best_params is None:
        raise RuntimeError("No candidate solution produced by machine-epsilon solver.")

    success = bool(best_norm <= tol)
    return best_params, best_residual, success, best_norm
    

    


if __name__ == "__main__":
    bh_data = import_bh_data("data/bh_table_1.txt")
    print(f"Loaded columns: {list(bh_data.keys())}")
    if DEBUG:
        for key in bh_data.keys():
            try:
                print(f"{key} type: {bh_data[key][0].dtype}")
            except:
                print(f"{key} type: {type(bh_data[key][0])}")

    M = bh_data["M"]
    sigma_gc = bh_data["sigma_gc"]
    M_equiv_err = 0.5*(bh_data["dM_low"]+bh_data["dM_high"])
    sigma_gc_equiv_err = 0.5*(bh_data["sigma_gc_low"]+bh_data["sigma_gc_high"])
    N_bh = len(M)

    solver = optx.LevenbergMarquardt(rtol=sys.float_info.epsilon, atol=sys.float_info.epsilon)
    M_mean_param = np.zeros_like(bh_data["M"])
    M_sigma_param = np.zeros_like(bh_data["M"])
    M_shape_param = np.zeros_like(bh_data["M"])
    sigmagc_mean_param = np.zeros_like(bh_data["M"])
    sigmagc_sigma_param = np.zeros_like(bh_data["M"])
    sigmagc_shape_param = np.zeros_like(bh_data["M"])
    central_mode = "mean_logx"
    for index, galaxy in enumerate(bh_data["Galaxy"]):
        M_params, M_residual, M_success, M_inf_norm = fit_logskewnormal_non_destructive_machine_epsilon(
            bh_data["M"][index],
            bh_data["dM_low"][index],
            bh_data["dM_high"][index],
            central_mode=central_mode,
            tol=sys.float_info.epsilon,
        )
        sigmagc_params, sigmagc_residual, sigmagc_success, sigmagc_inf_norm = fit_logskewnormal_non_destructive_machine_epsilon(
            bh_data["sigma_gc"][index],
            bh_data["sigma_gc_low"][index],
            bh_data["sigma_gc_high"][index],
            central_mode=central_mode,
            tol=sys.float_info.epsilon,
        )
        M_mean_param[index], M_sigma_param[index], M_shape_param[index] = M_params
        sigmagc_mean_param[index], sigmagc_sigma_param[index], sigmagc_shape_param[index] = sigmagc_params
        if not M_success:
            print(f"Warning: M fit did not reach machine epsilon for {galaxy} (inf-norm={M_inf_norm}).")
        if not sigmagc_success:
            print(f"Warning: sigma_gc fit did not reach machine epsilon for {galaxy} (inf-norm={sigmagc_inf_norm}).")
        if DEBUG:
            print(f"Galaxy {galaxy}: M={M_params}, sigma_gc={sigmagc_params}")
        print(jnp.max(jnp.abs(M_residual)))
        print(jnp.max(jnp.abs(sigmagc_residual)))
    exit()