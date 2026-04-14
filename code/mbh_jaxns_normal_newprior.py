import os
from pathlib import Path
import pickle
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
from jaxns import Prior, Model, NestedSampler
import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions
import matplotlib.pyplot as plt
from quadax import quadcc
from utils import save_nested_sampler_results, load_nested_sampler_results, import_bh_data
from normalization_integral import interval_length_jaxed, uniform_normalization_outer

DEBUG = False


def compute_x_interval_bounds(a, b, xmin, xmax, ymin, ymax):
    """
    Computes the intersection between the conditions for acceptability of x 
    that define the support of x in the prior given the values of a,b:
    - xmin <= x <= xmax
    - ymin <= a*x+b <= ymax

    Returns the lower and upper endpoint and the length of the intersection 
    interval to facilitate remapping Uniform(0,1) to the support of x.
    """
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
    # Create a mask where the a=0 case is valid (i.e. the line is within the y-bounds
    a0_valid = jnp.where(a == 0.0, (ymin <= b) & (b <= ymax), True)
    # Finally, estimates the length of the intersection interval
    width = jnp.where(a0_valid, jnp.maximum(0.0, high - low), 0.0)
    return low, high, width


def _length_power_integrand_b(b, a, xmin, xmax, ymin, ymax, n_obs):
    length = interval_length_jaxed(a, b, xmin, xmax, ymin, ymax)
    return jnp.power(jnp.maximum(length, 0.0), n_obs)


def _b_mass_for_a(a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    length = interval_length_jaxed(a, b, xmin, xmax, ymin, ymax)
    return jnp.power(jnp.maximum(length, 0.0), n_obs)
    val, _ = quadcc(
        _length_power_integrand_b,
        (bmin, bmax),
        args=(a, xmin, xmax, ymin, ymax, n_obs),
        epsabs=os.sys.float_info.epsilon,
        epsrel=os.sys.float_info.epsilon,
    )
    return val


def _unnorm_p_a(a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    k = 0.5 * (n_obs - 3.0)
    return jnp.power(1.0 + a * a, k) * _b_mass_for_a(a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)


def _cdf_a(a, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs, norm_c):
    a_clamped = jnp.clip(a, amin, amax)
    numer, _ = quadcc(
        _unnorm_p_a,
        (amin, a_clamped),
        args=(bmin, bmax, xmin, xmax, ymin, ymax, n_obs),
        epsabs=os.sys.float_info.epsilon,
        epsrel=os.sys.float_info.epsilon,
    )
    cdf_val = jnp.where(norm_c > 0.0, numer / norm_c, 0.0)
    return jnp.where(a <= amin, 0.0, jnp.where(a >= amax, 1.0, cdf_val))


@jit
def _inverse_cdf_a(u, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs, norm_c):
    u = jnp.clip(u, 0.0, 1.0)

    def body(_, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        c_mid = _cdf_a(mid, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs, norm_c)
        go_right = c_mid < u
        lo = jnp.where(go_right, mid, lo)
        hi = jnp.where(go_right, hi, mid)
        return lo, hi

    lo, hi = jax.lax.fori_loop(0, 64, body, (amin, amax))
    return 0.5 * (lo + hi)


def _cdf_b_given_a(b, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs, b_mass):
    b_clamped = jnp.clip(b, bmin, bmax)
    numer, _ = quadcc(
        _length_power_integrand_b,
        (bmin, b_clamped),
        args=(a, xmin, xmax, ymin, ymax, n_obs),
        epsabs=os.sys.float_info.epsilon,
        epsrel=os.sys.float_info.epsilon,
    )
    cdf_val = jnp.where(b_mass > 0.0, numer / b_mass, 0.0)
    return jnp.where(b <= bmin, 0.0, jnp.where(b >= bmax, 1.0, cdf_val))


@jit
def _inverse_cdf_b_given_a(u, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    u = jnp.clip(u, 0.0, 1.0)
    b_mass = _b_mass_for_a(a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)

    def body(_, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        c_mid = _cdf_b_given_a(mid, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs, b_mass)
        go_right = c_mid < u
        lo = jnp.where(go_right, mid, lo)
        hi = jnp.where(go_right, hi, mid)
        return lo, hi

    lo, hi = jax.lax.fori_loop(0, 64, body, (bmin, bmax))
    return 0.5 * (lo + hi)

@jit
def linear_correlation_exp(sigma_gc, a, b):
    return jnp.power(10.0, b) * jnp.power(sigma_gc/200.0, a)

if __name__ == "__main__":
    bh_data = import_bh_data("data/bh_table_1.txt")
    print(f"Loaded columns: {list(bh_data.keys())}")
    if DEBUG:
        for key in bh_data.keys():
            try:
                print(f"{key} type: {bh_data[key][0].dtype}")
            except:
                print(f"{key} type: {type(bh_data[key][0])}")

    print(bh_data["sigma_gc"])
    print(bh_data["M"])

    M = bh_data["M"]
    sigma_gc = bh_data["sigma_gc"]
    M_equiv_err = 0.5*(bh_data["dM_low"]+bh_data["dM_high"])
    sigma_gc_equiv_err = 0.5*(bh_data["sigma_gc_low"]+bh_data["sigma_gc_high"])
    N_bh = len(M)

    @jit
    def log_likelihood_normal(a, b, true_sigma_gc):
        M_max = 1.0e+12
        predicted_M = linear_correlation_exp(true_sigma_gc, a, b)
        in_bounds = jnp.all((predicted_M >= 0.0) & (predicted_M <= M_max))
        log_like = jnp.sum(
            tfpd.Normal(predicted_M, M_equiv_err).log_prob(M)
            + tfpd.Normal(true_sigma_gc, sigma_gc_equiv_err).log_prob(sigma_gc)
        ) - N_bh * jnp.log(M_max)
        return jnp.where(in_bounds, log_like, -jnp.inf)
    
    amin = -35
    amax = 35
    bmin = -20
    bmax = 20
    sigma_gc_min_log = -15.0
    sigma_gc_max_log = jnp.log10(2.99792458e5)

    # y = log10(M) bounds entering the prior support constraints.
    ymin = 0.0
    ymax = 18.0

    # Prior normalization for diagnostics/consistency checks.
    prior_norm_C = uniform_normalization_outer(
        amin,
        amax,
        bmin,
        bmax,
        sigma_gc_min_log,
        sigma_gc_max_log,
        ymin,
        ymax,
        N_bh,
    )
    print("Uninformative prior normalization C:", prior_norm_C)

    def prior_model_uniform():
        a = yield Prior(tfpd.Uniform(amin, amax), name="a")
        b = yield Prior(tfpd.Uniform(bmin, bmax), name="b")
        true_sigma_gc = yield Prior(tfpd.Sample(tfpd.Uniform(10**sigma_gc_min_log, 10**sigma_gc_max_log), sample_shape=(N_bh,)), name=r"\sigma_{gc}^{true}")
        return a, b, true_sigma_gc

    def prior_model_uninformative():
        # Draw latent uniforms and map through on-the-fly inverse-CDF transforms.
        u_a = yield Prior(tfpd.Uniform(0.0, 1.0), name="u_a")
        a_value = _inverse_cdf_a(
            u_a,
            amin,
            amax,
            bmin,
            bmax,
            sigma_gc_min_log,
            sigma_gc_max_log,
            ymin,
            ymax,
            N_bh,
            prior_norm_C,
        )
        a = yield Prior(a_value, name="a")

        u_b = yield Prior(tfpd.Uniform(0.0, 1.0), name="u_b")
        b_value = _inverse_cdf_b_given_a(
            u_b,
            a_value,
            bmin,
            bmax,
            sigma_gc_min_log,
            sigma_gc_max_log,
            ymin,
            ymax,
            N_bh,
        )
        b = yield Prior(b_value, name="b")

        low_x, _, width_x = _compute_x_interval_bounds(
            a_value,
            b_value,
            sigma_gc_min_log,
            sigma_gc_max_log,
            ymin,
            ymax,
        )
        width_x = jnp.maximum(width_x, 0.0)

        u_x = yield Prior(
            tfpd.Sample(tfpd.Uniform(0.0, 1.0), sample_shape=(N_bh,)),
            name="u_x",
        )
        x_true = low_x + width_x * u_x
        true_sigma_gc_value = jnp.power(10.0, x_true)
        true_sigma_gc = yield Prior(true_sigma_gc_value, name=r"\sigma_{gc}^{true}")
        return a, b, true_sigma_gc
    
    model = Model(prior_model_uninformative, log_likelihood_normal)
    model.sanity_check(random.PRNGKey(0), S=10)

    ns = NestedSampler(model, s=1000, k=model.U_ndims, num_live_points=model.U_ndims*1000)
    termination_reason, state = jax.jit(ns)(random.PRNGKey(2))
    results = ns.to_results(termination_reason, state=state)
    save_nested_sampler_results(results, "results/gaussian_ns_results.pkl")
    np.savez("results/gaussian_ns_results.npz",
         log_Z=np.asarray(results.log_Z_mean),
         log_L=np.asarray(results.log_L_samples),
         U=np.asarray(results.U_samples))
    #posterior = resample(random.PRNGKey(1), results, S=5000)
    ns.summary(results)
    ns.plot_diagnostics(results)
    ns.plot_cornerplot(results, save_name='results/gaussian_full_corner.png')

    exit()
