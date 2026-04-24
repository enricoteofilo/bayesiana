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
from utils import save_nested_sampler_results, load_nested_sampler_results, import_bh_data
from uninformative_prior_linear import (PriorA_UninformLinearJAXNS, 
                                        PriorBgivenA_UninformLinearJAXNS,
                                        normalization_prob_a,
                                        build_cdf_a_lut,
                                        x_bounds_scalars
                                        )

DEBUG = False

@jit
def linear_correlation_exp(sigma_gc, a, b):
    return jnp.power(10.0, b) * jnp.power(sigma_gc/200.0, a)

@jit
def linear_correlation(sigma_gc, a, b):
    return b + a * jnp.log10(sigma_gc/200.0)

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

    amin, amax = -jnp.inf, jnp.inf
    bmin, bmax = -1.0e3, 1.0e3
    log_sigma_gc_min, log_sigma_gc_max = jnp.log10(jnp.finfo(np.float64).eps), jnp.log10(2.99792458e5)
    log_M_min, log_M_max = 2.0, 18.0

    a_normalization = normalization_prob_a(amin, amax, bmin, bmax, log_sigma_gc_min, log_sigma_gc_max, 
                                           log_M_min, log_M_max, N_bh, limit = 20*int(1e6)
                                           )
    
    a_grid, cdf_table, pdf_table = build_cdf_a_lut(a_normalization, amin, amax, bmin, bmax, log_sigma_gc_min, 
                                                   log_sigma_gc_max, log_M_min, log_M_max, N_bh, n_coarse_grid=1000, 
                                                    tol=jnp.finfo(np.float64).eps, 
                                                    max_points=5*int(1e6), use_linear=False
                                                    )

    @jit
    def log_likelihood_normal(a, b, true_sigma_gc):
        predicted_M = jnp.exp(linear_correlation(true_sigma_gc, a, b))
        log_like = jnp.sum(
            tfpd.Normal(predicted_M, M_equiv_err).log_prob(M)
            + tfpd.Normal(true_sigma_gc, sigma_gc_equiv_err).log_prob(sigma_gc)
        )
        return log_like
    
    def prior_linear_uninformative():
        a = yield PriorA_UninformLinearJAXNS(a_grid, cdf_table, pdf_table, a_normalization, amin, 
                                             amax, bmin, bmax, log_sigma_gc_min, log_sigma_gc_max, 
                                             log_M_min, log_M_max, N_bh, name="a")
        b = yield PriorBgivenA_UninformLinearJAXNS(a, bmin, bmax, log_sigma_gc_min, log_sigma_gc_max, 
                                                log_M_min, log_M_max, N_bh, name="b")
        x_low, x_high = x_bounds_scalars(a,b,log_sigma_gc_min, log_sigma_gc_max, log_M_min, log_M_max)
        true_sigma_gc = yield Prior(tfpd.Sample(tfpd.Uniform(10**log_sigma_gc_min, 10**log_sigma_gc_max), 
                                                sample_shape=(N_bh,)), name=r"\sigma_{gc}^{true}")
        return a, b, true_sigma_gc
    
    model = Model(prior_linear_uninformative, log_likelihood_normal)
    model.sanity_check(random.PRNGKey(0), S=10)

    ns = NestedSampler(model, s=1000, k=model.U_ndims, num_live_points=model.U_ndims*1000)
    termination_reason, state = jax.jit(ns)(random.PRNGKey(2))
    results = ns.to_results(termination_reason, state=state)
    save_nested_sampler_results(results, "results/gaussian_uninformative_ns_results.pkl")
    np.savez("results/gaussian_uninformative_ns_results.npz",
         log_Z=np.asarray(results.log_Z_mean),
         log_L=np.asarray(results.log_L_samples),
         U=np.asarray(results.U_samples))
    #posterior = resample(random.PRNGKey(1), results, S=5000)
    ns.summary(results)
    ns.plot_diagnostics(results)
    ns.plot_cornerplot(results, save_name='results/gaussian_uninformative_full_corner.png')

    exit()
