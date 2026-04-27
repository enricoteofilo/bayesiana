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
from jaxns import Prior, Model, NestedSampler, resample
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
from my_models import linear_correlation
import seaborn as sns

DEBUG = False


def linear_uninformative_gaussian_all(galaxy_names, M, M_equiv_err, sigma_gc, sigma_gc_equiv_err):
    unique_names, inverse, counts = np.unique(galaxy_names, return_inverse=True, return_counts=True)
    weights = jnp.asarray(1.0 / counts[inverse].astype(np.float64), dtype=jnp.float64)
    N_bh = len(unique_names)
    amin, amax = -jnp.inf, jnp.inf
    bmin, bmax = -1/(10*jnp.finfo(jnp.float64).eps), 1/(10*jnp.finfo(jnp.float64).eps)
    log_sigma_gc_min, log_sigma_gc_max = -3.0-jnp.log10(200.0), jnp.log10(2.99792458e5) - jnp.log10(200.0) #
    log_M_min, log_M_max = 0.0, 18.0

    a_normalization = normalization_prob_a(amin, amax, bmin, bmax, log_sigma_gc_min, log_sigma_gc_max, 
                                           log_M_min, log_M_max, N_bh, limit = 20*int(1e6)
                                           )
    
    if not np.isfinite(amin) or not np.isfinite(amax): use_linear = False
    else: use_linear = True
    
    a_grid, cdf_table, pdf_table = build_cdf_a_lut(a_normalization, amin, amax, bmin, bmax, log_sigma_gc_min, 
                                                   log_sigma_gc_max, log_M_min, log_M_max, N_bh, n_coarse_grid=1000, 
                                                    tol=jnp.finfo(np.float64).eps, 
                                                    max_points=5*int(1e6), use_linear=use_linear
                                                    )
    
    @jit
    def log_likelihood_normal(a, b, log_true_sigma_gc):
        predicted_M = 10**linear_correlation(log_true_sigma_gc[inverse], a, b)
        true_sigma_gc = 10**(log_true_sigma_gc[inverse]+jnp.log10(200.0))
        log_like = jnp.sum(
            tfpd.Normal(predicted_M, M_equiv_err).log_prob(M)
            + tfpd.Normal(true_sigma_gc, sigma_gc_equiv_err).log_prob(sigma_gc)
            + jnp.log(weights)
        )
        return log_like
    
    def prior_linear_uninformative():
        a = yield PriorA_UninformLinearJAXNS(a_grid, cdf_table, pdf_table, a_normalization, amin, 
                                             amax, bmin, bmax, log_sigma_gc_min, log_sigma_gc_max, 
                                             log_M_min, log_M_max, N_bh, name=r"$a$")
        b = yield PriorBgivenA_UninformLinearJAXNS(a, bmin, bmax, log_sigma_gc_min, log_sigma_gc_max, 
                                                log_M_min, log_M_max, N_bh, name=r"$b$")
        log_sigma_gc_low, log_sigma_gc_high = x_bounds_scalars(a,b,log_sigma_gc_min, log_sigma_gc_max, log_M_min, log_M_max)
        log_true_sigma_gc = yield Prior(tfpd.Sample(tfpd.Uniform(log_sigma_gc_low, log_sigma_gc_high), 
                                                sample_shape=(N_bh,)), name=r"$\log\left(\frac{\sigma_{gc}^{true}}{200\,\rm{km/s}}\right)$")
        return a, b, log_true_sigma_gc
    
    model = Model(prior_linear_uninformative, log_likelihood_normal)
    model.sanity_check(random.PRNGKey(0), S=10)

    jaxns_istance = NestedSampler(model, k=model.U_ndims, num_live_points=model.U_ndims*1000, #parameter_estimation=True, 
                       difficult_model=True, verbose=True)
    return jaxns_istance

if __name__ == "__main__":
    bh_data = import_bh_data("data/bh_table_1_all.txt")
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
    galaxy_names = bh_data["Galaxy"]

    jaxns_istance = linear_uninformative_gaussian_all(galaxy_names, M, M_equiv_err, sigma_gc, sigma_gc_equiv_err)
    
    termination_reason, state = jax.jit(jaxns_istance)(random.PRNGKey(2))
    results = jaxns_istance.to_results(termination_reason, state=state)
    save_nested_sampler_results(results, "results/gaussian_uninformative_jaxns_results.pkl")
    save_nested_sampler_results(termination_reason, "results/gaussian_uninformative_jaxns_termination.pkl")
    save_nested_sampler_results(state, "results/gaussian_uninformative_jaxns_state.pkl")
    np.savez("results/gaussian_uninformative_jaxns_npz.npz",
         log_Z=np.asarray(results.log_Z_mean),
         log_L=np.asarray(results.log_L_samples),
         U=np.asarray(results.U_samples))
    #posterior = resample(random.PRNGKey(1), results, S=5000)
    jaxns_istance.summary(results)
    jaxns_istance.plot_cornerplot(results, save_name='results/gaussian_uninformative_jaxns_corner.png')
    jaxns_istance.plot_diagnostics(results)

    exit()
