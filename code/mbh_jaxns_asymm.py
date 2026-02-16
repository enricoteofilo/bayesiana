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
from utils import save_nested_sampler_results, load_nested_sampler_results, logskewnorm_logpdf, skewnorm_cdf

DEBUG = False

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

    M = bh_data["M"]
    sigma_gc = bh_data["sigma_gc"]
    M_equiv_err = 0.5*(bh_data["dM_low"]+bh_data["dM_high"])
    sigma_gc_equiv_err = 0.5*(bh_data["sigma_gc_low"]+bh_data["sigma_gc_high"])
    N_bh = len(M)

    M_logskew_alpha = jnp.zeros_like(M)
    sigma_logskew_alpha = jnp.zeros_like(M)
    M_logskew_scale = jnp.log(M_equiv_err)
    sigma_logskew_scale = jnp.log(sigma_gc_equiv_err)

    @jit
    def log_likelihood_logskewnormal(a, b, true_sigma_gc):
        M_max = 1.0e+12
        predicted_M = linear_correlation_exp(true_sigma_gc, a, b)
        in_bounds = jnp.all((predicted_M >= 0.0) & (predicted_M <= M_max))
        log_like = jnp.sum(
            logskewnorm_logpdf(M, mean=jnp.log(predicted_M), scale=M_logskew_scale, shape=M_logskew_alpha)
            + logskewnorm_logpdf(sigma_gc, mean=jnp.log(predicted_M), scale=sigma_logskew_scale, shape=sigma_logskew_alpha)
        ) - N_bh * jnp.log(M_max)
        return jnp.where(in_bounds, log_like, -jnp.inf)
    
    def prior_model_logskewnormal():
        a = yield Prior(tfpd.Uniform(-35.0, 35.0), name="a")
        b = yield Prior(tfpd.Uniform(-20.0, 40.0), name="b")
        true_sigma_gc = yield Prior(tfpd.Sample(tfpd.Uniform(0.0, 1.0e+3), sample_shape=(N_bh,)), name="true_sigma_gc")
        return a, b, true_sigma_gc
    
    model = Model(prior_model_logskewnormal, log_likelihood_logskewnormal)
    model.sanity_check(random.PRNGKey(0), S=10)

    ns = NestedSampler(model, s=1000, k=model.U_ndims, num_live_points=model.U_ndims*10000)
    termination_reason, state = jax.jit(ns)(random.PRNGKey(2))
    results = ns.to_results(termination_reason, state=state)
    save_nested_sampler_results(results, "results/logskewnormal_ns_results.pkl")
    np.savez("results/logskewnormal_ns_results.npz",
         log_Z=np.asarray(results.log_Z_mean),
         log_L=np.asarray(results.log_L_samples),
         U=np.asarray(results.U_samples))
    #posterior = resample(random.PRNGKey(1), results, S=5000)
    ns.summary(results)
    ns.plot_diagnostics(results)
    ns.plot_cornerplot(results, save_name='results/logskewnormal_full_corner.png')

    exit()
