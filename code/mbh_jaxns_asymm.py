import os
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
from jaxns import Prior, Model, NestedSampler
from jaxns import resample
import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions
import matplotlib.pyplot as plt

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
    return jnp.exp(b) * jnp.power(sigma_gc/200.0, a)

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

    plt.figure('bh_data', figsize=(6, 4), dpi=600)
    plt.title('Acquired data for $M_{BH}-\sigma_{gc}$ correlation')
    plt.errorbar(bh_data["sigma_gc"], bh_data["M"], xerr=[bh_data["sigma_gc_low"], bh_data["sigma_gc_high"]], 
                 yerr=[bh_data["dM_low"], bh_data["dM_high"]], fmt='.', label='Observed Data', 
                 markersize=4.0, capsize=2.0, linestyle='None', color='black', alpha=0.95)
    plt.xlabel(r'$\sigma_{gc}$ [km/s]')
    plt.ylabel(r'$M_{bh}$ [$M_\odot$]')
    plt.legend(loc='best')
    plt.xlim(50, 500)
    plt.ylim(1e6, 1e10)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('figures/bh_data.png')
    plt.close()

    M = bh_data["M"]
    sigma_gc = bh_data["sigma_gc"]
    M_equiv_err = 0.5*(bh_data["dM_low"]+bh_data["dM_high"])
    sigma_gc_equiv_err = 0.5*(bh_data["sigma_gc_low"]+bh_data["sigma_gc_high"])
    N_bh = len(M)

    @jit
    def log_likelihood_normal(a, b, true_sigma_gc):
        M_max = 1.0e+12
        return jnp.sum(tfpd.Normal(linear_correlation_exp(true_sigma_gc, a, b), M_equiv_err).log_prob(M)+
        tfpd.Normal(true_sigma_gc, sigma_gc_equiv_err).log_prob(sigma_gc)) - N_bh * jnp.log(M_max)
    
    def prior_model_normal():
        a = yield Prior(tfpd.Uniform(-10.0, 10.0), name="a")
        b = yield Prior(tfpd.Uniform(-5.0, 5.0), name="b")
        true_sigma_gc = yield Prior(tfpd.Sample(tfpd.Uniform(1.0e-14, 3.0e+14), sample_shape=(N_bh,)), name="true_sigma_gc")
        return a, b, true_sigma_gc
    
    model = Model(prior_model_normal, log_likelihood_normal)
    model.sanity_check(random.PRNGKey(0), S=10)

    ns = NestedSampler(model, s=1000, k=model.U_ndims, num_live_points=model.U_ndims*1000)
    termination_reason, state = jax.jit(ns)(random.PRNGKey(2))
    results = ns.to_results(termination_reason, state=state)
    np.savez("results/asymmetric_ns_results.npz",
         log_Z=np.asarray(results.log_Z_estimate),
         log_L=np.asarray(results.log_L_samples),
         U=np.asarray(results.U_samples))
    #posterior = resample(random.PRNGKey(1), results, S=5000)
    ns.summary(results)
    ns.plot_diagnostics(results)
    ns.plot_cornerplot(results, save_name='results/asymmetric_full_corner.png')

    exit()
