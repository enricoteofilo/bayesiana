"""
--- PLOTTING SCRIPT ---
This file is intended to be used for creating the final plots avoiding 
the need to re-run the entire data processing and inference steps.

Author: Francesco Enrico Teofilo

"""
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
from jaxns import resample
import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions
import matplotlib.pyplot as plt
from utils import load_nested_sampler_results, import_bh_data
from my_models import  linear_correlation
from mbh_jaxns_newprior_normal import linear_uninformative_gaussian
from mbh_jaxns_newprior_normal_all import linear_uninformative_gaussian_all
import seaborn as sns

DEBUG = False

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

    plt.figure('bh_data', figsize=(6, 4), dpi=600)
    plt.title(r'Acquired data for $M_{BH}-\sigma_{gc}$ correlation')
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


    ### --- LINEAR MODEL + GAUSSIAN LIKELIHOOD


    gaussian_uninformative_jaxns = linear_uninformative_gaussian_all(bh_data["Galaxy"],M, M_equiv_err, sigma_gc, sigma_gc_equiv_err)

    gaussian_uninformative_jaxns_results = load_nested_sampler_results("results/gaussian_uninformative_jaxns_results.pkl")
    gaussian_uninformative_jaxns_termination = load_nested_sampler_results("results/gaussian_uninformative_jaxns_termination.pkl")
    gaussian_uninformative_jaxns_state = load_nested_sampler_results("results/gaussian_uninformative_jaxns_state.pkl")
    gaussian_uninformative_jaxns_npz = np.load("results/gaussian_uninformative_jaxns_npz.npz")
    gaussian_uninformative_jaxns_log_Z = gaussian_uninformative_jaxns_npz["log_Z"]
    gaussian_uninformative_jaxns_log_L = gaussian_uninformative_jaxns_npz["log_L"]
    gaussian_uninformative_jaxns_U = gaussian_uninformative_jaxns_npz["U"]
    gaussian_uninformative_jaxns.plot_cornerplot(gaussian_uninformative_jaxns_results, 
                                                 variables=["$a$","$b$"], 
                                                 save_name='results/gaussian_uninformative_jaxns_corner_ab.png',
                                                 kde_overlay=True)

    linear_uninformative_gaussian_jaxns_posterior_samples = resample(
        random.PRNGKey(42),
        gaussian_uninformative_jaxns_results.samples, 
        gaussian_uninformative_jaxns_results.log_dp_mean,
        S=int(jnp.floor(gaussian_uninformative_jaxns_results.ESS))
        )
    print(linear_uninformative_gaussian_jaxns_posterior_samples.keys())


    ### --- POSTERIOR PREDICTIVE PLOT: LINEAR MODEL + GAUSSIAN LIKELIHOOD ---
    linear_uninformative_gaussian_jaxns_corrfig, linear_uninformative_gaussian_jaxns_axs = plt.subplots(
        1, 1, figsize=(16, 9), dpi=150)
    linear_uninformative_gaussian_jaxns_corrfig.suptitle(r'$M_{BH}-\sigma_{gc}$ correlation: linear model'+
                                                         '\n'+r'with Gaussian likelihood'
                                                         )
    linear_uninformative_gaussian_jaxns_axs.errorbar(bh_data["sigma_gc"], bh_data["M"], 
                                                     xerr=[bh_data["sigma_gc_low"], bh_data["sigma_gc_high"]], 
                                                    yerr=[bh_data["dM_low"], bh_data["dM_high"]], fmt='.', 
                                                    label='Observed Data', markersize=4.0, capsize=2.0, 
                                                    linestyle='None', color='black', alpha=0.95)
    print([att for att in dir(gaussian_uninformative_jaxns_results) if not att.startswith("_")])
    x_arr = np.logspace(np.log10(50), np.log10(500), 10)
    a_samples = linear_uninformative_gaussian_jaxns_posterior_samples["$a$"]
    b_samples = linear_uninformative_gaussian_jaxns_posterior_samples["$b$"]
    @jax.jit
    def compute_y_batch(a, b, x):
        log_x_scaled = jnp.log10(x) - jnp.log10(200.0)
        return jax.vmap(lambda a_i, b_i: jnp.exp(jnp.log(10) * linear_correlation(log_x_scaled, a_i, b_i)))(a, b)
    y = np.asarray(compute_y_batch(a_samples, b_samples, x_arr))
    
    y_median = np.median(y, axis=0)
    y_cred_int_low = np.percentile(y, 5, axis=0)
    y_cred_int_high = np.percentile(y, 95, axis=0)
    linear_uninformative_gaussian_jaxns_axs.plot(x_arr, y_median, color='red', label='Posterior Median')
    linear_uninformative_gaussian_jaxns_axs.fill_between(x_arr, y_cred_int_low, y_cred_int_high, 
                                                         color='lightskyblue', alpha=1.0, label='90% C.L.')
    linear_uninformative_gaussian_jaxns_axs.set_xlabel(r'$\sigma_{gc}$ [km/s]')
    linear_uninformative_gaussian_jaxns_axs.set_ylabel(r'$M_{bh}$ [$M_\odot$]')
    linear_uninformative_gaussian_jaxns_axs.legend(loc='best')
    linear_uninformative_gaussian_jaxns_axs.set_xlim(50, 500)
    linear_uninformative_gaussian_jaxns_axs.set_ylim(1e6, 1e10)
    linear_uninformative_gaussian_jaxns_axs.set_xscale('log')
    linear_uninformative_gaussian_jaxns_axs.set_yscale('log')
    plt.show()


    exit()
    
