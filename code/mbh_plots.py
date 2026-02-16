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
from utils import function_pair, newton_solver, save_nested_sampler_results, load_nested_sampler_results
from mbh_jaxns_normal import linear_correlation_exp, import_bh_data

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
    N_bh = len(M)

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

    exit()