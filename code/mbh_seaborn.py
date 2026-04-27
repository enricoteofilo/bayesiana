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
import pandas as pd

DEBUG = False

if __name__ == "__main__":
    bh_data = import_bh_data("data/bh_table_1_all.txt")
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
    galaxy_names = bh_data["Galaxy"]
    unique_names, inverse, counts = np.unique(galaxy_names, return_inverse=True, return_counts=True)


    ### --- LINEAR MODEL + GAUSSIAN LIKELIHOOD
    gaussian_uninformative_jaxns = linear_uninformative_gaussian_all(galaxy_names, M, M_equiv_err, sigma_gc, sigma_gc_equiv_err)

    gaussian_uninformative_jaxns_results = load_nested_sampler_results("results/gaussian_uninformative_jaxns_results.pkl")
    gaussian_uninformative_jaxns_termination = load_nested_sampler_results("results/gaussian_uninformative_jaxns_termination.pkl")
    gaussian_uninformative_jaxns_state = load_nested_sampler_results("results/gaussian_uninformative_jaxns_state.pkl")
    gaussian_uninformative_jaxns_npz = np.load("results/gaussian_uninformative_jaxns_npz.npz")
    gaussian_uninformative_jaxns_log_Z = gaussian_uninformative_jaxns_npz["log_Z"]
    gaussian_uninformative_jaxns_log_L = gaussian_uninformative_jaxns_npz["log_L"]
    gaussian_uninformative_jaxns_U = gaussian_uninformative_jaxns_npz["U"]

    linear_uninformative_gaussian_jaxns_posterior_samples = resample(
        random.PRNGKey(42),
        gaussian_uninformative_jaxns_results.samples, 
        gaussian_uninformative_jaxns_results.log_dp_mean,
        S=int(jnp.floor(gaussian_uninformative_jaxns_results.ESS))
        )
    print(linear_uninformative_gaussian_jaxns_posterior_samples.keys())

    plot_data = {
        "a": linear_uninformative_gaussian_jaxns_posterior_samples["$a$"],
        "b": linear_uninformative_gaussian_jaxns_posterior_samples["$b$"]
    }

    # 3. Unpack the 2D sigma array into multiple 1D columns
    sigma_array = linear_uninformative_gaussian_jaxns_posterior_samples[r"$\log\left(\frac{\sigma_{gc}^{true}}{200\,\rm{km/s}}\right)$"]
    Nn_bh = sigma_array.shape[1]

    # This creates a new column for each component, e.g., $\sigma$_0, $\sigma$_1, etc.
    for i in range(Nn_bh):
        plot_data[r"$\sigma_{GC}^{%s}\,[\rm{km/s}]$" %unique_names[i]] = jnp.exp(jnp.log(10)*(sigma_array[:, i] + jnp.log10(200.0)))

    # 4. Convert to a Pandas DataFrame (Seaborn handles DataFrames much better)
    df = pd.DataFrame(plot_data) 
    
    print('Initiating plot...')
    g = sns.PairGrid(df, 
                     #corner=True,
                     )
    g.map_upper(sns.kdeplot,fill=False, thresh=0.0,
                common_norm=True,
                levels=jsp.special.erfc(jnp.linspace(5, 0, 6)/jnp.sqrt(2)), #jnp.exp(-0.5 * jnp.linspace(5, 0, 6)**2),
                )
    g.map_upper(sns.scatterplot, 
                s=3, 
                alpha=0.4, 
                color=".2", 
                marker="."
                )
    g.map_lower(sns.kdeplot,fill=True, thresh=0.0,
                common_norm=True,
                levels=100,
                cmap="mako"
                )
    g.map_diag(sns.histplot, lw=1.0, legend=False,
               kde=True,
               common_norm=True, 
               fill=True
               )
    
    '''
    for ax in g.diag_axes:
        y_max = 0
        
        # 1. Extract the maximum y-value from the KDE boundary line
        for line in ax.get_lines():
            y_max = max(y_max, line.get_ydata().max())
            
        # 2. Extract the maximum y-value from the KDE fill (PolyCollection)
        #    This is a fallback in case you ever set lw=0 
        for collection in ax.collections:
            for path in collection.get_paths():
                if len(path.vertices) > 0:
                    y_max = max(y_max, path.vertices[:, 1].max())

        # 3. Force the axis limits to exactly match the KDE peak
        if y_max > 0:
            ax.set_ylim(0, y_max)
    '''
    plt.show()

    exit()
    
