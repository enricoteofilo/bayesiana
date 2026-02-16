import os, sys
from pathlib import Path
# Force JAX to ignore TPU/GPU backends in this environment.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import numpy as np
import scipy as sp
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
import matplotlib.pyplot as plt
from utils import logskewnorm_logpdf, skewnorm_logpdf 


def test_logskewnorm_is_normalized():
    # Integrate over t = log(x), with x = exp(t), dx = exp(t) dt.
    # This avoids singular behavior near x=0 and improves numerical stability.
    def integrate_logpdf(loc, scale, shape, t_min=-50.0, t_max=50.0, n=200000):
        t = np.linspace(t_min, t_max, n)
        x = np.exp(t)
        logf = np.asarray(logskewnorm_logpdf(x, loc=loc, scale=scale, shape=shape))
        y = np.exp(logf + t)
        return np.trapezoid(y, t)

    params = [
        (0.0, 1.0, 0.0),
        (0.3, 0.7, 2.0),
        (-1.2, 1.8, -3.5),
        (2.0, 0.2, 5.0),
    ]

    for loc, scale, shape in params:
        integral = integrate_logpdf(loc, scale, shape)
        assert np.isfinite(integral)
        assert abs(integral - 1.0) < 5e-4

if __name__ == "__main__":
    x = jnp.logspace(-2, 3, 1000)
    
    omega = 1.0
    meanlog = jnp.log(100)
    alpha = -6.0

    y_lognorm = sp.stats.lognorm.pdf(x, omega, loc=0.0, scale=jnp.exp(meanlog))
    y_logskewnorm_noskew = jnp.exp(tfpd.Normal(meanlog, omega).log_prob(jnp.log(x))-jnp.log(x))
    y_logskewnorm_noskew_custom = jnp.exp(logskewnorm_logpdf(x, loc=meanlog, scale=omega, shape=0.0))
    y_logskewnorm_skew = jnp.exp(logskewnorm_logpdf(x, loc=meanlog, scale=omega, shape=alpha))
    plt.figure("Lognormal vs Log-Skew-Normal PDF", figsize=(16, 12), dpi=300)
    plt.title("Lognormal vs Log-Skew-Normal PDF")
    plt.plot(x, y_lognorm, label="LogNormal")
    plt.plot(x, y_logskewnorm_noskew, label=r"LogSkewNormal $(\alpha=0)$", linestyle="dashed")
    plt.plot(x, y_logskewnorm_noskew_custom, label=r"LogSkewNormal $(\alpha=0)$ (custom)", linestyle="dotted")
    plt.plot(x, y_logskewnorm_skew, label=r"LogSkewNormal $(\alpha=%.1lf)$ (custom)" % alpha)
    plt.legend(loc='best')
    plt.xscale("log")
    #plt.yscale("log")
    plt.xlabel("x")
    plt.ylabel("p(x)")

    plt.figure(2)
    plt.title("Lognormal vs Log-Skew-Normal PDF")
    plt.plot(x, jnp.abs(y_lognorm-y_logskewnorm_skew), label="LogNormal-LogSkewNormal")
    plt.legend(loc='best')
    plt.xscale("log")
    plt.xlabel("x")

    x = jnp.linspace(-4.0, 4.0, 1000)
    sigma = 1.0
    mu = 0.0
    alpha = np.linspace(-2.0, 2.0, 5)
    plt.figure(3)
    plt.title("SkewNormal examples")
    for a in alpha:
        plt.plot(x, jnp.exp(skewnorm_logpdf(x, mean=mu, sigma=sigma, shape=a)), label=r"SkewNormal $(\alpha=%.2lf)$" %a)
    plt.legend(loc='best')
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.show()

