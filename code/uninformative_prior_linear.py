import os
import math
from functools import partial

# Keep computations in FP64 and on CPU for reproducibility.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
from quadax import quadcc

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

@jax.jit
def x_bounds_scalars(a,b,xmin,xmax,ymin,ymax):
    r"""
    Bounds of the set of values of acceptable values of x:

    .. math::
        \mathbb{I}(x;a,b,x_{min},x_{max},y_{min},y_{max}) = \begin{cases}
        1 & \text{if } x_{min} \leq x \leq x_{max} \text{ and } y_{min} \leq ax+b \leq y_{max} \\
        0 & \text{otherwise}
        \end{cases}
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
    return low, high

@jax.jit
def Lx(a,b,xmin,xmax,ymin,ymax):
    r"""
    The integral of the characteristic function over the full set of acceptable values of x:
    .. math::
        L(a,b;x_{min},x_{max},y_{min},y_{max}) = \int_{x_{min}}^{x_{max}} dx^{\prime}\,\mathbb{I}(x^{\prime};a,b,x_{min},x_{max},y_{min},y_{max})
    """
    low, high = x_bounds_scalars(a,b,xmin,xmax,ymin,ymax)
    return jnp.maximum(0.0, high - low)

@jax.jit
def prob_x_given_ab(x,a,b,xmin,xmax,ymin,ymax):
    r"""
    The marginal probability density function of the uninformative prior for x:
    .. math::
        PDF(x|a,b;x_{min},x_{max},y_{min},y_{max}) = \frac{\mathbb{I}(x;a,b,x_{min},x_{max},y_{min},y_{max})}
        {L(a,b;x_{min},x_{max},y_{min},y_{max})}

    `x` can be a scalar or an array (broadcasting is supported).
    """
    existence = (xmin <= xmax) & (ymin <= ymax)
    x_low_bound, x_high_bound = x_bounds_scalars(a,b,xmin,xmax,ymin,ymax)
    norm = jnp.maximum(0.0, x_high_bound - x_low_bound)
    acceptable = (norm > 0.0) & (xmin <= x) & (x <= xmax) & (ymin <= a*x+b) & (a*x+b <= ymax) & (x_low_bound <= xmax) & (x_high_bound >= xmin)
    safe_norm = jnp.where(norm > 0.0, norm, 1.0)
    return jnp.where(existence, jnp.where(acceptable, 1.0 / safe_norm, 0.0), jnp.nan)

@partial(jax.jit, static_argnames=['n_obs'])
def unnorm_prob_b_given_a(a, b, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    existence = (xmin <= xmax) & (ymin <= ymax) & (bmin <= bmax)
    x_low_bound, x_high_bound = x_bounds_scalars(a,b,xmin,xmax,ymin,ymax)
    length = jnp.maximum(0.0, x_high_bound - x_low_bound)
    in_bounds = (bmin <= b) & (b <= bmax) & (x_low_bound < x_high_bound) & (xmin <= x_low_bound) & (x_high_bound <= xmax)
    return jnp.where(existence, jnp.where(in_bounds, jnp.exp(n_obs*jnp.log(length)), 0.0), jnp.nan)

def integral_unnorm_prob_b_given_a(a, b, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    integrand = lambda b: unnorm_prob_b_given_a(a, b, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)
    result = quadcc(integrand, bmin, bmax)
    return result

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    xmin = -10.0
    xmax = jnp.log10(3.0e5)+5
    ymin = 0.0
    ymax = 18.0+5
    amin = -100.0
    amax = 100.0
    bmin = -35.0
    bmax = 35.0
    Nobs = 10

    plt.figure('prob_x')
    print("Plotting p(x|a,b) for different values of a...")
    x = jnp.linspace(xmin-5.0, xmax+5.0, 100)
    a = jnp.logspace(-1.5,1.0,10)
    b = 2.0
    for a_temp in a:
        prob_x_ab = prob_x_given_ab(x,a_temp,b,xmin,xmax,ymin,ymax)
        print(f"a={a_temp:.2f}, prob_x_ab={prob_x_ab}")
        plt.plot(x, prob_x_ab, label=f'a={a_temp:.2f}')
    plt.legend(loc='best')

    plt.figure('prob_b_unnorm')
    print("Plotting p(b|a) for different values of a...")
    a = jnp.logspace(-1.5,1.0,10)
    b = jnp.linspace(-5, 25.0, 100)
    for a_temp in a:
        prob_b_given_a = unnorm_prob_b_given_a(a_temp, b, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
        plt.plot(b, prob_b_given_a, label=f'a={a_temp:.2f}')
    plt.legend(loc='best')


    plt.show()

    exit()



    