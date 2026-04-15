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
    valid_bounds = (xmin <= xmax) & (ymin <= ymax) & (bmin <= bmax) & (n_obs >= 0)
    x_low_bound, x_high_bound = x_bounds_scalars(a,b,xmin,xmax,ymin,ymax)
    length = jnp.maximum(0.0, x_high_bound - x_low_bound)
    b_low_bound = jnp.maximum(bmin, ymin - jnp.maximum(a*xmin, a*xmax))
    b_high_bound = jnp.minimum(bmax, ymax - jnp.minimum(a*xmin, a*xmax))
    in_bounds = (b_low_bound <= b) & (b <= b_high_bound) & (x_low_bound < x_high_bound) & (xmin <= x_low_bound) & (x_high_bound <= xmax)
    return jnp.where(valid_bounds, jnp.where(in_bounds, jnp.exp(n_obs*jnp.log(length)), 0.0), jnp.nan)

def integral_unnorm_prob_b_given_a_scalar(b, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    """Scalar-kernel for unnormalized conditional cumlative function for the 
    uninformative prior.
    
    .. math::
        F(b;a) = \int_{b_{min}}^{b} db^{\prime}\,\mathbb{I}(b^{\prime};b_{min},b_{max})
        L(a,b^{\prime};x_{min},x_{max},y_{min},y_{max})^N
    
    """
    if jnp.ndim(b) != 0 or jnp.ndim(a) != 0:
        raise ValueError("integral_unnorm_prob_b_given_a_scalar expects scalar `b` and scalar `a`.")

    valid_bounds = (xmin <= xmax) & (ymin <= ymax) & (bmin <= bmax) & (n_obs >= 0)
    x_low_bound, x_high_bound = x_bounds_scalars(a,b,xmin,xmax,ymin,ymax)
    b_low_bound = jnp.maximum(bmin, ymin - jnp.maximum(a*xmin, a*xmax))
    b_high_bound = jnp.minimum(bmax, ymax - jnp.minimum(a*xmin, a*xmax))
    b_clipped = jnp.maximum(bmin, b)

    def b_integral_helper(_):
        integral, _ = quadcc(
            unnorm_prob_b_given_a,
            (bmin, b), # integrates from bmin to b_hi.
            args=(a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs), #args passes the fixed parameters into the integrand
            epsabs=os.sys.float_info.epsilon,
            epsrel=os.sys.float_info.epsilon,
        )
        return jnp.clip(integral, 0.0, +jnp.inf)
    # Clipping avoids tiny negative values caused by floating-point/integration noise 
    # and enforces the expected nonnegative integral.

    # jax.lax.cond chooses what to do conditional on `b <= b_low_bound`: Here:
    # If True, it returns 0.0 immediately (zero probability outside the support).
    # If False, it calls `b_integral_helper` to compute the integral over the 
    # non-empty interval [b_low_bound, b].
    # Prevents problematic zero-width integrations
    integral = jax.lax.cond(
        b > b_low_bound, # Condition to check if b is within the valid range for integration
        b_integral_helper,
        lambda _: jnp.asarray(0.0, dtype=jnp.float64),
        operand=None,
    )
    # This works because under your vmap each in_bounds is scalar-shaped bool[].
    # If in_bounds were vector-shaped, lax.cond would not be appropriate; you would need 
    # elementwise logic (for example jnp.where) or vmap over a scalar cond.
    return jnp.where(valid_bounds, integral, jnp.nan)


def integral_unnorm_prob_b_given_a(b, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    """Vectorized version of the integral of the unnormalized conditional cumulative 
    probability function for the uninformative prior for the linear model y=ax+b:

    .. math::
        F(b;a) = \int_{b_{min}}^{b} db^{\prime}\,mathbb{I}(b^{\prime};b_{min},b_{max})L(a,b^{\prime};x_{min},x_{max},y_{min},y_{max})^N
            
    Computes the integral for a range of `b` values, given a fixed `a` and other parameters. 
    It uses JAX's `vmap` to vectorize the computation over the `b` values, 
    allowing for efficient evaluation across an array of `b` inputs.
    """
    b = jnp.asarray(b)
    # If b is a scalar calls the scalar helper function directly
    if b.ndim == 0:
        return integral_unnorm_prob_b_given_a_scalar(b,a,bmin,bmax,xmin,xmax,\
                ymin,ymax,n_obs)
    # If b is an array, applies the scalar helper function to each element 
    # with parallelization handled via `jax.vmap`.
    return jax.vmap(
        lambda b_iter: integral_unnorm_prob_b_given_a_scalar(
            b_iter, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs
        )
    )(b) #Iterating only on the b variable, since the other parameters are 
    #fixed for the integral in b.

def prob_b_given_a(a, b, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    valid_bounds = (xmin <= xmax) & (ymin <= ymax) & (bmin <= bmax) & (n_obs >= 0)
    x_low_bound, x_high_bound = x_bounds_scalars(a,b,xmin,xmax,ymin,ymax)
    length = jnp.maximum(0.0, x_high_bound - x_low_bound)
    b_low_bound = jnp.maximum(bmin, ymin - jnp.maximum(a*xmin, a*xmax))
    b_high_bound = jnp.minimum(bmax, ymax - jnp.minimum(a*xmin, a*xmax))
    in_bounds = (b_low_bound <= b) & (b <= b_high_bound) & (x_low_bound < x_high_bound) & (xmin <= x_low_bound) & (x_high_bound <= xmax)
    normalization = integral_unnorm_prob_b_given_a(bmax, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)
    return jnp.where(valid_bounds, jnp.where(in_bounds & (normalization>0.0), jnp.exp(n_obs*jnp.log(length))/normalization, 0.0), jnp.nan)

def conditional_cdf_b_given_a(b, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    valid_bounds = (xmin <= xmax) & (ymin <= ymax) & (bmin <= bmax) & (n_obs >= 0)
    normalization = integral_unnorm_prob_b_given_a(bmax, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)
    cumulative_unnorm = integral_unnorm_prob_b_given_a(b, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)
    return jnp.where(valid_bounds, jnp.where(normalization>0.0, cumulative_unnorm/normalization, 0.0), jnp.nan)



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
    x = jnp.linspace(xmin-5.0, xmax+5.0, 1000)
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
    b = jnp.linspace(bmin, bmax, 2500)
    for a_temp in a:
        y = unnorm_prob_b_given_a(a_temp, b, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
        plt.plot(b, y, label=f'a={a_temp:.2f}')
    plt.legend(loc='best')

    plt.figure('cumulative_prob_b_unnorm')
    print("Plotting F(b|a) for different values of a...")
    a = jnp.logspace(-1.5,1.0,10)
    b = jnp.linspace(bmin, bmax, 2500)
    for a_temp in a:
        y = integral_unnorm_prob_b_given_a(b, a_temp, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
        plt.plot(b, y, label=f'a={a_temp:.2f}')
    plt.legend(loc='best')

    plt.figure('prob_b')
    print("Plotting p(b|a) for different values of a...")
    a = jnp.linspace(1.25,3.75,10)
    a = a[((a<0) & (a*xmin>=ymin-bmax) & (a*xmax<=ymax-bmin) & (amin<=a) & (a<=amax)) |
          ((a==0) & (amin<=a) & (a<=amax)) | 
          ((a>0) & (a*xmin<=ymax-bmin) & (a*xmax>=ymin-bmax) & (amin<=a) & (a<=amax))]
    b = jnp.linspace(bmin, bmax, 2500)
    for a_temp in a:
        y = prob_b_given_a(a_temp, b, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
        plt.plot(b, y, label=f'a={a_temp:.2f}')
    plt.legend(loc='best')

    a_fixed_test = 2.5
    print(f"Plotting p(b|a={a_fixed_test:.2f})...")
    plt.figure('prob_b_fixed_a')
    b = jnp.linspace(bmin, bmax, 2500)
    y = prob_b_given_a(a_fixed_test, b, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
    plt.plot(b, y, label=f'a={a_fixed_test:.2f}')
    plt.legend(loc='best')
    
    plt.figure('conditional_cdf_b')
    print("Plotting CDF(b|a) for different values of a...")
    a = jnp.linspace(1.25,3.75,10)
    a = a[((a<0) & (a*xmin>=ymin-bmax) & (a*xmax<=ymax-bmin) & (amin<=a) & (a<=amax)) |
          ((a==0) & (amin<=a) & (a<=amax)) | 
          ((a>0) & (a*xmin<=ymax-bmin) & (a*xmax>=ymin-bmax) & (amin<=a) & (a<=amax))]
    b = jnp.linspace(bmin, bmax, 2500)
    for a_temp in a:
        y = conditional_cdf_b_given_a(a_temp, b, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
        plt.plot(b, y, label=f'a={a_temp:.2f}') 
    plt.legend(loc='best')




    plt.show()

    exit()



    