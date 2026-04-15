import os
import math
from functools import partial
# Keep computations in FP64 and on CPU for reproducibility.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
from quadax import quadcc, quadgk, quadts, romberg
import numpy as np
from scipy.integrate import quad

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

## CONDITIONAL PROBABILITY p(x|a,b)
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

## CONDITIONAL PROBABILITY p(b|a)
@partial(jax.jit, static_argnames=['n_obs'])
def unnorm_prob_b_given_a(b, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    r"""
    The unnormalized conditional prior for b given a, under the uninformative joint 
    prior for the linear model y=ax+b:

    .. math::
        p(b|a) \propto \mathbb{I}(b;b_{min},b_{max})L(a,b;x_{min},x_{max},y_{min},y_{max})^N
        
        
    """
    valid_bounds = (xmin <= xmax) & (ymin <= ymax) & (bmin <= bmax) & (n_obs >= 0)
    x_low_bound, x_high_bound = x_bounds_scalars(a,b,xmin,xmax,ymin,ymax)
    length = x_high_bound - x_low_bound #jnp.maximum(0.0, x_high_bound - x_low_bound)
    b_low_bound = jnp.maximum(bmin, ymin - jnp.maximum(a*xmin, a*xmax))
    b_high_bound = jnp.minimum(bmax, ymax - jnp.minimum(a*xmin, a*xmax))
    in_bounds = (b_low_bound <= b) & (b <= b_high_bound) & (x_low_bound < x_high_bound) & (xmin <= x_low_bound) & (x_high_bound <= xmax)
    return jnp.where(valid_bounds, jnp.where(in_bounds, jnp.exp(n_obs*jnp.log(length)), 0.0), jnp.nan)


@partial(jax.jit, static_argnames=['n_obs', 'n_grid'])
def build_cdf_lut(a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs, n_grid=2000):
    r"""
    Helper function. Builds a lookup table for the CDF(b|a) evaluating it on a grid 
    internally.

    .. math::
        CDF(b|a) = \int_{b_{min}}^{b} db^{\prime}\,\mathbb{I}(b^{\prime};b_{min},b_{max})L(a,b^{\prime};x_{min},x_{max},y_{min},y_{max})^N

    Evaluates the unnormalized PDF on the grid ``linspace(bmin, bmax, n_grid)``,
    clips to non-negative, and accumulates the probability density with the trapezoidal 
    rule so that the resulting function is non-decreasing by construction.

    Returns
    -------
    b_grid : 1-D array of shape ``(n_grid,)``, the grid used to build the look-up table
    cdf    : 1-D array of shape ``(n_grid,)``, values in [0, 1]
    """
    b_grid = jnp.linspace(bmin, bmax, n_grid)
    pdf_vals = jnp.clip(unnorm_prob_b_given_a(b_grid, a, bmin, bmax, xmin, xmax,
                                     ymin, ymax, n_obs), 0.0, jnp.inf)

    db = jnp.diff(b_grid)
    avg_pdf = 0.5 * (pdf_vals[:-1] + pdf_vals[1:])
    increments = avg_pdf * db                       # non-negative
    cumulative = jnp.concatenate([jnp.zeros(1), jnp.cumsum(increments)])

    normalization = cumulative[-1]
    cdf = jnp.where(normalization > 0.0, cumulative / normalization, 0.0)
    cdf = jnp.clip(cdf, 0.0, 1.0)
    return b_grid, cdf


@partial(jax.jit, static_argnames=['n_obs', 'n_grid'])
def cdf_b_given_a(b, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs, n_grid=2000):
    r"""
    Returns the CDF(b|a) evaluated at arbitrary **b** values for fixed **a**.

    Builds an internal grid of size *n_grid*, computes the non-decreasing CDF 
    via cumulative trapezoidal integration, and returns the CDF evaluated at the
    requested *b* value(s) through direct index computation + linear
    interpolation (O(1) per point, avoids ``searchsorted``).

    Parameters
    ----------
    b      : scalar or array — query point(s)
    n_grid : int (static) — internal grid resolution (default 2000)
    """
    valid_bounds = (xmin <= xmax) & (ymin <= ymax) & (bmin <= bmax) & (n_obs >= 0)
    b_grid, cdf_table = build_cdf_lut(a, bmin, bmax, xmin, xmax, ymin, ymax,
                                         n_obs, n_grid)
    # Direct index computation exploiting the uniform grid (O(1) per point).
    t = (b - bmin) / jnp.where(bmax > bmin, bmax - bmin, 1.0)
    t = jnp.clip(t, 0.0, 1.0)
    idx_f = t * (n_grid - 1)
    idx_lo = jnp.floor(idx_f).astype(jnp.int64)
    idx_lo = jnp.clip(idx_lo, 0, n_grid - 2)
    frac = idx_f - idx_lo
    result = cdf_table[idx_lo] * (1.0 - frac) + cdf_table[idx_lo + 1] * frac
    return jnp.where(valid_bounds, result, jnp.nan)


@partial(jax.jit, static_argnames=['n_obs'])
def cdf_b_given_a_monotone(a, b_grid, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    r"""Monotone CDF of b|a via cumulative trapezoidal integration on a sorted grid.

    This function evaluates the unnormalized
    PDF once on the whole grid and accumulates with the trapezoidal rule.

    Because the PDF is clipped to be non-negative, every trapezoidal increment
    is >= 0 and the cumulative sum is **guaranteed non-decreasing**.  The result
    is normalized by the total to lie in [0, 1].

    Parameters
    ----------
    b_grid : 1-D array, **must be sorted in ascending order**.
    """
    valid_bounds = (xmin <= xmax) & (ymin <= ymax) & (bmin <= bmax) & (n_obs >= 0)

    # Evaluate the unnormalized PDF on the whole grid (broadcasts over b_grid)
    pdf_vals = unnorm_prob_b_given_a(b_grid, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)

    # Guard against tiny negative values from floating-point noise
    pdf_vals = jnp.maximum(pdf_vals, 0.0)

    # Cumulative trapezoidal rule — monotone by construction
    db = jnp.diff(b_grid)
    avg_pdf = 0.5 * (pdf_vals[:-1] + pdf_vals[1:])
    increments = avg_pdf * db                       # non-negative
    cumulative = jnp.concatenate([jnp.zeros(1), jnp.cumsum(increments)])

    # Normalize to [0, 1]
    total = cumulative[-1]
    cdf = jnp.where(total > 0.0, cumulative / total, 0.0)
    cdf = jnp.clip(cdf, 0.0, 1.0)

    return jnp.where(valid_bounds, cdf, jnp.nan)

@partial(jax.jit, static_argnames=['n_obs', 'n_grid'])
def quantile_b_given_a(u, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs, n_grid=2000):
    r"""Quantile function :math:`b=CDF^{-1}(u|a)`.

    Maps :math:`u \sim \mathrm{Uniform}(0,1)` to *b* such that
    :math:`\mathrm{CDF}(b\mid a) = u`.

    Internally builds a grid of size *n_grid*, computes the monotone CDF, and
    inverts via linear interpolation

    Parameters
    ----------
    u      : scalar or array — quantile(s) in [0, 1]
    n_grid : int (static) — internal grid resolution (default 2000)
    """
    valid_bounds = (xmin <= xmax) & (ymin <= ymax) & (bmin <= bmax) & (n_obs >= 0)
    #Creating a LUT for the CDF
    b_grid, cdf_table = build_cdf_lut(a, bmin, bmax, xmin, xmax, ymin, ymax,
                                         n_obs, n_grid)
    result = jnp.interp(u, cdf_table, b_grid)
    return jnp.where(valid_bounds, result, jnp.nan)

## PURE QUADAX INDEPENDENT-QUADRATURES IMPLEMENTATION for CDF(b|a): QUITE NOISY AND UNSTABLE!
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
        return integral_unnorm_prob_b_given_a_scalar(b,a,bmin,bmax,xmin,xmax,
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
    r"""
    Returns the conditional prior :math:`\pi(b|a)`, under the uninformative joint 
    prior for the linear model :math:`y=ax+b`:

    .. math::
        \pi(b|a) = \frac{1}{C_{b}(a)} \mathbb{I}(b;b_{min},b_{max})L(a,b;x_{min},x_{max},y_{min},y_{max})^N
        
    
    """
    valid_bounds = (xmin <= xmax) & (ymin <= ymax) & (bmin <= bmax) & (n_obs >= 0)
    x_low_bound, x_high_bound = x_bounds_scalars(a,b,xmin,xmax,ymin,ymax)
    length = jnp.maximum(0.0, x_high_bound - x_low_bound)
    b_low_bound = jnp.maximum(bmin, ymin - jnp.maximum(a*xmin, a*xmax))
    b_high_bound = jnp.minimum(bmax, ymax - jnp.minimum(a*xmin, a*xmax))
    in_bounds = (b_low_bound <= b) & (b <= b_high_bound) & (x_low_bound < x_high_bound) & (xmin <= x_low_bound) & (x_high_bound <= xmax)
    normalization = integral_unnorm_prob_b_given_a(bmax, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)
    return jnp.where(valid_bounds, jnp.where(in_bounds & (normalization>0.0), jnp.exp(n_obs*jnp.log(length))/normalization, 0.0), jnp.nan)


## SCIPY IMPLEMENTATION  for CDF(b|a) FOR CROSS-CHECKS 
def integral_unnorm_prob_b_given_a_scipy(b, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    """Logically identical to :func:`integral_unnorm_prob_b_given_a` but uses
    ``scipy.integrate.quad`` instead of ``quadax.quadcc``.

    This version is **not** JAX-traceable and does **not** use ``vmap``.
    It loops over ``b`` values in plain Python, which makes it suitable as a
    reference / cross-check implementation.

    .. math::
        F(b;a) = \\int_{b_{min}}^{b} db^{\\prime}\\,
        \\mathbb{I}(b^{\\prime};b_{min},b_{max})\\,
        L(a,b^{\\prime};x_{min},x_{max},y_{min},y_{max})^N
    """
    def _integrand(b_prime):
        return float(unnorm_prob_b_given_a(b_prime, a, bmin, bmax,
                                           xmin, xmax, ymin, ymax, n_obs))

    b = np.asarray(b, dtype=np.float64)
    scalar_input = b.ndim == 0
    b = np.atleast_1d(b)

    valid_bounds = (xmin <= xmax) and (ymin <= ymax) and (bmin <= bmax) and (n_obs >= 0)
    b_low_bound = float(jnp.maximum(bmin, ymin - jnp.maximum(a * xmin, a * xmax)))

    results = np.empty_like(b)
    for i, b_val in enumerate(b):
        if not valid_bounds:
            results[i] = np.nan
        elif b_val <= b_low_bound:
            results[i] = 0.0
        else:
            integral, _ = quad(_integrand, float(bmin), float(b_val),
                               epsabs=os.sys.float_info.epsilon,
                               epsrel=os.sys.float_info.epsilon,
                               limit=200)
            results[i] = max(integral, 0.0)

    if scalar_input:
        return jnp.asarray(results[0], dtype=jnp.float64)
    return jnp.asarray(results, dtype=jnp.float64)

def cdf_b_given_a_scipy(b, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    """Logically identical to :func:`cdf_b_given_a` but uses
    ``scipy.integrate.quad`` via :func:`integral_unnorm_prob_b_given_a_scipy`.

    Not JAX-traceable; intended as a reference / cross-check implementation.
    """
    valid_bounds = (xmin <= xmax) and (ymin <= ymax) and (bmin <= bmax) and (n_obs >= 0)
    if not valid_bounds:
        b = np.atleast_1d(np.asarray(b, dtype=np.float64))
        return jnp.full_like(b, jnp.nan)
    normalization = integral_unnorm_prob_b_given_a_scipy(bmax, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)
    cumulative_unnorm = integral_unnorm_prob_b_given_a_scipy(b, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)
    ratio = jnp.where(normalization > 0.0, cumulative_unnorm / normalization, 0.0)
    return jnp.clip(ratio, 0.0, 1.0)


## MAIN JUST FOR TESTING
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
        y = unnorm_prob_b_given_a(b, a_temp, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
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

    plt.figure('conditional_cdf_b')
    print("Plotting CDF(b|a) for different values of a...")
    a = jnp.linspace(1.25,3.75,10)
    a = a[((a<0) & (a*xmin>=ymin-bmax) & (a*xmax<=ymax-bmin) & (amin<=a) & (a<=amax)) |
          ((a==0) & (amin<=a) & (a<=amax)) | 
          ((a>0) & (a*xmin<=ymax-bmin) & (a*xmax>=ymin-bmax) & (amin<=a) & (a<=amax))]
    b = jnp.linspace(bmin, bmax, 10000)
    for a_temp in a:
        y1 = cdf_b_given_a(b, a_temp, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
        y2 = cdf_b_given_a_scipy(b, a_temp, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
        #y3 = cdf_b_given_a_monotone(b, a_temp, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
        plt.plot(b, y1, label=f'a={a_temp:.2f} (LUT)')
        plt.plot(b, y2, label=f'a={a_temp:.2f} (scipy)', linestyle='dashed')
        #plt.plot(b, y3, label=f'a={a_temp:.2f} (monotone)', linestyle='dotted')
    plt.legend(loc='best')

    plt.show()
    exit()

    # --- round-trip test: quantile(cdf(b)) ≈ b ---
    print("Round-trip test: quantile(cdf(b)) ≈ b ...")
    a_test = 2.5
    b_test = jnp.linspace(bmin, bmax, 500)
    u_test = cdf_b_given_a(b_test, a_test, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
    b_roundtrip = quantile_b_given_a(u_test, a_test, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
    # Only check within the support (where 0 < CDF < 1)
    mask = (u_test > 1e-6) & (u_test < 1.0 - 1e-6)
    max_err = jnp.max(jnp.abs(b_roundtrip[mask] - b_test[mask]))
    print(f"  max |quantile(cdf(b)) - b| inside support = {max_err:.2e}")

    plt.figure('quantile_b')
    print("Plotting quantile(u|a) for different values of a...")
    a = jnp.linspace(1.25,3.75,10)
    a = a[((a<0) & (a*xmin>=ymin-bmax) & (a*xmax<=ymax-bmin) & (amin<=a) & (a<=amax)) |
          ((a==0) & (amin<=a) & (a<=amax)) | 
          ((a>0) & (a*xmin<=ymax-bmin) & (a*xmax>=ymin-bmax) & (amin<=a) & (a<=amax))]
    u = jnp.linspace(0.0, 1.0, 500)
    for a_temp in a:
        b_out = quantile_b_given_a(u, a_temp, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
        plt.plot(u, b_out, label=f'a={a_temp:.2f}')
    plt.xlabel('u')
    plt.ylabel('b = quantile(u|a)')
    plt.legend(loc='best')

    plt.show()
    exit()