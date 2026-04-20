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
import optimistix as optx

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
        p(b|a) \propto f(b;a)=\mathbb{I}(b;b_{min},b_{max})L(a,b;x_{min},x_{max},y_{min},y_{max})^N
        
        
    """
    valid_bounds = (xmin <= xmax) & (ymin <= ymax) & (bmin <= bmax) & (n_obs >= 0)
    x_low_bound, x_high_bound = x_bounds_scalars(a,b,xmin,xmax,ymin,ymax)
    length = x_high_bound - x_low_bound #jnp.maximum(0.0, x_high_bound - x_low_bound)
    b_low_bound = jnp.maximum(bmin, ymin - jnp.maximum(a*xmin, a*xmax))
    b_high_bound = jnp.minimum(bmax, ymax - jnp.minimum(a*xmin, a*xmax))
    in_bounds = (b_low_bound <= b) & (b <= b_high_bound) & (x_low_bound < x_high_bound) & (xmin <= x_low_bound) & (x_high_bound <= xmax)
    return jnp.where(valid_bounds, jnp.where(in_bounds, jnp.exp(n_obs*jnp.log(length)), 0.0), jnp.nan)


@partial(jax.jit, static_argnames=['n_obs'])
def cdf_b_given_a_helper(b, a, xmin, xmax, ymin, ymax, n_obs):
    """
    Evaluates the analytical results of the integral:

    .. math::
        F(b;a) = \int_{b_{min}}^{b} db^{\prime}\,\mathbb{I}(b^{\prime};b_{min},b_{max})L(a,b^{\prime};x_{min},x_{max},y_{min},y_{max})^N

    .. math::
        L_{flat} = \begin{cases}
        x_{max} - x_{min} & \text{if } (y_{max} - a x_{max}) \geq (y_{min} - a x_{min}) \\
        \frac{y_{max} - y_{min}}{|a|} & \text{otherwise}
        \end{cases}
    
        
    """
    ax_max = jnp.maximum(a*xmin, a*xmax)
    ax_min = jnp.minimum(a*xmin, a*xmax)
    b1 = ymin - ax_max
    b2 = jnp.minimum(ymin - ax_min, ymax - ax_max)
    b3 = jnp.maximum(ymin - ax_min, ymax - ax_max)
    b4 = ymax - ax_min
    Lflat = jnp.where((ymax -ax_max) >= (ymin - ax_min), xmax - xmin, (ymax-ymin)/jnp.abs(a))
    result = jnp.where( a == 0.0, jnp.where(b<b1, 0.0, jnp.where((b2<=b) & (b<=b3), (b-b2)*jnp.exp(n_obs*jnp.log(Lflat)), (b3-b2)*jnp.exp(n_obs*jnp.log(Lflat)))), 
                       jnp.where(b<=b1, 0.0, 
                                 jnp.where(b<=b2, jnp.exp((n_obs+1)*jnp.log(b-b1)-n_obs*jnp.log(jnp.abs(a))-jnp.log(n_obs+1)),
                                           jnp.where(b<=b3,
                                                     jnp.exp((n_obs+1)*jnp.log(b2-b1)-n_obs*jnp.log(jnp.abs(a))-jnp.log(n_obs+1))+
                                                     (b-b2)*jnp.exp(n_obs*jnp.log(Lflat)),
                                                     jnp.where(b<b4,
                                                               jnp.exp((n_obs+1)*jnp.log(b2-b1)-n_obs*jnp.log(jnp.abs(a))-jnp.log(n_obs+1))+
                                                                (b3-b2)*jnp.exp(n_obs*jnp.log(Lflat))+
                                                                jnp.exp((n_obs+1)*jnp.log(b4-b3)-n_obs*jnp.log(jnp.abs(a))-jnp.log(n_obs+1))-
                                                                jnp.exp((n_obs+1)*jnp.log(b4-b)-n_obs*jnp.log(jnp.abs(a))-jnp.log(n_obs+1)),
                                                                jnp.exp((n_obs+1)*jnp.log(b2-b1)-n_obs*jnp.log(jnp.abs(a))-jnp.log(n_obs+1))+
                                                                (b3-b2)*jnp.exp(n_obs*jnp.log(Lflat))+
                                                                jnp.exp((n_obs+1)*jnp.log(b4-b3)-n_obs*jnp.log(jnp.abs(a))-jnp.log(n_obs+1))
                                                               )
                                                     )
                                           )
                                 )
                       ) 
    return jnp.clip(result, 0.0, +jnp.inf)


@partial(jax.jit, static_argnames=['n_obs'])
def prob_b_given_a(b, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    r"""
    The conditional prior for b given a, under the uninformative joint 
    prior for the linear model y=ax+b:

    .. math::
        p(b|a) = \frac{\mathbb{I}(b;b_{min},b_{max})L(a,b;x_{min},x_{max},y_{min},y_{max})^N}{F(b_{high bound};a)-F(b_{low bound};a)}
        
        
    """
    valid_bounds = (xmin <= xmax) & (ymin <= ymax) & (bmin <= bmax) & (n_obs >= 0)
    x_low_bound, x_high_bound = x_bounds_scalars(a,b,xmin,xmax,ymin,ymax)
    b_low_bound = jnp.maximum(bmin, ymin - jnp.maximum(a*xmin, a*xmax))
    b_high_bound = jnp.minimum(bmax, ymax - jnp.minimum(a*xmin, a*xmax))
    ax_max = jnp.maximum(a*xmin, a*xmax)
    ax_min = jnp.minimum(a*xmin, a*xmax)
    b1 = ymin - ax_max
    b2 = jnp.minimum(ymin - ax_min, ymax - ax_max)
    b3 = jnp.maximum(ymin - ax_min, ymax - ax_max)
    b4 = ymax - ax_min
    Lflat = jnp.where((ymax -ax_max) >= (ymin - ax_min), xmax - xmin, (ymax-ymin)/jnp.abs(a))
    prob = jnp.where( a == 0.0, jnp.where((b2<=b) & (b<=b3), jnp.exp(n_obs*jnp.log(Lflat)), 0.0), 
                       jnp.where(b<=b1, 0.0, 
                                 jnp.where(b<=b2, jnp.exp(n_obs*jnp.log(b-b1)-n_obs*jnp.log(jnp.abs(a))),
                                           jnp.where(b<=b3,
                                                     jnp.exp(n_obs*jnp.log(Lflat)),
                                                     jnp.where(b<b4,
                                                                jnp.exp(n_obs*jnp.log(b4-b)-n_obs*jnp.log(jnp.abs(a))),
                                                                0.0
                                                               )
                                                     )
                                           )
                                 )
                       ) 
    in_bounds = (b_low_bound <= b) & (b <= b_high_bound) & (x_low_bound < x_high_bound) & (xmin <= x_low_bound) & (x_high_bound <= xmax)
    denominator = (cdf_b_given_a_helper(b_high_bound, a, xmin, xmax, ymin, ymax, n_obs)-
                  cdf_b_given_a_helper(b_low_bound, a, xmin, xmax, ymin, ymax, n_obs))
    return jnp.where(valid_bounds, jnp.where(in_bounds & (denominator>0.0), prob/denominator, 0.0), jnp.nan)


@partial(jax.jit, static_argnames=['n_obs'])
def cdf_b_given_a(b, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    """
    Returns the normalized conditional cumulative density function for b given a,
    under the uninformative joint prior for the linear model y=ax+b:

    .. math::
        CDF(b|a) = \frac{F(b;a)-F(b_{low bound};a)}{F(b_{high bound};a)-F(b_{low bound};a)}

    .. math::
        L_{flat} = \begin{cases}
        x_{max} - x_{min} & \text{if } (y_{max} - a x_{max}) \geq (y_{min} - a x_{min}) \\
        \frac{y_{max} - y_{min}}{|a|} & \text{otherwise}
        \end{cases}
    
        
    """
    b_low_bound = jnp.maximum(bmin, ymin - jnp.maximum(a*xmin, a*xmax))
    b_high_bound = jnp.minimum(bmax, ymax - jnp.minimum(a*xmin, a*xmax))
    numerator = (cdf_b_given_a_helper(b, a, xmin, xmax, ymin, ymax, n_obs)-
                  cdf_b_given_a_helper(b_low_bound, a, xmin, xmax, ymin, ymax, n_obs))    
    denominator = (cdf_b_given_a_helper(b_high_bound, a, xmin, xmax, ymin, ymax, n_obs)-
                  cdf_b_given_a_helper(b_low_bound, a, xmin, xmax, ymin, ymax, n_obs))
    cdf = jnp.where(denominator > 0.0,
                    jnp.where(b<=bmin, 0.0,
                              jnp.where(b<bmax, numerator/denominator,
                                        1.0)),
                    jnp.nan)
    return jnp.clip(cdf, 0.0, 1.0)

@partial(jax.jit)
def inversion_cdf_b_given_a_residual(b, args):
    F_target, a, xmin, xmax, ymin, ymax, n_obs = args
    return cdf_b_given_a_helper(b, a, xmin, xmax, ymin, ymax, n_obs) - F_target

@partial(jax.jit, static_argnames=['n_obs'])
def inverse_cdf_b_given_a(ub, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    r"""Quantile function :math:`b=CDF^{-1}(u|a)`.

    Maps :math:`u \sim \mathrm{Uniform}(0,1)` to *b* such that
    :math:`\mathrm{CDF}(b\mid a) = u`.

    Parameters
    ----------
    u      : scalar or array — quantile(s) in [0, 1]
    """
    valid_bounds = (xmin <= xmax) & (ymin <= ymax) & (bmin <= bmax) & (n_obs >= 0)
    b_low_bound = jnp.maximum(bmin, ymin - jnp.maximum(a*xmin, a*xmax))
    b_high_bound = jnp.minimum(bmax, ymax - jnp.minimum(a*xmin, a*xmax))
    safe_epsilon = 0.5*jnp.finfo(jnp.float64).eps
    ub_clipped = jnp.clip(ub, 0.0 + safe_epsilon, 1.0 - safe_epsilon)
    F_low = cdf_b_given_a_helper(b_low_bound, a, xmin, xmax, ymin, ymax, n_obs)
    F_high = cdf_b_given_a_helper(b_high_bound, a, xmin, xmax, ymin, ymax, n_obs)
    F_target = jnp.clip(F_low + ub_clipped * (F_high - F_low), 0.0, +jnp.inf)
    b_result = jnp.where((ub >= 0.0) & (ub < safe_epsilon), b_low_bound,
                         jnp.where((ub > 1.0 - safe_epsilon) & (ub <= 1.0), b_high_bound,
                                   jnp.where((0.0 < ub) & (ub < 1.0),
                                   optx.root_find(inversion_cdf_b_given_a_residual,
                                                    optx.Bisection(rtol=os.sys.float_info.epsilon,
                                                                    atol=os.sys.float_info.epsilon
                                                                    ),
                                                    y0=0.5*(b_high_bound+b_low_bound),
                                                    args=(F_target, a, xmin, xmax, ymin, ymax, n_obs),
                                                    options={"lower": b_low_bound-safe_epsilon, "upper": b_high_bound+safe_epsilon}, #dictionary with bounds
                                                    max_steps=512,
                                                    throw=False                                           
                                                ).value,
                                                jnp.nan
                                            )
                                    )
                        )
    b_result = jnp.clip(b_result, b_low_bound, b_high_bound)
    return jnp.where(valid_bounds, b_result, jnp.nan)

## PROBABILITY p(a)
@partial(jax.jit, static_argnames=['n_obs'])
def unnorm_prob_a(a, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    r"""
    The unnormalized marginalized prior for `a`, under the uninformative joint 
    prior for the linear model y=ax+b:

    .. math::
        \pi(a) \propto f(a)=\mathbb{I}(a;a_{min},a_{max})(1+a^2)^{\frac{N-3}{2}}(F(b_{high bound};a)-F(b_{low bound};a))
        
        
    """
    valid_bounds = (xmin <= xmax) & (ymin <= ymax) & (bmin <= bmax) & (amin<= amax) & (n_obs >= 0)
    b_low_bound = jnp.maximum(bmin, ymin - jnp.maximum(a*xmin, a*xmax))
    b_high_bound = jnp.minimum(bmax, ymax - jnp.minimum(a*xmin, a*xmax))
    in_bounds = (amin <= a) & (a <= amax) & (b_low_bound <= b_high_bound) & (((a<0) & (a*xmin>=ymin-bmax) & (a*xmax<=ymax-bmin) & (amin<=a) & (a<=amax)) |
          ((a==0) & (amin<=a) & (a<=amax)) | 
          ((a>0) & (a*xmin<=ymax-bmin) & (a*xmax>=ymin-bmax) & (amin<=a) & (a<=amax)))
    output = jnp.where(valid_bounds, jnp.clip(jnp.where(in_bounds,
                                            jnp.exp(0.5*(n_obs-3)*jnp.log(1+a**2)+jnp.log(
                                                cdf_b_given_a_helper(b_high_bound, a, xmin, xmax, ymin, ymax, n_obs)-
                                                cdf_b_given_a_helper(b_low_bound, a, xmin, xmax, ymin, ymax, n_obs)
                                                )),
                                            0.0),0.0,+jnp.inf),
                    jnp.nan)
    return output

def normalization_prob_a(amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    r"""
    Numerically evaluates the integral necessary to normalize the marginalized prior 
    for `a`, under the uninformative joint prior for the linear model y=ax+b:

    .. math::
        Z = \int_{-\infty}^{\infty}da\mathbb{I}(a;a_{min},a_{max})(1+a^2)^{\frac{N-3}{2}}(F(b_{high bound};a)-F(b_{low bound};a))
        
    """
    Z, _ = quad(unnorm_prob_a, amin, amax, args=(amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs), 
                epsabs=os.sys.float_info.epsilon , epsrel=os.sys.float_info.epsilon)
    return Z

@partial(jax.jit, static_argnames=['normalization','amin','amax','bmin','bmax',
                                   'xmin','xmax','ymin','ymax','n_obs'])
def prob_a(a, normalization, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    r"""
    The marginalized prior for `a`, under the uninformative joint 
    prior for the linear model y=ax+b:

    .. math::
        \pi(a)=\frac{\mathbb{I}(a;a_{min},a_{max})(1+a^2)^{\frac{N-3}{2}}(F(b_{high bound};a)-F(b_{low bound};a))}
        {\int_{-\infty}^{\infty}da\mathbb{I}(a;a_{min},a_{max})(1+a^2)^{\frac{N-3}{2}}(F(b_{high bound};a)-F(b_{low bound};a))}
        
    """
    return jnp.where(normalization > 0.0, unnorm_prob_a(a, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)/normalization, jnp.nan)

# --- Finding the effective bounds for the support of :math:`CDF(a)` ---
def find_effective_bounds_a():
    return 0.0

# --- Building the LUT for :math:`CDF(a)` with a non-uniform grid ---
"""
PSEUDOALGORITHM:
Pass 1: Coarse uniform grid (n = 1000–2000) over [a_eff_min, a_eff_max]
        → trapezoidal cumulative sum → rough CDF → rough quantile

Pass 2: Place n_final points CDF-equispaced using the rough quantile:
          u_grid = linspace(0, 1, n_final)
          a_grid = rough_quantile(u_grid)
        → evaluate unnorm_prob_a on a_grid
        → trapezoidal cumulative sum → refined CDF + store PDF values

Pass 3 (optional): Repeat pass 2 using the refined quantile
        → even better grid placement

Termination: compare CDF from pass k vs pass k-1 at test points,
             or check max |CDF_k(a) - CDF_{k-1}(a)| < ε
"""
def build_cdf_a_lut_iterative():
    return 0.0


# --- Hermite interpolation ---
def hermite_interp_cdf_a(a, a_grid, cdf_table, pdf_table):
    """
    Cubic Hermite spline interpolation for :math:`CDF(a)`

    See https://en.wikipedia.org/wiki/Cubic_Hermite_spline
    
    
    """

    return 0.0


# --- Evaluate the quantile function Q(ua)=a starting from the CDF(a) LUT ---
@partial(jax.jit, static_argnames=['normalization', 'amin', 'amax', 'bmin', 'bmax', 'xmin', 
                                   'xmax', 'ymin', 'ymax', 'n_obs', 'newton_steps'])
def quantile_a(ua, a_grid, cdf_table, pdf_table, normalization, amin, amax, bmin, 
               bmax, xmin, xmax, ymin, ymax, n_obs, newton_steps=4):
    """
    Finds :math:`a=CDF(u_{a})` from the interpolation of an existing look-up table 
    for :math:`CDF(a)` and refining the initial guess.
    
    1. Linear interpolation of the existing LUT grid provides initial guess
    2. Newton steps where the CDF value is estimated via Hermite interpolation of the
        existing LUT, and the PDF value is estimated from the normalized probability function.

    We exploit the knowledge of the first derivative of :math:`CDF(a)` marginal pdf 
    to speed up Newton method computation.

    """
    # Start from the interpolation of the existing LUT for CDF(a)
    a_init = jnp.interp(ua, cdf_table, a_grid) #interpolates using the CDF(a) as `x` and the `a` as `y`
    # Refining the initial guess with a few steps of Newton's method
    # We exploit the knowledge of the marginal pdf :math:`\pi(a)` to help speed up Newton
    def newton_step(a, _):
        cdf_value = hermite_interp_cdf_a(a, a_grid, cdf_table, pdf_table)
        pdf_value = unnorm_prob_a(a, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)/normalization
        safe_pdf_val = jnp.clip(pdf_value, jnp.finfo(jnp.float64).tiny, jnp.inf)


    return 0.0






## p(b|a) JAXED WITH LUTS
@partial(jax.jit, static_argnames=['n_obs', 'n_grid'])
def build_cdf_b_given_a_lut(a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs, n_grid=2000):
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
def cdf_b_given_a_lut(b, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs, n_grid=2000):
    r"""
    Returns the CDF(b|a) evaluated at arbitrary **b** values for fixed **a**.

    Builds an internal grid of size *n_grid*, computes the non-decreasing CDF 
    via cumulative trapezoidal integration, and returns the CDF evaluated at the
    requested *b* value(s) through direct index computation + linear
    interpolation (O(1) per point).

    Parameters
    ----------
    b      : scalar or array — query point(s)
    n_grid : int (static) — internal grid resolution (default 2000)
    """
    valid_bounds = (xmin <= xmax) & (ymin <= ymax) & (bmin <= bmax) & (n_obs >= 0)
    b_grid, cdf_table = build_cdf_b_given_a_lut(a, bmin, bmax, xmin, xmax, ymin, ymax,
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

@partial(jax.jit, static_argnames=['n_obs', 'n_grid'])
def quantile_b_given_a_lut(u, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs, n_grid=2000):
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
    b_grid, cdf_table = build_cdf_b_given_a_lut(a, bmin, bmax, xmin, xmax, ymin, ymax,
                                         n_obs, n_grid)
    result = jnp.interp(u, cdf_table, b_grid)
    return jnp.where(valid_bounds, result, jnp.nan)


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
    xmin = -15.0
    xmax = jnp.log10(3.0e5)+1
    ymin = -1.0
    ymax = 25.0
    amin = -jnp.inf
    amax = jnp.inf
    bmin = -1000
    bmax = 1000
    Nobs = 10

    plt.figure('prob_x')
    plt.title('p(x|a,b) for different values of a')
    print("Plotting p(x|a,b) for different values of a...")
    x = jnp.linspace(xmin-5.0, xmax+5.0, 1000)
    a = jnp.logspace(-1.5,1.0,10)
    b = 2.0
    for a_temp in a:
        prob_x_ab = prob_x_given_ab(x,a_temp,b,xmin,xmax,ymin,ymax)
        #print(f"a={a_temp:.2f}, prob_x_ab={prob_x_ab}")
        plt.plot(x, prob_x_ab, label=f'a={a_temp:.2f}')
    plt.legend(loc='best')

    plt.figure('prob_b_unnorm')
    plt.title('unnormalized p(b|a) for different values of a')
    print("Plotting p(b|a) for different values of a...")
    a = jnp.logspace(-1.5,1.0,10)
    b = jnp.linspace(bmin, bmax, 2500)
    for a_temp in a:
        y = unnorm_prob_b_given_a(b, a_temp, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
        plt.plot(b, y, label=f'a={a_temp:.2f}')
    plt.legend(loc='best')

    plt.figure('prob_b')
    print("Plotting p(b|a) for different values of a...")
    plt.title('p(b|a) for different values of a')
    a = jnp.linspace(1.25,3.75,10)
    a = a[((a<0) & (a*xmin>=ymin-bmax) & (a*xmax<=ymax-bmin) & (amin<=a) & (a<=amax)) |
          ((a==0) & (amin<=a) & (a<=amax)) | 
          ((a>0) & (a*xmin<=ymax-bmin) & (a*xmax>=ymin-bmax) & (amin<=a) & (a<=amax))]
    b = jnp.linspace(bmin, bmax, 2500)
    for a_temp in a:
        y = prob_b_given_a(b, a_temp, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
        plt.plot(b, y, label=f'a={a_temp:.2f}')
    plt.legend(loc='best')

    plt.figure('conditional_cdf_b')
    print("Plotting CDF(b|a) for different values of a...")
    plt.title('CDF(b|a) for different values of a')
    a = jnp.linspace(1.25,3.75,10)
    a = a[((a<0) & (a*xmin>=ymin-bmax) & (a*xmax<=ymax-bmin) & (amin<=a) & (a<=amax)) |
          ((a==0) & (amin<=a) & (a<=amax)) | 
          ((a>0) & (a*xmin<=ymax-bmin) & (a*xmax>=ymin-bmax) & (amin<=a) & (a<=amax))]
    b = jnp.linspace(bmin, bmax, 10000)
    for a_temp in a:
        y1 = cdf_b_given_a(b, a_temp, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
        y2 = cdf_b_given_a_lut(b, a_temp, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
        #y3 = cdf_b_given_a_monotone(b, a_temp, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
        plt.plot(b, y1, label=f'a={a_temp:.2f} (exact)')
        plt.plot(b, y2, label=f'a={a_temp:.2f} (LUT)', linestyle='dashed')
        #plt.plot(b, y3, label=f'a={a_temp:.2f} (monotone)', linestyle='dotted')
    plt.legend(loc='best')

    plt.figure('expected_quantile_function_b')
    print("Plotting CDF^{-1}(b|u,a) for different values of a...")
    a = jnp.linspace(1.25,3.75,10)
    a = a[((a<0) & (a*xmin>=ymin-bmax) & (a*xmax<=ymax-bmin) & (amin<=a) & (a<=amax)) |
          ((a==0) & (amin<=a) & (a<=amax)) | 
          ((a>0) & (a*xmin<=ymax-bmin) & (a*xmax>=ymin-bmax) & (amin<=a) & (a<=amax))]
    b = jnp.linspace(bmin, bmax, 10000)
    #ub = jnp.linspace(0.0, 1.0, 10000)
    ub = 0.5+0.5*jnp.sort(jnp.concatenate([-jax.nn.sigmoid(jnp.logspace(-16.0, 16.0, 5000)), jax.nn.sigmoid(jnp.logspace(-16.0, 16.0, 5000))])) # to avoid numerical issues at the edges
    # --- Plotting loop ---
    for a_temp in a:
        y1 = cdf_b_given_a(b, a_temp, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
        y2 = jax.vmap(inverse_cdf_b_given_a, in_axes=(0, None, None, None, None, None, None, None, None))(
    ub, a_temp, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
        plt.plot(y1, b, label=f'a={a_temp:.2f} (expected)')
        plt.plot(ub, y2, label=f'a={a_temp:.2f} (root find)', linestyle='dashed')
    plt.legend(loc='best')

    plt.figure('unnorm_prob_a')
    print("Plotting f(a) with respect to a...")
    a = jnp.linspace(-10,10,2000)
    y = unnorm_prob_a(a, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
# --- Plotting loop ---
    plt.plot(a, y, label='f(a)')
    plt.legend(loc='best')

    plt.show()
    exit()