import os
import math
from functools import partial
import warnings
# Keep computations in FP64 and on CPU for reproducibility.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import jax
import jax.numpy as jnp
from quadax import quadcc, quadgk, quadts, romberg
import numpy as np
from scipy.integrate import quad
import optimistix as optx
from jaxns.framework.special_priors import SpecialPrior

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

## CONDITIONAL PROBABILITY p(x|a,b,c)
@jax.jit
def find_x_bounds(c,a,b,xmin, xmax, ymin, ymax):
    _4ac_min = jnp.minimum(4*c*(ymax-b),4*c*(ymin-b))
    _4ac_max = jnp.maximum(4*c*(ymax-b),4*c*(ymin-b))
    delta_min = a**2 + _4ac_min
    delta_max = a**2 + _4ac_max
    if delta_max < 0.0:
        xa = -jnp.inf
        xb = jnp.inf
        xc = jnp.inf
        xd = jnp.inf
    else:
        x_vertex = -0.5*a/c
        sqrt_delta_max = 0.5*jnp.sqrt(delta_max)/jnp.abs(c)
        xa = x_vertex - sqrt_delta_max
        xd = x_vertex + sqrt_delta_max
        if delta_min < 0.0:
            xb = 0.5*(xa+xd)
            xc = xb
        else:
            sqrt_delta_min = 0.5*jnp.sqrt(delta_min)/jnp.abs(c)
            xb = x_vertex - sqrt_delta_min
            xc = x_vertex + sqrt_delta_min
    return xa, xb, xc, xd


@jax.jit
def prob_x_given_bounds(x,xa,xb,xc,xd):
    r"""
    Defines a probability distribution which is uniform over a possibly disjoint 
    union of two intervals :math:`[x_a, x_b] \cup [x_c, x_d]` and zero elsewhere.
    """
    valid_bounds = (xa <= xb) & (xb <= xc) & (xc <= xd) & ((xa < xb) | (xc < xd))
    return jnp.where(not valid_bounds, jnp.nan, jnp.where((xa<=x<=xb) | (xc<=x<=xd), 1.0/(xd-xc+xb-xa), 0.0))

@jax.jit
def cdf_x_given_bounds(x,xa,xb,xc,xd):
    valid_bounds = (xa <= xb) & (xb <= xc) & (xc <= xd) & (xa < xd)
    L = (xd-xc+xb-xa)
    m = 1.0/L
    q = jnp.where(xa<=x<=xb, -xa*m, jnp.where(xc<=x<=xd, (-xc+xb-xa)*m, (xb-xa)*m))
    const = jnp.where(xb<=x<=xc, m*(xb-xa), jnp.where(x<xd, 0.0, 1.0))
    return jnp.where(not valid_bounds, jnp.nan,
                    jnp.where((xa<=x<xb) | (xc<x<=xd), m*x+q, const)
                    )

@jax.jit
def quantile_x_from_ux(u, xa, xb, xc, xd):
    valid_bounds = (xa <= xb) & (xb <= xc) & (xc <= xd) & (xa < xd) & (0.0 <= u) & (u <= 1.0)
    L = (xd-xc+xb-xa)
    return jnp.where(not valid_bounds, jnp.nan, 
                     jnp.where(u<=(xb-xa)/L, xa+L*u, xa+xc-xb+L*u)
                     )

@jax.jit
def Lx(u, xa, xb, xc, xd):
    valid_bounds = (xa <= xb) & (xb <= xc) & (xc <= xd) & (xa < xd) & (0.0 <= u) & (u <= 1.0)
    return jnp.where(not valid_bounds, jnp.nan, xd-xc+xb-xa)

class PriorX_UninformQuadraticJAXNS(SpecialPrior):
    """
    Marginal probability :math:`\pi(x|a,b,c)`
    """
    def __init__(self, c, a, b, cmin, cmax, amin, amax, bmin, 
                 bmax, xmin, xmax, ymin, ymax, name=None):
        super().__init__(name=name)
        self.xa, self.xb, self.xc, self.xd = find_x_bounds(c, a, b, xmin, xmax, ymin, ymax)

    def _dtype(self): return jnp.float64
    def _base_shape(self): return ()
    def _shape(self): return ()

    def _forward(self, U):
        """
        Quantile function (inverse CDF)
        """
        return quantile_x_from_ux(U, self.xa, self.xb, self.xc, self.xd)
    
    def _inverse(self, X):
        """
        CDF (cumulative density function)
        """
        return cdf_x_given_bounds(X, self.xa, self.xb, self.xc, self.xd)
    
    def _log_prob(self, X):
        """
        Logarithm of the PDF (probability density function)
        """
        L =self.xd-self.xc+self.xb-self.xa
        valid_bounds = (self.xa <= self.xb) & (self.xb <= self.xc) & (self.xc <= self.xd) & ((self.xa < self.xb) | (self.xc < self.xd))
        return jnp.where(not valid_bounds, jnp.nan, jnp.where((self.xa<=X<=self.xb) | (self.xc<=X<=self.xd), -jnp.log(L), -jnp.inf))
    

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

def normalization_prob_a(amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs, 
                         epsabs=os.sys.float_info.epsilon, epsrel=os.sys.float_info.epsilon, 
                         limit=int(1e6)):
    r"""
    Numerically evaluates the integral necessary to normalize the marginalized prior 
    for `a`, under the uninformative joint prior for the linear model y=ax+b:

    .. math::
        Z = \int_{-\infty}^{\infty}da\mathbb{I}(a;a_{min},a_{max})(1+a^2)^{\frac{N-3}{2}}
        (F(b_{high bound};a)-F(b_{low bound};a))


    To try and provide accurate numerical integration even when the bounds :math:`a_{min}` 
    and :math:`a_{max}` on `a` are very large or infinite, the integration is performed 
    after applying the transformation :math:`a = \tan\left(\frac{\pi}{2}t\right)`, 
    which maps :math:`t \in (-1, 1)` to :math:`a \in (-\infty, +\infty)`.
    The probability density function is changed according to:

    .. math::
        \pi_{a}(a)da = \pi_{a}(a(t))\left|\frac{da}{dt}\right|dt = \pi_{t}(t)dt
        
    """
    if ((amin != -np.inf) and (amax != +np.inf) and (np.abs(amin) > 0.01*np.finfo(np.float64).max) 
        and (np.abs(amax) > 0.01*np.finfo(np.float64).max)):
        # Direct integration in a-space fails when amin/amax are near ±float64_max
        # because scipy.integrate.quad cannot handle ranges of order 1e307.
        t_epsilon = 1.5*jnp.finfo(jnp.float64).eps
        tmin = max(float(a_to_t_map(jnp.asarray(amin, dtype=jnp.float64))), -1.0 + t_epsilon)
        tmax = min(float(a_to_t_map(jnp.asarray(amax, dtype=jnp.float64))), 1.0 - t_epsilon)

        def integrand_t(t):
            a = float(np.tan(0.5 * np.pi * t))
            f_a = float(unnorm_prob_a(jnp.asarray(a, dtype=jnp.float64),
                                    amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs))
            jacobian = 0.5 * np.pi * (1.0 + a ** 2)
            val = f_a * jacobian
            return 0.0 if (not np.isfinite(val) or val < 0.0) else val
        
    else:
        tmin = amin
        tmax = amax
        def integrand_t(t):
            val = unnorm_prob_a(t, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)
            return 0.0 if (not np.isfinite(val) or val < 0.0) else val
        
    # Insert a=0 as a known kink point if it lies in the domain
    points = [0.0] if (np.isfinite(amin) and np.isfinite(amax) and tmin <= 0.0 <= tmax) else None

    Z, _ = quad(integrand_t, tmin, tmax, epsabs=epsabs, epsrel=epsrel, limit=limit, points=points)
    return Z

@partial(jax.jit, static_argnames=['n_obs'])
def prob_a(a, normalization, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    r"""

    The marginalized prior for `a`, under the uninformative joint 
    prior for the linear model y=ax+b:

    .. math::
        \pi(a)=\frac{\mathbb{I}(a;a_{min},a_{max})(1+a^2)^{\frac{N-3}{2}}(F(b_{high bound};a)-F(b_{low bound};a))}
        {\int_{-\infty}^{\infty}da\mathbb{I}(a;a_{min},a_{max})(1+a^2)^{\frac{N-3}{2}}(F(b_{high bound};a)-F(b_{low bound};a))}
        
    """
    return jnp.where(normalization > 0.0, 
                     unnorm_prob_a(a, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)/normalization, 
                     jnp.nan
                     )

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
@jax.jit
def t_to_a_map(t):
    t = jnp.clip(t, -1.0, +1.0)
    return jnp.where( t<= -1.0, -jnp.inf, jnp.where(t>= 1.0, +jnp.inf, jnp.tan(0.5*jnp.pi*t)))

@jax.jit
def a_to_t_map(a):
    inv_pi = 1.0 / jnp.pi
    return 2*inv_pi*jnp.arctan(a)

@partial(jax.jit, static_argnames=['n_obs', 'use_linear'])
def prob_a_of_t(t, normalization, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs, use_linear=True):
    t = jnp.clip(t, -1.0, +1.0)
    if use_linear:
        a = amin + 0.5 * (amax - amin) * (t + 1.0)
        jacobian = 0.5 * (amax - amin)
    else:
        a = t_to_a_map(t)
        # Guard against 0*inf=NaN at t=±1 where a=±inf but prob_a=0
        jacobian = jnp.where(jnp.isfinite(a), 0.5 * jnp.pi * (1.0 + a ** 2), 0.0)
    return prob_a(a, normalization, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs) * jacobian

### MY IMPLEMENTATION
@jax.jit
def cdf_from_pdf_over_grid(t_grid, pdf_t):
    r"""Build a normalized, non-decreasing CDF via the trapezoidal rule 
    starting from a normalized `pdf`.

    Parameters
    ----------
    t_grid : sorted array — abscissa.
    pdf_t  : array — PDF values at *t_grid* points.

    Returns
    -------
    cdf_values : array, shape ``(len(t_grid),)``, values in ``[0, 1]``.
    """
    dt = jnp.diff(t_grid)
    trapezoid_area = jnp.clip(0.5 * (pdf_t[:-1] + pdf_t[1:]) * dt, 0.0, np.inf)
    cumulative = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(trapezoid_area)])
    cdf_values = jnp.where(cumulative[-1] <= 0.0, jnp.nan, 
                           jnp.clip(jnp.where(cumulative[-1] != 1.0, cumulative / cumulative[-1], 
                                              cumulative
                                              ), 
                                    0.0, 1.0
                                    )
                            )
    return cdf_values

@partial(jax.jit, static_argnames=['n_grid'])
def build_cdf_equispaced_t_grid(x_grid, cdf_vals, xmin, xmax, n_grid=2000):
    r"""
    Creates a grid of *n_grid* points equally spaced in :math:`u = CDF(x)` 
    space. Each element of the array is :math:`u_j = j/(n-1)`

    Then for each :math:`u_j = j/(n-1)` finds :math:`x_j = Q(u_j)` via linear
    interpolation of the *(cdf_vals, t_grid)* table.

    Returns a sorted array of *n_grid* x-values clipped to ``[xmin, xamx]`` 
    and the corresponding uniform CDF values.
    """
    u_uniform = jnp.linspace(0.0, 1.0, n_grid)
    x_new_grid = jnp.interp(u_uniform, cdf_vals, x_grid)
    return jnp.clip(jnp.sort(x_new_grid), xmin, xmax), u_uniform

def inject_zero_anchor_point(t_grid, pdf_t_grid, t0_anchor, normalization, amin, 
                             amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs, 
                             use_linear=True
                             ):
    p0 = np.array(prob_a_of_t(jnp.asarray(t0_anchor, dtype=np.float64), normalization, 
                              amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs, 
                                use_linear=use_linear),
                        dtype=np.float64)
    np.nan_to_num(p0, nan=0.0, copy=False)
    p0 = np.clip(p0, 0.0, np.inf)
    t_grid = np.concatenate([t_grid, np.array([t0_anchor], dtype=np.float64)])
    pdf_t_grid = np.concatenate([pdf_t_grid, np.array([p0], dtype=np.float64)])
    indices_sorted = np.argsort(t_grid, kind='stable')
    t_grid = t_grid[indices_sorted]
    pdf_t_grid = pdf_t_grid[indices_sorted]
    return t_grid, pdf_t_grid


def adaptive_mesh_refinement_cdf(t_grid, pdf_t_grid, normalization,
                          amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs,
                          tol=jnp.finfo(np.float64).eps, max_points=int(1e5),
                          use_linear = True
                          ):
    t_array = np.array(t_grid, dtype=np.float64)
    pdf_t_array = np.array(pdf_t_grid, dtype=np.float64)

    converged = False
    n_points = len(t_array)
    while (n_points <= max_points):
        # Estimate the increment between 
        dt = np.diff(t_array)
        t_mid = 0.5 * (t_array[:-1] + t_array[1:])
        # Evaluates the pdf at the midpoints of the existing grid
        prob_mid = np.array(
                        prob_a_of_t(jnp.asarray(t_mid, dtype=jnp.float64),
                                    normalization, amin, amax, bmin, bmax,
                                    xmin, xmax, ymin, ymax, n_obs, 
                                    use_linear=use_linear
                                    ),
                        dtype=np.float64,
                    )
        np.nan_to_num(prob_mid, nan=0.0, copy=False)
        prob_mid = np.clip(prob_mid, 0.0, np.inf)
        # Estimates the error via the Newton-Cotes formula 
        # for estimating numerical integration error
        newton_cotes_err = (2.0 * dt/3.0) * np.abs(prob_mid - 0.5 * (pdf_t_array[:-1] + pdf_t_array[1:]))
        not_within_tol = newton_cotes_err > tol
        n_bad = int(np.sum(not_within_tol))
        # If all points are within the tolerance, stop the refinement
        if n_bad == 0:
            converged = True 
            break
        # Otherwise keeps on until the maximum allowed number of grid 
        # points is reached
        slots_left = max_points - n_points
        # If the `slots_left` are not enough to refine the grid to guarantee 
        # the required precision, the refinement prioritezes the points with 
        # the hightest estimated error
        if n_bad > slots_left:
            # Sort from worst error to last (`-1` in the indices)
            # `[:slots_left]` selects the first `slots_left` grid intervals 
            # with worst estimated error
            worst = np.argsort(newton_cotes_err)[::-1][:slots_left]
            # Creates the mask for the intervals to refine
            refine_mask = np.zeros(len(newton_cotes_err), dtype=bool)
            refine_mask[worst] = True
            not_within_tol = refine_mask
        # Appends the new points of the refined grid to the end of the arrays
        t_array = np.concatenate([t_array, t_mid[not_within_tol]])
        pdf_t_array = np.concatenate([pdf_t_array, prob_mid[not_within_tol]])
        # Sorts the arrays to mantain an ordered `t_grid` array
        indices_sorted = np.argsort(t_array, kind='stable')
        # Refined grid at the end of this pass. Either output of the AMR step 
        # or input of the next pass if neither convergence criterion is met
        t_array = t_array[indices_sorted]
        pdf_t_array = pdf_t_array[indices_sorted]
        n_points = len(t_array)

        # Throw a warning if the loop is exited before the required precision 
        # is reached
        if not converged:
            warnings.warn(
                f"adaptive_mesh_refinement_cdf: reached max_points={max_points} before "
                f"satisfying tol={tol:.3e}. The CDF grid may not meet the requested precision. "
                f"Consider increasing max_points or relaxing tol.",
                RuntimeWarning,
                stacklevel=2,
            )

    return t_array, pdf_t_array                      

def build_cdf_a_lut(normalization, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs, 
                    n_coarse_grid=2000, tol=jnp.finfo(np.float64).eps, max_points=int(1e5),
                    use_linear = True
                    ):
    if use_linear:
        tmin = -1.0
        tmax = 1.0
    else:
        # Clamp tmin/tmax away from ±1: for very large finite bounds (e.g. ±0.45·fmax)
        # a_to_t_map rounds to exactly ±1.0 in float64, causing t_to_a_map→±inf and
        # 0*inf=NaN in the PDF.  t_eps=1e-7 keeps |a| ≲ 2/(π·1e-7) ≈ 6.4e6.
        t_epsilon = 1.5*jnp.finfo(jnp.float64).eps
        tmin = float(np.clip(float(a_to_t_map(jnp.asarray(amin, dtype=jnp.float64))),
                            -1.0 + t_epsilon, 1.0 - t_epsilon))
        tmax = float(np.clip(float(a_to_t_map(jnp.asarray(amax, dtype=jnp.float64))),
                            -1.0 + t_epsilon, 1.0 - t_epsilon))
    if amin <= 0.0 <= amax:
        t0_anchor = 0.0 if not use_linear else (-(amin + amax) / (amax - amin))
    else:
        t0_anchor = None

    # PASS 1: coarse uniform grid
    t_grid = np.linspace(tmin, tmax, n_coarse_grid)
    pdf_t = prob_a_of_t(t_grid, normalization, amin, amax, bmin, bmax, xmin, xmax, 
                        ymin, ymax, n_obs, use_linear=use_linear
                        )
    cdf_table = cdf_from_pdf_over_grid(t_grid, pdf_t)

    # PASS 2: CDF-equispaced grid
    t_grid, _ = build_cdf_equispaced_t_grid(t_grid, cdf_table, tmin, tmax, n_grid=n_coarse_grid)
    pdf_t = prob_a_of_t(t_grid, normalization, amin, amax, bmin, bmax, xmin, xmax, 
                        ymin, ymax, n_obs, use_linear=use_linear
                        )
    if t0_anchor is not None:
        t_grid, pdf_t = inject_zero_anchor_point(t_grid, pdf_t, t0_anchor, normalization,
                                                amin, amax, bmin, bmax, xmin, xmax, 
                                                ymin, ymax, n_obs, use_linear=use_linear
                                                )
    cdf_table = cdf_from_pdf_over_grid(t_grid, pdf_t)

    # PASS 3: CDF-equispaced grid
    t_grid, _ = build_cdf_equispaced_t_grid(t_grid, cdf_table, tmin, tmax, n_grid=2*n_coarse_grid)
    pdf_t = prob_a_of_t(t_grid, normalization, amin, amax, bmin, bmax, xmin, xmax, 
                        ymin, ymax, n_obs, use_linear=use_linear
                        )
    cdf_table = cdf_from_pdf_over_grid(t_grid, pdf_t)

    # PASS 4+: Adaptive Mesh Refinement (AMR)
    t_grid, pdf_t = adaptive_mesh_refinement_cdf(t_grid, pdf_t, normalization,
                          amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs,
                          tol=tol, max_points=max_points, use_linear=use_linear
                          )

    # PASS 5: Additional Adaptive Mesh Refinement
    cutdown = len(t_grid) // n_coarse_grid
    t_grid = np.concatenate([t_grid[::cutdown], t_grid[-1:]])
    pdf_t = np.concatenate([pdf_t[::cutdown], pdf_t[-1:]])
    if t0_anchor is not None:
        t_grid, pdf_t = inject_zero_anchor_point(t_grid, pdf_t, t0_anchor, normalization,
                                                amin, amax, bmin, bmax, xmin, xmax, 
                                                ymin, ymax, n_obs, use_linear=use_linear
                                                )
    t_grid, pdf_t = adaptive_mesh_refinement_cdf(t_grid, pdf_t, normalization,
                        amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs,
                        tol=tol, max_points=max_points, use_linear=use_linear
                        )
    # Preparing the output for use with JAX code
    t_grid = jnp.asarray(t_grid, dtype=jnp.float64)
    pdf_t = jnp.asarray(pdf_t, dtype=jnp.float64)
    # Reconverting from `t` to `a`
    if use_linear:
        a_grid = amin + 0.5 * (amax - amin) * (t_grid + 1.0)
    else:
        a_grid = t_to_a_map(t_grid)
    #Recomputing the CDF over the final refined grid
    cdf_table = cdf_from_pdf_over_grid(t_grid, pdf_t)
    pdf_a = prob_a(a_grid, normalization, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)
    return a_grid, cdf_table, pdf_a


# --- Evaluate the quantile function Q(ua)=a starting from the CDF(a) LUT ---
@jax.jit
def quantile_a_from_cdf_table(ua, a_grid, cdf_table):
    """
    Finds :math:`a=CDF(u_{a})` from the linear interpolation of an existing 
    look-up table for :math:`CDF(a)`.

    Uses :func:`jnp.interp` to interpolate using the CDF(a) as `x` and the `a` as `y`
    """
    return jnp.interp(ua, cdf_table, a_grid)

'''
@partial(jax.jit, static_argnames=['amin', 'amax', 'bmin', 'bmax', 'xmin', 
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
    valid_bounds = ((xmin <= xmax) & (ymin <= ymax) & (bmin <= bmax) & (amin<= amax) & 
                    (n_obs >= 0) & (ua >= 0.0) & (ua <= 1.0))
    a_low, a_high = a_grid[0], a_grid[-1]
    # Start from the interpolation of the existing LUT for CDF(a)
    # `jnp.interp` here interpolates using the CDF(a) as `x` and the `a` as `y`
    a_init = jnp.interp(ua, cdf_table, a_grid) 
    # Refining the initial guess with a few steps of Newton's method
    # We exploit the knowledge of the marginal pdf :math:`\pi(a)` to help speed up Newton
    def newton_step(a, _):
        # More copmplex interpolation scheme to find `a`
        cdf_value = hermite_interp_cdf_a(a, a_grid, cdf_table, pdf_table)
        # The PDF we know analytically is the first derivative of the CDF(a)
        pdf_value = unnorm_prob_a(a, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)/normalization
        safe_pdf_val = jnp.clip(pdf_value, jnp.finfo(jnp.float64).tiny, jnp.inf)
        # Residual
        delta = (cdf_value - ua) / safe_pdf_val
        # Newton update step
        a_updated = jnp.clip(a - delta, a_low, a_high)
        return a_updated, None
    
    a_refined, _ = jax.lax.scan(newton_step, a_init, None, length=newton_steps)

    return jnp.where(valid_bounds, a_refined, jnp.nan)
'''

## JAXNS-compatible implementations
class PriorA_UninformLinearJAXNS(SpecialPrior):
    """
    Marginal probability :math:`\pi(a)` for the slope of the linear correlation model
    """
    def __init__(self, a_grid, cdf_table, pdf_table, normalization, amin, amax, bmin, 
                 bmax, xmin, xmax, ymin, ymax, n_obs, name=None):
        super().__init__(name=name)
        self.a_grid = jnp.asarray(a_grid, dtype=jnp.float64)
        self.cdf_table = jnp.asarray(cdf_table, dtype=jnp.float64)
        self.pdf_table = jnp.asarray(pdf_table, dtype=jnp.float64)
        self.normalization = float(normalization)
        self.amin = float(amin)
        self.amax = float(amax)
        self.bmin = float(bmin)
        self.bmax = float(bmax)
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)
        self.n_obs = int(n_obs)

    def _dtype(self): return jnp.float64
    def _base_shape(self): return ()
    def _shape(self): return ()

    def _forward(self, U):
        return quantile_a_from_cdf_table(U, self.a_grid, self.cdf_table)
    
    def _inverse(self, X):
        # Linear interpolation of the existing LUT for CDF(a)
        return jnp.interp(X, self.a_grid, self.cdf_table)
    
    def _log_prob(self, X):
        return jnp.log(prob_a(X, self.normalization, self.amin, self.amax, self.bmin, 
                              self.bmax, self.xmin, self.xmax, self.ymin, self.ymax, self.n_obs))
    
class PriorBgivenA_UninformLinearJAXNS(SpecialPrior):
    """
    Conditional probability :math:`\pi(b|a)` for the intercept `b` of the linear correlation model
    given a value of the slope `a`.
    """
    def __init__(self, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs, name=None):
        super().__init__(name=name)
        self.a = a
        self.bmin, self.bmax = float(bmin), float(bmax)
        self.xmin, self.xmax = float(xmin), float(xmax)
        self.ymin, self.ymax = float(ymin), float(ymax)
        self.n_obs = int(n_obs)

    def _dtype(self): return jnp.float64
    def _base_shape(self): return ()
    def _shape(self): return ()

    def _forward(self, U):
        return inverse_cdf_b_given_a(U, self.a, self.bmin, self.bmax, self.xmin, 
                                        self.xmax, self.ymin, self.ymax, self.n_obs)
    
    def _inverse(self, X):
        return cdf_b_given_a(X, self.a, self.bmin, self.bmax, self.xmin, 
                                self.xmax, self.ymin, self.ymax, self.n_obs)
    
    def _log_prob(self, X):
        return jnp.log(prob_b_given_a(X, self.a, self.bmin, self.bmax, self.xmin, 
                                        self.xmax, self.ymin, self.ymax, self.n_obs))
        

## MAIN JUST FOR TESTING
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    Nobs = 12
    amin, amax = -jnp.inf, jnp.inf
    bmin, bmax = -1/(10*jnp.finfo(jnp.float64).eps), 1/(10*jnp.finfo(jnp.float64).eps)
    xmin, xmax = -3.0-jnp.log10(200.0), jnp.log10(2.99792458e5) - jnp.log10(200.0) #
    ymin, ymax = 0.0, 18.0

    a_normalization = normalization_prob_a(amin, amax, bmin, bmax, xmin, xmax, 
                                           ymin, ymax, Nobs, limit = int(1e8)
                                           )
    
    if not np.isfinite(amin) or not np.isfinite(amax): use_linear = False
    else: use_linear = True
    
    a_grid, cdf_table, pdf_table = build_cdf_a_lut(a_normalization, amin, amax, bmin, bmax, xmin, 
                                                   xmax, ymin, ymax, Nobs, n_coarse_grid=1000, 
                                                    tol=jnp.finfo(np.float64).eps, 
                                                    max_points=int(1e7), use_linear=use_linear
                                                    )

    plt.figure('prob_x')
    plt.title('p(x|a,b) for different values of a')
    print("Plotting p(x|a,b) for different values of a...")
    x = jnp.linspace(xmin-5.0, xmax+5.0, 1000)
    a = jnp.logspace(-5.0,5.0,10)
    b = 2.0
    for a_temp in a:
        prob_x_ab = prob_x_given_ab(x,a_temp,b,xmin,xmax,ymin,ymax)
        #print(f"a={a_temp:.2f}, prob_x_ab={prob_x_ab}")
        plt.plot(x, prob_x_ab, label=f'a={a_temp:.2f}')
    plt.legend(loc='best')

    plt.figure('prob_b_unnorm')
    plt.title('unnormalized p(b|a) for different values of a')
    print("Plotting p(b|a) for different values of a...")
    a = jnp.logspace(-5.0,5.0,10)
    b = jnp.linspace(bmin, bmax, 2500)
    for a_temp in a:
        y = unnorm_prob_b_given_a(b, a_temp, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
        plt.plot(b, y, label=f'a={a_temp:.2f}')
    plt.legend(loc='best')

    plt.figure('prob_b')
    print("Plotting p(b|a) for different values of a...")
    plt.title('p(b|a) for different values of a')
    a = jnp.logspace(-5.0,5.0,10)
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
    a = jnp.logspace(-5.0,5.0,10)
    a = a[((a<0) & (a*xmin>=ymin-bmax) & (a*xmax<=ymax-bmin) & (amin<=a) & (a<=amax)) |
          ((a==0) & (amin<=a) & (a<=amax)) | 
          ((a>0) & (a*xmin<=ymax-bmin) & (a*xmax>=ymin-bmax) & (amin<=a) & (a<=amax))]
    b = jnp.linspace(bmin, bmax, 10000)
    for a_temp in a:
        y1 = cdf_b_given_a(b, a_temp, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
        #y2 = cdf_b_given_a_lut(b, a_temp, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
        #y3 = cdf_b_given_a_monotone(b, a_temp, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
        plt.plot(b, y1, label=f'a={a_temp:.2f} (exact)')
        #plt.plot(b, y2, label=f'a={a_temp:.2f} (LUT)', linestyle='dashed')
        #plt.plot(b, y3, label=f'a={a_temp:.2f} (monotone)', linestyle='dotted')
    plt.legend(loc='best')

    plt.figure('expected_quantile_function_b')
    print("Plotting CDF^{-1}(b|u,a) for different values of a...")
    a = jnp.logspace(-5.0,5.0,10)
    a = a[((a<0) & (a*xmin>=ymin-bmax) & (a*xmax<=ymax-bmin) & (amin<=a) & (a<=amax)) |
          ((a==0) & (amin<=a) & (a<=amax)) | 
          ((a>0) & (a*xmin<=ymax-bmin) & (a*xmax>=ymin-bmax) & (amin<=a) & (a<=amax))]
    b = jnp.linspace(bmin, bmax, 10000)
    ub = 0.5+0.5*jnp.sort(jnp.concatenate([-jax.nn.sigmoid(jnp.logspace(-16.0, 16.0, 5000)), jax.nn.sigmoid(jnp.logspace(-16.0, 16.0, 5000))])) # to avoid numerical issues at the edges
    # --- Plotting loop ---
    for a_temp in a:
        y1 = cdf_b_given_a(b, a_temp, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
        y2 = jax.vmap(inverse_cdf_b_given_a, in_axes=(0, None, None, None, None, None, None, None, None))(
    ub, a_temp, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
        plt.plot(y1, b, label=f'a={a_temp:.2f} (expected)')
        plt.plot(ub, y2, label=f'a={a_temp:.2f} (root find)', linestyle='dashed')
    plt.legend(loc='best')

    """
    plt.figure('unnorm_prob_a')
    print("Plotting f(a) with respect to a...")
    a = jnp.linspace(-10,10,2000)
    y = unnorm_prob_a(a, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, Nobs)
    plt.plot(a, y, label='f(a)')
    plt.legend(loc='best')
    """

    a_normalization = normalization_prob_a(amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, Nobs, 
                                           limit = 20*int(1e6))
    print(f"Normalization of f(a) is {a_normalization:.3e}")
    print("Plotting p(a)analytical and CDF(a) from look-up-table")
    fig, axs = plt.subplots(2, 1, layout='constrained')
    fig.suptitle(r'$CDF(a)$ from look-up-table')
    axs_limits = [-15.0,15.0]#[-0.04, 0.04]
    cdf_lims = [-0.1, 1.1]#[0.495, 0.505]
    #pdf_lims = [0.064, 0.073]

    a_grid, cdf_table, pdf_table = build_cdf_a_lut(a_normalization, amin, amax, bmin, bmax, xmin, 
                                                   xmax, ymin, ymax, Nobs, n_coarse_grid=1000, 
                                                   tol=jnp.finfo(np.float64).eps, 
                                                   max_points=5*int(1e6), use_linear=False
                                                   )
    axs[0].plot(a_grid, pdf_table, label=r'$\pi(a) analytical$', color='red'#, 
                #marker='.', markersize=2.0, markerfacecolor='black', markeredgecolor='black'
                )
    axs[0].set_ylabel(r'$\pi(a)$')
    #axs[0].set_ylim(pdf_lims)
    axs[1].plot(a_grid, cdf_table, label=r'$CDF(a)$ LUT', color='red'#, 
                #marker='.', markersize=2.0, markerfacecolor='black', markeredgecolor='black'
                )
    axs[1].set_xlabel(r'$a$')
    axs[1].set_ylabel(r'$CDF(a)$')
    axs[1].set_ylim(cdf_lims)
    axs[1].axhline(0.0, color='black', linestyle='dotted', linewidth=1.0)
    axs[1].axhline(1.0, color='black', linestyle='dotted', linewidth=1.0)
    for ax in axs:
        ax.set_xlim(axs_limits)
        ax.axvline(0.0, color='black', linestyle='dashed', linewidth=1.0)
        ax.legend(loc='best')


    ua = jnp.linspace(0.0, 1.0, 10000)
    a_from_ua = quantile_a_from_cdf_table(ua, a_grid, cdf_table)

    fig_quantile, ax_quantile = plt.subplots(1,1,layout='constrained')
    fig_quantile.suptitle('Quantile function from look-up-table')
    ax_quantile.plot(ua, a_from_ua, label=r'$Q(u_a)$ from LUT', color='red')
    ax_quantile.set_xlabel(r'$u_a$')
    ax_quantile.set_ylabel(r'$a$')
    ax_quantile.legend(loc='best')

    plt.show()
    exit()