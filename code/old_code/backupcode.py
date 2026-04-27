def x_characteristic(x,a,b,xmin,xmax,ymin,ymax):
    r"""
    The characteristic function of the set of values of acceptable values of x:
    .. math::
        \mathbb{I}(x;a,b,x_{min},x_{max},y_{min},y_{max}) = \begin{cases}
        1 & \text{if } x_{min} \leq x \leq x_{max} \text{ and } y_{min} \leq ax+b \leq y_{max} \\
        0 & \text{otherwise}
        \end{cases}
    """
    return jnp.where((xmin <= x) & (x <= xmax) & (ymin <= a*x+b) & (a*x+b <= ymax), 1.0, 0.0)

def x_char_integrated(x,a,b,xmin,xmax,ymin,ymax):
    r"""
    The integral of the characteristic function of the set of acceptable values of x:
    .. math::
        \int_{x_{min}}^{x} dx^{\prime}\,\mathbb{I}(x^{\prime};a,b,x_{min},x_{max},y_{min},y_{max})
    """
    if (x<xmin) | (x>xmax) | (ymin>a*x+b) | (a*x+b>ymax):
        return 0.0
    else:
        integral, _ = quadcc(x_characteristic, (xmin, x), args=(a,b,xmin,xmax,ymin,ymax), epsabs=os.sys.float_info.epsilon, epsrel=os.sys.float_info.epsilon)
        return integral
    
def L(a,b,xmin,xmax,ymin,ymax):
    r"""
    The integral of the characteristic function over the full set of acceptable values of x:
    .. math::
        L(a,b;x_{min},x_{max},y_{min},y_{max}) = \int_{x_{min}}^{x_{max}} dx^{\prime}\,\mathbb{I}(x^{\prime};a,b,x_{min},x_{max},y_{min},y_{max})
    """
    return x_char_integrated(xmax,a,b,xmin,xmax,ymin,ymax)

def cdf_x(x,a,b,xmin,xmax,ymin,ymax):
    r"""
    The marginal cumulative distribution function of the uninformative prior for x:
    .. math::
        CDF(x|a,b;x_{min},x_{max},y_{min},y_{max}) = \frac{\int_{x_{min}}^{x} dx^{\prime}\,\mathbb{I}(x^{\prime};a,b,x_{min},x_{max},y_{min},y_{max})}
        {L(a,b;x_{min},x_{max},y_{min},y_{max})}
    """
    return x_char_integrated(x,a,b,xmin,xmax,ymin,ymax) / L(a,b,xmin,xmax,ymin,ymax)

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



### UNINFORMATIVE PRIOR FOR LINEAR MODEL WITHOUT INTRINSIC SCATTER
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