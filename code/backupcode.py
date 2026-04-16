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