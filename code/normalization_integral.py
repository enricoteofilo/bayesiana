import os
from dataclasses import dataclass
from functools import partial

# Keep computations in FP64 and on CPU for reproducibility.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
from quadax import quadcc

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

def interval_length_scalars(a,b,xmin,xmax,ymin,ymax):
    r"""
    Evaluation of the result of the following integral:

    .. math::

        \int_{x_{\min}}^{x_{\max}} dx \int_{y_{\min}}^{y_{\max}} dy\,\delta(y - ax - b)

    Inputs are already reshaped to be mutually broadcastable.
    """

    if ymax < ymin or xmax < xmin:
        raise ValueError("Invalid bounds. Expected ymin < ymax and xmin < xmax.")
    
    if a>0:
        left = jnp.max([xmin, (ymin-b)/a])
        right = jnp.min([xmax, (ymax-b)/a])
        return jnp.max([0.0, right - left])
    elif a == 0 and ymin<=b<=ymax:
        return xmax-xmin
    elif a<0:
        left = jnp.max([xmin, (ymax-b)/a])
        right = jnp.min([xmax, (ymin-b)/a])
        return jnp.max([0.0, right - left])
    else:
        return 0.0


@jax.jit
def interval_length_jaxed(a, b, xmin, xmax, ymin, ymax):
    r"""
    JIT-compiled core for evaluation of the result of the following
    integral:

    .. math::

        L(a,b) =\int_{x_{\min}}^{x_{\max}} dx \int_{y_{\min}}^{y_{\max}} dy\,\delta(y - ax - b)

    Inputs are already reshaped to be mutually broadcastable.
    """
    # First mask the a=0 case to avoid division by zero
    safe_a = jnp.where(a == 0.0, 1.0, a)

    # When a > 0
    left_pos = jnp.maximum(xmin, (ymin - b) / safe_a)
    right_pos = jnp.minimum(xmax, (ymax - b) / safe_a)
    len_pos = jnp.maximum(0.0, right_pos - left_pos)

    # When a < 0
    left_neg = jnp.maximum(xmin, (ymax - b) / safe_a)
    right_neg = jnp.minimum(xmax, (ymin - b) / safe_a)
    len_neg = jnp.maximum(0.0, right_neg - left_neg)

    # When a = 0
    len_zero = jnp.where((ymin <= b) & (b <= ymax), xmax - xmin, 0.0)

    length = jnp.where(a > 0.0, len_pos, jnp.where(a < 0.0, len_neg, len_zero))
    invalid_bounds = (ymax < ymin) | (xmax < xmin)
    return jnp.where(invalid_bounds, jnp.nan, length)


def interval_length_broadcast(a, b, xmin, xmax, ymin, ymax):
    r"""
    Vectorized/JAX-friendly version of interval_length_scalars.

    Each input can be a scalar or a 1D array.
    - If all inputs are arrays with sizes (Na, Nb, N1, N2, N3, N4),
      output has shape (Na, Nb, N1, N2, N3, N4).
    - If one or more inputs are scalars, the corresponding axes are removed
      from the output via squeeze.

    For invalid bounds (ymax < ymin or xmax < xmin), this function returns NaN
    at the corresponding broadcasted entries.
    """
    values = [a, b, xmin, xmax, ymin, ymax]
    names = ["a", "b", "xmin", "xmax", "ymin", "ymax"]
    arrays = [jnp.asarray(v, dtype=jnp.float64) for v in values]

    for name, arr in zip(names, arrays):
        if arr.ndim > 1:
            raise ValueError(f"Input {name} must be a scalar or a 1D array.")

    is_scalar = [arr.ndim == 0 for arr in arrays]

    reshaped = []
    for axis, arr in enumerate(arrays):
        if arr.ndim == 0:
            shape = [1, 1, 1, 1, 1, 1]
            reshaped.append(jnp.reshape(arr, shape))
        else:
            shape = [1, 1, 1, 1, 1, 1]
            shape[axis] = arr.shape[0]
            reshaped.append(jnp.reshape(arr, shape))

    out = interval_length_jaxed(*reshaped)

    squeeze_axes = tuple(i for i, s in enumerate(is_scalar) if s)
    if squeeze_axes:
        out = jnp.squeeze(out, axis=squeeze_axes)
    return out

def uniform_normalization_integrand(a,b,xmin,xmax,ymin,ymax,n_obs):
    r"""
    Return the integrand of the normalization integral for the linear 
    model y=ax+b uninformative prior.

    .. math::
     f(a,b;x_{min},x_{max},y_{min},y_{max},N) = (1+a^2)^{\frac{N-3}{2}}L(a,b;x_{min},x_{max},y_{min},y_{max})^N
    
    """
    return (1+a**2)**(0.5*(n_obs-3.0)) * interval_length_jaxed(a,b,xmin,xmax,ymin,ymax)**n_obs

def uniform_normalization_inner(a,bmin,bmax,xmin,xmax,ymin,ymax,n_obs):
    r"""
    Estimates the integral over `b` of the normalization integrand:
    
    .. math::
        \int_{b_{min}}^{b_{max}} db\,f(a,b;x_{min},x_{max},y_{min},y_{max},N)
    
    
    """
    integral, _ = quadcc(uniform_normalization_integrand, (bmin, bmax), args=(a,xmin,xmax,ymin,ymax,n_obs), epsabs=os.sys.float_info.epsilon, epsrel=os.sys.float_info.epsilon)
    return integral

def uniform_normalization_outer(amin,amax,bmin,bmax,xmin,xmax,ymin,ymax,n_obs):
    integral, _ = quadcc(uniform_normalization_inner, (amin, amax), args=(bmin,bmax,xmin,xmax,ymin,ymax,n_obs), epsabs=os.sys.float_info.epsilon, epsrel=os.sys.float_info.epsilon)
    return integral


@jax.jit
def _x_feasible_interval(a, b, xmin, xmax, ymin, ymax):
    """Return the admissible x-interval [left, right] under y = a*x + b."""
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    xmin = jnp.asarray(xmin, dtype=jnp.float64)
    xmax = jnp.asarray(xmax, dtype=jnp.float64)
    ymin = jnp.asarray(ymin, dtype=jnp.float64)
    ymax = jnp.asarray(ymax, dtype=jnp.float64)

    # Avoid division by zero for a=0 while keeping branch-free JAX code.
    safe_a = jnp.where(a == 0.0, 1.0, a)

    left_pos = jnp.maximum(xmin, (ymin - b) / safe_a)
    right_pos = jnp.minimum(xmax, (ymax - b) / safe_a)

    left_neg = jnp.maximum(xmin, (ymax - b) / safe_a)
    right_neg = jnp.minimum(xmax, (ymin - b) / safe_a)

    left = jnp.where(a > 0.0, left_pos, jnp.where(a < 0.0, left_neg, xmin))
    right = jnp.where(a > 0.0, right_pos, jnp.where(a < 0.0, right_neg, xmax))

    # For a=0, the line is y=b. Support exists iff ymin <= b <= ymax.
    has_support_zero = (ymin <= b) & (b <= ymax)
    right = jnp.where((a == 0.0) & (~has_support_zero), left, right)
    return left, right

@jax.jit
def x_characteristic(x,a,b,xmin,xmax,ymin,ymax):
    r"""
    The characteristic function of the set of values of acceptable values of x:

    .. math::
        \mathbb{I}(x;a,b,x_{min},x_{max},y_{min},y_{max}) = \begin{cases}
        1 & \text{if } x_{min} \leq x \leq x_{max} \text{ and } y_{min} \leq ax+b \leq y_{max} \\
        0 & \text{otherwise}
        \end{cases}

    `x` can be a scalar or an array (broadcasting is supported).
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    xmin = jnp.asarray(xmin, dtype=jnp.float64)
    xmax = jnp.asarray(xmax, dtype=jnp.float64)
    ymin = jnp.asarray(ymin, dtype=jnp.float64)
    ymax = jnp.asarray(ymax, dtype=jnp.float64)

    valid_bounds = (xmin <= xmax) & (ymin <= ymax)
    mask = (xmin <= x) & (x <= xmax) & (ymin <= a*x+b) & (a*x+b <= ymax) & valid_bounds
    return jnp.where(mask, 1.0, 0.0)

@jax.jit
def x_char_integrated(x,a,b,xmin,xmax,ymin,ymax):
    r"""
    The integral of the characteristic function of the set of acceptable values of x:
    .. math::
        \int_{x_{min}}^{x} dx^{\prime}\,\mathbb{I}(x^{\prime};a,b,x_{min},x_{max},y_{min},y_{max})

    Closed-form, branch-free implementation. `x` can be a scalar or an array.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    xmin = jnp.asarray(xmin, dtype=jnp.float64)
    xmax = jnp.asarray(xmax, dtype=jnp.float64)
    ymin = jnp.asarray(ymin, dtype=jnp.float64)
    ymax = jnp.asarray(ymax, dtype=jnp.float64)

    left, right = _x_feasible_interval(a, b, xmin, xmax, ymin, ymax)
    integrated = jnp.maximum(0.0, jnp.minimum(x, right) - left)
    invalid_bounds = (xmax < xmin) | (ymax < ymin)
    return jnp.where(invalid_bounds, jnp.nan, integrated)
    
@jax.jit
def Lx(a,b,xmin,xmax,ymin,ymax):
    r"""
    The integral of the characteristic function over the full set of acceptable values of x:
    .. math::
        L(a,b;x_{min},x_{max},y_{min},y_{max}) = \int_{x_{min}}^{x_{max}} dx^{\prime}\,\mathbb{I}(x^{\prime};a,b,x_{min},x_{max},y_{min},y_{max})
    """
    return x_char_integrated(xmax, a, b, xmin, xmax, ymin, ymax)

@jax.jit
def cdf_x(x,a,b,xmin,xmax,ymin,ymax):
    r"""
    The marginal cumulative distribution function of the uninformative prior for x:
    .. math::
        CDF(x|a,b;x_{min},x_{max},y_{min},y_{max}) = \frac{\int_{x_{min}}^{x} dx^{\prime}\,\mathbb{I}(x^{\prime};a,b,x_{min},x_{max},y_{min},y_{max})}
        {L(a,b;x_{min},x_{max},y_{min},y_{max})}

    `x` can be a scalar or an array (broadcasting is supported).
    """
    numerator = x_char_integrated(x, a, b, xmin, xmax, ymin, ymax)
    denominator = Lx(a, b, xmin, xmax, ymin, ymax)
    safe_den = jnp.where(denominator > 0.0, denominator, 1.0)
    cdf = jnp.where(denominator > 0.0, numerator / safe_den, jnp.nan)
    return jnp.clip(cdf, 0.0, 1.0)


@jax.jit
def inv_cdf_x(ux,a,b,xmin,xmax,ymin,ymax):
        r"""
        Inverse of the marginal cumulative distribution function for x
        (quantile/PPF), at fixed `a, b, xmin, xmax, ymin, ymax`.

        .. math::
                x = CDF^{-1}(u_{x}|a,b;x_{min},x_{max},y_{min},y_{max}),\quad u_{x}\in[0,1]

        This is a closed-form implementation with no iterative solver.
        `ux` can be a scalar or an array (broadcasting is supported).

        Behavior:
        - `ux` is clipped to `[0, 1]` to account for numerical instabilities.
        - If the admissible x-support has zero length (or bounds are invalid),
            returns NaN.
        """
        ux = jnp.asarray(ux, dtype=jnp.float64)
        xmin = jnp.asarray(xmin, dtype=jnp.float64)
        xmax = jnp.asarray(xmax, dtype=jnp.float64)
        ymin = jnp.asarray(ymin, dtype=jnp.float64)
        ymax = jnp.asarray(ymax, dtype=jnp.float64)

        left, right = _x_feasible_interval(a, b, xmin, xmax, ymin, ymax)
        length = jnp.maximum(0.0, right - left)

        ux_clipped = jnp.clip(ux, 0.0, 1.0)
        x_quantile = left + ux_clipped * length

        invalid_bounds = (xmax < xmin) | (ymax < ymin)
        valid = (~invalid_bounds) & (length > 0.0)
        return jnp.where(valid, x_quantile, jnp.nan)

@jax.jit
def unnorm_prob_b_given_a(b,a,bmin,bmax,xmin,xmax,ymin,ymax,n_obs):
    r"""
    The unnormalized conditional prior for b given a, under the uninformative joint 
    prior for the linear model y=ax+b:

    .. math::
        p(b|a) \propto \mathbb{I}(b;b_{min},b_{max})L(a,b;x_{min},x_{max},y_{min},y_{max})^N
        
        
    """
    b = jnp.asarray(b, dtype=jnp.float64)
    a = jnp.asarray(a, dtype=jnp.float64)
    bmin = jnp.asarray(bmin, dtype=jnp.float64)
    bmax = jnp.asarray(bmax, dtype=jnp.float64)
    xmin = jnp.asarray(xmin, dtype=jnp.float64)
    xmax = jnp.asarray(xmax, dtype=jnp.float64)
    ymin = jnp.asarray(ymin, dtype=jnp.float64)
    ymax = jnp.asarray(ymax, dtype=jnp.float64)
    n_obs = jnp.asarray(n_obs, dtype=jnp.float64)

    in_bounds = (bmin <= b) & (b <= bmax) & (xmin < xmax) & (ymin < ymax) & (bmin < bmax)
    length = Lx(a, b, xmin, xmax, ymin, ymax)

    safe_length = jnp.maximum(length, 0.0)
    log_term = jnp.where(safe_length > 0.0, n_obs * jnp.log(safe_length), -jnp.inf)
    unnorm = jnp.exp(log_term)
    return jnp.where(in_bounds, unnorm, 0.0)


@jax.jit
def integrated_unnorm_prob_b_given_a_scalar(b,a,bmin,bmax,xmin,xmax,ymin,ymax,n_obs):
    """Scalar-kernel for integral in b.
    
    .. math::
        F(b;a) = \int_{b_{min}}^{b} db^{\prime}\,\mathbb{I}(b^{\prime};b_{min},b_{max})
        L(a,b^{\prime};x_{min},x_{max},y_{min},y_{max})^N
    
    """
    b = jnp.asarray(b, dtype=jnp.float64)
    a = jnp.asarray(a, dtype=jnp.float64)
    bmin = jnp.asarray(bmin, dtype=jnp.float64)
    bmax = jnp.asarray(bmax, dtype=jnp.float64)
    xmin = jnp.asarray(xmin, dtype=jnp.float64)
    xmax = jnp.asarray(xmax, dtype=jnp.float64)
    ymin = jnp.asarray(ymin, dtype=jnp.float64)
    ymax = jnp.asarray(ymax, dtype=jnp.float64)
    n_obs = jnp.asarray(n_obs, dtype=jnp.int64)

    invalid_bounds = (xmin >= xmax) | (ymin >= ymax) | (bmin >= bmax) | (b<bmin) | (b>bmax) | (n_obs < 0)
    b_hi = jnp.clip(b, bmin, bmax)
 
    # This creates an inner function that computes the integral only when needed.
    # The underscore argument is a dummy input required because jax.lax.cond expects 
    # branch functions with a uniform signature.
    def integral_helper(_):
        integral, _ = quadcc(
            unnorm_prob_b_given_a,
            (bmin, b_hi), # integrates from bmin to b_hi.
            args=(a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs), #args passes the fixed parameters into the integrand
            epsabs=os.sys.float_info.epsilon,
            epsrel=os.sys.float_info.epsilon,
        )
        return jnp.clip(integral, 0.0, +jnp.inf)
    # Clipping avoids tiny negative values caused by floating-point/integration noise 
    # and enforces the expected nonnegative integral.

    # jax.lax.cond chooses what to do conditional on `b_hi <= bmin`: Here:
    # If True, it returns 0 immediately (empty interval).
    # If False, it calls `integral_helper` to compute the integral over the 
    # non-empty interval [bmin, b_hi].
    # Prevents problematic zero-width integrations
    integral = jax.lax.cond(
        b_hi <= bmin,
        lambda _: jnp.asarray(0.0, dtype=jnp.float64),
        integral_helper,
        operand=None,
    )

    return jnp.where(invalid_bounds, jnp.nan, integral)


def integrated_unnorm_prob_b_given_a(b,a,bmin,bmax,xmin,xmax,ymin,ymax,n_obs):
    r"""
    Provides the integral of the unnormalized conditional prior for b given a, under 
    the uninformative joint prior for the linear model y=ax+b:

    .. math::
        F(b;a) = \int_{b_{min}}^{b} db^{\prime}\,mathbb{I}(b^{\prime};b_{min},b_{max})L(a,b^{\prime};x_{min},x_{max},y_{min},y_{max})^N
        
    """
    b = jnp.asarray(b, dtype=jnp.float64)

    # If b is a scalar calls the scalar helper function directly
    if b.ndim == 0:
        return integrated_unnorm_prob_b_given_a_scalar(b, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)

    # If b is an array, applies the scalar helper function to each element 
    # with parallelization handled via `jax.vmap`.
    return jax.vmap(
        lambda b_iter: integrated_unnorm_prob_b_given_a_scalar(
            b_iter, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs
        )
    )(b) #Iterating only on the b variable, since the other parameters are fixed for the integral in b.

def cdf_b_given_a(b,a,bmin,bmax,xmin,xmax,ymin,ymax,n_obs):
    r"""
    The cumulative distribution function of the conditional prior for b given a, under the 
    uninformative joint prior for the linear model y=ax+b:

    .. math::
        CDF(b;a) = \frac{\int_{b_{min}}^{b} db^{\prime}\,\mathbb{I}(b^{\prime};b_{min},b_{max})L(a,b^{\prime};x_{min},x_{max},y_{min},y_{max})^N}
        {\int_{b_{min}}^{b_{max}} db^{\prime}\,\mathbb{I}(b^{\prime};b_{min},b_{max})L(a,b^{\prime};x_{min},x_{max},y_{min},y_{max})^N}
    """
    numerator = integrated_unnorm_prob_b_given_a(b, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)
    denominator = integrated_unnorm_prob_b_given_a(bmax, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)
    return jnp.where(numerator == 0.0, 0.0, jnp.where(denominator < 0.0, jnp.nan, numerator / denominator))


@jax.jit
def power_nonnegative_fast(base, exponent):
    """Stable power for nonnegative base with explicit base=0 handling."""
    base = jnp.asarray(base, dtype=jnp.float64)
    exponent = jnp.asarray(exponent, dtype=jnp.float64)
    return jnp.where(
        base > 0.0,
        jnp.exp(exponent * jnp.log(base)),
        jnp.where(exponent == 0.0, 1.0, 0.0),
    )


@jax.jit
def unnorm_prob_b_given_a_fast(b,a,bmin,bmax,xmin,xmax,ymin,ymax,n_obs):
    r"""
    Fast closed-form counterpart of `unnorm_prob_b_given_a`.

    Same quantity:
    .. math::
        p(b|a) \propto \mathbb{I}(b;b_{min},b_{max})L(a,b;\cdots)^N

    Safeguards:
    - Returns 0.0 for degenerate intervals (zero support measure).
    - Returns NaN for invalid bounds or n_obs < 0.
    """
    b = jnp.asarray(b, dtype=jnp.float64)
    a = jnp.asarray(a, dtype=jnp.float64)
    bmin = jnp.asarray(bmin, dtype=jnp.float64)
    bmax = jnp.asarray(bmax, dtype=jnp.float64)
    xmin = jnp.asarray(xmin, dtype=jnp.float64)
    xmax = jnp.asarray(xmax, dtype=jnp.float64)
    ymin = jnp.asarray(ymin, dtype=jnp.float64)
    ymax = jnp.asarray(ymax, dtype=jnp.float64)
    n_obs = jnp.asarray(n_obs, dtype=jnp.float64)

    invalid = (xmin > xmax) | (ymin > ymax) | (bmin > bmax) | (n_obs < 0.0)
    degenerate = (xmin == xmax) | (ymin == ymax) | (bmin == bmax)
    in_bounds = (bmin <= b) & (b <= bmax)

    length = jnp.maximum(0.0, Lx(a, b, xmin, xmax, ymin, ymax))
    unnorm = power_nonnegative_fast(length, n_obs)
    out = jnp.where(in_bounds & (~degenerate), unnorm, 0.0)
    return jnp.where(invalid, jnp.nan, out)


@jax.jit
def _integrated_prob_b_given_a_fast_scalar(b,a,bmin,bmax,xmin,xmax,ymin,ymax,n_obs):
    r"""
    Closed-form fast scalar integral:

    .. math::
        F(b;a)=\int_{b_{min}}^{b} db'\,\mathbb{I}(b';b_{min},b_{max})L(a,b';\cdots)^N

    
    """
    b = jnp.asarray(b, dtype=jnp.float64)
    a = jnp.asarray(a, dtype=jnp.float64)
    bmin = jnp.asarray(bmin, dtype=jnp.float64)
    bmax = jnp.asarray(bmax, dtype=jnp.float64)
    xmin = jnp.asarray(xmin, dtype=jnp.float64)
    xmax = jnp.asarray(xmax, dtype=jnp.float64)
    ymin = jnp.asarray(ymin, dtype=jnp.float64)
    ymax = jnp.asarray(ymax, dtype=jnp.float64)
    n_obs = jnp.asarray(n_obs, dtype=jnp.float64)

    invalid = (xmin > xmax) | (ymin > ymax) | (bmin > bmax) | (n_obs < 0.0)
    degenerate = (xmin == xmax) | (ymin == ymax) | (bmin == bmax)
    b_hi = jnp.clip(b, bmin, bmax)
    n_plus_1 = n_obs + 1.0

    # a = 0: Lx is constant on active b range.
    x_width = jnp.maximum(0.0, xmax - xmin)
    active_lo = jnp.maximum(bmin, ymin)
    active_hi = jnp.minimum(b_hi, jnp.minimum(bmax, ymax))
    active_len = jnp.maximum(0.0, active_hi - active_lo)
    integral_a0 = power_nonnegative_fast(x_width, n_obs) * active_len

    # a != 0: exact piecewise integral of overlap(b)^N.
    abs_a = jnp.abs(a)
    ay_lo = ymin
    ay_hi = ymax
    by_lo0 = jnp.minimum(a * xmin, a * xmax)
    by_hi0 = jnp.maximum(a * xmin, a * xmax)

    width_a = ay_hi - ay_lo
    width_b = by_hi0 - by_lo0
    peak = jnp.minimum(width_a, width_b)

    u1 = ay_lo - by_hi0
    u2 = jnp.minimum(ay_lo - by_lo0, ay_hi - by_hi0)
    u3 = jnp.maximum(ay_lo - by_lo0, ay_hi - by_hi0)
    u4 = ay_hi - by_lo0

    lo = bmin
    hi = b_hi

    l1 = jnp.maximum(lo, u1)
    r1 = jnp.minimum(hi, u2)
    z1l = jnp.maximum(0.0, l1 - u1)
    z1r = jnp.maximum(0.0, r1 - u1)
    i1 = jnp.where(
        r1 > l1,
        (power_nonnegative_fast(z1r, n_plus_1) - power_nonnegative_fast(z1l, n_plus_1)) / n_plus_1,
        0.0,
    )

    l2 = jnp.maximum(lo, u2)
    r2 = jnp.minimum(hi, u3)
    i2 = jnp.where(r2 > l2, power_nonnegative_fast(peak, n_obs) * (r2 - l2), 0.0)

    l3 = jnp.maximum(lo, u3)
    r3 = jnp.minimum(hi, u4)
    z3l = jnp.maximum(0.0, u4 - l3)
    z3r = jnp.maximum(0.0, u4 - r3)
    i3 = jnp.where(
        r3 > l3,
        (power_nonnegative_fast(z3l, n_plus_1) - power_nonnegative_fast(z3r, n_plus_1)) / n_plus_1,
        0.0,
    )

    overlap_power_integral = i1 + i2 + i3
    scale = power_nonnegative_fast(1.0 / abs_a, n_obs)
    integral_an = overlap_power_integral * scale

    integral = jnp.where(a == 0.0, integral_a0, integral_an)
    integral = jnp.where(b_hi <= bmin, 0.0, integral)
    integral = jnp.where(degenerate, 0.0, integral)
    return jnp.where(invalid, jnp.nan, jnp.clip(integral, 0.0, +jnp.inf))


def integrated_prob_b_given_a_fast(b,a,bmin,bmax,xmin,xmax,ymin,ymax,n_obs):
    r"""
    Fast vectorized counterpart of `integrated_prob_b_given_a`.
    Uses exact closed-form expressions (no adaptive quadrature per query).
    """
    b = jnp.asarray(b, dtype=jnp.float64)
    if b.ndim == 0:
        return _integrated_prob_b_given_a_fast_scalar(b, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)
    return jax.vmap(
        lambda bi: _integrated_prob_b_given_a_fast_scalar(
            bi, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs
        )
    )(b)


@jax.jit
def cdf_b_given_a_fast(b,a,bmin,bmax,xmin,xmax,ymin,ymax,n_obs):
    r"""
    Normalized CDF in b built from the fast closed-form integral.
    """
    num = integrated_prob_b_given_a_fast(b, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)
    den = integrated_prob_b_given_a_fast(bmax, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)
    safe_den = jnp.where(den > 0.0, den, 1.0)
    cdf = jnp.where(den > 0.0, num / safe_den, jnp.nan)
    return jnp.clip(cdf, 0.0, 1.0)


def inv_cdf_b_given_a_fast(u,a,bmin,bmax,xmin,xmax,ymin,ymax,n_obs,n_iter=64):
    r"""
    Numerical inverse-CDF in b using bisection over fast CDF evaluations.

    This is intended for repeated sampling when the fast closed-form CDF is used.
    """
    u = jnp.asarray(u, dtype=jnp.float64)
    u = jnp.clip(u, 0.0, 1.0)

    def _inv_scalar(ui):
        lo = jnp.asarray(bmin, dtype=jnp.float64)
        hi = jnp.asarray(bmax, dtype=jnp.float64)

        def body_fun(_, state):
            l, h = state
            m = 0.5 * (l + h)
            fm = cdf_b_given_a_fast(m, a, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)
            l_new = jnp.where(fm < ui, m, l)
            h_new = jnp.where(fm < ui, h, m)
            return (l_new, h_new)

        l_fin, h_fin = jax.lax.fori_loop(0, n_iter, body_fun, (lo, hi))
        return 0.5 * (l_fin + h_fin)

    if u.ndim == 0:
        return _inv_scalar(u)
    return jax.vmap(_inv_scalar)(u)

def unnorm_prob_a(a, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    r"""
    The unnormalized marginal prior for a under the uninformative joint prior for the linear model y=ax+b:

    .. math::
        p(a) \propto (1+a^2)^{\frac{N_{obs}-3}{2}}\mathbb{I}(a;a_{min},a_{max})\int_{b_{min}}^{b_{max}} db\,\mathbb{I}(b;b_{min},b_{max})L(a,b;x_{min},x_{max},y_{min},y_{max})^N
    """
    a = jnp.asarray(a, dtype=jnp.float64)
    amin = jnp.asarray(amin, dtype=jnp.float64)
    amax = jnp.asarray(amax, dtype=jnp.float64)
    bmin = jnp.asarray(bmin, dtype=jnp.float64)
    bmax = jnp.asarray(bmax, dtype=jnp.float64)
    xmin = jnp.asarray(xmin, dtype=jnp.float64)
    xmax = jnp.asarray(xmax, dtype=jnp.float64)
    ymin = jnp.asarray(ymin, dtype=jnp.float64)
    ymax = jnp.asarray(ymax, dtype=jnp.float64)
    n_obs = jnp.asarray(n_obs, dtype=jnp.float64)

    in_bounds = (amin <= a) & (a <= amax) & (xmin < xmax) & (ymin < ymax) & (bmin < bmax)
    condition = (a==0) | ((a!=0) & (jnp.minimum(a*xmax,a*xmin) <= ymax - bmin) & (jnp.maximum(a*xmax,a*xmin) >= ymin - bmax))

    def unbounded_prob_a(a):
        return jnp.exp(0.5*(n_obs-3)*jnp.log(1+a**2))*integrated_unnorm_prob_b_given_a(bmax,a,bmin,bmax,xmin,xmax,ymin,ymax,n_obs)

    return jnp.where(in_bounds & condition, unbounded_prob_a(a), 0.0)

def integrated_unnorm_prob_a_scalar(a, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    r"""
    The integral of the unnormalized marginal prior for a under the uninformative joint prior for the linear model y=ax+b:

    .. math::
        F(a) = \int_{a_{min}}^{a} da'\,(1+a'^2)^{\frac{N_{obs}-3}{2}}\mathbb{I}(a';a_{min},a_{max})\int_{b_{min}}^{b_{max}} db\,\mathbb{I}(b;b_{min},b_{max})L(a',b;x_{min},x_{max},y_{min},y_{max})^N
    """
    a = jnp.asarray(a, dtype=jnp.float64)
    amin = jnp.asarray(amin, dtype=jnp.float64)
    amax = jnp.asarray(amax, dtype=jnp.float64)
    bmin = jnp.asarray(bmin, dtype=jnp.float64)
    bmax = jnp.asarray(bmax, dtype=jnp.float64)
    xmin = jnp.asarray(xmin, dtype=jnp.float64)
    xmax = jnp.asarray(xmax, dtype=jnp.float64)
    ymin = jnp.asarray(ymin, dtype=jnp.float64)
    ymax = jnp.asarray(ymax, dtype=jnp.float64)
    n_obs = jnp.asarray(n_obs, dtype=jnp.float64)

    invalid_bounds = (xmin >= xmax) | (ymin >= ymax) | (bmin >= bmax) | (amin >= amax) | (n_obs < 0)
    a_hi = jnp.clip(a, amin, amax)

    def integral_helper(_):
        integral, _ = quadcc(
            unnorm_prob_a,
            (amin, a_hi), # integrates from amin to a_hi.
            args=(amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs), #args passes the fixed parameters into the integrand
            epsabs=os.sys.float_info.epsilon,
            epsrel=os.sys.float_info.epsilon,
        )
        return jnp.clip(integral, 0.0, +jnp.inf)

    integral = jax.lax.cond(
        a_hi <= amin,
        lambda _: jnp.asarray(0.0, dtype=jnp.float64),
        integral_helper,
        operand=None,
    )

    return jnp.where(invalid_bounds, jnp.nan, integral)

def cdf_a(a, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    r"""
    The cumulative distribution function of the marginal prior for a under the uninformative joint prior for the linear model y=ax+b:

    .. math::
        CDF(a) = \frac{\int_{a_{min}}^{a} da'\,(1+a'^2)^{\frac{N_{obs}-3}{2}}\mathbb{I}(a';a_{min},a_{max})\int_{b_{min}}^{b_{max}} db\,\mathbb{I}(b;b_{min},b_{max})L(a',b;x_{min},x_{max},y_{min},y_{max})^N}
        {\int_{a_{min}}^{a_{max}} da'\,(1+a'^2)^{\frac{N_{obs}-3}{2}}\mathbb{I}(a';a_{min},a_{max})\int_{b_{min}}^{b_{max}} db\,\mathbb{I}(b;b_{min},b_{max})L(a',b;x_{min},x_{max},y_{min},y_{max})^N}
    """
    num = integrated_unnorm_prob_a(a, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)
    den = integrated_unnorm_prob_a(amax, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)
    safe_den = jnp.where(den > 0.0, den, 1.0)
    cdf = jnp.where(den >= 0.0, num / safe_den, 0.0)
    return jnp.clip(cdf, 0.0, 1.0)

def integrated_unnorm_prob_a(a, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs):
    r"""
    Parallelized array public wrapper.
    """
    a = jnp.asarray(a, dtype=jnp.float64)

    # If a is a scalar calls the scalar helper function directly
    if a.ndim == 0:
        return integrated_unnorm_prob_a_scalar(a, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs)

    # If a is an array, applies the scalar helper function to each element 
    # with parallelization handled via `jax.vmap`.
    return jax.vmap(
        lambda a_iter: integrated_unnorm_prob_a_scalar(
            a_iter, amin, amax, bmin, bmax, xmin, xmax, ymin, ymax, n_obs
        )
    )(a) #Iterating only on the a variable, since the other parameters are fixed for the integral in a.

    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #result = uniform_normalization_outer(-1.0e4, 1.0e4, -100, 100, -15, 9, 0, 18, 10)
    #print("Normalization integral result:", result)


    a_test = jnp.linspace(-5, 5, 1000)
    plt.plot(a_test, cdf_a(a_test, -10, 10, -100, 100, -15, 9, 0, 18, 10))
    plt.show()