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