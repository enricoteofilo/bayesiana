import os, sys
from pathlib import Path
# Force JAX to ignore TPU/GPU backends in this environment.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
jnp = jax.numpy
jsp = jax.scipy
random = jax.random
grad = jax.grad
jit = jax.jit
vmap = jax.vmap

def function_pair(callable_1, callable_2, x, y, a, b, A_target, B_target):
    """
    Function to cast the problem of finding solutions to the non linear system
    to finding the roots of a 2-components vector function of 2 variables.
    
    :param callable_1: Description
    :param callable_2: Description
    :param x: Description
    :param y: Description
    :param a: Description
    :param b: Description
    :param A_target: Description
    :param B_target: Description
    """
    return jnp.array([callable_1(x, y, a, b)-A_target, callable_2(x, y, a, b)-B_target])

def newton_solver(callable_1, callable_2, guess_tuple, a, b, A_target, B_target, max_iter=1000, tol=sys.float_info.epsilon, damping=0.0):
    """
    A simple implementation of the Newton-Raphson method to solve a system of two non-linear equations.
    """
    # Wrapper of the input function pair to ensure the jacobian is computed 
    # with respect to the correct variables (x,y) and not the other args
    def F(v):
        x, y = v #unpacks the function inputs
        return function_pair(callable_1, callable_2, x, y, a, b, A_target, B_target)

    # Helper function for one Newton solver iteration. We exploit `jax`
    # autodifferentiation to accelerate the Jacobian computation.
    def body(v, _):
        J = jax.jacobian(function_pair)(callable_1, callable_2, *v, a, b, A_target, B_target)  # 2x2
        # Evaluating the function at the current guess. If zero within the tolerance,
        # the system is considered solved.
        Fv = function_pair(callable_1, callable_2, *v, a, b, A_target, B_target)
        # The Newton step: the vector function is linearized around the current guess 
        # and we compute the displacement such that the residuals of the linearized 
        # system are zero. A damping term can be added to improve numerical stability.
        delta = jnp.linalg.solve(J + damping * jnp.eye(2), Fv)
        # We update the guess with the estimated displacement and return the new guess 
        # and the norm of the displacement.
        return v - delta, jnp.linalg.norm(delta)
    # We use `jax.lax.scan` to iterate the Newton solver step,
    # and v is the last guess while deltas is the array of the displacement norms
    last_v, (traj, deltas) = jax.lax.scan(body, guess_tuple, None, length=max_iter)
    idx = jnp.argmin(jnp.where(deltas < tol, jnp.arange(max_iter), max_iter))
    return jax.lax.dynamic_index_in_dim(traj, idx, keepdims=False)

