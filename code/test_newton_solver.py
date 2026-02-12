import jax, sys
jnp = jax.numpy

from utils import newton_solver

# Easy system: x + y = 1, x - y = 0 -> solution (0.5, 0.5)
def easy_f(x, y, a, b):
    return x + y

def easy_g(x, y, a, b):
    return x - y

# Trickier system: enforce y = exp(x) + sin(5x) and x^2 + exp(-y) + y = 2.5.
# Define residuals that should be driven to zero.
def hard_f(x, y, a, b):
    return y - (jnp.exp(x) + jnp.sin(5.0 * x))

def hard_g(x, y, a, b):
    return x**2 + jnp.exp(-y) + y - 2.5


def run_case(name, f, g, guess, A_target, B_target, max_iter=100, damping=0.0):
    sol = newton_solver(f, g, guess, a=0.0, b=0.0, A_target=A_target, B_target=B_target,
                        max_iter=max_iter, tol=jnp.finfo(jnp.float64).eps, damping=damping)
    resid = jnp.linalg.norm(jnp.array([f(*sol, 0.0, 0.0) - A_target,
                                       g(*sol, 0.0, 0.0) - B_target]))
    print(f"{name}: solution {sol}, residual {resid}")


def main():
    # Easy: start near the solution
    run_case(
        name="easy",
        f=easy_f,
        g=easy_g,
        guess=jnp.array([0.2, 0.8]),
        A_target=1.0,
        B_target=0.0,
        max_iter=20,
        damping=0.0,
    )

    # Harder: exponential + quadratic coupling. Use a reasonable guess.
    run_case(
        name="hard",
        f=hard_f,
        g=hard_g,
        guess=jnp.array([0.3, 12.6]),
        A_target=0.0,
        B_target=1.5,
        max_iter=2000,
        damping=sys.float_info.epsilon,  # damping to handle oscillations and stiffness
    )

if __name__ == "__main__":
    main()
