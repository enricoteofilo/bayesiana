import os
from dataclasses import dataclass
from functools import partial

# Keep computations in FP64 and on CPU for reproducibility.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
from quadax import quadcc
from scipy.special import hyp2f1

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


@dataclass(frozen=True)
class IntegralBounds:
	"""Container for integration limits with strict monotonicity checks."""

	a_min: float
	a_max: float
	b_min: float
	b_max: float
	x_min: float
	x_max: float
	y_min: float
	y_max: float

	def validate(self) -> None:
		"""Validate that all lower bounds are strictly smaller than upper bounds."""

		if not (self.a_min < self.a_max):
			raise ValueError("Expected a_min < a_max.")
		if not (self.b_min < self.b_max):
			raise ValueError("Expected b_min < b_max.")
		if not (self.x_min < self.x_max):
			raise ValueError("Expected x_min < x_max.")
		if not (self.y_min < self.y_max):
			raise ValueError("Expected y_min < y_max.")


@jax.jit
def _interval_overlap_length(l1: float, r1: float, l2: float, r2: float) -> float:
	"""Return max(0, min(r1,r2)-max(l1,l2)), the overlap length of two intervals."""

	left = jnp.maximum(l1, l2)
	right = jnp.minimum(r1, r2)
	return jnp.maximum(0.0, right - left)


@partial(jax.jit, static_argnames=("n_obs",))
def _segment_integral_linear_power(vl: float, vr: float, delta: float, n_obs: int) -> float:
	"""
	Integrate (linear interpolation between vl and vr)^n_obs over a segment of length delta.
	The closed form is exact for linear profiles and includes a near-flat safeguard.
	"""
	vl = jnp.maximum(vl, 0.0)
	vr = jnp.maximum(vr, 0.0)
	delta = jnp.maximum(delta, 0.0)

	dv = vr - vl
	n_obs_f = jnp.asarray(n_obs, dtype=jnp.float64)
	n1 = n_obs_f + 1.0

	scale = jnp.maximum(1.0, jnp.maximum(jnp.abs(vl), jnp.abs(vr)))
	near_flat = jnp.abs(dv) <= (64.0 * jnp.finfo(jnp.float64).eps * scale)

	flat_val = delta * jnp.power(0.5 * (vl + vr), n_obs_f)
	safe_dv = jnp.where(near_flat, 1.0, dv)
	exact_val = delta * (jnp.power(vr, n1) - jnp.power(vl, n1)) / (n1 * safe_dv)

	out = jnp.where(near_flat, flat_val, exact_val)
	return jnp.where(delta > 0.0, out, 0.0)


@partial(jax.jit, static_argnames=("n_obs",))
def _b_integral_exact(
	a: float,
	b_min: float,
	b_max: float,
	x_min: float,
	x_max: float,
	y_min: float,
	y_max: float,
	n_obs: int,
) -> float:
	"""
	Exact integral over b of [x-overlap length]^n_obs for fixed a.

	The b axis is partitioned by the 4 breakpoints where interval endpoints align,
	so the overlap profile is linear on each subinterval and integrated analytically.
	"""
	abs_a = jnp.abs(a)
	safe_abs_a = jnp.where(abs_a > 0.0, abs_a, 1.0)

	ax0 = a * x_min
	ax1 = a * x_max
	p = jnp.minimum(ax0, ax1)
	q = jnp.maximum(ax0, ax1)

	b1 = y_min - p
	b2 = y_max - p
	b3 = y_min - q
	b4 = y_max - q

	nodes = jnp.sort(jnp.asarray([b_min, b_max, b1, b2, b3, b4], dtype=jnp.float64))

	def body_fun(i: int, acc: float) -> float:
		left = jnp.maximum(nodes[i], b_min)
		right = jnp.minimum(nodes[i + 1], b_max)
		delta = jnp.maximum(right - left, 0.0)

		# z = b + a x maps x-range to [b+p, b+q].
		gl = _interval_overlap_length(left + p, left + q, y_min, y_max)
		gr = _interval_overlap_length(right + p, right + q, y_min, y_max)

		# Convert z-overlap to x-overlap by dividing by |a|.
		wl = gl / safe_abs_a
		wr = gr / safe_abs_a

		return acc + _segment_integral_linear_power(wl, wr, delta, n_obs)

	nonzero_a_val = jax.lax.fori_loop(0, 5, body_fun, 0.0)

	x_len = x_max - x_min
	b_overlap_at_a0 = _interval_overlap_length(b_min, b_max, y_min, y_max)
	a0_val = jnp.power(x_len, jnp.asarray(n_obs, dtype=jnp.float64)) * b_overlap_at_a0

	return jnp.where(abs_a > 0.0, nonzero_a_val, a0_val)


@partial(jax.jit, static_argnames=("n_obs",))
def _outer_integrand(
	a: float,
	b_min: float,
	b_max: float,
	x_min: float,
	x_max: float,
	y_min: float,
	y_max: float,
	n_obs: int,
) -> float:
	"""Return the a-integrand: (1+a^2)^((N-3)/2) times the exact b-integral term."""

	b_part = _b_integral_exact(a, b_min, b_max, x_min, x_max, y_min, y_max, n_obs)
	exponent = 0.5 * (jnp.asarray(n_obs, dtype=jnp.float64) - 3.0)
	a_weight = jnp.exp(exponent * jnp.log1p(a * a))
	return a_weight * b_part


def normalization_integral(
	bounds: IntegralBounds,
	n_obs: int,
	epsabs: float = 1e-13,
	epsrel: float = 1e-13,
) -> float:
	"""
	Compute
	  integral_{a_min}^{a_max} da (1+a^2)^((N-3)/2)
	  integral_{b_min}^{b_max} db [ integral_{x_min}^{x_max} Theta(...) Theta(...) dx ]^N

	Strategy:
	- exact (analytic) integration in b for each fixed a,
	- adaptive high-accuracy 1D integration in a (FP64).
	"""
	bounds.validate()
	if n_obs <= 2:
		raise ValueError("Expected n_obs > 2.")
	if epsabs <= 0.0 or epsrel <= 0.0:
		raise ValueError("Expected epsabs > 0 and epsrel > 0.")

	def integrand(a_scalar):
		a_val = jnp.asarray(a_scalar, dtype=jnp.float64)
		return _outer_integrand(
			a_val,
			jnp.asarray(bounds.b_min, dtype=jnp.float64),
			jnp.asarray(bounds.b_max, dtype=jnp.float64),
			jnp.asarray(bounds.x_min, dtype=jnp.float64),
			jnp.asarray(bounds.x_max, dtype=jnp.float64),
			jnp.asarray(bounds.y_min, dtype=jnp.float64),
			jnp.asarray(bounds.y_max, dtype=jnp.float64),
			n_obs,
		)

	# Split at a=0 to help adaptive quadrature handle the abs(a) branch smoothly.
	if bounds.a_min < 0.0 < bounds.a_max:
		left_val, _ = quadcc(integrand, [bounds.a_min, 0.0], epsabs=0.5 * epsabs, epsrel=epsrel)
		right_val, _ = quadcc(integrand, [0.0, bounds.a_max], epsabs=0.5 * epsabs, epsrel=epsrel)
		return float(left_val + right_val)

	val, _ = quadcc(integrand, [bounds.a_min, bounds.a_max], epsabs=epsabs, epsrel=epsrel)
	return float(val)


def analytic_full_visibility_case(bounds: IntegralBounds, n_obs: int) -> float:
	"""
	Closed-form reference when y-bounds are wide enough that both Heavisides are
	always 1 for all (a,b,x) in bounds.

	In this regime, inner x-integral is always (x_max-x_min), so
	  I = (x_range)^N * (b_range) * integral_{a_min}^{a_max}(1+a^2)^((N-3)/2) da.
	"""
	bounds.validate()
	if n_obs <= 2:
		raise ValueError("Expected n_obs > 2.")

	prods = [
		bounds.a_min * bounds.x_min,
		bounds.a_min * bounds.x_max,
		bounds.a_max * bounds.x_min,
		bounds.a_max * bounds.x_max,
	]
	ax_min = min(prods)
	ax_max = max(prods)

	required_y_min = bounds.b_min + ax_min
	required_y_max = bounds.b_max + ax_max
	if bounds.y_min > required_y_min or bounds.y_max < required_y_max:
		raise ValueError(
			"Bounds do not satisfy full-visibility regime required for this closed form."
		)

	x_len = bounds.x_max - bounds.x_min
	b_len = bounds.b_max - bounds.b_min
	p = 0.5 * (float(n_obs) - 3.0)

	def antiderivative(a: float) -> float:
		return float(a * hyp2f1(0.5, -p, 1.5, -(a * a)))

	a_part = antiderivative(bounds.a_max) - antiderivative(bounds.a_min)
	return (x_len ** n_obs) * b_len * a_part


if __name__ == "__main__":
	# Generic example.
	ex_bounds = IntegralBounds(
		a_min=-2.75,
		a_max=1.90,
		b_min=-1.20,
		b_max=2.10,
		x_min=-0.40,
		x_max=1.60,
		y_min=-0.85,
		y_max=1.75,
	)
	n_ex = 7
	value = normalization_integral(ex_bounds, n_ex, epsabs=1e-13, epsrel=1e-13)
	print(f"Integral value (generic case): {value:.16e}")

	# Machine-precision validation against a true closed-form regime.
	check_bounds = IntegralBounds(
		a_min=-2.0,
		a_max=2.5,
		b_min=-1.0,
		b_max=1.5,
		x_min=-0.5,
		x_max=1.0,
		y_min=-10.0,
		y_max=10.0,
	)
	n_check = 11
	numerical = normalization_integral(check_bounds, n_check, epsabs=1e-14, epsrel=1e-14)
	analytical = analytic_full_visibility_case(check_bounds, n_check)
	abs_err = abs(numerical - analytical)
	rel_err = abs_err / abs(analytical)

	print(f"Integral value (closed-form regime): {numerical:.16e}")
	print(f"Analytical value:                  {analytical:.16e}")
	print(f"Absolute error:                    {abs_err:.3e}")
	print(f"Relative error:                    {rel_err:.3e}")
