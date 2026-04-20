"""Benchmark kbn_cumsum (NumPy loop) vs kbn_cumsum_jax (JAX scan)."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
from pathlib import Path
_HERE = Path(__file__).resolve().parent
import time
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import matplotlib.pyplot as plt
from utils import kbn_cumsum, kbn_cumsum_jax

sizes = jnp.logspace(jnp.log10(2), 6, num=25, dtype=int)
n_warmup = 10
n_repeats = 100

def mad(arr):
    """Median absolute deviation."""
    return np.median(np.abs(arr - np.median(arr)))

# Warm up JIT (compile once with a dummy array)
dummy = jnp.ones(max(sizes), dtype=jnp.float64)
kbn_cumsum_jax(dummy).block_until_ready()

med_np = []
mad_np = []
med_jax = []
mad_jax = []
med_ratio = []
mad_ratio = []

for n in sizes:
    inc_np = np.random.default_rng(0).standard_normal(n)
    inc_jax = jnp.asarray(inc_np)

    # --- NumPy loop ---
    dts_np = np.empty(n_repeats)
    for k in range(n_repeats):
        t0 = time.perf_counter()
        kbn_cumsum(inc_np)
        dts_np[k] = time.perf_counter() - t0

    # --- JAX scan (re-compile only if shape changes) ---
    kbn_cumsum_jax(inc_jax).block_until_ready()  # warm up for this size
    dts_jax = np.empty(n_repeats)
    for k in range(n_repeats):
        t0 = time.perf_counter()
        kbn_cumsum_jax(inc_jax).block_until_ready()
        dts_jax[k] = time.perf_counter() - t0

    med_np.append(np.median(dts_np))
    mad_np.append(mad(dts_np))
    med_jax.append(np.median(dts_jax))
    mad_jax.append(mad(dts_jax))

    # Per-repeat speedup ratios
    ratios = dts_np / dts_jax
    med_ratio.append(np.median(ratios))
    mad_ratio.append(mad(ratios))

    print(f"n={n:>7d}  numpy={med_np[-1]:.6f}±{mad_np[-1]:.6f}s  "
          f"jax={med_jax[-1]:.6f}±{mad_jax[-1]:.6f}s  "
          f"speedup={med_ratio[-1]:.1f}±{mad_ratio[-1]:.1f}x")

med_np = np.array(med_np)
mad_np = np.array(mad_np)
med_jax = np.array(med_jax)
mad_jax = np.array(mad_jax)
med_ratio = np.array(med_ratio)
mad_ratio = np.array(mad_ratio)
sizes_arr = np.array(sizes)

# --- Correctness check ---
inc = np.random.default_rng(42).standard_normal(1000)
res_np = kbn_cumsum(inc)
res_jax = np.asarray(kbn_cumsum_jax(jnp.asarray(inc)))
max_diff = np.max(np.abs(res_np - res_jax))
print(f"\nMax abs difference (n=1000): {max_diff:.2e}")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.loglog(sizes_arr, med_np, "o-", label="kbn_cumsum (NumPy loop)")
ax1.fill_between(sizes_arr, med_np - mad_np, med_np + mad_np, alpha=0.25)
ax1.loglog(sizes_arr, med_jax, "s-", label="kbn_cumsum_jax (JAX scan)")
ax1.fill_between(sizes_arr, med_jax - mad_jax, med_jax + mad_jax, alpha=0.25)
ax1.set_xlabel("Array length")
ax1.set_ylabel("Median time (s)")
ax1.set_title("Runtime (shaded = MAD)")
ax1.legend()
ax1.grid(True, which="both", ls="--", alpha=0.5)

ax2.semilogx(sizes_arr, med_ratio, "D-", color="tab:green")
ax2.fill_between(sizes_arr, med_ratio - mad_ratio, med_ratio + mad_ratio,
                 color="tab:green", alpha=0.25)
ax2.set_xlabel("Array length")
ax2.set_ylabel("Speedup (NumPy / JAX)")
ax2.set_title("Speedup factor (shaded = MAD)")
ax2.axhline(1.0, color="gray", ls="--", alpha=0.5)
ax2.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig(_HERE / "../figures/kbn_benchmark.png", dpi=600)
plt.close()
exit()