# hyper_corr

Fast, numba-accelerated correlation coefficients with SciPy-compatible results. `hyper_corr` provides drop-in replacements for common bivariate statistics—Pearson's *r*, Spearman's ρ, Kendall's τ, Chatterjee's ξ, and Somers' *D*—plus specialized variants that exploit pre-sorted inputs and known tie structure for maximum throughput.

## Features
- **Numba-accelerated kernels** for high-volume or repeated correlation evaluations.
- **SciPy-style return types** (`SignificanceResult`/`SomersDResult`) so existing code can adopt the faster implementations without large refactors.
- **Tie-aware and tie-free variants** for Kendall, Spearman, Chatterjee, and Somers to match your data assumptions.
- **Deterministic numeric behavior** that mirrors SciPy's handling of undefined cases (returns `nan` when variance is zero).

## Installation
The library targets Python 3.8+ and depends on NumPy and Numba.

```bash
pip install numba numpy
# from source
pip install -e .
```

## Quick start
```python
import numpy as np
from hyper_corr import pearsonr, spearmanr, kendalltau, chatterjeexi, somersd

rng = np.random.default_rng(seed=0)
x = rng.normal(size=5000)
y = x * 0.75 + rng.normal(scale=0.25, size=5000)

print(pearsonr(x, y))          # linear correlation
print(spearmanr(x, y))         # rank correlation (auto tie handling)
print(kendalltau(x, y))        # Kendall's tau with automatic tie detection
print(chatterjeexi(x, y))      # Chatterjee's xi with automatic tie detection
print(somersd(x, y))           # Somers' D with automatic tie detection
```

### Performance-focused variants
If you already have sorted data or know whether ties exist, call the specialized kernels directly for additional speed:

```python
# Example: tie-free Spearman's rho with pre-sorted x
idx = np.argsort(x, kind="stable")
x_sorted = x[idx]
y_ordered = y[idx]

from hyper_corr import spearmanr_noties
rho, pvalue = spearmanr_noties(x_sorted, y_ordered, len(x_sorted))
```

## Development
Benchmarks and usage experiments live in the `bench/` and `examples/` folders. Packaging metadata is defined in `pyproject.toml`. Contributions should keep the public API exports in `hyper_corr/__init__.py` up to date.

## License
Released under the MIT License. See [LICENSE](./LICENSE) for details.
