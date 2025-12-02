# hyper_corr -- Hyper-fast Correlation Functions

Hyper-fast, numba-accelerated correlation coefficients with SciPy-compatible results. `hyper_corr` provides drop-in replacements for common bivariate statistics—Pearson's *r*, Spearman's ρ, Kendall's τ, Chatterjee's ξ, and Somers' *D*—plus specialized variants that exploit pre-sorted inputs and known tie structure for maximum throughput.
For sample sizes of N = 50 (with continuous tie-free data) speedups over Scipy range from approximately x150 to x1500 times faster.

## Features
- **Numba-accelerated kernels** for high-volume or repeated correlation evaluations.
- **SciPy-style return types** (`SignificanceResult`/`SomersDResult`) from the general functions so existing code can adopt the faster implementations without large refactors.
- **Tie-aware and tie-free variants** for Kendall, Spearman, Chatterjee, and Somers to match your data assumptions for extreme performance.

## Installation
The library targets Python 3.8+ and depends on NumPy and Numba.

```bash
pip install numba numpy

#If you wish to use the included benchmarks for comparison to SciPy
pip install scipy

# optional for fast math optimizations on Intel CPUs
pip install icc_rt

#Install hyper-corr from pypi with pip
pip install hyper-corr

# or local install from source
pip install -e .

```

## Quick Start
```python
import numpy as np
from hyper_corr import pearsonr, spearmanr, kendalltau, chatterjeexi, somersd

rng = np.random.default_rng(seed=0)
x = rng.normal(size=500)
y = x * 0.75 + rng.normal(scale=0.25, size=500)

#Sorting by x not needed.
print(pearsonr(x, y))          # Pearson's r linear correlation
#Rank correlations with sorting and auto tie handling
print(spearmanr(x, y))         # Spearman's rho
print(kendalltau(x, y))        # Kendall's tau
print(chatterjeexi(x, y))      # Chatterjee's xi
print(somersd(x, y))           # Somers' D
```

### Performance-focused Variants
If you already have sorted data, and know whether ties exist, call the specialized kernels directly for the fastest speeds:

```python
from hyper_corr import spearmanr_noties, spearmanr_ties

# Example: tie-free Spearman's rho with pre-sorted x
idx = np.argsort(x, kind="stable")
x_sorted = x[idx]
y_ordered = y[idx]

rho, pvalue = spearmanr_noties(x_sorted, y_ordered, len(x_sorted))

# Example: Spearman's rho with pre-sorted x with ties
x_sorted = np.round(x_sorted, 1); y_ordered = np.round(y_ordered, 1)
rho, pvalue = spearmanr_ties(x_sorted, y_ordered, len(x_sorted))
```

### Optimal Use Case
Many small/medium repeated slices of pre-sorted large arrays with known tie structure.

```python
N = 1_000_000
rng = np.random.default_rng(0)
x = rng.normal(size=N); y = rng.normal(size=N)

W = 25              # window size
M = N - W + 1       # Number of windows

taus = np.empty(M, dtype=np.float64)
pvals = np.empty(M, dtype=np.float64)

ind = np.argsort(y, kind="stable")
x_sorted = x[ind]; y_ordered = y[ind]   # y in the same order as sorted x
    
ties = ((N-np.unique(x).size)>0) or ((N-np.unique(y).size)>0)

for i in range(M):
    xw = x_ordered[i:i+W]
    yw = y_sorted[i:i+W]
    if ties:
        tau, p = kendalltau_ties(xw, yw, W)
    else:
        tau, p = kendalltau_noties(xw, yw, W)
    taus[i] = tau
    pvals[i] = p
```

## Notes
- Data should be pre-cleaned. Sample data is not checked for real values or the fact that correlations must have n>2. nan is not taken into account. Speed was considered to be of the utmost importance.
- For the *_tie/_notie functions x MUST be sorted and y MUST be ordered by that sort.
- *_tie/_notie functions output stat and pvalue not SciPy return types as they are incompatible with Numba.
- The first run of the included correlation functions are slower than future runs due to Numba compilation.

## Development
Benchmarks and usage experiments live in the `bench/` and `examples/` folders. Packaging metadata is defined in `pyproject.toml`. Contributions should keep the public API exports in `hyper_corr/__init__.py` up to date.

## Provenance and Licensing
Several kernels and statistical routines in `hyper_corr` originate from or were adapted from corresponding SciPy implementations. Those upstream sources are distributed under the BSD-3-Clause license, and their terms continue to apply to the derived portions of this project. The BSD-3-Clause obligations coexist with the MIT License that governs the rest of the codebase; using or redistributing `hyper_corr` should account for both license notices. Upstream attribution details live in [`THIRD_PARTY_LICENSES.md`](./THIRD_PARTY_LICENSES.md), and the bundled BSD-3-Clause text itself is stored in [`licenses/SciPy_LICENSE.txt`](./licenses/SciPy_LICENSE.txt).

## License
Released under the MIT License alongside the third-party terms noted above. See [LICENSE](./LICENSE) and [THIRD_PARTY_LICENSES.md](./THIRD_PARTY_LICENSES.md) for details.
