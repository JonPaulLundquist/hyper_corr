# ltest — Shift-invariant two-sample CvM (the L-test)

Two-sample L-test (shift-invariant Cramér–von Mises variant).
    
Performs the two-sample L-test. The L-test is a shift-invariant modification of the Cramér–von Mises (CvM) two-sample test [1] that minimizes the integral squared difference between two empirical CDFs (L_2 squared distance), called U, by optimizing a scalar location shift, s, between samples. For independent samples X = {X_i}_{i=1..n} and Y = {Y_j}_{j=1..m}, the null hypothesis is:
    
    H0: ∃ s ∈ ℝ such that F_X(t) = F_Y(t + s) for all t
    
i.e., samples X and Y are draws from the same (unspecified) continuous distribution up to a location difference.

It returns a Monte-Carlo p-value, its uncertainty, a shift estimate, its uncertainty, and the minimized statistic.
    
Motivation
----------
Originally developed for ultra–high-energy cosmic-ray (UHECR) composition work comparing X_max distributions to model predictions, where model means have larger uncertainty than higher moments (such that there is significant CI overlap) [2]. The L-test is useful whenever relative location may be biased (instrument/location/seasonal effects, unknown inter-experiment offsets), and/or shape differences (variance, skew, tails, multimodality) are of primary interest. See Appendix B (“L-test”) of the author’s Ph.D. thesis for background and derivations [3].

## Install
```bash
pip install numpy scipy
# local install from source
pip install -e .
```

## Usage
```python
import numpy as np
from ltest import ltest

rng = np.random.default_rng(0)
x = rng.normal(size=200)
y = rng.normal(loc=0.3, scale=1.2, size=220)

l_p, l_p_err, l_shift, shift_boot, shift_err, l_stat = ltest(
    x, y, B=1000, tol_p=0.05, tol_s=0.05, workers=None, brute=False
)
print(l_p, l_p_err, l_shift, shift_err)
```

## Notes
- The L statistic equals the minimized version of Eq. (9) in Anderson [6] when including a free location parameter. Minimization is performed numerically (scalar search).
- The L-shift is generally interpretable as a location offset only when the L-test fails to reject (i.e., shapes appear compatible) or the two parent distributions are symmetrical. Under shape mismatch, it is a nuisance alignment chosen to minimize the ECDF distance and should not be interpreted as a population location difference.
- Parallel bootstrap with early stopping by relative error on *p* or on the shift uncertainty.
- Optional “brute” search of *s* via rank-change breakpoints (slower).
- See `examples/` for Type I/II power and shift-accuracy experiments.
- Run tests via `tests/test_basic.py`. Uses pytest -q.
- On Windows/macOS, protect the entry point when using multiprocessing:
  ```python
  if __name__ == "__main__":
      # call ltest(...)
  ```

This is new code based upon the statistical distribution test proposed in the author's 2017 Ph.D. thesis [3] and was used for the conference paper 'Study of UHECR Composition Using Telescope Array’s Middle Drum Detector and Surface Array in Hybrid Mode', 34th ICRC, 2016. Additionally, the paper The Astrophysical Journal, 858:76 (27pp), 2018 May 10 used a similar principle.

## References
[1] https://en.wikipedia.org/wiki/Cramer-von_Mises_criterion<br>
[2] Abbasi, R. U., Thomson, G. B., <xmax> uncertainty from extrapolation of cosmic ray air shower parameters arXiv:1605.05241<br>
[3] Lundquist, J.P. Energy Anisotropies of Proton-Like Ultra-High Energy Cosmic Rays, Ph.D. Thesis, University of Utah, 2017<br>
[4] Pebay, P.P. (2008) Formulas for Robust, One-Pass Parallel Computation of Covariances and Arbitrary-Order Statistical Moments, Sandia National Laboratories Technical Report<br>
[5] Scholz, F. W and Stephens, M. A. (1987), K-Sample Anderson-Darling Tests, Journal of the American Statistical Association, Vol. 82, pp. 918-924<br>
[6] Anderson, T.W., On the distribution of the two-sampleCramer-von-Mises criterion. The Annals of Mathematical Statistics, pp. 1148-1159<br>

## Dependencies
- Python ≥ 3.8  
- NumPy ≥ 1.23  
- SciPy ≥ 1.9

## License
This project is licensed under the MIT license.  
See the full text in [LICENSE](./LICENSE).
