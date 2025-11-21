#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jon Paul Lundquist
"""
Created on Sun Oct  5 17:58:20 2025

    Hyper-fast Pearson's r correlation

    Pearson's r is the classical linear correlation coefficient measuring the 
    strength of a linear relationship between two variables. It quantifies how well 
    paired samples (x_i, y_i) fit a straight-line trend by normalizing their 
    covariance with their standard deviations. r ranges from -1 to +1: +1 indicates 
    a perfectly increasing linear relationship, -1 a perfectly decreasing linear 
    relationship, and 0 indicates no linear association. Unlike rank-based measures 
    (e.g., Kendall's tau or Spearman's rho), Pearson's r is sensitive to outliers 
    and assumes that the relationship is linear under the null hypothesis of no 
    correlation.

    This implementation provides a drop-in, Numba-accelerated computation of 
    Pearson's r with SciPy-compatible two-sided p-values and efficient O(n) 
    fused-accumulator evaluation. It is designed for large-scale statistical 
    pipelines, Monte Carlo simulations, permutation-based significance tests, and 
    any workload requiring extremely fast repeated correlation evaluations.

    Parameters
    ----------
    x, y : 1-D array_like 
        samples of equal length (n ≥ 3). Inputs are treated as numeric; NaNs/Infs are not supported.
    
    pvals : {True, False}
        - Flag for p-value calculation. Default: pvals=True. Returns nan if pvals=False.
    
    Returns
    -------
    res : SignificanceResult
        An object with the attributes:    
        statistic : float
               Pearson's r statistic.
        pvalue : float
               Two-sided p-value for the null hypothesis of no association, H0: r=0.
               
    See Also
    --------
    scipy.stats.spearmanr
    
    Dependencies
    ------------
    - Python ≥ 3.8
    - NumPy ≥ 1.23
    - Numba ≥ 0.61
    - Scipy ≥ 1.9 (for benchmarks)

    Installation
    ------------
    pip install numpy numba scipy # or: conda install numpy numba scipy
    pip install hyper-corr
    
    # optional for fast math optimizations on Intel CPUs
    pip install icc_rt #or: conda install -c icc_rt
    
    Benchmarks (illustrative; environment-dependent): See pearsonr_bench.py
    ------------------------------------------------
    CPU: Ultra 9 275HX, Python 3.13.5, NumPy 2.1.3, Numba 0.61.2, SciPy 1.16.0

           N |  SciPy(ms) |  Hyper(ms) |   Speed× |      IQR |    Δr(max) |    Δp(max)
    -----------------------------------------------------------------------------------------
          25 |      0.112 |      0.001 |   173.81 |     5.27 |  1.110e-16 |  1.654e-14
          50 |      0.109 |      0.001 |   162.93 |    13.71 |  1.110e-16 |  1.532e-14
          75 |      0.113 |      0.001 |   163.71 |     8.35 |  8.327e-17 |  2.496e-13
         100 |      0.109 |      0.001 |   158.12 |    12.81 |  5.551e-17 |  8.304e-14
         200 |      0.112 |      0.001 |   161.72 |    11.07 |  2.776e-17 |  8.149e-14
         300 |      0.107 |      0.001 |   154.91 |    12.95 |  4.163e-17 |  1.401e-13
         400 |      0.110 |      0.001 |   157.88 |    10.18 |  3.123e-17 |  8.108e-13
         500 |      0.106 |      0.001 |   157.68 |    10.48 |  5.551e-17 |  1.062e-13
        1000 |      0.109 |      0.001 |   147.83 |    18.36 |  2.602e-17 |  1.029e-12
        2000 |      0.113 |      0.001 |   125.57 |     8.84 |  2.998e-17 |  8.079e-05
        3000 |      0.105 |      0.001 |   102.76 |     7.45 |  2.776e-17 |  5.433e-05
        4000 |      0.113 |      0.001 |    96.64 |     6.60 |  3.643e-17 |  4.075e-05
        5000 |      0.117 |      0.001 |    99.62 |     7.93 |  3.123e-17 |  2.631e-05
       10000 |      0.132 |      0.002 |    72.92 |    12.85 |  4.510e-17 |  1.630e-05
       20000 |      0.154 |      0.003 |    45.41 |     4.10 |  2.711e-17 |  8.030e-06
       30000 |      0.185 |      0.005 |    38.19 |     8.96 |  6.765e-17 |  5.431e-06
       40000 |      0.224 |      0.006 |    35.48 |     4.50 |  4.250e-17 |  4.066e-06
       50000 |      0.268 |      0.008 |    33.51 |     3.20 |  2.472e-17 |  3.168e-06
      100000 |      0.479 |      0.016 |    30.88 |     2.46 |  4.163e-17 |  1.466e-06
      200000 |      1.033 |      0.037 |    27.91 |     2.21 |  5.074e-17 |  8.071e-07
      300000 |      1.613 |      0.066 |    24.54 |     2.94 |  3.383e-17 |  5.429e-07
      400000 |      2.250 |      0.098 |    22.98 |     2.87 |  8.674e-17 |  4.062e-07
      500000 |      2.944 |      0.120 |    24.46 |     1.67 |  1.913e-16 |  3.258e-07
     1000000 |      7.472 |      0.274 |    28.12 |    22.77 |  3.441e-16 |  1.626e-07

     
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    .. [2] Virtanen, P., et al., SciPy 1.0: Fundamental Algorithms for 
           Scientific Computing in Python. Nature Methods, 17(3), 261-272  
           
@author: Jon Paul Lundquist
"""

from numba import njit
import math
from ._utils import _incbet
from collections import namedtuple
SignificanceResult = namedtuple("SignificanceResult", ["statistic", "pvalue"])

@njit(cache=True, nogil=True, fastmath=True)
def _pearsonr(x, y, pvals=True):
    """
    Compute Pearson r and two‐tailed p‐value.
    x, y must be 1D arrays of the same length ≥ 3.
    """

    n = x.size
    
    Sx = 0.0
    Sy = 0.0
    Sxx = 0.0
    Syy = 0.0
    Sxy = 0.0
    for i in range(n):
        xi = x[i]; yi = y[i]
        Sx  += xi
        Sy  += yi
        Sxx += xi * xi
        Syy += yi * yi
        Sxy += xi * yi
    
    # means
    invn = 1.0 / n
    mx = Sx * invn
    my = Sy * invn

    # covariance numerator and variance denominators
    cov = Sxy - Sx * my
    vx = Sxx - Sx * mx
    vy = Syy - Sy * my

    if vx <= 0.0 or vy <= 0.0:
        # one vector constant -> r undefined; match SciPy: r=nan, p=nan
        return math.nan, math.nan

    r = cov / math.sqrt(vx * vy)
    
    # clamp for numerical safety
    if r >  1.0: r =  1.0
    if r < -1.0: r = -1.0

    # *** GUARD FOR PERFECT CORRELATION ***
    if math.fabs(r) == 1.0:
        # exact linear relationship → infinite t → p = 0
        return r, 0.0
    
    if pvals:
        # Student‐t statistic and p‐value
        df = n - 2
        
        if df >= 1000:
            # r * sqrt(df) is close to t for large df
            z = math.fabs(r) * math.sqrt(df)
            # two-sided normal tail
            pvalue = math.erfc(z / 1.4142135623730951)
            
        else:
            # guard denom for r extremely close to ±1
            r2 = r * r
            if r2 > 1.0 - 1e-15:
                r2 = 1.0 - 1e-15
            t2 = r2 * df / (1.0 - r2)
            x_arg = df / (df + t2)
            pvalue = _incbet(0.5 * df, 0.5, x_arg)
            
    else:
        pvalue = math.nan
    
    return r, pvalue

def pearsonr(x, y, pvals=True):
    r, pvalue = _pearsonr(x, y, pvals=pvals)
    
    #Mirroring scipy output for compatibility. Slows things down slightly.
    res = SignificanceResult(statistic=float(r), pvalue=float(pvalue))
    
    return res
