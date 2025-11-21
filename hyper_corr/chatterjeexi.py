#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jon Paul Lundquist
"""
Created on Sun Oct  5 18:07:51 2025

    Hyper-fast Chatterjee's xi (ξ) correlation

    Chatterjee's xi (ξ) correlation coefficient is a measure of association 
    between two variables; ξ tends to zero with independent variables and tends 
    to 1 when there is a strong association. The ξ correlation is effective even 
    with non-monotonic associations [1].
    
    For maximum speed on repeated calls, use the specialized kernels and you must 
    supply pre-sorted inputs:
    
        idx = np.argsort(y, kind="quicksort")  # keep equal y contiguous
        y_sorted = y[idx]
        x_ordered = x[idx]
    
        #tie-aware
        xi, pvalue = chatterjeexi_ties(x_ordered, y_sorted, n) 
    
        # no ties 
        xi, pvalue = chatterjeexi_noties(x_ordered, n)
    
    If you can’t guarantee the preconditions, use `chatterjeexi(x, y, ties="auto",
    sorted_y=False)`. It sorts as needed and dispatches to the right kernel, but 
    is much slower than calling the kernels directly.
    
    chatterjeexi_noties() is faster than scipy for any N and is up to ~×190 times faster 
    for N=25 to ~x3.5 times faster for extremely large arrays.
    
    Warning!
    --------
    The xi statistic is very sensitive to the tie-breaking method with many ties. 
    Numpy quicksort must be used for agreement with scipy. The actual xi statistic 
    should be the average of randomized tie-breaking results [1] -- this is not 
    implemented in this package or scipy. With many ties the differences can be 
    very large.
    
    When to use
    -----------
    - **Many small/medium repeated slices** of large arrays with known tie
      structure → `chatterjeexi_ties`/ `chatterjeexi_noties` (fastest).
    - **One-off or unknown inputs** → `chatterjeexi` (convenience wrapper).
        
   Parameters
   ----------
   x, y : 1-D array_like 
       The independent and dependent samples of equal length (n ≥ 3). y is assumed
       to be drawn from a continuous distribution with no ties. The difference in 
       p-value is generally negligible and this assumption speeds calculation. Ties
       are more likely to violate this assumption and xi and p-value deviate from
       the continuous assumption.
   
    pvals : {True, False}
        - Flag for p-value calculation. Default: pvals=True. Returns nan if pvals=False.

   For the specialized kernels chatterjeexi_ties() and chatterjeexi_noties():
       n : Length of the arrays      
       - chatterjeexi_ties(x, y, n)
       - chatterjeexi_noties(x, y, n)`: assumes y has no ties (continuous); 
         ties in x are allowed and are broken deterministically.
       
   for chatterjeexi()
       ties : {"auto", True, False} 
       Choose tie-aware kernel automatically or force a variant.
           - "auto": Default. Detect ties and pick the proper kernel
           - True:   Force tie-aware kernel
           - False:  Force no-ties kernel
   
   Returns
   -------
   for chatterjeexi_ties() / chatterjeexi_noties() :
       xi : Chatterjee's xi statistic (0 to 1).
       pvalue : One-sided "greater" tail p-value for the null hypothesis of no 
                association, H0: ξ=0.
   
   for chatterjeexi():
       res : SignificanceResult
       An object with the attributes:    
           statistic : float
              Chatterjee's xi statistic (0 to 1).
           pvalue : float
              One-sided "greater" tail p-value for the null hypothesis of no 
              association, H0: ξ=0.

    Notes
    -----

    - The statistic is not symmetric in `x` and `y`: "...we may want to understand 
    if `y` is a function `x`, and not just if one of the variables is a function 
    of the other."[1]
    - Inputs are treated as numeric; NaNs are not supported.

    See Also
    --------
    scipy.stats.chatterjeexi
        
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

    Benchmarks (illustrative; environment-dependent): See chatterjeexi_bench.py
    ------------------------------------------------
    CPU: Ultra 9 275HX, Python 3.13.5, NumPy 2.1.3, Numba 0.61.2, SciPy 1.16.0

    With ties (`chatterjeexi_ties`):
           N |  SciPy(ms) |  Hyper(ms) |   Speed× |      IQR |   Δxi(max) |    Δp(max)
    -----------------------------------------------------------------------------------------
          25 |      0.203 |      0.001 |   157.43 |     3.20 |  0.000e+00 |  8.049e-16
          50 |      0.204 |      0.002 |   116.70 |     1.92 |  0.000e+00 |  6.661e-16
          75 |      0.208 |      0.002 |    92.92 |     4.31 |  0.000e+00 |  8.882e-16
         100 |      0.209 |      0.003 |    76.23 |     1.84 |  0.000e+00 |  8.049e-16
         200 |      0.224 |      0.005 |    43.13 |     1.47 |  0.000e+00 |  4.996e-16
         300 |      0.240 |      0.008 |    31.16 |     0.83 |  0.000e+00 |  9.437e-16
         400 |      0.249 |      0.011 |    23.37 |     0.42 |  0.000e+00 |  6.939e-16
         500 |      0.259 |      0.014 |    18.05 |     0.71 |  0.000e+00 |  8.882e-16
        1000 |      0.312 |      0.060 |     5.14 |     0.30 |  0.000e+00 |  7.772e-16
        2000 |      0.396 |      0.080 |     5.07 |     0.28 |  0.000e+00 |  6.661e-16
        3000 |      0.628 |      0.140 |     4.50 |     0.26 |  0.000e+00 |  8.482e-14
        4000 |      0.897 |      0.242 |     3.67 |     0.35 |  0.000e+00 |  5.723e-14
        5000 |      1.003 |      0.273 |     3.66 |     0.39 |  0.000e+00 |  7.983e-14
       10000 |      1.610 |      0.347 |     4.61 |     0.35 |  0.000e+00 |  3.076e-13
       20000 |      2.809 |      0.602 |     4.72 |     0.19 |  0.000e+00 |  3.505e-13
       30000 |      3.936 |      0.897 |     4.48 |     0.52 |  0.000e+00 |  5.920e-13
       40000 |      5.551 |      1.193 |     4.71 |     0.41 |  0.000e+00 |  1.360e-12
       50000 |      6.683 |      1.485 |     4.53 |     0.23 |  0.000e+00 |  8.799e-13
      100000 |     13.580 |      2.836 |     4.78 |     0.74 |  0.000e+00 |  3.180e-12
      200000 |     26.425 |      6.327 |     4.21 |     0.30 |  0.000e+00 |  3.295e-12
      300000 |     44.057 |     11.762 |     3.77 |     0.35 |  0.000e+00 |  6.137e-12
      400000 |     59.514 |     17.070 |     3.50 |     0.27 |  0.000e+00 |  6.073e-12
      500000 |     73.980 |     23.307 |     3.22 |     0.34 |  0.000e+00 |  7.005e-12
     1000000 |    176.737 |     71.727 |     2.49 |     0.39 |  0.000e+00 |  1.967e-11
 
    No ties (`chatterjeexi_noties`):
           N |  SciPy(ms) |  Hyper(ms) |   Speed× |      IQR |   Δxi(max) |    Δp(max)
    -----------------------------------------------------------------------------------------
          25 |      0.182 |      0.001 |   190.71 |     5.52 |  0.000e+00 |  3.608e-16
          50 |      0.254 |      0.001 |   173.33 |    18.70 |  0.000e+00 |  1.110e-16
          75 |      0.285 |      0.002 |   135.80 |     7.39 |  0.000e+00 |  6.661e-16
         100 |      0.259 |      0.002 |   113.48 |     5.34 |  0.000e+00 |  3.608e-16
         200 |      0.282 |      0.004 |    66.00 |     2.95 |  0.000e+00 |  3.331e-16
         300 |      0.306 |      0.007 |    47.44 |     2.27 |  0.000e+00 |  2.220e-16
         400 |      0.344 |      0.010 |    37.21 |     2.22 |  0.000e+00 |  1.110e-16
         500 |      0.363 |      0.013 |    27.78 |     1.57 |  0.000e+00 |  9.437e-16
        1000 |      0.432 |      0.062 |     7.02 |     0.49 |  0.000e+00 |  8.882e-16
        2000 |      0.591 |      0.084 |     6.93 |     0.72 |  0.000e+00 |  8.882e-16
        3000 |      0.718 |      0.117 |     6.18 |     0.35 |  0.000e+00 |  5.862e-14
        4000 |      0.860 |      0.154 |     5.59 |     0.24 |  0.000e+00 |  2.309e-14
        5000 |      1.047 |      0.194 |     5.41 |     0.22 |  0.000e+00 |  1.343e-14
       10000 |      1.914 |      0.414 |     4.67 |     0.24 |  0.000e+00 |  1.762e-14
       20000 |      3.774 |      0.877 |     4.30 |     0.16 |  0.000e+00 |  3.189e-13
       30000 |      5.620 |      1.366 |     4.18 |     0.19 |  0.000e+00 |  2.569e-13
       40000 |      9.113 |      2.189 |     4.14 |     0.09 |  0.000e+00 |  4.368e-13
       50000 |     10.575 |      2.553 |     4.13 |     0.20 |  0.000e+00 |  4.479e-13
      100000 |     20.908 |      5.039 |     4.13 |     0.16 |  0.000e+00 |  1.586e-13
      200000 |     46.556 |     10.037 |     4.16 |     0.56 |  0.000e+00 |  8.105e-14
      300000 |     67.353 |     14.817 |     4.61 |     0.61 |  0.000e+00 |  1.055e-14
      400000 |    110.038 |     27.143 |     4.04 |     0.19 |  0.000e+00 |  1.242e-11
      500000 |    144.025 |     36.543 |     3.93 |     0.13 |  0.000e+00 |  1.008e-11
     1000000 |    325.526 |     89.429 |     3.49 |     0.32 |  0.000e+00 |  3.085e-12
 
    `chatterjeexi` (wrapper) with unsorted random tie/no_tie input:
       N |  SciPy(ms) |  Hyper(ms) |   Speed× |      IQR |   Δxi(max) |    Δp(max)
    -----------------------------------------------------------------------------------------
          25 |      0.203 |      0.004 |    58.07 |     1.52 |  0.000e+00 |  5.065e-16
          50 |      0.206 |      0.003 |    59.64 |     1.25 |  0.000e+00 |  5.551e-16
          75 |      0.211 |      0.004 |    47.31 |     1.32 |  0.000e+00 |  6.661e-16
         100 |      0.218 |      0.005 |    42.30 |     1.96 |  0.000e+00 |  3.608e-16
         200 |      0.226 |      0.009 |    25.00 |     0.43 |  0.000e+00 |  2.776e-16
         300 |      0.240 |      0.013 |    18.81 |     0.83 |  0.000e+00 |  5.551e-16
         400 |      0.251 |      0.018 |    14.00 |     0.51 |  0.000e+00 |  7.772e-16
         500 |      0.262 |      0.023 |    11.48 |     1.01 |  0.000e+00 |  9.992e-16
        1000 |      0.315 |      0.079 |     3.96 |     0.78 |  0.000e+00 |  6.661e-16
        2000 |      0.395 |      0.103 |     3.92 |     0.18 |  0.000e+00 |  8.327e-16
        3000 |      0.581 |      0.156 |     3.66 |     0.22 |  0.000e+00 |  9.670e-14
        4000 |      0.712 |      0.207 |     3.34 |     0.13 |  0.000e+00 |  4.131e-14
        5000 |      0.865 |      0.264 |     3.26 |     0.07 |  0.000e+00 |  9.415e-14
       10000 |      1.134 |      0.305 |     3.65 |     1.04 |  0.000e+00 |  1.653e-13
       20000 |      2.013 |      0.598 |     3.37 |     1.26 |  0.000e+00 |  3.187e-13
       30000 |      4.695 |      1.819 |     2.59 |     1.14 |  0.000e+00 |  5.322e-13
       40000 |      6.491 |      2.504 |     2.61 |     0.86 |  0.000e+00 |  6.224e-13
       50000 |      4.609 |      1.341 |     3.39 |     1.14 |  0.000e+00 |  6.041e-13
      100000 |     17.050 |      6.836 |     2.51 |     0.86 |  0.000e+00 |  1.224e-12
      200000 |     38.328 |     16.011 |     2.40 |     0.81 |  0.000e+00 |  1.727e-12
      300000 |     34.421 |     14.287 |     2.41 |     0.35 |  0.000e+00 |  3.255e-12
      400000 |     46.393 |     21.365 |     2.27 |     0.09 |  0.000e+00 |  1.229e-11
      500000 |    114.924 |     52.799 |     2.16 |     0.04 |  0.000e+00 |  9.905e-12
     1000000 |    138.368 |     87.068 |     1.90 |     0.10 |  0.000e+00 |  1.484e-11

    'chatterjeexi' (wrapper) with unsorted samples with ties:
           N |  SciPy(ms) |  Hyper(ms) |   Speed× |      IQR |   Δxi(max) |    Δp(max)
    -----------------------------------------------------------------------------------------
          25 |      0.212 |      0.003 |    74.31 |     1.72 |  0.000e+00 |  8.049e-16
          50 |      0.216 |      0.004 |    60.25 |     1.18 |  0.000e+00 |  6.661e-16
          75 |      0.214 |      0.005 |    46.89 |     1.06 |  0.000e+00 |  8.882e-16
         100 |      0.216 |      0.005 |    41.83 |     1.01 |  0.000e+00 |  8.049e-16
         200 |      0.231 |      0.009 |    25.92 |     0.69 |  0.000e+00 |  4.996e-16
         300 |      0.249 |      0.013 |    19.45 |     0.74 |  0.000e+00 |  9.437e-16
         400 |      0.257 |      0.018 |    14.75 |     0.82 |  0.000e+00 |  6.939e-16
         500 |      0.269 |      0.023 |    11.67 |     0.57 |  0.000e+00 |  8.882e-16
        1000 |      0.327 |      0.077 |     4.22 |     0.20 |  0.000e+00 |  7.772e-16
        2000 |      0.406 |      0.108 |     3.79 |     0.12 |  0.000e+00 |  6.661e-16
        3000 |      0.505 |      0.151 |     3.39 |     0.09 |  0.000e+00 |  8.482e-14
        4000 |      0.587 |      0.189 |     3.09 |     0.14 |  0.000e+00 |  5.723e-14
        5000 |      0.708 |      0.224 |     3.18 |     0.17 |  0.000e+00 |  7.983e-14
       10000 |      1.146 |      0.288 |     4.01 |     0.20 |  0.000e+00 |  3.076e-13
       20000 |      2.046 |      0.519 |     3.99 |     0.21 |  0.000e+00 |  3.505e-13
       30000 |      2.855 |      0.780 |     3.70 |     0.28 |  0.000e+00 |  5.920e-13
       40000 |      3.930 |      1.071 |     3.65 |     0.25 |  0.000e+00 |  1.360e-12
       50000 |      4.872 |      1.377 |     3.55 |     0.24 |  0.000e+00 |  8.799e-13
      100000 |      9.838 |      2.803 |     3.57 |     0.31 |  0.000e+00 |  3.180e-12
      200000 |     21.060 |      6.268 |     3.34 |     0.11 |  0.000e+00 |  3.295e-12
      300000 |     35.457 |     13.053 |     2.74 |     0.12 |  0.000e+00 |  6.137e-12
      400000 |     53.400 |     22.856 |     2.35 |     0.08 |  0.000e+00 |  6.073e-12
      500000 |     68.329 |     31.954 |     2.15 |     0.08 |  0.000e+00 |  7.005e-12
     1000000 |    148.799 |     78.839 |     1.90 |     0.13 |  0.000e+00 |  1.967e-11

    'chatterjeexi' (wrapper) with unsorted samples without any ties:
           N |  SciPy(ms) |  Hyper(ms) |   Speed× |      IQR |   Δxi(max) |    Δp(max)
    -----------------------------------------------------------------------------------------
          25 |      0.221 |      0.003 |    65.15 |     2.39 |  0.000e+00 |  3.608e-16
          50 |      0.227 |      0.004 |    60.03 |     0.95 |  0.000e+00 |  1.110e-16
          75 |      0.228 |      0.005 |    45.93 |     1.18 |  0.000e+00 |  6.661e-16
         100 |      0.232 |      0.006 |    40.94 |     1.56 |  0.000e+00 |  3.608e-16
         200 |      0.248 |      0.010 |    25.24 |     0.52 |  0.000e+00 |  3.331e-16
         300 |      0.266 |      0.014 |    18.32 |     1.03 |  0.000e+00 |  2.220e-16
         400 |      0.279 |      0.021 |    13.41 |     0.46 |  0.000e+00 |  1.110e-16
         500 |      0.296 |      0.031 |     9.58 |     0.51 |  0.000e+00 |  9.437e-16
        1000 |      0.368 |      0.120 |     3.07 |     0.09 |  0.000e+00 |  8.882e-16
        2000 |      0.490 |      0.120 |     4.08 |     0.25 |  0.000e+00 |  8.882e-16
        3000 |      0.627 |      0.173 |     3.65 |     0.13 |  0.000e+00 |  5.862e-14
        4000 |      0.751 |      0.229 |     3.29 |     0.10 |  0.000e+00 |  2.309e-14
        5000 |      0.941 |      0.291 |     3.23 |     0.11 |  0.000e+00 |  1.343e-14
       10000 |      1.743 |      0.617 |     2.83 |     0.10 |  0.000e+00 |  1.762e-14
       20000 |      3.496 |      1.328 |     2.64 |     0.05 |  0.000e+00 |  3.189e-13
       30000 |      5.275 |      2.064 |     2.55 |     0.05 |  0.000e+00 |  2.569e-13
       40000 |      7.198 |      2.807 |     2.55 |     0.09 |  0.000e+00 |  4.368e-13
       50000 |      8.842 |      3.536 |     2.53 |     0.05 |  0.000e+00 |  4.479e-13
      100000 |     18.641 |      7.694 |     2.43 |     0.03 |  0.000e+00 |  1.586e-13
      200000 |     40.489 |     17.928 |     2.28 |     0.06 |  0.000e+00 |  8.105e-14
      300000 |     64.621 |     27.178 |     2.40 |     0.07 |  0.000e+00 |  1.055e-14
      400000 |     97.963 |     43.422 |     2.25 |     0.03 |  0.000e+00 |  1.242e-11
      500000 |    124.901 |     57.928 |     2.15 |     0.03 |  0.000e+00 |  1.008e-11
     1000000 |    271.340 |    139.220 |     1.93 |     0.06 |  0.000e+00 |  3.085e-12
        
    References
    ----------
    .. [1] Chatterjee, Sourav. "A new coefficient of correlation." Journal of
           the American Statistical Association 116.536 (2021): 2009-2022.
           :doi:`10.1080/01621459.2020.1758115`.
    .. [2] Virtanen, P., et al., SciPy 1.0: Fundamental Algorithms for 
           Scientific Computing in Python. Nature Methods, 17(3), 261-272  
               
@author: Jon Paul Lundquist
"""

from numba import njit
import numpy as np
import math
from collections import namedtuple
SignificanceResult = namedtuple("SignificanceResult", ["statistic", "pvalue"])

from ._utils import _argsort

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _rankdata(a, n):
    """
    'max'-tie ranks (tie block gets the *last* position).
    """
    #idx = np.argsort(a)
    idx = _argsort(a, n)
    
    r = np.empty(n, np.int64)
    # walk in sorted order; inside a tie block, all get rank = block end index + 1
    i = 0
    while i < n:
        j = i + 1
        vi = a[idx[i]]
        while j < n and a[idx[j]] == vi:
            j += 1
        rank_val = j  # 1-based, max position
        for k in range(i, j):
            r[idx[k]] = rank_val
        i = j
    return r

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _ranks_no_ties(a, n):
    # 1-based ranks for distinct values
    #idx = np.argsort(a)
    idx = _argsort(a, n)
    r = np.empty(n, np.int64)
    for k in range(n):
        r[idx[k]] = k + 1
    return r

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _xi_statistic(y_ordered, n):
    """
    Assumes input y is already ordered by the x-sort permutation.
    Ties are permitted.
    Produces:
      - r: max-tie rank of y
      - l: max-tie rank of -y
      - xi: 1 - 3 * sum |r[i]-r[i-1]| / (n^2 - 1)
    """

    r = _rankdata(y_ordered, n)
    # sum |r[i] - r[i-1]|
    s = 0.0
    for i in range(1, n):
        d = r[i] - r[i-1]
        s += d if d >= 0 else -d
    
    l = _rankdata(-y_ordered, n)
    
    den = 2 * np.sum((n - l) * l)
    xi = 1 - n * s / den
    return xi, r, l

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _xi_statistic_no_y_ties(y_ordered, n):

    r = _ranks_no_ties(y_ordered, n)   # no tie logic needed
    # sum |r[i]-r[i-1]|
    s = 0.0
    for i in range(1, n):
        d = r[i] - r[i-1]
        s += d if d >= 0 else -d
    xi = 1.0 - 3.0 * s / (n*n - 1.0)
    return xi, r

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _xi_std(r, l, n):
    """
    Compute asymptotic std of Xi in O(n log n) for the sort +
    O(n) for the single accumulation pass.
    """
    # 1) copy & sort r into u
    idx = _argsort(r, n)
    u = np.empty(n, dtype=np.float64)
    for i in range(n):
        #u[i] = r[i]
        u[i] = r[idx[i]]
    #u.sort()  # in-place sort
    
    # 2) one pass over sorted u to accumulate an, bn, cn
    s       = 0.0      # running sum v_j
    an_acc  = 0.0
    bn_acc  = 0.0
    cn_acc  = 0.0

    # j is 0-based here; i = j+1 in the math
    for j in range(n):
        uj = u[j]
        s  += uj
        i1 = j + 1
        coef = 2*n - 2*i1 + 1
        an_acc += coef * uj * uj
        cn_acc += coef * uj
        tmp     = s + (n - i1)*uj
        bn_acc += tmp * tmp

    inv_n2 = 1.0 / (n**2)
    inv_n3 = inv_n2 / n
    inv_n4 = inv_n3 / n
    inv_n5 = inv_n4 / n

    an = an_acc * inv_n4
    bn = bn_acc * inv_n5
    cn = cn_acc * inv_n3

    # 3) one pass over l to accumulate dn
    dn_acc = 0.0
    for i in range(n):
        li = l[i]
        dn_acc += li * (n - li)
    dn = dn_acc * inv_n3

    tau2 = (an - 2.0*bn + cn*cn) / (dn*dn)
    return math.sqrt(tau2/n)

@njit(cache=True, nogil=True, fastmath=True)
def chatterjeexi_noties(x_sorted, y_ordered, n, pvals=True):

    xi, r = _xi_statistic_no_y_ties(y_ordered, n)
    
    if pvals:
        l = (n - r + 1.0)                                     # ranks of -y (unique ⇒ exact reverse)
        sd = _xi_std(r, l, n)                                 # your finite-n variance routine

        #sd = 0.6324555320336759 / math.sqrt(n) # Asymptotic, continuous case
        if sd == 0.0:
            return xi, 0.0 if abs(xi) > 0 else 1.0
        pvalue = 0.5 * math.erfc((xi / sd) / 1.4142135623730951)
    else:
        pvalue = np.nan

    return xi, pvalue

@njit(cache=True, nogil=True, fastmath=True)
def chatterjeexi_ties(x_sorted, y_ordered, n, pvals=True):

    xi, r, l = _xi_statistic(y_ordered, n)
    
    if pvals:
        sd = _xi_std(r, l, n)
        if sd == 0.0:
            return xi, 0.0 if abs(xi) > 0 else 1.0
        pvalue = 0.5 * math.erfc((xi / sd) / 1.4142135623730951)

    else:
        pvalue = np.nan
        
    return xi, pvalue
    
def chatterjeexi(x, y, *, pvals=True, ties="auto", sorted_x=False):
    # x = np.asarray(x); y = np.asarray(y)
    # if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size<3:
    #     raise ValueError("x and y must be 1-D arrays of the same length greater than two")
    
    n = x.size
    
    if not sorted_x:
        idx = np.argsort(x, kind='quicksort') #Numba's sort causes disagreement with scipy.  
        x_sorted = x[idx]
        y_ordered = y[idx]
    else:
        x_sorted = x
        y_ordered = y

    if ties == "auto":
        has_ties = True 
        
    else:
        has_ties = bool(ties)
    
    if has_ties:
        tau, pvalue = chatterjeexi_ties(x_sorted, y_ordered, n, pvals=pvals)

    else:
        tau, pvalue = chatterjeexi_noties(x_sorted, y_ordered, n, pvals=pvals)

    res = SignificanceResult(statistic=float(tau), pvalue=float(pvalue))
    
    return res
