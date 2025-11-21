#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jon Paul Lundquist
"""
Created on Sun Oct  5 15:33:13 2025

    Hyper-fast Kendall’s tau (τ) correlation
    
    Kendall’s tau (τ) is a ranked correlation that is a nonparametric measure 
    of the monotonicity between two variables by comparing concordant vs. 
    discordant pairs. With no ties it is the normalized sum of the sign of the 
    slopes between all data pairs. τ ranges from −1 to +1: +1 indicates a 
    perfectly increasing monotonic relationship, −1 a perfectly decreasing one, 
    and 0 indicates no monotonic association [1,2,3].
    
    This module implements:
    - kendalltau_ties() τ-b (tie-aware) when ties are present or unknown,
    - kendalltau_noties() τ-a (no-ties) when both `x` and `y` are strictly monotone.
    - kendalltau (wrapper) a drop in replacement for scipy.kendalltau when data
      is sanitized but ties are unknown and/or data is not sorted by 'x'.
    
    For maximum speed on repeated calls, use the specialized kernels and you must 
    supply sanitized (no nan or inf) pre-sorted inputs:
    
        #Main arrays
        ind = np.argsort(x, kind="stable")  # keep equal x contiguous
        y_ordered = y[ind]
        x_sorted = x[ind]
        
        #Best use case: Run on many slices of main large arrays
        idx = some choice of sub-array
        
        # tie-aware (τ-b)
        tau, pvalue = kendalltau_ties(x_sorted[idx], y_ordered[idx], n)
    
        # no ties (τ-a)
        tau, pvalue = kendalltau_noties(x_sorted[idx], y_ordered[idx], n)
    
    If you can’t guarantee the preconditions, use `kendalltau(x, y, ties="auto",
    sorted_x=False)`. It sorts as needed and dispatches to the right kernel, but 
    is much slower than calling the kernels directly.
    
    kendalltau_ties() is up to ~x120 times faster than scipy for N=25 samples. 
    kendalltau_noties() is up to ~×430 times faster for N=25 samples.
    
    When to use
    -----------
    - Ideal case: **Many small/medium repeated slices** of pre-sorted large arrays 
      with known tie structure → `kendalltau_ties` / `kendalltau_noties` (fastest).
    - **One-off or unknown inputs** → `kendalltau` (convenience wrapper). Can be
        significantly slower.
              
    Parameters
    ----------
    x, y : 1-D array_like 
        samples of equal length (n ≥ 3). Inputs are treated as numeric; NaNs/Infs are not supported.
    
    pvals : {True, False}
        - Flag for p-value calculation. Default: pvals=True. Returns nan if pvals=False.
    
    For the specialized kernels kendalltau_ties() and kendalltau_noties():
        n : Length of the arrays      
        - `kendalltau_ties(x_sorted, y_ordered, n)`: `x_sorted` ascending; `y_ordered`
          must be `y` permuted by the same `ind`.
        - `kendalltau_noties(x_sorted, y_ordered, n)`: no ties in either variable.
        
    for kendalltau()
        ties : {"auto", True, False} 
        Choose tie-aware kernel automatically or force a variant.
            - "auto": Default. Detect ties and pick the proper kernel
            - True:   Force tie-aware kernel
            - False:  Force no-ties kernel
        sorted_x : bool
            Default: False. If True, you promise x is sorted ascending and y is 
            permuted by the same order. (True only if you pre-sorted upstream.)
    
    Returns
    -------
    for kendalltau_ties() / kendalltau_noties() :
        tau : Kendall's tau statistic.
        pvalue : Two-sided p-value for the null hypothesis of no association, 
                 H0: τ=0.
    
    for kendalltau() :
        res : SignificanceResult
        An object with the attributes:    
            statistic : float
               Kendall's tau statistic.
            pvalue : float
               Two-sided p-value for the null hypothesis of no association, H0: τ=0.
    
    Notes
    -----
    - Exact p-values are used for very small n / near-perfect orderings via a
      precomputed table (n ≤ 300); otherwise a normal approximation is used.
    
    See Also
    --------
    scipy.stats.kendalltau
    
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
    
    Benchmarks (illustrative; environment-dependent): See kendalltau_bench.py
    ------------------------------------------------
    CPU: Ultra 9 275HX, Python 3.13.5, NumPy 2.1.3, Numba 0.61.2, SciPy 1.16.0

    No ties (`kendalltau_noties`). Compared to scipy.kendall method='exact' for N<=300:
           N |  SciPy(ms) |  Hyper(ms) |   Speed× |      IQR |    Δτ(max) |    Δp(max)
    -----------------------------------------------------------------------------------------
          25 |      0.177 |      0.000 |   434.83 |    26.88 |  1.110e-16 |  6.661e-16
          50 |      0.384 |      0.001 |   634.72 |    95.45 |  2.776e-17 |  6.661e-16
          75 |      0.772 |      0.001 |   814.35 |    43.17 |  2.776e-17 |  5.551e-16
         100 |      1.290 |      0.001 |  1199.07 |    76.56 |  2.082e-17 |  7.772e-16
         200 |      7.743 |      0.002 |  3719.50 |   168.64 |  2.776e-17 |  2.331e-15
         300 |     21.847 |      0.003 |  7833.13 |   145.31 |  1.388e-17 |  7.772e-16
         400 |      0.187 |      0.004 |    49.75 |     1.60 |  6.939e-18 |  1.665e-16
         500 |      0.255 |      0.006 |    42.60 |     2.33 |  1.388e-17 |  2.220e-16
        1000 |      0.276 |      0.021 |    12.83 |     0.46 |  3.469e-18 |  1.665e-16
        2000 |      0.456 |      0.100 |     4.56 |     0.12 |  3.469e-18 |  1.110e-16
        3000 |      0.579 |      0.171 |     3.39 |     0.14 |  3.469e-18 |  2.220e-16
        4000 |      0.701 |      0.241 |     2.90 |     0.09 |  1.735e-18 |  2.220e-16
        5000 |      0.876 |      0.312 |     2.78 |     0.11 |  1.735e-18 |  2.220e-16
       10000 |      1.645 |      0.708 |     2.28 |     0.06 |  1.735e-18 |  1.110e-16
       20000 |      3.136 |      1.532 |     2.06 |     0.09 |  8.674e-19 |  1.110e-16
       30000 |      4.729 |      2.401 |     1.95 |     0.08 |  8.674e-19 |  1.110e-16
       40000 |      6.466 |      3.268 |     1.98 |     0.07 |  8.674e-19 |  1.665e-16
       50000 |      8.165 |      4.185 |     1.92 |     0.05 |  8.674e-19 |  2.220e-16
      100000 |     16.533 |      8.990 |     1.84 |     0.07 |  8.674e-19 |  2.220e-16
      200000 |     34.032 |     18.689 |     1.82 |     0.04 |  4.337e-19 |  1.943e-16
      300000 |     57.046 |     29.774 |     1.90 |     0.08 |  2.168e-19 |  1.110e-16
      400000 |     79.231 |     42.273 |     1.91 |     0.08 |  4.337e-19 |  1.110e-16
      500000 |     92.687 |     48.983 |     1.87 |     0.06 |  2.168e-19 |  1.665e-16
     1000000 |    197.746 |    107.477 |     1.89 |     0.09 |  2.168e-19 |  1.110e-16
         
    With ties (`kendalltau_ties`):
           N |  SciPy(ms) |  Hyper(ms) |   Speed× |      IQR |    Δτ(max) |    Δp(max)
    -----------------------------------------------------------------------------------------
          25 |      0.121 |      0.001 |   122.66 |     5.02 |  2.776e-17 |  2.220e-16
          50 |      0.115 |      0.001 |    95.09 |     2.16 |  5.551e-17 |  1.665e-16
          75 |      0.116 |      0.002 |    75.30 |     2.37 |  1.388e-17 |  1.665e-16
         100 |      0.116 |      0.002 |    62.30 |     2.90 |  1.388e-17 |  1.110e-16
         200 |      0.122 |      0.003 |    37.38 |     1.65 |  1.388e-17 |  1.110e-16
         300 |      0.129 |      0.005 |    27.09 |     0.75 |  1.388e-17 |  1.110e-16
         400 |      0.136 |      0.006 |    21.09 |     1.01 |  1.388e-17 |  1.665e-16
         500 |      0.142 |      0.008 |    17.83 |     0.44 |  1.388e-17 |  1.110e-16
        1000 |      0.182 |      0.020 |     9.30 |     0.66 |  6.939e-18 |  1.110e-16
        2000 |      0.233 |      0.042 |     5.66 |     0.21 |  3.469e-18 |  1.110e-16
        3000 |      0.285 |      0.063 |     4.57 |     0.21 |  6.939e-18 |  2.220e-16
        4000 |      0.343 |      0.083 |     4.12 |     0.08 |  3.469e-18 |  1.665e-16
        5000 |      0.403 |      0.101 |     3.99 |     0.13 |  1.735e-18 |  1.110e-16
       10000 |      0.618 |      0.174 |     3.58 |     0.11 |  3.469e-18 |  1.665e-16
       20000 |      1.067 |      0.342 |     3.13 |     0.07 |  1.735e-18 |  1.665e-16
       30000 |      1.505 |      0.512 |     2.94 |     0.06 |  8.674e-19 |  1.388e-16
       40000 |      2.006 |      0.674 |     3.00 |     0.13 |  8.674e-19 |  1.110e-16
       50000 |      2.443 |      0.843 |     2.88 |     0.07 |  8.674e-19 |  2.220e-16
      100000 |      4.734 |      1.687 |     2.80 |     0.05 |  8.674e-19 |  1.110e-16
      200000 |      9.481 |      3.369 |     2.82 |     0.06 |  4.337e-19 |  1.110e-16
      300000 |     15.224 |      5.423 |     2.77 |     0.09 |  4.337e-19 |  1.110e-16
      400000 |     21.107 |      7.602 |     2.78 |     0.09 |  4.337e-19 |  2.220e-16
      500000 |     27.687 |     10.408 |     2.70 |     0.38 |  2.168e-19 |  1.110e-16
     1000000 |     68.201 |     27.528 |     2.44 |     0.12 |  2.168e-19 |  2.220e-16
  
    `kendalltau` (wrapper) with unsorted samples without any ties:
           N |  SciPy(ms) |  Hyper(ms) |   Speed× |      IQR |    Δτ(max) |    Δp(max)
    -----------------------------------------------------------------------------------------
          25 |      0.177 |      0.002 |   103.57 |     2.42 |  1.110e-16 |  6.661e-16
          50 |      0.289 |      0.002 |   133.15 |     3.41 |  2.776e-17 |  6.661e-16
          75 |      0.490 |      0.003 |   182.78 |     7.89 |  2.776e-17 |  5.551e-16
         100 |      0.895 |      0.003 |   273.28 |     8.14 |  2.082e-17 |  7.772e-16
         200 |      5.761 |      0.007 |   856.19 |    49.92 |  2.776e-17 |  2.331e-15
         300 |     18.860 |      0.011 |  1693.36 |    69.22 |  1.388e-17 |  7.772e-16
         400 |      0.145 |      0.012 |    12.23 |     0.54 |  6.939e-18 |  1.665e-16
         500 |      0.150 |      0.016 |     9.39 |     0.28 |  1.388e-17 |  2.220e-16
        1000 |      0.183 |      0.036 |     5.13 |     0.24 |  3.469e-18 |  1.665e-16
        2000 |      0.272 |      0.074 |     3.68 |     0.10 |  3.469e-18 |  1.110e-16
        3000 |      0.362 |      0.121 |     2.99 |     0.09 |  3.469e-18 |  2.220e-16
        4000 |      0.444 |      0.166 |     2.66 |     0.06 |  1.735e-18 |  2.220e-16
        5000 |      0.547 |      0.221 |     2.48 |     0.06 |  1.735e-18 |  1.388e-16
       10000 |      1.027 |      0.499 |     2.07 |     0.05 |  1.735e-18 |  1.110e-16
       20000 |      2.065 |      1.074 |     1.92 |     0.03 |  8.674e-19 |  1.110e-16
       30000 |      3.065 |      1.655 |     1.85 |     0.03 |  8.674e-19 |  1.110e-16
       40000 |      4.242 |      2.284 |     1.86 |     0.02 |  8.674e-19 |  1.665e-16
       50000 |      5.401 |      2.955 |     1.83 |     0.03 |  8.674e-19 |  2.220e-16
      100000 |     11.405 |      6.375 |     1.78 |     0.02 |  8.674e-19 |  2.220e-16
      200000 |     24.584 |     14.254 |     1.72 |     0.02 |  4.337e-19 |  1.943e-16
      300000 |     39.230 |     23.904 |     1.64 |     0.02 |  2.168e-19 |  1.110e-16
      400000 |     52.634 |     33.750 |     1.56 |     0.04 |  4.337e-19 |  1.110e-16
      500000 |     69.264 |     47.399 |     1.46 |     0.07 |  2.168e-19 |  1.665e-16
     1000000 |    164.725 |    137.065 |     1.21 |     0.05 |  2.168e-19 |  1.110e-16
     
    `kendalltau` (wrapper) with unsorted samples with ties:
            N |  SciPy(ms) |  Hyper(ms) |   Speed× |      IQR |    Δτ(max) |    Δp(max)
     -----------------------------------------------------------------------------------------
          25 |      0.118 |      0.002 |    52.33 |     1.26 |  2.776e-17 |  2.220e-16
          50 |      0.120 |      0.003 |    46.15 |     1.08 |  5.551e-17 |  1.665e-16
          75 |      0.121 |      0.003 |    39.82 |     0.51 |  1.388e-17 |  1.665e-16
         100 |      0.121 |      0.004 |    34.38 |     0.84 |  1.388e-17 |  1.110e-16
         200 |      0.132 |      0.006 |    22.96 |     0.90 |  1.388e-17 |  1.110e-16
         300 |      0.140 |      0.008 |    17.24 |     0.44 |  1.388e-17 |  1.110e-16
         400 |      0.149 |      0.012 |    12.75 |     0.43 |  1.388e-17 |  1.665e-16
         500 |      0.153 |      0.015 |    10.09 |     0.36 |  1.388e-17 |  1.110e-16
        1000 |      0.180 |      0.035 |     5.21 |     0.17 |  6.939e-18 |  1.110e-16
        2000 |      0.244 |      0.068 |     3.63 |     0.15 |  3.469e-18 |  1.110e-16
        3000 |      0.304 |      0.099 |     3.05 |     0.06 |  6.939e-18 |  2.220e-16
        4000 |      0.356 |      0.130 |     2.73 |     0.05 |  3.469e-18 |  1.665e-16
        5000 |      0.417 |      0.158 |     2.67 |     0.09 |  1.735e-18 |  1.110e-16
       10000 |      0.667 |      0.259 |     2.59 |     0.08 |  3.469e-18 |  1.665e-16
       20000 |      1.157 |      0.475 |     2.43 |     0.11 |  1.735e-18 |  1.665e-16
       30000 |      1.613 |      0.698 |     2.32 |     0.14 |  8.674e-19 |  1.388e-16
       40000 |      2.182 |      0.926 |     2.36 |     0.13 |  8.674e-19 |  1.110e-16
       50000 |      2.659 |      1.141 |     2.31 |     0.16 |  8.674e-19 |  2.220e-16
      100000 |      5.226 |      2.274 |     2.29 |     0.13 |  8.674e-19 |  1.110e-16
      200000 |     10.839 |      4.727 |     2.27 |     0.07 |  4.337e-19 |  1.110e-16
      300000 |     16.866 |      7.694 |     2.18 |     0.10 |  4.337e-19 |  1.110e-16
      400000 |     23.698 |     11.352 |     2.06 |     0.10 |  4.337e-19 |  2.220e-16
      500000 |     30.924 |     17.263 |     1.82 |     0.14 |  2.168e-19 |  1.110e-16
     1000000 |     72.361 |     45.067 |     1.63 |     0.13 |  2.168e-19 |  2.220e-16

    `kendalltau` (wrapper) with unsorted random tie/no_tie samples:
           N |  SciPy(ms) |  Hyper(ms) |   Speed× |      IQR |    Δτ(max) |    Δp(max)
    -----------------------------------------------------------------------------------------
          25 |      0.122 |      0.002 |    54.39 |    36.51 |  1.110e-16 |  6.661e-16
          50 |      0.121 |      0.002 |    47.93 |    81.47 |  2.776e-17 |  4.441e-16
          75 |      0.487 |      0.003 |   172.19 |   139.79 |  2.776e-17 |  6.661e-16
         100 |      0.123 |      0.003 |    35.42 |   222.56 |  2.776e-17 |  4.441e-16
         200 |      5.312 |      0.007 |   798.75 |    76.62 |  1.388e-17 |  2.109e-15
         300 |      0.141 |      0.008 |    17.72 |  1525.92 |  1.388e-17 |  6.661e-16
         400 |      0.138 |      0.013 |    10.62 |     1.07 |  6.939e-18 |  1.110e-16
         500 |      0.148 |      0.015 |     9.93 |     0.88 |  6.939e-18 |  2.220e-16
        1000 |      0.175 |      0.033 |     5.22 |     0.29 |  3.469e-18 |  1.665e-16
        2000 |      0.239 |      0.067 |     3.60 |     0.10 |  3.469e-18 |  1.665e-16
        3000 |      0.342 |      0.115 |     3.04 |     0.10 |  6.939e-18 |  2.220e-16
        4000 |      0.428 |      0.160 |     2.68 |     0.10 |  1.735e-18 |  1.388e-16
        5000 |      0.535 |      0.213 |     2.54 |     0.17 |  3.469e-18 |  2.220e-16
       10000 |      0.659 |      0.260 |     2.52 |     0.53 |  8.674e-19 |  1.110e-16
       20000 |      1.157 |      0.524 |     2.22 |     0.51 |  1.735e-18 |  1.665e-16
       30000 |      2.948 |      1.593 |     1.88 |     0.43 |  8.674e-19 |  1.110e-16
       40000 |      4.133 |      2.237 |     1.86 |     0.46 |  8.674e-19 |  2.220e-16
       50000 |      2.609 |      1.161 |     2.16 |     0.51 |  8.674e-19 |  1.665e-16
      100000 |     10.690 |      6.016 |     1.78 |     0.52 |  8.674e-19 |  2.220e-16
      200000 |     22.565 |     13.112 |     1.73 |     0.51 |  8.674e-19 |  1.665e-16
      300000 |     18.364 |      8.007 |     2.04 |     0.56 |  2.168e-19 |  1.665e-16
      400000 |     24.510 |     12.183 |     1.86 |     0.53 |  4.337e-19 |  2.220e-16
      500000 |     68.852 |     44.766 |     1.55 |     0.41 |  4.337e-19 |  1.110e-16
     1000000 |     88.122 |     53.717 |     1.42 |     0.42 |  2.168e-19 |  2.220e-16
     
    References
    ----------
    .. [1] Kendall, M. G., A New Measure of Rank Correlation, Biometrika Vol. 30, 
           No. 1/2, pp. 81-93, 1938.
    .. [2] Kendall, M. G., The treatment of ties in ranking problems, Biometrika 
           Vol. 33, No. 3, pp. 239-251. 1945.
    .. [3] Kendall, M. G., Rank Correlation Methods (4th Edition), 
           Charles Griffin & Co., 1970.
    .. [4] Virtanen, P., et al., SciPy 1.0: Fundamental Algorithms for 
           Scientific Computing in Python. Nature Methods, 17(3), 261-272    
               
@author: Jon Paul Lundquist
"""

from numba import njit
import numpy as np
import math
from importlib.resources import files, as_file

from collections import namedtuple
SignificanceResult = namedtuple("SignificanceResult", ["statistic", "pvalue"])

from ._utils import _has_dups_quicksort, _has_dups_sorted, _has_dups_argsort, _argsort

with as_file(files("hyper_corr.data").joinpath("kendall_p_flat.npy")) as pf, \
     as_file(files("hyper_corr.data").joinpath("kendall_p_offsets.npy")) as of, \
     as_file(files("hyper_corr.data").joinpath("kendall_p_lo.npy")) as lf, \
     as_file(files("hyper_corr.data").joinpath("kendall_p_hi.npy")) as hf:
    P_FLAT   = np.load(pf, mmap_mode="r", allow_pickle=False)
    OFFSETS  = np.load(of, mmap_mode="r", allow_pickle=False)
    LO       = np.load(lf, mmap_mode="r", allow_pickle=False)
    HI       = np.load(hf, mmap_mode="r", allow_pickle=False)

EXPLOG = np.zeros(171, np.float64)
acc = 0.0  # log(k!)
EXPLOG[0] = 2.0  # 2/0! = 2
for k in range(1, 171):
    acc += math.log(k)          # log(k!)
    EXPLOG[k] = math.exp(0.6931471805599453 - acc)  # 2 - log(k!) in exp

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _kendall_p_exact(n, c):
    # fold to effective tail
    m_pairs = n * (n - 1) // 2
    half    = m_pairs // 2
    c_eff   = c if c <= half else (m_pairs - c)

    # midpoint (exact 1.0)
    if 4 * c_eff == n * (n - 1):
        return 1.0

    # ragged lookup when available
    if n <= 300:
        lo = int(LO[n]); hi = int(HI[n])
        if c_eff < lo:
            return 0.0      # below stored band → numerically zero
        if c_eff > hi:
            return 1.0      # above stored band → numerically one
        off = int(OFFSETS[n])
        return float(P_FLAT[off + (c_eff - lo)])

    # exact fallback (two-sided)
    new = np.zeros(c_eff + 1, np.float64)
    new[0] = 1.0
    if c_eff >= 1:
        new[1] = 1.0
    for j in range(3, n + 1):
        # prefix-sum / divide-by-j pass
        s = 0.0
        for i in range(c_eff + 1):
            s += new[i]
            new[i] = s / j
        limit = (c_eff + 1) - j
        if limit > 0:
            for i in range(j, c_eff + 1):
                new[i] -= new[i - j]

    p = 0.0
    for i in range(c_eff + 1):
        p += new[i]
    #p *= 2.0
    if p < 0.0: p = 0.0
    if p > 1.0: p = 1.0
    return p

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _sorted_ranks(segments, x, n):
    segments[0] = 1  # First element always starts a new segment
    for i in range(1, n):
        segments[i] = segments[i-1] + (x[i] != x[i-1])
    return segments

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _ind_ranks(segments, x, ind, n):
    segments[0] = 1  # First element always starts a new segment
    for i in range(1, n):
        segments[i] = segments[i-1] + (x[ind[i]] != x[ind[i-1]])
    return segments

@njit(cache=True, nogil=True, fastmath=True)
def _ranks(a, n):
    """
    Map float array 'a' to 1-based integer ranks (equal values share a rank).
    Returns (rank:int32[:], max_rank:int32).
    """
    idx = _argsort(a, n)
    rank = np.empty(n, np.int32)
    r = 1
    rank[idx[0]] = r
    for k in range(1, n):
        if a[idx[k]] != a[idx[k-1]]:
            r += 1
        rank[idx[k]] = r
    return rank, r  # r is max rank

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _dis_mergestable(xy, dst, n):
    """
    Count inversions with iterative mergesort, using caller-provided buffer.
    returns inv_count.
    """

    src = xy.copy()
    inv = np.int64(0)
    width = 1
    while width < n:
        i = 0
        while i < n:
            mid = i + width
            end = i + width + width
            if mid > n: mid = n
            if end > n: end = n

            j = i
            k = mid
            p = i
            while j < mid and k < end:
                if src[j] <= src[k]:
                    dst[p] = src[j]
                    j += 1
                else:
                    dst[p] = src[k] 
                    inv += (mid - j) 
                    k += 1
                p += 1
            while j < mid: 
                dst[p] = src[j]
                j += 1; p += 1
            while k < end: 
                dst[p] = src[k]
                k += 1; p += 1

            i = end

        tmp = src; src = dst; dst = tmp
        width <<= 1

    return int(inv)

@njit(cache=True, nogil=True, fastmath=True)
def _ties_and_dis(x_rank, y_rank, x_max, y_max, n):
    cx = np.zeros(x_max + 1, np.int64)
    cy = np.zeros(y_max + 1, np.int64)
    for i in range(n):
        cx[x_rank[i]] += 1
        cy[y_rank[i]] += 1

    # stable scatter by x
    tmp = np.empty(n, np.int32)
    pos = 0
    xtie = np.int64(0)
    x0 = np.float64(0.0)
    x1 = np.float64(0.0)
    for k in range(1, x_max + 1):
        c = cx[k]; cx[k] = pos; pos += c
        if c > 1:
            f = c * (c - 1)
            xtie += f
            x0 += f * (c - 2)
            x1 += f * (2 * c + 5)
    
    xtie = xtie // 2 #integer divide by two
    for i in range(n):
        k = x_rank[i]
        p = cx[k]
        tmp[p] = i
        cx[k] = p + 1

    # stable scatter by y -> perm
    perm = np.empty(n, np.int32)
    pos = np.int64(0)
    ytie = np.int64(0)
    y0 = np.float64(0.0)
    y1 = np.float64(0.0)
    for k in range(1, y_max + 1):
        c = cy[k]; cy[k] = pos; pos += c
        if c > 1:
            f = c * (c - 1)
            ytie += f
            y0 += f * (c - 2)
            y1 += f * (2 * c + 5)
    
    ytie = ytie // 2 #integer divide by two
    for i in range(n):
        j = tmp[i]
        k = y_rank[j]
        p = cy[k]
        perm[p] = j
        cy[k] = p + 1

    # Boundaries from perm itself (contiguous by y)
    y_start = np.empty(y_max + 2, np.int32)
    y_start[1] = 0
    g = 1                       # current y-rank (1..y_max)
    for i in range(1, n):
        if y_rank[perm[i]] != y_rank[perm[i-1]]:
            g += 1
            y_start[g] = i
    y_start[y_max + 1] = n

    # ntie (within each y group, x equal runs)
    ntie = np.int64(0)
    for k in range(1, y_max + 1):
        a = y_start[k]; b = y_start[k + 1]
        if b - a >= 2:
            # perm[a:b] are indices with this y; they are x-secondary-stable
            # Scan runs of equal x_rank
            run = 1
            for t in range(a + 1, b):
                if x_rank[perm[t]] == x_rank[perm[t - 1]]:
                    run += 1
                else:
                    if run > 1:
                        ntie += run * (run - 1) // 2
                    run = 1
            if run > 1:
                ntie += run * (run - 1) // 2

    # discordant pairs via Fenwick over x ranks, processing y groups in order.
    # Important: insert a group's x after querying it, so ties on y are excluded.
    bit = np.zeros(x_max + 1, np.int64)
    seen = np.int64(0)
    dis  = np.int64(0)
    for k in range(1, y_max + 1):
        a = y_start[k]; b = y_start[k + 1]
        # first query all in group
        for t in range(a, b):
            r = x_rank[perm[t]]
            # strictly greater x seen so far
            # Instead of calling _bit_sum()
            s = 0
            ii = r
            while ii > 0:
                s += bit[ii]
                ii -= ii & -ii
            dis += (seen-s)
        # then add this group's x ranks to the tree
        for t in range(a, b):
            r = x_rank[perm[t]]
            ii = r
            while ii <= x_max:
                bit[ii] += 1
                ii += ii & -ii
        seen += (b - a)

    return xtie, x0, x1, ytie, y0, y1, ntie, dis

# ---------- main: Kendall tau-b ----------

@njit(nogil=True, fastmath=True)
def kendalltau_ties(x_sorted, y_ordered, n, pvals=True):
    """
    - x_sorted must be sorted and of course y must be ordered by x.
        ind = np.argsort(x, kind='stable')
        y = y[ind]
        x = x[ind]
    """
    # x ranks, equal x share the same rank
    segments = np.empty(n, np.int32)
    x_rank = _sorted_ranks(segments, x_sorted, n)

    x_max  = int(x_rank[-1])
     # y ranks, equal y share the same rank
    y_rank, y_max = _ranks(y_ordered, n)
    
    xtie, x0, x1, ytie, y0, y1, ntie, dis = _ties_and_dis(x_rank, y_rank, 
                                                          x_max, y_max, n)
    
    # Assemble tau-b and p (matches your formulas)
    m = n * (n - 1.0)
    tot = int(m) // 2
    con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
    denom = math.sqrt(tot - xtie) * math.sqrt(tot - ytie)
    tau = con_minus_dis / denom if denom > 0 else 0.0
    if tau > 1.0: tau = 1.0
    if tau < -1.0: tau = -1.0

    if pvals:
        if (xtie == 0 and ytie == 0) and (n <= 300 or min(dis, tot - dis) <= 1):
            pvalue = _kendall_p_exact(n, min(dis, tot - dis))
        else:
            var = ((m * (2*n + 5) - x1 - y1) / 18.0
                   + (2.0 * xtie * ytie) / m
                   + (x0 * y0) / (9.0 * m * (n - 2)))
            pvalue = math.erfc(abs(con_minus_dis) / math.sqrt(var) / 1.4142135623730951)
    else:
        pvalue = math.nan
        
    return float(tau), float(pvalue)

@njit(nogil=True, fastmath=True)
def _kendalltau_ties_unsorted(x, y, n, idx=None, pvals=True):
    if idx is None:
        idx = _argsort(x, n)
    
    # x ranks, equal x share the same rank
    segments = np.empty(n, np.int32)
    x_rank = _ind_ranks(segments, x, idx, n)

    x_max  = int(x_rank[-1])
    # y ranks, equal y share the same rank
    y_ordered = np.empty(n, np.float64)
    for i in range(n):
        y_ordered[i] = y[idx[i]]
    y_rank, y_max = _ranks(y_ordered, n)
    
    xtie, x0, x1, ytie, y0, y1, ntie, dis = _ties_and_dis(x_rank, y_rank, 
                                                          x_max, y_max, n)
    
    # Assemble tau-b and p (matches your formulas)
    m = n * (n - 1.0)
    tot = int(m) // 2
    con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
    denom = math.sqrt(tot - xtie) * math.sqrt(tot - ytie)
    tau = con_minus_dis / denom if denom > 0 else 0.0
    if tau > 1.0: tau = 1.0
    if tau < -1.0: tau = -1.0

    if pvals:
        if (xtie == 0 and ytie == 0) and (n <= 300 or min(dis, tot - dis) <= 1):
            pvalue = _kendall_p_exact(n, min(dis, tot - dis))
        else:
            var = ((m * (2*n + 5) - x1 - y1) / 18.0
                   + (2.0 * xtie * ytie) / m
                   + (x0 * y0) / (9.0 * m * (n - 2)))
            pvalue = math.erfc(abs(con_minus_dis) / math.sqrt(var) / 1.4142135623730951)
    else:
        pvalue = math.nan
        
    return float(tau), float(pvalue)

@njit(nogil=True, fastmath=True)
def kendalltau_noties(x_sorted, y_ordered, n, pvals=True):
    """
    Use ONLY if there are no ties or if you want to sacrifice some accuracy for speed
    - x_sorted must be sorted and of course y must be ordered by x.
        ind = np.argsort(x)
        y = y[ind]
        x = x[ind]
    """
    
    buf = np.empty(n, np.float64)
    dis = _dis_mergestable(y_ordered, buf, n)
    
    m = n * (n - 1.)
    tot = int(m) // 2  # total number of pairs 
    con_minus_dis = tot - 2 * dis
    tau = con_minus_dis / float(tot)
    
    if pvals:
        if (n <= 300 or min(dis, tot-dis) <= 1):
            pvalue = _kendall_p_exact(n, min(dis, tot-dis))
        else:
            var = (m * (2*n + 5)) / 18
            pvalue = math.erfc(abs(con_minus_dis) / math.sqrt(var) / 1.4142135623730951)  # using erfc for two-tailed p-value
    else:
        pvalue = math.nan
        
    return float(tau), float(pvalue)

@njit(nogil=True, fastmath=True)
def _kendalltau_noties_unsorted(x, y, n, idx=None, pvals=True):
    """
    Use ONLY if there are no ties or if you want to sacrifice some accuracy for speed
    """
    if idx is None:
        idx = _argsort(x, n)  
        
    y_ordered = np.empty(n, np.float64)
    for i in range(n):
        y_ordered[i] = y[idx[i]]

    tau, pvalue = kendalltau_noties(x, y_ordered, n, pvals=pvals)
        
    return float(tau), float(pvalue)

def kendalltau(x, y, pvals=True, ties='auto', sorted_x=False):
    #Routing to kendalltau_ties without ties checking for n>300 is faster
    #for randomly generated tie/no_tie samples. See kendelltau_bench.py
    
    #commented checks.
    # x = np.asarray(x, dtype=np.float64)
    # y = np.asarray(y, dtype=np.float64)
    # if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size<3:
    #     raise ValueError("x and y must be 1-D arrays of the same length greater than two")
    
    n = x.size
    
    if not sorted_x:
        if ties=='auto':
            #Just routing to kendalltau_ties with no ties checking for n>300 is 
            #faster and results in correct answers as well...
            
            if n<=300:
                idx, x_dups = _has_dups_argsort(x, n, early_exit=False)
                if x_dups:
                    tau, pvalue = _kendalltau_ties_unsorted(x, y, n, idx=idx, pvals=pvals)
                    
                else:
                    y_dups = _has_dups_quicksort(y, n)
                    if y_dups:
                        tau, pvalue = _kendalltau_ties_unsorted(x, y, n, idx=idx, pvals=pvals)
                        
                    else:
                        tau, pvalue = _kendalltau_noties_unsorted(x, y, n, idx=idx, pvals=pvals)
                    
            else:
               tau, pvalue = _kendalltau_ties_unsorted(x, y, n, pvals=pvals)
                
        elif ties==False:
            tau, pvalue = _kendalltau_noties_unsorted(x, y, n, pvals=pvals)
            
        elif ties==True:
            tau, pvalue = _kendalltau_ties_unsorted(x, y, n, pvals=pvals)
            
    else:
        if ties=='auto':
            #Just routing to kendalltau_ties with no ties checking for n>300 is 
            #faster and results in correct answers as well...
            
            if n<=300:
                x_dups = _has_dups_sorted(x, n)
                if x_dups:
                    tau, pvalue = kendalltau_ties(x, y, n, pvals=pvals)
                    
                else:
                    y_dups = _has_dups_quicksort(y, n)
                    if y_dups:
                        tau, pvalue = kendalltau_ties(x, y, n, pvals=pvals)
                    
                    else:
                        tau, pvalue = kendalltau_noties(x, y, n, pvals=pvals)
                    
            else:
                tau, pvalue = kendalltau_ties(x, y, n, pvals=pvals)
                
        elif ties==False:
            tau, pvalue = kendalltau_noties(x, y, n, pvals=pvals)
            
        elif ties==True:
            tau, pvalue = kendalltau_ties(x, y, n, pvals=pvals)

    #Mirroring scipy output for compatibility. Slows things down slightly.
    res = SignificanceResult(statistic=float(tau), pvalue=float(pvalue))
    
    return res