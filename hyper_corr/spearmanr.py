#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jon Paul Lundquist
"""
Created on Sun Oct  5 18:01:25 2025

    Hyper-fast Spearman's rho (ρ) correlation

    Spearman’s rho (ρ) measures rank-order (monotonic) association between two 
    variables. It is computed as Pearson’s r (linear correlation) on the ranked 
    samples (with ties assigned average ranks). ρ ranges from −1 to +1: +1 
    indicates a perfectly increasing monotonic relationship, −1 a perfectly 
    decreasing one, and 0 indicates no monotonic association [1].

    This module implements:
    - spearmanr_ties when ties are present,
    - spearmanr_noties when both `x` and `y` are strictly monotone.
    - spearmanr when ties are unknown.
    
    For maximum speed on repeated calls, use the specialized kernels and you must 
    supply pre-sorted inputs:
    
        idx = np.argsort(x, kind="stable")  # keep equal x contiguous
        y_ordered = y[idx]
        x_sorted = x[idx]
    
        # tie-aware
        rho, pvalue = spearmanr_ties(x_sorted, y_ordered, n)
    
        # no ties
        rho, pvalue = spearmanr_noties(x_sorted, y_ordered, n)
    
    If you can’t guarantee the preconditions, use `spearmanr(x, y, ties="auto",
    sorted_x=False)`. It sorts as needed and dispatches to the right kernel, but 
    is much slower than calling the kernels directly.
    
    All versions are faster than scipy for any N with Numba enabled.
    spearmanr_ties() is x200 (for N=25) to x5 times faster than scipy. 
    spearmanr_noties() is ~×220 (for N=25) to x6 times faster.
    spearmanr() is ~x100 (for N=25) to x3 times faster than scipy.
    
    With Numba removed: 
    spearmanr_ties() is ~x10 (for N=25) faster than scipy. Slower than scipy for 
    N>=3000 (possibly due to loops Numba likes). About 55% scipy speed at N=1,000,000.
    spearmanr_noties() is ~x30 (for N=25) faster than scipy. Slower than scipy for 
    N>=3000 (possibly due to loops Numba likes). Goes back up to 90% scipy speed by N=1,000,000.
    
    When to use
    -----------
    - **Many small/medium repeated slices** of pre-sorted large arrays with known tie
      structure → `spearmanr_ties` / `spearmanr_noties` (fastest).
    - **One-off or unknown inputs** → `spearmanr` (convenience wrapper).

    Parameters
    ----------
    x, y : 1-D array_like 
        samples of equal length (n ≥ 3). Inputs are treated as numeric; NaNs/Infs are not supported.
    
    pvals : {True, False}
        - Flag for p-value calculation. Default: pvals=True. Returns nan if pvals=False.

    For the specialized kernels spearmanr_ties() and spearmanr_noties():
        n : Length of the arrays      
        - `spearmanr_ties(x_sorted, y_ordered, n)`: `x_sorted` ascending; `y_ordered`
          must be `y` permuted by the same `idx`.
        - `spearmanr_noties(x_sorted, y_ordered, n)`: no ties in either variable.
        
    for spearmanr()
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
    for spearmanr_ties() / spearmanr_noties() :
        rho : Spearman’s rho statistic.
        pvalue : Two-sided p-value for the null hypothesis of no association, 
                 H0: ρ=0.
    
    for spearmanr() :
        res : SignificanceResult
        An object with the attributes:    
            statistic : float
               Spearman's rho statistic.
            pvalue : float
               Two-sided p-value for the null hypothesis of no association, H0: ρ=0.
    
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
    
    Benchmarks (illustrative; environment-dependent): See spearmanr_bench.py
    ------------------------------------------------
    CPU: Ultra 9 275HX, Python 3.13.5, NumPy 2.1.3, Numba 0.61.2, SciPy 1.16.0

    With ties (`spearmanr_ties`):
           N |  SciPy(ms) |  Hyper(ms) |   Speed× |      IQR |  Δrho(max) |    Δp(max)
    -----------------------------------------------------------------------------------------
          25 |      0.141 |      0.001 |   192.56 |    13.91 |  1.110e-16 |  2.220e-14
          50 |      0.142 |      0.001 |   163.83 |     7.34 |  5.551e-17 |  1.343e-14
          75 |      0.146 |      0.001 |   141.39 |     8.62 |  2.776e-17 |  3.542e-14
         100 |      0.151 |      0.001 |   127.67 |     5.67 |  2.776e-17 |  4.763e-14
         200 |      0.151 |      0.002 |    76.28 |     4.00 |  2.776e-17 |  1.949e-12
         300 |      0.157 |      0.003 |    58.14 |     1.55 |  1.388e-17 |  1.002e-13
         400 |      0.162 |      0.004 |    46.09 |     1.72 |  1.388e-17 |  1.723e-13
         500 |      0.168 |      0.004 |    38.89 |     1.71 |  1.388e-17 |  1.629e-12
        1000 |      0.195 |      0.009 |    20.70 |     0.58 |  6.939e-18 |  1.748e-12
        2000 |      0.238 |      0.025 |     9.86 |     0.54 |  6.939e-18 |  2.446e-12
        3000 |      0.287 |      0.035 |     8.26 |     0.49 |  1.388e-17 |  1.428e-12
        4000 |      0.322 |      0.044 |     7.40 |     0.38 |  6.939e-18 |  4.579e-12
        5000 |      0.377 |      0.050 |     7.60 |     0.60 |  5.204e-18 |  3.434e-12
       10000 |      0.597 |      0.070 |     8.49 |     0.51 |  3.469e-18 |  3.857e-12
       20000 |      1.049 |      0.123 |     8.50 |     0.48 |  3.469e-18 |  1.080e-11
       30000 |      1.448 |      0.182 |     7.89 |     0.59 |  1.735e-18 |  3.049e-11
       40000 |      1.956 |      0.233 |     8.37 |     0.82 |  1.735e-18 |  3.919e-11
       50000 |      2.407 |      0.300 |     8.03 |     0.59 |  8.674e-19 |  6.041e-11
      100000 |      4.773 |      0.651 |     7.30 |     0.74 |  3.799e-16 |  2.238e-11
      200000 |     12.384 |      1.514 |     7.47 |     1.54 |  4.337e-19 |  1.529e-10
      300000 |     21.468 |      2.861 |     7.39 |     1.02 |  3.339e-16 |  1.299e-09
      400000 |     29.550 |      4.169 |     7.03 |     0.60 |  5.182e-16 |  2.835e-10
      500000 |     37.926 |      5.850 |     6.50 |     0.68 |  3.147e-16 |  1.691e-10
     1000000 |     81.573 |     16.011 |     5.03 |     0.63 |  2.132e-13 |  1.072e-09
 
    No ties (`spearmanr_noties`):
           N |  SciPy(ms) |  Hyper(ms) |   Speed× |      IQR |  Δrho(max) |    Δp(max)
    -----------------------------------------------------------------------------------------
          25 |      0.141 |      0.001 |   217.27 |    10.96 |  2.776e-17 |  1.654e-14
          50 |      0.146 |      0.001 |   193.87 |     7.08 |  2.776e-17 |  1.747e-13
          75 |      0.142 |      0.001 |   159.63 |     9.32 |  2.776e-17 |  2.098e-14
         100 |      0.142 |      0.001 |   141.61 |     5.12 |  2.776e-17 |  4.458e-13
         200 |      0.149 |      0.002 |    88.53 |     4.44 |  2.776e-17 |  3.680e-14
         300 |      0.158 |      0.002 |    67.42 |     2.95 |  1.388e-17 |  1.825e-13
         400 |      0.165 |      0.003 |    54.19 |     2.15 |  1.388e-17 |  1.660e-13
         500 |      0.169 |      0.004 |    44.06 |     2.35 |  6.939e-18 |  2.631e-12
        1000 |      0.208 |      0.009 |    22.67 |     0.71 |  4.163e-17 |  2.791e-12
        2000 |      0.262 |      0.022 |    11.66 |     0.92 |  6.939e-18 |  1.176e-12
        3000 |      0.330 |      0.033 |    10.17 |     0.54 |  1.388e-17 |  1.813e-12
        4000 |      0.393 |      0.045 |     8.69 |     0.48 |  2.776e-17 |  1.893e-11
        5000 |      0.474 |      0.056 |     8.45 |     0.46 |  3.469e-18 |  9.546e-11
       10000 |      0.832 |      0.118 |     7.07 |     0.42 |  3.469e-18 |  3.389e-12
       20000 |      1.601 |      0.251 |     6.40 |     0.22 |  3.469e-18 |  1.078e-11
       30000 |      2.364 |      0.396 |     5.95 |     0.20 |  1.735e-18 |  8.548e-12
       40000 |      3.191 |      0.540 |     5.93 |     0.17 |  1.735e-18 |  3.685e-11
       50000 |      4.088 |      0.708 |     5.77 |     0.10 |  1.735e-18 |  7.294e-11
      100000 |      8.451 |      1.534 |     5.52 |     0.15 |  3.782e-16 |  2.284e-11
      200000 |     18.733 |      3.535 |     5.39 |     0.44 |  4.337e-19 |  1.383e-10
      300000 |     31.429 |      5.246 |     5.97 |     0.21 |  4.337e-19 |  4.171e-10
      400000 |     54.046 |      7.662 |     7.08 |     0.24 |  1.878e-16 |  2.765e-10
      500000 |     69.204 |     10.069 |     6.85 |     0.22 |  1.921e-16 |  1.868e-10
     1000000 |    136.977 |     24.243 |     5.70 |     0.35 |  3.600e-16 |  9.129e-10
 
    `spearmanr` (wrapper) with unsorted random tie/no_tie input:
           N |  SciPy(ms) |  Hyper(ms) |   Speed× |      IQR |  Δrho(max) |    Δp(max)
    -----------------------------------------------------------------------------------------
          25 |      0.151 |      0.002 |    97.78 |     4.82 |  5.551e-17 |  2.220e-14
          50 |      0.149 |      0.002 |    89.56 |     4.21 |  1.110e-16 |  1.747e-13
          75 |      0.154 |      0.002 |    76.99 |     4.92 |  8.327e-17 |  3.286e-14
         100 |      0.155 |      0.002 |    68.13 |     3.91 |  5.551e-17 |  7.527e-14
         200 |      0.166 |      0.004 |    41.36 |     2.61 |  2.776e-17 |  1.030e-13
         300 |      0.182 |      0.005 |    32.65 |     2.97 |  2.776e-17 |  1.096e-13
         400 |      0.192 |      0.008 |    25.49 |     2.50 |  1.388e-17 |  1.327e-13
         500 |      0.196 |      0.010 |    19.91 |     1.97 |  1.388e-17 |  2.661e-13
        1000 |      0.240 |      0.035 |     6.76 |     1.57 |  6.939e-18 |  7.451e-13
        2000 |      0.308 |      0.048 |     6.44 |     0.49 |  3.469e-18 |  6.317e-13
        3000 |      0.455 |      0.071 |     6.20 |     0.44 |  6.939e-18 |  2.102e-12
        4000 |      0.548 |      0.095 |     5.68 |     0.35 |  3.469e-18 |  3.201e-12
        5000 |      0.693 |      0.121 |     5.72 |     0.23 |  6.939e-18 |  4.078e-12
       10000 |      0.920 |      0.136 |     6.64 |     1.93 |  3.469e-18 |  3.978e-12
       20000 |      1.692 |      0.262 |     6.28 |     2.69 |  1.735e-18 |  1.869e-10
       30000 |      3.823 |      0.845 |     4.72 |     2.63 |  1.735e-18 |  7.951e-12
       40000 |      5.224 |      1.131 |     4.67 |     1.84 |  1.735e-18 |  2.716e-11
       50000 |      3.948 |      0.654 |     5.90 |     2.38 |  1.735e-18 |  2.346e-11
      100000 |     14.395 |      3.269 |     4.50 |     2.06 |  3.773e-16 |  9.780e-11
      200000 |     34.945 |      7.358 |     4.76 |     1.24 |  4.337e-19 |  1.533e-10
      300000 |     30.328 |      5.504 |     5.37 |     1.30 |  3.339e-16 |  1.809e-09
      400000 |     40.127 |      8.143 |     4.81 |     0.71 |  2.125e-14 |  2.862e-10
      500000 |     97.369 |     22.596 |     4.26 |     0.30 |  3.196e-14 |  2.667e-10
     1000000 |    119.343 |     43.888 |     3.33 |     0.37 |  1.439e-13 |  1.148e-09

    'spearmanr' (wrapper) with unsorted samples with ties:
           N |  SciPy(ms) |  Hyper(ms) |   Speed× |      IQR |  Δrho(max) |    Δp(max)
    -----------------------------------------------------------------------------------------
          25 |      0.148 |      0.001 |   107.11 |     7.20 |  1.110e-16 |  2.220e-14
          50 |      0.147 |      0.002 |    88.28 |     2.25 |  5.551e-17 |  1.343e-14
          75 |      0.152 |      0.002 |    76.04 |     3.17 |  2.776e-17 |  3.542e-14
         100 |      0.152 |      0.002 |    67.72 |     2.60 |  2.776e-17 |  4.763e-14
         200 |      0.164 |      0.004 |    43.09 |     1.12 |  2.776e-17 |  1.949e-12
         300 |      0.177 |      0.005 |    32.70 |     0.99 |  1.388e-17 |  1.002e-13
         400 |      0.186 |      0.007 |    26.11 |     0.80 |  1.388e-17 |  1.723e-13
         500 |      0.208 |      0.010 |    20.82 |     0.82 |  1.388e-17 |  1.629e-12
        1000 |      0.245 |      0.033 |     7.38 |     0.65 |  6.939e-18 |  1.748e-12
        2000 |      0.309 |      0.048 |     6.37 |     0.37 |  6.939e-18 |  2.446e-12
        3000 |      0.382 |      0.066 |     5.90 |     0.26 |  1.388e-17 |  1.428e-12
        4000 |      0.469 |      0.086 |     5.47 |     0.22 |  6.939e-18 |  4.579e-12
        5000 |      0.560 |      0.101 |     5.48 |     0.28 |  5.204e-18 |  3.434e-12
       10000 |      0.920 |      0.128 |     7.19 |     0.34 |  3.469e-18 |  3.857e-12
       20000 |      1.682 |      0.228 |     7.32 |     0.53 |  3.469e-18 |  1.080e-11
       30000 |      2.374 |      0.342 |     6.99 |     0.60 |  1.735e-18 |  3.049e-11
       40000 |      3.269 |      0.464 |     7.03 |     0.63 |  1.735e-18 |  3.919e-11
       50000 |      4.020 |      0.592 |     6.81 |     0.87 |  8.674e-19 |  6.041e-11
      100000 |      8.095 |      1.239 |     6.54 |     0.67 |  3.799e-16 |  2.238e-11
      200000 |     17.145 |      2.792 |     6.16 |     0.39 |  4.337e-19 |  1.529e-10
      300000 |     27.436 |      4.879 |     5.64 |     0.54 |  3.344e-16 |  1.299e-09
      400000 |     37.192 |      7.759 |     4.86 |     0.30 |  3.284e-16 |  2.835e-10
      500000 |     48.787 |     11.397 |     4.24 |     0.19 |  3.378e-16 |  1.691e-10
     1000000 |    116.482 |     36.382 |     3.20 |     0.74 |  2.116e-13 |  1.072e-09

    'spearmanr' (wrapper) with unsorted samples without any ties:
           N |  SciPy(ms) |  Hyper(ms) |   Speed× |      IQR |  Δrho(max) |    Δp(max)
    -----------------------------------------------------------------------------------------
          25 |      0.147 |      0.001 |   106.28 |     6.19 |  2.776e-17 |  1.654e-14
          50 |      0.147 |      0.002 |    86.76 |     3.24 |  2.776e-17 |  1.747e-13
          75 |      0.155 |      0.002 |    75.08 |     6.11 |  2.776e-17 |  2.098e-14
         100 |      0.152 |      0.002 |    62.76 |     1.49 |  2.776e-17 |  4.458e-13
         200 |      0.163 |      0.004 |    39.82 |     1.75 |  2.776e-17 |  3.680e-14
         300 |      0.181 |      0.006 |    29.93 |     1.27 |  1.388e-17 |  1.825e-13
         400 |      0.198 |      0.008 |    23.87 |     0.71 |  1.388e-17 |  1.660e-13
         500 |      0.202 |      0.011 |    18.67 |     0.91 |  6.939e-18 |  2.631e-12
        1000 |      0.259 |      0.050 |     5.24 |     0.49 |  6.939e-18 |  2.791e-12
        2000 |      0.355 |      0.051 |     7.00 |     0.24 |  6.939e-18 |  1.176e-12
        3000 |      0.479 |      0.076 |     6.35 |     0.33 |  1.388e-17 |  1.813e-12
        4000 |      0.579 |      0.101 |     5.76 |     0.20 |  3.469e-18 |  1.893e-11
        5000 |      0.726 |      0.127 |     5.71 |     0.27 |  3.469e-18 |  9.546e-11
       10000 |      1.341 |      0.258 |     5.16 |     0.10 |  3.469e-18 |  3.389e-12
       20000 |      2.702 |      0.564 |     4.79 |     0.11 |  3.469e-18 |  1.078e-11
       30000 |      4.021 |      0.903 |     4.46 |     0.11 |  1.735e-18 |  8.548e-12
       40000 |      5.543 |      1.212 |     4.58 |     0.07 |  1.735e-18 |  3.685e-11
       50000 |      7.050 |      1.570 |     4.49 |     0.11 |  1.735e-18 |  7.294e-11
      100000 |     14.794 |      3.373 |     4.40 |     0.14 |  3.782e-16 |  2.284e-11
      200000 |     32.625 |      7.486 |     4.38 |     0.11 |  4.337e-19 |  1.383e-10
      300000 |     53.635 |     11.995 |     4.43 |     0.15 |  4.337e-19 |  4.171e-10
      400000 |     75.125 |     17.809 |     4.26 |     0.06 |  3.290e-14 |  2.765e-10
      500000 |    104.919 |     24.325 |     4.27 |     0.25 |  3.027e-14 |  2.153e-10
     1000000 |    221.612 |     68.484 |     3.23 |     0.38 |  1.322e-14 |  9.129e-10
     
    References
    ----------
    .. [1] Spearman, C. The Proof and Measurement of Association between Two Things, 
           The American Journal of Psychology. 15 (1): 72–101.
    .. [2] Virtanen, P., et al., SciPy 1.0: Fundamental Algorithms for 
           Scientific Computing in Python. Nature Methods, 17(3), 261-272         
    
@author: Jon Paul Lundquist
"""

from numba import njit
import numpy as np
import math
from ._utils import _incbet, _argsort

from collections import namedtuple
SignificanceResult = namedtuple("SignificanceResult", ["statistic", "pvalue"])

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _rankdata_avg_ties(y, n):
    #idx = np.argsort(y, kind='quicksort')                 # stable not required here
    idx = _argsort(y, n)
    ry = np.empty(n, np.float64)
    Syy = 0.0

    i = 0
    while i < n:
        j = i + 1
        yi = y[idx[i]]
        while j < n and y[idx[j]] == yi:
            j += 1
        r = 0.5 * (i + 1 + j)              # average rank for the tie block
        # write block and accumulate Sxx
        for k in range(i, j):
            ry[idx[k]] = r
        m = j - i
        Syy += m * r * r                    # faster than summing r*r in the loop
        i = j
    return ry, Syy

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _rankdata_avg_ties_sorted(x_sorted, n):
    """
    x_sorted ascending. Return:
      rx : average ranks (float64) in the SAME order as x_sorted
      Sxx: sum(rx**2)
    """
    rx = np.empty(n, np.float64)
    Sxx = 0.0
    i = 0
    while i < n:
        j = i + 1
        xi = x_sorted[i]
        while j < n and x_sorted[j] == xi:
            j += 1
        # average rank for positions i..j-1 (1-based ranks are i+1..j)
        r = 0.5 * (i + 1 + j)
        m = j - i
        for k in range(i, j):
            rx[k] = r
        Sxx += m * r * r
        i = j
    return rx, Sxx

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _rankdata_noties(y, n):
    #idx = np.argsort(y, kind='quicksort')
    idx = _argsort(y, n)
    ry = np.empty(n, np.float64)
    for k in range(n):
        ry[idx[k]] = k + 1.0
    Syy = (n * (n + 1.0) * (2.0 * n + 1.0)) / 6.0
    return ry, Syy

@njit(nogil=True, fastmath=True)
def spearmanr_ties(x_sorted, y_ordered, n, pvals=True):
    """
    Spearman's rho assuming:
      - x_sorted is ascending (ties allowed),
      - y_ordered is y permuted by the same ordering as x.
    Returns (rho, pvalue).
    """
    # Ranks 'average' handling
    ry, Syy = _rankdata_avg_ties(y_ordered, n)
    rx, Sxx = _rankdata_avg_ties_sorted(x_sorted, n)

    # sums of ranks (same with ties)
    Sy = 0.5 * n * (n + 1.0)
    Sxy = 0.0
    for k in range(n):
        Sxy += rx[k] * ry[k]

    invn = 1.0 / n
    num  = Sxy - Sy * Sy * invn
    denx = Sxx - Sy * Sy * invn
    deny = Syy - Sy * Sy * invn
    if denx <= 0.0 or deny <= 0.0:
        return math.nan, math.nan

    rho = num / math.sqrt(denx * deny)
    if rho >  1.0: rho =  1.0
    if rho < -1.0: rho = -1.0

    if pvals:
        if math.fabs(rho) == 1.0:
            # exact two-sided permutation p-value for perfect monotone match
            return rho, math.exp(0.6931471805599453 - math.lgamma(n+1)) #2.0 / math.gamma(n + 1.0)
    
        # Two-sided p via Student-t -> regularized incomplete beta
        df  = n - 2
        t2  = (rho * rho) * df / (1.0 - rho * rho)
        x_arg = df / (df + t2)
        pvalue = _incbet(df * 0.5, 0.5, x_arg)
    
    else:
        pvalue = math.nan

    return rho, pvalue

@njit(nogil=True, fastmath=True)
def _spearmanr_ties_unsorted(x, y, n, pvals=True):
    """
    Spearman's rho assuming:
      - x_sorted is ascending (ties allowed),
      - y_ordered is y permuted by the same ordering as x.
    Returns (rho, pvalue).
    """
    #idx = np.argsort(x, kind='quicksort')
    idx = _argsort(x, n)
    x_sorted = np.empty(n, np.float64)
    for i in range(n):
        x_sorted[i] = x[idx[i]]
        
    y_ordered = np.empty(n, np.float64)
    for i in range(n):
        y_ordered[i] = y[idx[i]]
        
    rho, pvalue = spearmanr_ties(x_sorted, y_ordered, n, pvals=True)
        
    return rho, pvalue

@njit(nogil=True, fastmath=True)
def spearmanr_noties(x_sorted, y_ordered, n, pvals=True):
    """
    Spearman's rho assuming:
      - x_sorted is ascending (no ties allowed),
      - y_ordered is y permuted by the same ordering as x_sorted.
    Returns (rho, pvalue).
    """

    # argsort of y_ordered
    #idx = np.argsort(y_ordered, kind='quicksort')  # O(n log n)
    idx = _argsort(y_ordered, n)
    # inverse permutation: invperm[pos_in_y_ordered] = rank_index (0..n-1)
    invperm = np.empty(n, np.int64)
    for r in range(n):
        invperm[idx[r]] = r

    # sums
    Sy  = 0.5 * n * (n + 1.0)
    Sxx = (n * (n + 1.0) * (2.0 * n + 1.0)) / 6.0  # closed form
    Syy = Sxx                                       # no ties in y → same

    # Sxy = Σ (k+1) * (invperm[k]+1), k iterates in x_sorted order
    Sxy = np.int64(0)
    for k in range(n):
        Sxy += k* invperm[k]
    
    Sxy = float(Sxy) + float(n) * float(n)
    
    invn = 1.0 / n
    num  = Sxy - Sy * Sy * invn
    denx = Sxx - Sy * Sy * invn
    deny = Syy - Sy * Sy * invn
    if denx <= 0.0 or deny <= 0.0:
        return math.nan, math.nan

    rho = num / math.sqrt(denx * deny)
    if rho >  1.0: rho =  1.0
    if rho < -1.0: rho = -1.0

    if pvals:
        if math.fabs(rho) == 1.0:
            # exact two-sided permutation p-value for perfect monotone match
            return rho, math.exp(0.6931471805599453 - math.lgamma(n+1)) #2.0 / math.gamma(n + 1.0)
    
        # Two-sided p via Student-t -> regularized incomplete beta (SciPy-compatible)
        df  = n - 2
        t2  = (rho * rho) * df / (1.0 - rho * rho)
        x_arg = df / (df + t2)
        pvalue = _incbet(df * 0.5, 0.5, x_arg)
        
    else:
        pvalue = math.nan
        
    return rho, pvalue

@njit(nogil=True, fastmath=True)
def _spearmanr_noties_unsorted(x, y, n, pvals=True):
    """
    Spearman's rho assuming:
      - x_sorted is ascending (no ties allowed),
      - y_ordered is y permuted by the same ordering as x_sorted.
    Returns (rho, pvalue).
    """
    #idx = np.argsort(x, kind='quicksort')
    idx = _argsort(x, n)
    y_ordered = np.empty(n, np.float64)
    for i in range(n):
        y_ordered[i] = y[idx[i]]
        
    rho, pvalue = spearmanr_noties(x, y_ordered, n, pvals=pvals)
        
    return rho, pvalue

def spearmanr(x, y, pvals=True, ties='auto', sorted_x=False):
    #Routing to spearmanr_ties without ties checking is faster
    #for randomly generated tie/no_tie samples. See spearmanr_bench.py
    
    #commented checks.
    # x = np.asarray(x, dtype=np.float64)
    # y = np.asarray(y, dtype=np.float64)
    # if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size<3:
    #     raise ValueError("x and y must be 1-D arrays of the same length greater than two")
    
    n = x.size
    
    if not sorted_x:
        if (ties=='auto') or (ties is True):
            #Just routing to spearmanr_ties with no ties checking is faster
            #and results in correct answers as well...
            rho, pvalue = _spearmanr_ties_unsorted(x, y, n, pvals=pvals)
            
        else:
            rho, pvalue = _spearmanr_noties_unsorted(x, y, n, pvals=pvals)
            
    else:
        if (ties=='auto') or (ties is True):
            #Just routing to spearmanr_ties with no ties checking is faster
            #and results in correct answers as well...
            rho, pvalue = spearmanr_ties(x, y, n, pvals=pvals)
    
        else:
            rho, pvalue = spearmanr_noties(x, y, n, pvals=pvals)

    #Mirroring scipy output for compatibility. Slows things down slightly.
    res = SignificanceResult(statistic=float(rho), pvalue=float(pvalue))
    
    return res