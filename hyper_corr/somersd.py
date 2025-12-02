#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jon Paul Lundquist
# Derived and adapted from SciPy's scipy.stats.somersd (BSD-3-Clause);
# see licenses/SciPy_LICENSE.txt for the original license.
"""
Created on Sun Oct  5 18:16:57 2025

    Hyper-fast Somers' D correlation

    This module implements:
    - somersd_ties when ties are present,
    - somersd_noties when both `x` and `y` are strictly monotone.
    - somersd when ties are unknown.
    
    For maximum speed on repeated calls, use the specialized kernels and you must 
    supply pre-sorted inputs:
    
        ind = np.argsort(x)
        y_ordered = y[ind]
        x_sorted = x[ind]
    
        # tie-aware
        D, pvalue = somersd_ties(x_sorted, y_ordered, n)
    
        # no ties
        D, pvalue = somersd_noties(x_sorted, y_ordered, n)
    
    If you can’t guarantee the preconditions, use `somersd(x, y, ties="auto",
    sorted_x=False)`. It sorts as needed and dispatches to the right kernel, but 
    is much slower than calling the kernels directly.
    
    Speedup over scipy increases with number of unique samples. ~200x for N=25.
    Can be over a hundred of thousans times faster than scipy for unique N 
    samples >= 400 (~150,000x).
    
    When to use
    -----------
    - **Many small/medium repeated slices** of pre-sorted large arrays with known tie
      structure → `somersd_ties` / `somersd_noties` (fastest).
    - **One-off or unknown inputs** → `somersd` (convenience wrapper).

    Parameters
    ----------
    x, y : 1-D array_like 
        samples of equal length (n ≥ 3). Inputs are treated as numeric; NaNs/Infs are not supported.
    
    pvals : {True, False}
        - Flag for p-value calculation. Default: pvals=True. Returns nan if pvals=False.

    For the specialized kernels spearmanr_ties() and spearmanr_noties():
        n : Length of the arrays      
        - `somersd_ties(x_sorted, y_ordered, n)`: `x_sorted` ascending; `y_ordered`
          must be `y` permuted by the same `ind`.
        - `somersd_noties(x_sorted, y_ordered, n)`: no ties in either variable.
        
    for somersd()
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
    for somersd_ties() / somersd_noties() :
        D : Spearman’s rho statistic.
        pvalue : Two-sided p-value for the null hypothesis of no association, 
                 H0: ρ=0.
        T : Contingency table.
    
    for somersd() :
        res : SomersDResult
        An object with the attributes:    
            statistic : float
               Somers' D statistic.
            pvalue : float
               Two-sided p-value for the null hypothesis of no association, H0: D=0.
            table : 2D array
                The contingency table from rankings x and y.
    
    See Also
    --------
    scipy.stats.somersd
    
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
    
    Benchmarks (illustrative; environment-dependent): See somersd_bench.py
    ------------------------------------------------
    CPU: Ultra 9 275HX, Python 3.13.5, NumPy 2.1.3, Numba 0.61.2, SciPy 1.16.0

    No ties ('somersd_noties')
     Unique N |  SciPy(ms) |  Hyper(ms) |   Speed× |      IQR |    ΔD(max) |    Δp(max)|  ΔTable(max)|
------------------------------------------------------------------------------------------------------------
           25 |      0.278 |      0.001 |    197.3 |      6.8 |  0.000e+00 | 1.388e-16 |  0.000e+00
           50 |      4.665 |      0.003 |   1567.3 |     36.9 |  0.000e+00 | 1.110e-16 |  0.000e+00
           75 |     24.782 |      0.006 |   4467.7 |     72.9 |  0.000e+00 | 1.665e-16 |  0.000e+00
          100 |     80.205 |      0.010 |   8359.2 |    151.4 |  0.000e+00 | 2.220e-16 |  0.000e+00
          200 |   1320.816 |      0.038 |  35161.1 |    996.1 |  0.000e+00 | 1.110e-16 |  0.000e+00
          300 |   6847.720 |      0.083 |  82570.2 |   1019.1 |  0.000e+00 | 1.110e-16 |  0.000e+00
          400 |  21884.291 |      0.143 | 151831.2 |   2185.7 |  0.000e+00 | 1.110e-16 |  0.000e+00
          500 |  53557.084 |      0.244 | 218860.2 |   3870.1 |  0.000e+00 | 1.110e-16 |  0.000e+00   
         
    Ties (`somersd_ties`) with 75% unique values (25% with ties):
     Unique N |  SciPy(ms) |  Hyper(ms) |   Speed× |      IQR |    ΔD(max) |    Δp(max)|  ΔTable(max)|
------------------------------------------------------------------------------------------------------------
           25 |      0.284 |      0.002 |    167.5 |      3.8 |  0.000e+00 | 1.665e-16 |  0.000e+00
           50 |      4.688 |      0.003 |   1372.2 |     43.3 |  0.000e+00 | 2.220e-16 |  0.000e+00
           75 |     24.966 |      0.007 |   3673.9 |     71.1 |  0.000e+00 | 2.220e-16 |  0.000e+00
          100 |     80.959 |      0.011 |   7560.5 |    257.6 |  0.000e+00 | 1.110e-16 |  0.000e+00
          200 |   1333.263 |      0.041 |  32442.9 |    639.4 |  0.000e+00 | 1.110e-16 |  0.000e+00
          300 |   6901.065 |      0.092 |  74661.9 |   1508.7 |  0.000e+00 | 1.110e-16 |  0.000e+00
          400 |  22013.698 |      0.163 | 134604.9 |   3956.1 |  0.000e+00 | 1.665e-16 |  0.000e+00
          500 |  53028.333 |      0.265 | 200976.9 |   4304.6 |  0.000e+00 | 1.110e-16 |  0.000e+00

`somersd` (wrapper) with unsorted random tie/no_tie input:
       Unique N |  SciPy(ms) |  Hyper(ms) |   Speed× |      IQR |    ΔD(max) |    Δp(max)|  ΔTable(max)|
--------------------------------------------------------------------------------------------------------------
             25 |      0.260 |      0.003 |     95.5 |      6.1 |  0.000e+00 | 1.110e-16 |  0.000e+00
             50 |      4.336 |      0.004 |   1005.3 |     64.6 |  0.000e+00 | 1.665e-16 |  0.000e+00
             75 |     22.915 |      0.007 |   3171.1 |    251.3 |  0.000e+00 | 1.110e-16 |  0.000e+00
            100 |     74.132 |      0.011 |   6457.2 |    620.3 |  0.000e+00 | 1.110e-16 |  0.000e+00
            200 |   1223.125 |      0.039 |  31696.0 |   2461.2 |  0.000e+00 | 1.110e-16 |  0.000e+00
            300 |   6712.357 |      0.092 |  76238.2 |   7949.6 |  0.000e+00 | 1.665e-16 |  0.000e+00
            400 |  24322.256 |      0.194 | 128395.6 |  14818.3 |  0.000e+00 | 1.110e-16 |  0.000e+00
            500 |  61791.708 |      0.332 | 186165.6 |  19893.6 |  0.000e+00 | 1.665e-16 |  0.000e+00

@author: Jon Paul Lundquist
"""

from numba import njit
import math
import numpy as np
from collections import namedtuple

SomersDResult = namedtuple("SomersDResult", ["statistic", "pvalue", "table"])

from ._utils import _argsort

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _ranks_sorted(a, n):

    ranks = np.empty(n, np.int64)
    r = 0
    ranks[0] = r
    for t in range(1, n):
        if a[t] != a[t-1]:
            r += 1
        ranks[t] = r
    return ranks, r + 1                   # (ranks, n_unique)

@njit(cache=True, nogil=True, fastmath=True)
def _ranks(a, n):

    ranks = np.empty(n, np.int64)
    idx = _argsort(a, n)
    r = 0
    ranks[idx[0]] = r
    for t in range(1, n):
        i = idx[t]
        j = idx[t-1]
        if a[i] != a[j]:
            r += 1
        ranks[i] = r
    return ranks, r + 1                   # (ranks, n_unique)

@njit(cache=True, nogil=True, fastmath=True)
def _contingency_table(x_sorted, y_ordered, n):
    """
    Produce the same 2D contingency table as `scipy.stats.contingency.crosstab(x, y)[1]`.
    Assumes x and y are 1D float64 arrays of equal length.
    Returns a 2D int64 array of shape (len(ux), len(uy)).
    """

    rx, nx = _ranks_sorted(x_sorted, n)
    ry, ny = _ranks(y_ordered, n)
    table = np.zeros((nx, ny), np.int64)
    for i in range(n):
        table[rx[i], ry[i]] += 1
    return table, nx, ny

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _build_ps(T, nx, ny):
    PS = np.zeros((nx, ny), dtype=np.int64)
    for i in range(nx):
        row_sum = 0
        for j in range(ny):
            row_sum += T[i, j]
            PS[i, j] = row_sum + (PS[i-1, j] if i > 0 else 0)
    return PS

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _UL(PS, i, j):
    return PS[i-1, j-1] if (i > 0 and j > 0) else 0

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _UR(PS, i, j, c):
    return (PS[i-1, c-1] - PS[i-1, j]) if (i > 0 and j < c-1) else 0

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _LL(PS, i, j, r):
    return (PS[r-1, j-1] - PS[i, j-1]) if (i < r-1 and j > 0) else 0

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _LR(PS, i, j, nx, ny, TOT):
    # TOT = PS[r-1, c-1]
    a = PS[i, j]
    b = PS[i, ny-1]
    d = PS[nx-1, j]
    return TOT - b - d + a  # guards implicit when i==r-1 or j==c-1

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _PQA(T, nx, ny):
    """
    Numba‐friendly (O(mn)) discordant‐pair count:
       Q = sum_{i, j} T[i,j] * (T[i+1:, :j].sum() + T[:i, j+1:].sum())
    """

    PS  = _build_ps(T, nx, ny)
    TOT = PS[nx-1, ny-1]

    Q_total = np.int64(0)
    P_total = np.int64(0)
    A_total = np.int64(0)
    for i in range(nx):
        for j in range(ny):
            cnt = T[i, j]
            if cnt == 0:
                continue
            aij = _UL(PS, i, j) + _LR(PS, i, j, nx, ny, TOT)
            dij = _LL(PS, i, j, nx) + _UR(PS, i, j, ny)
            Q_total += cnt * dij
            P_total += cnt * aij
            diff = aij - dij
            A_total += cnt * (diff * diff)
    return Q_total, P_total, A_total

# ---------- inversion count (Fenwick BIT) ----------
@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _bit_add(bit, idx):
    n = bit.size - 1
    while idx <= n:
        bit[idx] += 1
        idx += idx & -idx

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _bit_sum(bit, idx):
    s = 0
    while idx > 0:
        s += bit[idx]
        idx -= idx & -idx
    return s

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _inversion_count(perm, n):
    # perm values must be 0..n-1; Fenwick is 1-based
    bit = np.zeros(n + 1, np.int64)
    inv = 0
    # process from right to left
    for i in range(n - 1, -1, -1):
        p1 = perm[i] + 1
        inv += _bit_sum(bit, p1 - 1)  # count of elements < perm[i] to the right
        _bit_add(bit, p1)
    return inv
    
@njit(cache=True, nogil=True, fastmath=True)
def somersd_ties(x_sorted, y_ordered, n, pvals=True):

    T, nx, ny = _contingency_table(x_sorted, y_ordered, n)   
    if nx < 2 or ny < 2:
        return 0.0, 1.0, T
    
    QA, PA, AA = _PQA(T, nx, ny)

    # --- Branch: no ties in X → each row sum is 1 → Sri2 = n; denom = n(n-1)
    if nx == n:
        Sri2 = float(n)         # skip row-sum loop
        denom = float(n) * float(n - 1)
    else:
        Sri2 = 0.0
        for i in range(nx):
            s = 0  # int64 accumulator for the row sum
            for j in range(ny):
                s += T[i, j]
            Sri2 += float(s) * float(s)
  
    n2 = float(n * n)
    denom = float(n2 - Sri2)
    if denom == 0:
        return 0.0, 1.0, T
    
    numerator = float(PA - QA)
    if numerator == 0:
        return 0.0, 1.0, T
    
    D = numerator/denom
    
    if pvals:
        S = float(AA) - numerator**2/float(n)
        
        # Degenerate variance → infinite Z (if numerator!=0)
        if S <= 0.0:
            return D, 0.0, T
    
        # Normal approximation (two-sided)
        Z = numerator / math.sqrt(4.0 * S)
        aZ = abs(Z)
    
        # erfc is stable enough up to ~26 before underflow matters
        if aZ < 26.0:
            pvalue = math.erfc(aZ * 0.7071067811865475)
        else:
            # log tail: p ≈ 2 * φ(aZ)/aZ * (1 - 1/aZ^2 + 3/aZ^4 - 15/aZ^6 + ...)
            aZ2 = aZ * aZ
            logp = -0.5 * aZ2 - 0.5 * 1.8378770664093453 - math.log(aZ)
            inv = 1.0 / aZ2
            corr = 1.0 - inv + 3.0 * inv * inv - 15.0 * inv * inv * inv
            logp += math.log(corr) + 0.6931471805599453
            pvalue = math.exp(logp)
    
        # clamp
        if pvalue < 0.0: pvalue = 0.0
        if pvalue > 1.0: pvalue = 1.0
        
    else:
        pvalue = math.nan

    return D, pvalue, T

@njit(cache=True, nogil=True, fastmath=True)
def _somersd_ties_unsorted(x, y, n, idx=None, pvals=True):
    if idx is None:
        idx = _argsort(x, n)
        
    x_sorted = np.empty(n, np.float64)
    for i in range(n):
        x_sorted[i] = x[idx[i]]
        
    y_ordered = np.empty(n, np.float64)
    for i in range(n):
        y_ordered[i] = y[idx[i]]
        
    D, pvalue, T = somersd_ties(x_sorted, y_ordered, n, pvals=True)

    return D, pvalue, T

@njit(cache=True, nogil=True, fastmath=True)
def somersd_noties(x_sorted, y_ordered, n, pvals=True):
        
    # ranks for x and y (0..n-1)
    ry = np.empty(n, np.int64)

    idx = _argsort(y_ordered, n)
    for r in range(n):
        ry[idx[r]] = r
    
    # discordant pairs = inversion count
    d = _inversion_count(ry, n)
        
    # total_pairs = n(n-1)/2 as the denominator (tau_a)
    total_pairs = (n * (n - 1)) // 2
    S = total_pairs - (2 * d)          # S = C - D
    D = S / float(total_pairs)      # Somers' D == tau_a when no ties

    # Build contingency table (n x n, one 1 per row/col)
    T = np.zeros((n, n), np.int64)
    for k in range(n):
        T[k, ry[k]] = 1
            
    # p-value: same variance as tau_a in the no-ties case
    if pvals:
        # Use the same variance route as the ties path to match SciPy
        QA, PA, AA = _PQA(T, n, n)
        numerator = float(PA - QA)          # equals S_num
        S = float(AA) - (numerator * numerator) / float(n)
        if S <= 0.0:
            pvalue = 0.0
        else:
            z = numerator / math.sqrt(4 * S)
            pvalue = math.erfc(abs(z) * 0.7071067811865475)

    else:
        pvalue = math.nan
            
    return D, pvalue, T

@njit(cache=True, nogil=True, fastmath=True)
def _somersd_noties_unsorted(x, y, n, idx=None, pvals=True):
    if idx is None:
        idx = _argsort(x, n)
        
    y_ordered = np.empty(n, np.float64)
    for i in range(n):
        y_ordered[i] = y[idx[i]]
        
    D, pvalue, T = somersd_noties(x, y_ordered, n, pvals=pvals)
            
    return D, pvalue, T

def somersd(x, y, pvals=True, ties='auto', sorted_x=False):
    #Routing to somersd_ties without ties checking is faster
    #for randomly generated tie/no_tie samples. See spearmanr_bench.py
    
    #commented checks.
    # x = np.asarray(x, dtype=np.float64)
    # y = np.asarray(y, dtype=np.float64)
    # if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size<3:
    #     raise ValueError("x and y must be 1-D arrays of the same length greater than two")
    
    n = x.size
    
    if not sorted_x:
        idx = _argsort(x, n)
        if (ties == 'auto') or (ties is True):
            #Just routing to spearmanr_ties with no ties checking is faster
            #and results in correct answers as well...
            D, pvalue, T = _somersd_ties_unsorted(x, y, n, idx=idx, pvals=pvals)

        else:
            D, pvalue, T = _somersd_noties_unsorted(x, y, n, idx=idx, pvals=pvals)

    else:
        if (ties == 'auto') or (ties is True):
            #Just routing to spearmanr_ties with no ties checking is faster
            #and results in correct answers as well...
            D, pvalue, T = somersd_ties(x, y, n, pvals=pvals)
    
        else:
            D, pvalue, T = somersd_noties(x, y, n, pvals=pvals)
            
    #Mirroring scipy output for compatibility. Slows things down slightly.
    res = SomersDResult(statistic=float(D), pvalue=float(pvalue), table=T)
    
    return res
