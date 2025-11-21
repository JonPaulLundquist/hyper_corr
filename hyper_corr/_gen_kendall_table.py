#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 16:34:40 2025

@author: Jon Paul Lundquist
"""

import numpy as np
from numba import njit

# ---- exact p for fixed (n, c_eff) using your recurrence ----
@njit(nogil=True, fastmath=True)
def _kendall_exact_p(n, c_eff):
    """
    Exact two-sided p-value for given n and effective c:
    """
    new = np.zeros(c_eff + 1, np.float64)
    new[0] = 1.0
    if c_eff >= 1:
        new[1] = 1.0

    # recurrence
    for j in range(3, n + 1):
        # prefix-sum + normalize
        s = 0.0
        for i in range(c_eff + 1):
            s += new[i]
            new[i] = s / j
        # subtract shifted prefix for i >= j
        limit = (c_eff + 1) - j
        if limit > 0:
            # keep a copy of pre-shift segment [0:limit]
            temp = np.empty(limit, np.float64)
            for i in range(limit):
                temp[i] = new[i]
            for i in range(j, c_eff + 1):
                new[i] -= temp[i - j]

    # cumulative count up to c_eff
    p = 0.0
    for i in range(c_eff + 1):
        p += new[i]
    if p < 0.0: p = 0.0
    if p > 1.0: p = 1.0
    return p

@njit(nogil=True, fastmath=True)
def build_kendall_p(N_MAX=300, eps0=0.0, eps1=1.0):
    # rows indexed by n; each row length = floor(n(n-1)/4)+1 (use 1 at n<3)
    rows = []
    for n in range(0, N_MAX + 1):
        print(n)
        if n < 3:
            row = np.array([1.0], dtype=np.float64)
        else:
            cmax = (n * (n - 1)) // 4
            r = np.empty(cmax + 1, np.float64)
            for c in range(cmax + 1):
                r[c] = _kendall_exact_p(n, c)
            row = r
        rows.append(row)

    LO = np.zeros(N_MAX + 1, np.int64)
    HI = np.zeros(N_MAX + 1, np.int64)
    OFFSETS = np.zeros(N_MAX + 1, np.int64)
    SEGLEN = np.zeros(N_MAX + 1, np.int64)

    total = 0
    for n in range(0, N_MAX + 1):
        row = rows[n]
        l = 0
        while l < row.size and not (row[l] > eps0):
            l += 1
        if l == row.size:
            l = row.size - 1

        h = row.size - 1
        while h >= 0 and not (row[h] < eps1):
            h -= 1
        if h < l:
            h = l

        LO[n] = l
        HI[n] = h
        seg = h - l + 1
        SEGLEN[n] = seg
        OFFSETS[n] = total
        total += seg

    P_FLAT = np.empty(total, np.float64)
    for n in range(0, N_MAX + 1):
        l, h = int(LO[n]), int(HI[n])
        if SEGLEN[n] > 0:
            P_FLAT[OFFSETS[n]: OFFSETS[n] + SEGLEN[n]] = rows[n][l:h+1]

    return P_FLAT, OFFSETS, LO, HI

def save_kendall_p_table_trimmed(max_n: int, base_path: str):
    p_flat, offsets, lo, hi = build_kendall_p(max_n)
    np.save(base_path + "_flat.npy",    p_flat)
    np.save(base_path + "_offsets.npy", offsets)
    np.save(base_path + "_lo.npy",      lo)
    np.save(base_path + "_hi.npy",      hi)
    return p_flat.size