#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jon Paul Lundquist
"""
Created on Sun Oct  5 18:03:40 2025

@author: Jon Paul Lundquist
"""

from numba import njit, objmode
import math
import numpy as np

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _betacf(a, b, x, maxiter=1000, eps=3e-16):
    # Continued fraction for incomplete beta function
    FPMIN = 1e-30
    m2 = 0
    aa = 0.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d

    for m in range(1, maxiter + 1):
        m2 = 2 * m
        # even term
        aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        
        c = 1.0 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        
        d = 1.0 / d
        h *= d * c
        # odd term
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
        d = 1.0 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        
        c = 1.0 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break

    return h

@njit(cache=True, nogil=True, fastmath=True, inline='always')
def _incbet(a, b, x):
    # Regularized incomplete beta I_x(a,b)
    if x <= 0.0: return 0.0
    
    if x >= 1.0: return 1.0
    
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    lnum  = a*math.log(x) + b*math.log1p(-x) - lbeta

    if x < (a+1.0)/(a+b+2.0):
        # forward: I_x(a,b)
        return math.exp(lnum) * _betacf(a, b, x) / a
    else:
        # backward: 1 - I_{1-x}(b,a)
        return 1.0 - (math.exp(lnum) * _betacf(b, a, 1.0 - x) / b)
    
@njit(cache=True, nogil=True, fastmath=True)
def _has_dups_quicksort(a, n):
    """
    Early-exit duplicate check using 3-way quicksort partitioning.
    """

    # copy input so original isn't modified
    arr = np.empty_like(a)
    for i in range(n):
        arr[i] = a[i]

    # --- preallocate stack ---
    # depth bound ~ 2*log2(n)+8 is very safe
    max_depth = int(2.0 * math.log2(n + 1.0)) + 8
    lo_stack = np.empty(max_depth, np.int64)
    hi_stack = np.empty(max_depth, np.int64)
    top = 0
    lo_stack[0] = 0
    hi_stack[0] = n - 1

    while top >= 0:
        lo = lo_stack[top]
        hi = hi_stack[top]
        top -= 1
        if lo >= hi:
            continue

        # median-of-three pivot
        mid = (lo + hi) >> 1
        p0 = arr[lo]
        p1 = arr[mid]
        p2 = arr[hi]
        if p1 < p0:
            p0, p1 = p1, p0
        if p2 < p1:
            p1, p2 = p2, p1
            if p1 < p0:
                p0, p1 = p1, p0
        pivot = p1

        # 3-way partition
        lt = lo
        i  = lo
        gt = hi
        eq_seen = 0

        while i <= gt:
            v = arr[i]
            if v < pivot:
                arr[lt], arr[i] = arr[i], arr[lt]
                lt += 1
                i  += 1
            elif v > pivot:
                arr[i], arr[gt] = arr[gt], arr[i]
                gt -= 1
            else:
                eq_seen += 1
                if eq_seen >= 2:
                    return True
                i += 1

        # push left and right subarrays onto the stack
        if lo < lt - 1:
            top += 1
            lo_stack[top] = lo
            hi_stack[top] = lt - 1
        if gt + 1 < hi:
            top += 1
            lo_stack[top] = gt + 1
            hi_stack[top] = hi

    return False

@njit(cache=True, nogil=True, fastmath=True)
def _has_dups_sorted(a, n):
    for i in range(1, n):
        if a[i] == a[i - 1]:
            return True
    return False

@njit(cache=True, nogil=True, fastmath=True)
def _has_dups_argsort(a, n, early_exit=False):
    """
    Return:
        idx : int64 array of length n, such that a[idx] is in ascending order
        has_dup : bool, True if any duplicate values exist in a (by ==)
    Notes:
        - Input 'a' is NOT modified.
        - Unstable sort (like NumPy's quicksort), but produces a valid argsort.
        - Duplicate detection uses exact == comparisons (IEEE semantics).
    """
    # Copy 'a' (data) and build index array we will permute alongside
    arr = np.empty_like(a)
    for i in range(n):
        arr[i] = a[i]
    idx = np.arange(n, dtype=np.int64)

    # Preallocate explicit stacks for ranges [lo,hi]
    max_depth = int(2.0 * math.log2(n + 1.0)) + 8  # safe bound
    lo_stack = np.empty(max_depth, np.int64)
    hi_stack = np.empty(max_depth, np.int64)
    top = 0
    lo_stack[0] = 0
    hi_stack[0] = n - 1

    has_dup = False

    while top >= 0:
        lo = lo_stack[top]
        hi = hi_stack[top]
        top -= 1
        if lo >= hi:
            continue

        # --- median-of-three pivot selection ---
        mid = (lo + hi) >> 1
        p0 = arr[lo]
        p1 = arr[mid]
        p2 = arr[hi]
        if p1 < p0:
            p0, p1 = p1, p0
        if p2 < p1:
            p1, p2 = p2, p1
            if p1 < p0:
                p0, p1 = p1, p0
        pivot = p1

        # --- 3-way partition: [lo..lt-1] < p, [lt..gt] == p, [gt+1..hi] > p ---
        lt = lo
        i  = lo
        gt = hi

        eq_count = 0  # size of the == pivot bucket in this partition

        while i <= gt:
            v = arr[i]
            if v < pivot:
                # swap into < bucket
                arr[lt], arr[i] = arr[i], arr[lt]
                idx[lt], idx[i] = idx[i], idx[lt]
                lt += 1
                i  += 1
            elif v > pivot:
                # swap into > bucket
                arr[i], arr[gt] = arr[gt], arr[i]
                idx[i], idx[gt] = idx[gt], idx[i]
                gt -= 1
            else:
                # equal to pivot
                eq_count += 1
                i += 1

        if eq_count >= 2:
            has_dup = True  # record, but do NOT early-exit
            if early_exit:
                return np.empty(0, dtype=np.int64), has_dup

        # Left and right subranges to process
        left_lo,  left_hi  = lo,     lt - 1
        right_lo, right_hi = gt + 1, hi

        if left_lo < left_hi:
            top += 1
            lo_stack[top] = left_lo
            hi_stack[top] = left_hi
        if right_lo < right_hi:
            top += 1
            lo_stack[top] = right_lo
            hi_stack[top] = right_hi

    return idx, has_dup

# Numpy sort is faster for about n > 1000
@njit(cache=True)
def _argsort(a, n, threshold=1000):
    if n > threshold:
        # NumPy's C argsort via objmode (usually faster for large n)
        with objmode(inds='intp[:]'):
            inds = np.argsort(a, kind='quicksort')
    else:
        # Numba-lowered argsort (often faster for small n)
        ac = np.ascontiguousarray(a)   # cheap if already contiguous
        inds = np.argsort(ac)          # returns intp dtype
    return inds