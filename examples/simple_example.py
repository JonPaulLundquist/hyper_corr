#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 13:20:43 2025

    Rolling Kendall correlation on a large array with a small window.
    Speedup greater than x100 with hyper_corr.kendalltau_ties/noties over 
    scipy.stats.kendalltau
    
@author: Jon Paul Lundquist
"""

import numpy as np
from hyper_corr import kendalltau_noties, kendalltau_ties
import time
import scipy.stats as stats

N = 1_000_000
rng = np.random.default_rng(0)
x = rng.normal(size=N); y = rng.normal(size=N)

W = 25              # window size
M = N - W + 1

taus = np.empty(M, dtype=np.float64)
pvals = np.empty(M, dtype=np.float64)

t0 = time.perf_counter()

ind = np.argsort(y, kind="stable")
y_sorted = y[ind]; x_ordered = x[ind]   # x in the same order as y
    
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

t1 = time.perf_counter()
td1 = t1-t0

t2 = time.perf_counter()
for i in range(M):
    tau, p = stats.kendalltau(x[i:i+W], y[i:i+W])
    taus[i] = tau
    pvals[i] = p

t3 = time.perf_counter()
td2 = t3-t2

print(f"hyper_corr: {td1:.2f} s")
print(f"scipy.stats: {td2:.2f} s")
print(f"speedup: x{td2/td1:.2f} s")