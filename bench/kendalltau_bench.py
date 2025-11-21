#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 18:48:41 2025

@author: Jon Paul Lundquist
"""
import os
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

import gc, time
import numpy as np
import scipy.stats
import math
from hyper_corr import kendalltau, kendalltau_ties, kendalltau_noties

def measure_interleaved(scipy_fn, numba_fn, loops_s, loops_n, start_with="scipy"):
    """
    Time scipy_fn and numba_fn with alternating call order.
    Returns (t_s_seconds, t_n_seconds).
    """
    perf_ns = time.perf_counter_ns
    t_s = 0
    t_n = 0

    m = min(loops_s, loops_n)
    # Alternate for the first m iterations
    if start_with == "scipy":
        for _ in range(m):
            t0 = perf_ns(); _ = scipy_fn(); t_s += perf_ns() - t0
            t0 = perf_ns(); _ = numba_fn(); t_n += perf_ns() - t0
    else:
        for _ in range(m):
            t0 = perf_ns(); _ = numba_fn(); t_n += perf_ns() - t0
            t0 = perf_ns(); _ = scipy_fn(); t_s += perf_ns() - t0

    # Finish the remainder (if any) for the one with larger loop budget
    if loops_s > m:
        rem = loops_s - m
        for _ in range(rem):
            t0 = perf_ns(); _ = scipy_fn(); t_s += perf_ns() - t0
    elif loops_n > m:
        rem = loops_n - m
        for _ in range(rem):
            t0 = perf_ns(); _ = numba_fn(); t_n += perf_ns() - t0

    return (t_s * 1e-9), (t_n * 1e-9)

def choose_loops(fn, target_ms=500.0, max_loops=1_000_000, overshoot_tol=1.25):
    """
    Pick a loop count so total time ~ target_ms, with back-off if we overshoot.
    Returns loops (int).
    """
    perf = time.perf_counter

    # 1) Probe: 1 call (or 2 if the function is extremely fast)
    t0 = perf(); fn(); t1 = perf()
    dt = (t1 - t0) * 1e3
    if dt < 0.02:  # ultra-fast; average a couple calls
        t0 = perf(); 
        fn(); fn(); fn(); fn(); fn(); fn(); fn(); fn(); fn(); fn();
        t1 = perf()
        dt = (t1 - t0) * 1e3 / 10.0

    # Guard against timer resolution / zero
    dt = max(dt, 1e-6)

    # 2) Predict loops from probe
    loops = int(math.ceil(target_ms / dt))
    loops = max(1, min(loops, max_loops))

    # 3) Measure at predicted loops
    t0 = perf()
    for _ in range(loops):
        fn()
    t1 = perf()
    total_ms = (t1 - t0) * 1e3

    # 4) If we overshot by more than overshoot_tol, back off once
    if total_ms > overshoot_tol * target_ms and loops > 1:
        # Scale loops by the ratio target/actual
        new_loops = int(max(1, min(max_loops, math.floor(loops * (target_ms / total_ms)))))
        if new_loops != loops:
            loops = new_loops
            t0 = perf()
            for _ in range(loops):
                fn()
            t1 = perf()
            total_ms = (t1 - t0) * 1e3

    # 5) If we undershot badly (< target/overshoot_tol), bump once
    elif total_ms < (target_ms / overshoot_tol) and loops < max_loops:
        new_loops = int(min(max_loops, math.ceil(loops * (target_ms / max(total_ms, 1e-6)))))
        if new_loops != loops:
            loops = new_loops
            t0 = perf()
            for _ in range(loops):
                fn()
            t1 = perf()
            total_ms = (t1 - t0) * 1e3

    return loops

SIZES = (25, 50, 75, 100, 200, 300, 400, 500, 1_000, 2_000, 3_000, 4_000, 5_000,
         10_000, 20_000, 30_000, 40_000, 50_000, 100_000, 200_000, 300_000, 400_000, 
         500_000, 1_000_000)

TIES        = "unknown"   # False | True | "unknown"
BLOCKS      = 25
TARGET_MS   = 1000.0

rng = np.random.default_rng(123)

results = []
sizes_order = list(SIZES)

print(f"{'N':>8} | {'SciPy(ms)':>10} | {'Hyper(ms)':>10} | {'Speed×':>8} | {'IQR':>8} | {'Δτ(max)':>10} | {'Δp(max)':>10}")
print("-"*89)

for n in sizes_order:

    scipy_times, numba_times, ratios = [], [], []
    max_d_tau, max_d_p = 0.0, 0.0

    # Representative sample just for loop selection
    x0 = rng.normal(size=n); y0 = rng.normal(size=n)
    if TIES is True:
        x0 = np.round(x0, 1); y0 = np.round(y0, 1)
    
    if TIES is True or TIES is False:
        idx0 = np.argsort(x0, kind="mergesort")
        xs0, ys0 = x0[idx0], y0[idx0]
        nn0 = ys0.size
    else:
        xs0 = x0
        ys0 = y0

    if (n<=300) and (TIES == False):
        scipy_fn0 = (lambda xs=xs0, ys=ys0: scipy.stats.kendalltau(xs, ys, method='exact'))
    else:
        scipy_fn0 = (lambda xs=xs0, ys=ys0: scipy.stats.kendalltau(xs, ys))
        
    if TIES is True:
        numba_fn0 = (lambda xs=xs0, ys=ys0, nn=nn0: kendalltau_ties(xs, ys, nn))
    elif TIES is False:
        numba_fn0 = (lambda xs=xs0, ys=ys0, nn=nn0: kendalltau_noties(xs, ys, nn))
    else:  # "unknown" -> always use unified kendalltau
        numba_fn0 = (lambda xs=xs0, ys=ys0: kendalltau(xs, ys))

    # Choose loops on the EXACT per-block callables (more stable)
    loops_s = choose_loops(scipy_fn0, target_ms=TARGET_MS)
    loops_n = choose_loops(numba_fn0, target_ms=TARGET_MS)
    
    r = 1
    for k in range(BLOCKS):
        # --- fresh sample for this block (EXCLUDED from timing) ---
        x = rng.normal(size=n); y = rng.normal(size=n)
        if TIES is True:
            x = np.round(x, 1); y = np.round(y, 1)
            r = 0
            
        elif TIES == "unknown":
            # randomly create ties on ~50% of blocks
            r = rng.random()
            if r < 0.5:
                x = np.round(x, 1); y = np.round(y, 1)

        if TIES is True or TIES is False:
            idx = np.argsort(x, kind="mergesort")
            xs, ys = x[idx], y[idx]

        else:
            xs = x
            ys = y
            
        nn = ys.size
        # reference result for this block (EXCLUDED from timing)
        if (n<=300) and (r>=0.5):
            res_ref = scipy.stats.kendalltau(xs, ys, method='exact')
        else:
            res_ref = scipy.stats.kendalltau(xs, ys)
        tau_ref, p_ref = float(res_ref.statistic), float(res_ref.pvalue)

        # zero-arg callables for this block
        if (n<=300) and (r>=0.5):
            scipy_fn = (lambda xs=xs, ys=ys: scipy.stats.kendalltau(xs, ys, method='exact'))
        else:
            scipy_fn = (lambda xs=xs, ys=ys: scipy.stats.kendalltau(xs, ys))
            
        if TIES is True:
            numba_fn = (lambda xs=xs, ys=ys, nn=nn: kendalltau_ties(xs, ys, nn))
        elif TIES is False:
            numba_fn = (lambda xs=xs, ys=ys, nn=nn: kendalltau_noties(xs, ys, nn))
        else:  # "unknown"
            numba_fn = (lambda xs=xs, ys=ys, nn=nn: kendalltau(xs, ys))

        # --- base deltas once per block (EXCLUDED from timing) ---
        tau_chk, p_chk = numba_fn()
        dt = abs(tau_ref - tau_chk) 
        dp = abs(p_ref - p_chk)
        if dt > max_d_tau: max_d_tau = dt
        if dp > max_d_p:   max_d_p   = dp

        # Disable GC during timing
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled: gc.disable()

        # Warm per-block callables
        _ = scipy_fn()
        _ = numba_fn()

        # Interleave order per block to remove order bias
        start_with = "scipy" if (k % 2 == 0) else "numba"
        t_s, t_n = measure_interleaved(scipy_fn, numba_fn, loops_s, loops_n, start_with)

        per_call_s = t_s / loops_s
        per_call_n = t_n / loops_n
        scipy_times.append(per_call_s)
        numba_times.append(per_call_n)
        if per_call_n > 0:
            ratios.append(per_call_s / per_call_n)

        if gc_was_enabled: gc.enable()

    scipy_med = float(np.median(scipy_times))
    numba_med = float(np.median(numba_times))
    speed_med = float(np.median(ratios))
    speed_iqr = float(np.percentile(ratios, 75) - np.percentile(ratios, 25))

    print(f"{n:8d} | {scipy_med*1e3:10.3f} | {numba_med*1e3:10.3f} | {speed_med:8.2f} | "
          f"{speed_iqr:8.2f} | {max_d_tau:10.3e} | {max_d_p:10.3e}")