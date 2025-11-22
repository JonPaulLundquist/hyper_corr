#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 12:23:12 2025

@author: jplundquist
"""

import numpy as np
from hyper_corr import kendalltau_ties

def local_kendall_map(Z, dx, dy, win_phys_x, win_phys_y):
    """
    Compute Kendall's tau between local distance-to-center r and z within a
    rectangular physical window (±win_phys_x, ±win_phys_y) around each grid cell.

    Z:   2D array (ny, nx) of z-values on a regular grid with steps dx, dy
    dx, dy: grid spacing along x and y (float)
    win_phys_x, win_phys_y: half-window size in physical units (float)

    Returns:
      tau_map, p_map with same shape as Z
    """
    ny, nx = Z.shape
    # half-window radius in cells (clip at least 0)
    wx = int(win_phys_x // dx)
    wy = int(win_phys_y // dy)

    # upper bound on number of points in a window
    max_m = (2*wy + 1) * (2*wx + 1)

    tau_map = np.empty((ny, nx), np.float64)
    p_map   = np.empty((ny, nx), np.float64)

    # work buffers reused per-row 
    for iy in range(ny):
        # allocate per-iteration buffers
        z_buf = np.empty(max_m, np.float64)
        r_buf = np.empty(max_m, np.float64)

        for ix in range(nx):
            # window bounds (clamped to grid)
            j0 = max(0,  ix - wx); j1 = min(nx - 1, ix + wx)
            i0 = max(0,  iy - wy); i1 = min(ny - 1, iy + wy)

            # fill buffers
            m = 0
            for i in range(i0, i1 + 1):
                dyc = (i - iy) * dy
                for j in range(j0, j1 + 1):
                    dxc = (j - ix) * dx
                    z_buf[m] = Z[i, j]
                    # distance from (iy, ix) center
                    r_buf[m] = (dxc*dxc + dyc*dyc) ** 0.5
                    m += 1

            # stable sort z_buf, apply same perm to r_buf
            # (mergesort is stable in NumPy/Numba)
            ord_idx = np.argsort(z_buf[:m], kind='mergesort')  # 0..m-1
            z_sorted = np.empty(m, np.float64)
            r_ordered = np.empty(m, np.float64)
            for k in range(m):
                idx = ord_idx[k]
                z_sorted[k] = z_buf[idx]
                r_ordered[k] = r_buf[idx]

            # Kendall tau (ties version; x must be sorted → z_sorted)
            res = kendalltau_ties(z_sorted, r_ordered, m)
            tau_map[iy, ix] = res.statistic
            p_map[iy, ix]   = res.pvalue

    return tau_map, p_map

nx = ny = 1000
xv = np.linspace(-1.0, 1.0, nx)
yv = np.linspace(-1.0, 1.0, ny)
dx = xv[1] - xv[0]
dy = yv[1] - yv[0]

x, y = np.meshgrid(xv, yv, indexing='xy')
R0   = np.hypot(x, y)
rng  = np.random.default_rng(0)
z    = np.sin(7.0 * R0) + 0.15 * rng.normal(size=R0.shape)

# choose physical half-window ~0.005
win_phys = 0.005
tau_map, p_map = local_kendall_map(z, dx, dy, win_phys, win_phys)
