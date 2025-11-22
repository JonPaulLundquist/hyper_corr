#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 12:22:18 2025

@author: jplundquist
"""

import numpy as np
from hyper_corr import kendalltau_ties

rng = np.random.default_rng(0)

nx = ny = 1000                      # 1000 x 1000 grid -> 1e6 points
N = nx*ny
xv = np.linspace(-1.0, 1.0, nx)
yv = np.linspace(-1.0, 1.0, ny)
X, Y = np.meshgrid(xv, yv, indexing="xy")

# Flatten to 1D
x = X.ravel()
y = Y.ravel()

r_0 = np.hypot(x, y)
z = np.sin(7.0 * r_0) + 0.15 * rng.normal(size=r_0.size)

idx = np.argsort(z, kind="stable")
xs = x[idx]
ys = y[idx]
zs = z[idx]

tau = np.zeros(N)
p = np.zeros(N)
for i in range(N):
    ix = xs[i]
    iy = ys[i]
    gind = (abs(ix-xs)<0.005) & (abs(iy-ys)<0.005)
    r = np.hypot(xs[gind] - ix, ys[gind] - iy)
    tau[i], p[i] = kendalltau_ties(zs[gind], r, np.sum(gind))
