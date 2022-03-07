#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 13:34:17 2022

Grid sampling method

@author: igingell
"""
import numpy as np
from matplotlib import pyplot as plt

from matplotlib.colors import LogNorm
from matplotlib.tri import Triangulation, UniformTriRefiner, TriAnalyzer
import warnings

# Load data
lambdas_times = np.load(
    "/Users/jamesplank/Downloads/temp/lambdas_times.npy", allow_pickle=True
).item()
lambdas = np.load(
    "/Users/jamesplank/Downloads/temp/lambdas.npy", allow_pickle=True
).item()
TmaxA = np.load("/Users/jamesplank/Downloads/temp/tmaxs.npy", allow_pickle=True)

supX = lambdas_times[TmaxA[10]]  # < CHANGE THIS LINE TO CHANGE THE SET GRID

[X2, Y2] = np.meshgrid(supX, TmaxA)

lambda_sup = np.zeros(X2.shape)

for i in range(len(TmaxA)):
    lambda_sup[i, :] = np.interp(supX, lambdas_times[TmaxA[i]], lambdas[TmaxA[i]])

cmin = -2  # np.log10(np.min(lambda_sup[lambda_sup[:]>0]))
cmax = 2.1  # np.log10(np.max(lambda_sup[:]))

# clev = np.linspace(cmin,cmax,10)
# clev2 = np.linspace(cmin,cmax,21)

clev = np.arange(cmin, cmax, 0.5)
clev2 = np.arange(cmin, cmax, 1 / 8)


plt.clf()

fig, ax = plt.subplots(1, 1, num=1)
# plta = ax.pcolor(supX,np.log10(TmaxA),np.log10(lambda_sup),
#                 vmin=cmin,vmax=cmax)


#### Colourmesh
spacing = np.diff(np.log10(TmaxA))[0] / 2
spacing = np.logspace(
    np.log10(TmaxA[0]) - spacing, np.log10(TmaxA[-1]) + spacing, TmaxA.size + 1
)
Lspacing = [abs(t[1] - t[0]) for _, t in lambdas_times.items()]
Lspacing = [
    np.linspace(t[0] - Lspacing[i] / 2, t[-1] + Lspacing[i] / 2, t.size + 1)
    for i, (_, t) in enumerate(lambdas_times.items())
]

for i, T in enumerate(TmaxA):
    lambdas[T][lambdas[T] <= 0] = min(lambdas[T][lambdas[T] > 0])
    # Stack colormesh rows
    plta = ax.pcolormesh(
        Lspacing[i],
        np.log10([spacing[i], (spacing[i] + spacing[i + 1]) / 2, spacing[i + 1]]),
        np.column_stack([np.log10(lambdas[T]), np.log10(lambdas[T])]).T,
        vmin=cmin,
        vmax=cmax,
    )

# # Contours thick
# pltb = ax.contour(
#     supX,
#     np.log10(TmaxA),
#     np.log10(lambda_sup),
#     linestyles="solid",
#     colors=("k",),
#     linewidths=(1,),
#     levels=clev,
#     vmin=cmin,
#     vmax=cmax,
# )
# # Contours thin
# pltc = ax.contour(
#     supX,
#     np.log10(TmaxA),
#     np.log10(lambda_sup),
#     linestyles="dashed",
#     colors=("k",),
#     linewidths=(1,),
#     alpha=0.2,
#     levels=clev2,
#     vmin=cmin,
#     vmax=cmax,
# )

tri = Triangulation(X2.flatten(), np.log10(Y2).flatten())
UTR = UniformTriRefiner(tri)
tri_refi, lambda_refi = UTR.refine_field(np.log10(lambda_sup).flatten(), subdiv=4)

ax.tricontour(
    tri_refi,
    lambda_refi,
    colors="k",
    levels=clev,
    linewidths=1,
    linestyles="solid",
    vmin=cmin,
    vmax=cmax,
)
ax.tricontour(
    tri_refi,
    lambda_refi,
    colors="k",
    levels=clev2,
    linewidths=1,
    linestyles="dashed",
    alpha=0.2,
    vmin=cmin,
    vmax=cmax,
)

ax.set_xlabel("t")
ax.set_ylabel(r"$\log_{10} T_{\rm max}$")
cb = fig.colorbar(plta, ax=ax)
# cb.add_lines(pltb)
# cb.add_lines(pltc, erase=False)
cb.set_label(r"$\log_{10} \lambda_c$")
plt.tight_layout()
plt.show()
