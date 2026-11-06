# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:31:20 2026

@author: he820946
"""
# code to check whats going on for CTCS and the sin initial condition

import os
import matplotlib.pyplot as plt
import numpy as np
from advectionHelpers import l2error, sinWave

# Ensure plots folder exists
if not os.path.exists("plots"):
    os.makedirs("plots")


# ---------------------------------------------------------------
# Graph 1: Amplification factor |g| and arg(g) vs wavenumber
# ---------------------------------------------------------------
def amplificationFactor(outFile, cValues=None, nk=400):
    """Plots |A| and arg(A) against k*dx for CTCS at several Courant numbers.

    For the leapfrog (CTCS) scheme applied to the linear advection equation
    with centred differences, the amplification factor satisfies

        A^2 - 2 i c sin(k dx) g - 1 = 0

    giving the two branches

        A = i c sin(k dx) +/- sqrt(1 - c^2 sin^2(k dx))

    |A| = 1 for c <= 1 (neutral stability); |A| > 1 for c > 1.
    """
    if cValues is None:
        cValues = [0.4, 0.8, 1.0, 1.1, 1.3]

    kdx = np.linspace(0, np.pi, nk)
    s = np.sin(kdx)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for c in cValues:
        disc = 1.0 - c**2 * s**2
        root = np.sqrt(disc.astype(complex))


        APlus = 1j * c * s + root
        AMinus = 1j * c * s - root

        # Physical branch is the one continuous with g -> 1 as k -> 0
        axes[0].plot(kdx, np.abs(APlus), label=f"c = {c}")
        axes[1].plot(kdx, np.angle(APlus), label=f"c = {c}")

    axes[0].axhline(1.0, color="k", linestyle="--", alpha=0.5)
    axes[0].set_xlabel(r"$k\,\Delta x$")
    axes[0].set_ylabel(r"$|A|$")
    axes[0].set_title(r"Amplification magnitude $|A|$")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel(r"$k\,\Delta x$")
    axes[1].set_ylabel(r"$\arg(A)$")
    axes[1].set_title(r"Phase $\arg(A)$ (physical branch)")
    axes[1].legend(fontsize=8)

    fig.suptitle("CTCS amplification factor vs wavenumber")
    fig.tight_layout()
    fig.savefig("plots/" + outFile)
    plt.close(fig)


amplificationFactor("ctcsAmplification.pdf")

