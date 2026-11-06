# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:04:40 2026

@author: he820946
"""

import numpy as np
import matplotlib.pyplot as plt
from advectionHelpers import sinWave, cosBell_0_05, cond2

nx, nsteps = 40, 40
x = np.arange(nx) / nx
m = np.arange(nx // 2 + 1)
kdx = 2 * np.pi * m / nx

cs = np.linspace(0.05, 1.0, 400)
ics = [("sin", sinWave, "red"), ("cos bell", cosBell_0_05, "green"),
       ("square", cond2, "blue")]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- panel 1: per-mode factor for a few modes ---
for mm, colour in zip([1, 5, 10, 20], ["tab:blue", "tab:orange", "tab:green", "tab:purple"]):
    dtheta = np.arcsin(cs * np.sin(kdx[mm])) - cs * kdx[mm]
    ax1.plot(cs, 4 * np.sin(nsteps * dtheta / 2)**2, color=colour,
             label=f"m = {mm}")
ax1.set_xlabel("Courant number c")
ax1.set_ylabel(r"$4\sin^2(n\Delta\theta/2)$")
ax1.set_title(f"Per-mode error factor, n = {nsteps}")
ax1.legend(fontsize=8)

# --- panel 2: weighted total per IC ---
for name, init, colour in ics:
    amp = np.abs(np.fft.rfft(init(x)) / nx)
    total = []
    for c in cs:
        dtheta = np.arcsin(c * np.sin(kdx)) - c * kdx
        total.append(np.sum(amp**2 * 4 * np.sin(nsteps * dtheta / 2)**2))
    ax2.plot(cs, total, color=colour, label=name)
ax2.set_xlabel("Courant number c")
ax2.set_ylabel("weighted error")
ax2.set_title("Total error per IC (phase only, no growth)")
ax2.legend()

fig.tight_layout()
fig.savefig("plots/phaseErrorStructure.pdf")
