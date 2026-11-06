# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:11:28 2026

@author: he820946
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from advectionHelpers import cosBell_0_05, l2error, cond2, sinWave
from schemesBasic import FTBS, CTCS, FTCS
from SemiLagrangian import SLC

# Ensure plots folder exists
if not os.path.exists("plots"):
  os.makedirs("plots")

def errorsByResolution(init,schemes,outFile,nx_values=None,nu=0.5, u=1,
    endTime=1.0,xLim=[0, 1],
):
    """Creates a log-log convergence plot."""
    """
    Author: Áine Sweeny <a.sweeny@pgr.reading.ac.uk>
    Creates a log-log convergence plot of l2 error versus gird spacing
    for various schemes.
    Parameters:
        init(callable): Function to set the initial conditions
        schemes(list of callable): List of functions to evaluate the schemes
        outFile(string): the name of the output file
        nx_values(list of int, optional): Grid resolutions to be tested
        nu(float, optional): courant number to be used for the convergence test
        u (float): the wind speed
        endTime(float): the final time
        xLim(tuple): the maximum and minimum spatial dimensions
        
      
      Returns:
          None
                          
    """
    if nx_values is None:
        nx_values = [25, 50, 100, 200, 400]

    cols = [
        "blue",
        "red",
        "green",
        "cyan",
        "magenta",
        "brown",
    ]

    plt.figure(figsize=(8, 6))

    for i, scheme in enumerate(schemes):

        errors = []
        dxs = []

        for nx in nx_values:

            dx = (xLim[1] - xLim[0]) / nx
            dt = nu * dx / u
            nt = int(np.round(endTime / dt))
            actual_endTime = nt * dt

            x = np.arange(xLim[0], xLim[1], dx)

            phi0 = init(x)

            if scheme.__name__ == "CTCS":
                phi_prev = init(
                    (x - u * dt) % (xLim[1] - xLim[0])
                )
                phiLevels = [phi_prev, phi0.copy()]
            else:
                phiLevels = [phi0.copy()]

            failed = False

            for n in range(nt):
                try:
                    scheme(phiLevels, nu)

                    if np.any(np.isnan(phiLevels[-1])):
                        failed = True
                        break

                except Exception:
                    failed = True
                    break

            if failed:
                errors.append(np.nan)
            else:

                phi_exact = init(
                    (x - u * actual_endTime)
                    % (xLim[1] - xLim[0])
                )

                errors.append(
                    l2error(phiLevels[-1], phi_exact)
                )

                dxs.append(dx)

        errors = np.array(errors)
        dxs = np.array(dxs)

        valid = ~np.isnan(errors)

        plt.loglog(
            dxs[valid],
            errors[valid],
            "o-",
            color=cols[i % len(cols)],
            label=scheme.__name__,
        )

        # Estimate convergence order
        if np.sum(valid) > 2:
            slope = np.polyfit(
                np.log(dxs[valid]),
                np.log(errors[valid]),
                1,
            )[0]

            print(
                f"{scheme.__name__}: "
                f"order ≈ {slope:.2f}"
            )

    plt.xlabel(r"$\Delta x$")
    plt.ylabel(r"$L_2$ Error")
    plt.title(f"Convergence Test (C = {nu})")
    plt.legend()
    plt.grid(True, which="both")

    plt.savefig("plots/" + outFile)
    plt.close()
    
errorsByResolution(
    sinWave,
    [FTBS, CTCS, SLC],
    "convergence_sin.pdf",
    nx_values=[25, 50, 100, 200, 400],
    nu=0.5,
)