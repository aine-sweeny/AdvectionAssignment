# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:50:02 2026

@author: he820946
"""
# run L2 errors vs time to look at whats going on with the sin wave initial condition
# also check for different number of time steps if the p
import os
import matplotlib.pyplot as plt
import numpy as np
from advectionHelpers import cosBell_0_05, l2error, cond2, sinWave
from schemesBasic import FTBS, CTCS, FTCS
from SemiLagrangian import SLC

# Ensure plots folder exists
if not os.path.exists("plots"):
  os.makedirs("plots")
  
def amplitude_phase_error(phi_num, phi_exact):
    """
    Compute amplitude and phase errors for a periodic initial condition.
    
    Parameters:
        phi_num(ndarray): Numerical solution at the final time
        phi_exact(ndarray): Exact analytical solution at the same time

    Returns:
        amp_error    : relative amplitude error
        phase_error  : phase shift in grid points
    """
    n = len(phi_num)

    # FFT of exact and numerical solutions
    F_exact = np.fft.rfft(phi_exact)
    F_num = np.fft.rfft(phi_num)

    # Ignore zero mode and find dominant frequency
    k = np.argmax(np.abs(F_exact[1:])) + 1

    A_exact = np.abs(F_exact[k])
    A_num = np.abs(F_num[k])

    # Relative amplitude error
    amp_error = np.abs(A_num - A_exact) / A_exact

    # Phase difference
    phase_exact = np.angle(F_exact[k])
    phase_num = np.angle(F_num[k])

    phase_error = phase_num - phase_exact

    # Wrap to [-pi, pi]
    phase_error = (phase_error + np.pi) % (2*np.pi) - np.pi

    # Convert to grid-point shift
    phase_error = phase_error / (2*np.pi*k) * n

    return amp_error, abs(phase_error)


def errorsByCourant(init,schemes,outFile,nx=40,nu_values=None,u=1,endTime=1.34,
    xLim=[0, 1],yLim=None,phase_amp_analysis = False
):
  """
  Author: Áine Sweeny <a.sweeny@pgr.reading.ac.uk>
  Creates a graph of final L2 error versus Courant number for various schemes.
  Parameters:
      init(callable): Function to set the initial conditions
      outFile(string): the name of the output file
      nx(int): The number of points in space
      nu_values(linspace): evenly spaced values over the range of the Courant
                number
      u (float): the wind speed
      endTime(float): the final time
      xLim(tuple): the maximum and minimum spatial dimensions
      yLim (tuple): The minimum (yLim[0]) and maximum (yLim[1]) y values of
                    the graph. If None is specified then default limits
                    are used.
      phase_amp_analysis(callable): function to seperate l2 errors into amplitude
                    and phase errors
    
    Returns:
        None
                        
  """

  
  if nu_values is None:
    nu_values = np.linspace(0.1, 1.5, 30)

  dx = (xLim[1] - xLim[0]) / nx
  x = np.arange(xLim[0], xLim[1], dx)

  cols = [
      "blue",
      "red",
      "green",
      "cyan",
      "magenta",
      "brown",
      "purple",
      "pink",
      "orange",
      "gray",
  ]

  plt.figure(figsize=(8, 6))
  
  all_amp_errors = []
  all_phase_errors = []

  for i, scheme in enumerate(schemes):
    errors = []
    if phase_amp_analysis:
        amp_errors = []
        phase_errors = []

    for nu in nu_values:

        dt = nu * dx / u
        nt = int(np.round(endTime / dt))
        actual_endTime = nt * dt

        phi0 = init(x)
        phiLevels = [phi0.copy()]

        if scheme.__name__ == "CTCS":
            # intialise CTCS using analytic solution, not FTBS startup step
            phi_prev = init((x - u * dt) % (xLim[1] - xLim[0]))
            phiLevels = [phi_prev, phi0.copy()]

        failed = False

        for n in range(nt):

            try:
                scheme(phiLevels, nu)

                if np.any(np.isnan(phiLevels[-1])) or np.any(
                        np.abs(phiLevels[-1]) > 1e4
                        ):
                    failed = True
                    break

            except Exception:
                failed = True
                break

        if failed:
            errors.append(np.nan)

            if phase_amp_analysis:
                amp_errors.append(np.nan)
                phase_errors.append(np.nan)

        else:
            phiA = init(
                (x - u * actual_endTime) % (xLim[1] - xLim[0])
            )

            l2_val = l2error(phiLevels[-1], phiA)
            errors.append(l2_val)

            if phase_amp_analysis:
                amp_err, phase_err = amplitude_phase_error(
                    phiLevels[-1], phiA
                )

                amp_errors.append(amp_err)
                phase_errors.append(phase_err)

    # finished looping over all Courant numbers

    plt.semilogy(
        nu_values,
        errors,
        color=cols[i % len(cols)],
        label=scheme.__name__
    )

    if phase_amp_analysis:
        all_amp_errors.append(amp_errors)
        all_phase_errors.append(phase_errors)

  plt.axvline(
      1.0, color="k", linestyle="--", alpha=0.5, label="C = 1"
  )
  plt.xlim([nu_values[0], nu_values[-1]])
  if yLim is not None:
    plt.ylim(yLim)
  plt.legend()
  plt.xlabel("Courant Number (C)")
  plt.ylabel(r"Normalized $\ell_2$ Error at $T = 1.34$")
  plt.title("Error vs. Courant Number")
  plt.savefig("plots/" + outFile)
  plt.close()
  
  if phase_amp_analysis:

    # Amplitude error plot
    plt.figure(figsize=(8,6))

    for i, scheme in enumerate(schemes):
        plt.semilogy(
            nu_values,
            all_amp_errors[i],
            color=cols[i % len(cols)],
            label=scheme.__name__,
        )

    plt.axvline(
        1.0, color="k", linestyle="--", alpha=0.5
    )

    plt.xlabel("Courant Number")
    plt.ylabel("Relative Amplitude Error")
    plt.title("Amplitude Error vs Courant Number")
    plt.legend()

    plt.savefig(
        "plots/" + outFile.replace(".pdf", "_amp.pdf")
    )

    plt.close()


    # Phase error plot
    plt.figure(figsize=(8,6))

    for i, scheme in enumerate(schemes):
        plt.semilogy(
            nu_values,
            all_phase_errors[i],
            color=cols[i % len(cols)],
            label=scheme.__name__,
        )

    plt.axvline(
        1.0, color="k", linestyle="--", alpha=0.5
    )

    plt.xlabel("Courant Number")
    plt.ylabel("Phase Error (grid cells)")
    plt.title("Phase Error vs Courant Number")
    plt.legend()

    plt.savefig(
        "plots/" + outFile.replace(".pdf", "_phase.pdf")
    )

    plt.close()



schemes = [FTBS, CTCS, SLC]

# Range of Courant numbers from 0.1 to 1.5
nu_range = np.linspace(0.1, 1.4, 28)

# Generate plots usinginitial conditions
# only generate phase and amplitude plots for sin initial condition
errorsByCourant(
    cosBell_0_05,
    schemes,
    "cosErrorsByCourant.pdf",
    nx=40,
    nu_values=nu_range,
    u=1,
    endTime=1.34,
    phase_amp_analysis = False
)

errorsByCourant(
    cond2,
    schemes,
    "squareErrorsByCourant.pdf",
    nx=40,
    nu_values=nu_range,
    u=1,
    endTime=1.34,
    phase_amp_analysis= False
)

errorsByCourant(
    sinWave,
    schemes,
    "sinErrorsByCourant.pdf",
    nx=40,
    nu_values=nu_range,
    u=1,
    endTime=1.34,
    phase_amp_analysis=True
)

