# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:02:53 2026

@author: he820946
"""

"""
CTCS diagnostics for linear advection with a sine-wave initial condition.

Test 1:
    Plot the numerical solution, exact solution, and pointwise error
    at selected Courant numbers.

Test 2:
    Plot the L2 error at every time step and a consecutive-time-level
    difference diagnostic.

Requires:
    advectionHelpers.py containing sinWave and l2error
    schemesBasic.py containing CTCS

Save this script in the same directory as those modules and run it.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from advectionHelpers import sinWave, l2error
from schemesBasic import CTCS


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
NX = 40
U = 1.0
L = 1.0
T = 1.0

# Courant numbers for Test 1 and Test 2.
SELECTED_COURANTS = [0.4, 0.5, 0.7, 0.9]

# Make output directory if needed.
os.makedirs("plots", exist_ok=True)

dx = L / NX
x = np.arange(NX) * dx


def exact_solution(x_values, time):
    """Exact periodic solution of phi_t + U phi_x = 0."""
    return sinWave((x_values - U * time) % L)


def run_ctcs(courant, initialisation="analytic"):
    """
    Run CTCS to the fixed final time T.

    Parameters
    ----------
    courant : float
        Courant number C = U*dt/dx.
    initialisation : {"analytic", "ftcs"}
        "analytic": initialise at t=-dt and t=0 using the exact solution.
        "ftcs": start with phi(t=0) and use CTCS's built-in FTCS start-up.

    Returns
    -------
    times : ndarray
        Times corresponding to the saved numerical solutions.
    solutions : ndarray
        Numerical solutions, shape (number_of_times, NX).
    """
    if courant <= 0:
        raise ValueError("Courant number must be positive.")

    dt = courant * dx / U
    nt = int(round(T / dt))

    # Choose dt so that the simulation ends at T exactly.
    dt = T / nt
    courant_used = U * dt / dx

    phi0 = sinWave(x)

    if initialisation == "analytic":
        # At t=-dt, phi(x,-dt) = phi0(x + U*dt).
        phi_previous = sinWave((x + U * dt) % L)
        levels = [phi_previous.copy(), phi0.copy()]
        saved_solutions = [phi0.copy()]
        times = [0.0]

        # Current levels initially represent t=-dt and t=0.
        # Each CTCS call advances the newest level by one time step.
        # The two initial levels are at t=-dt and t=0, so only
        # nt-1 further updates are needed to reach T.
        for n in range(nt - 1):
            CTCS(levels, courant_used)
            time_now = (n + 1) * dt
            saved_solutions.append(levels[-1].copy())
            times.append(time_now)

    elif initialisation == "ftcs":
        levels = [phi0.copy()]
        saved_solutions = [phi0.copy()]
        times = [0.0]

        for n in range(nt):
            CTCS(levels, courant_used)
            time_now = (n + 1) * dt
            saved_solutions.append(levels[-1].copy())
            times.append(time_now)
    else:
        raise ValueError("initialisation must be 'analytic' or 'ftcs'.")

    return np.asarray(times), np.asarray(saved_solutions), courant_used


# ---------------------------------------------------------------------
# Test 1: final numerical solution and pointwise error
# ---------------------------------------------------------------------
def test_1_solution_and_pointwise_error():
    ncols = 2
    nrows = len(SELECTED_COURANTS)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(12, 3.2 * nrows), squeeze=False
    )

    for row, courant in enumerate(SELECTED_COURANTS):
        times, solutions, used_c = run_ctcs(courant)
        numerical = solutions[-1]
        exact = exact_solution(x, times[-1])
        pointwise_error = numerical - exact

        axes[row, 0].plot(x, exact, "k-", linewidth=2, label="Exact")
        axes[row, 0].plot(
            x, numerical, "ro--", markersize=3, label="CTCS"
        )
        axes[row, 0].set_title(
            f"Final solution: requested C={courant:.3f}, "
            f"actual C={used_c:.3f}"
        )
        axes[row, 0].set_xlabel("x")
        axes[row, 0].set_ylabel(r"$\phi$")
        axes[row, 0].legend()
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].axhline(0.0, color="k", linewidth=0.8)
        axes[row, 1].plot(
            x, pointwise_error, "b.-", markersize=5
        )
        axes[row, 1].set_title(
            f"Pointwise error; normalised L2 = {l2error(numerical, exact):.3e}"
        )
        axes[row, 1].set_xlabel("x")
        axes[row, 1].set_ylabel(r"$\phi_{\mathrm{CTCS}}-\phi_{\mathrm{exact}}$")
        axes[row, 1].grid(True, alpha=0.3)

    fig.suptitle("Test 1: CTCS sine-wave solution at T = 1", y=1.002)
    fig.tight_layout()
    fig.savefig("plots/ctcs_test1_solution_error.png", dpi=180,
                bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------
# Test 2: error growth at every time step
# ---------------------------------------------------------------------
def test_2_error_over_time():
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for courant in SELECTED_COURANTS:
        times, solutions, used_c = run_ctcs(courant)

        errors = []
        consecutive_differences = []

        for n, time_now in enumerate(times):
            exact = exact_solution(x, time_now)
            errors.append(l2error(solutions[n], exact))

            # Diagnostic only: this is not a pure computational-mode
            # amplitude, because the physical solution also changes.
            if n == 0:
                consecutive_differences.append(np.nan)
            else:
                difference = solutions[n] - solutions[n - 1]
                consecutive_differences.append(
                    np.sqrt(np.mean(difference**2))
                )

        axes[0].semilogy(
            times, np.maximum(errors, 1e-16),
            label=f"C={used_c:.3f}"
        )
        axes[1].plot(
            times, consecutive_differences,
            label=f"C={used_c:.3f}"
        )

    axes[0].set_ylabel("Normalised L2 error")
    axes[0].set_title("CTCS error at every time step")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Time")
    axes[1].set_ylabel(r"RMS($\phi^n-\phi^{n-1}$)")
    axes[1].set_title("Consecutive-time-level difference")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig("plots/ctcs_test2_error_over_time.png", dpi=180,
                bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    test_1_solution_and_pointwise_error()
    test_2_error_over_time()
    print("Finished. Figures saved in the 'plots' directory.")