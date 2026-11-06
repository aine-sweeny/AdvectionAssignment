# new numerical scheme for solving the advection equation
"""
Created on Fri Oct 16 10:57:04 2026

@author: he820946
"""

import numpy as np

def SLC(phiLevels, c):
    '''
    Solves the linear advection equation for one time step using
    a semi-Lagrangian scheme with cubic Lagrange interpolation.
    phiLevels is updated. Nothing is returned.

    Parameters:
        phiLevels ([1darray]): A list containing 1darray which is the latest 
        value of phi, to be updated.

        c (float): The Courant number

    Returns:
        None
    '''

    # Add new time level, copying from the most recent
    phiLevels.append(phiLevels[-1].copy())

    # Labels for time levels at n and n+1
    phiOld = phiLevels[0]
    phiNew = phiLevels[1]

    #update phi for all points
    nx = len(phiOld)

    for j in range(nx):

        # Find arrival/departure cell
        k = int(np.floor(j - c))

        # Find beta
        beta = j - c - k

        # Periodic indices
        km1 = (k - 1) % nx
        k0  = k % nx
        kp1 = (k + 1) % nx
        kp2 = (k + 2) % nx
        
        # #linear version
        # k = int(np.floor(i - c))

        # beta = i - k - c

        # k0 = k % nx
        # k1 = (k + 1) % nx

        # phiNew[i] = (
        #     (1 - beta)*phiOld[k0]
        #     + beta*phiOld[k1]
        # )
        
        # Cubic Lagrange interpolation
        phiNew[j] = (
            -(1/6)*beta*(1-beta)*(2-beta)*phiOld[km1]
            +0.5*(1+beta)*(1-beta)*(2-beta)*phiOld[k0]
            +0.5*(1+beta)*beta*(2-beta)*phiOld[kp1]
            -(1/6)*(1+beta)*beta*(1-beta)*phiOld[kp2]
        )

    # Remove the old value of phiLevels
    phiLevels.pop(0)

    return None
