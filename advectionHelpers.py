# Helper functions for testing advection schemes.
# Authors: Hilary Weller <h.weller@reading.ac.uk>

import numpy as np

def cosBell(x, a, b):
    '''
    Returns a cosine bell shape between parameters a and b
    Author: Hilary Weller <h.weller@reading.ac.uk>
    Parameters:
        x (1darray): The x-locations for calculating the cosine bell.
        a (float): The start of the non-zero bell
        b (float): The end of the non-zero bell
    Returns:
        phi : 1darray. A cosine bell shape between x=a and x=b
    '''
    return np.where((x > a) & (x < b), 
                    0.5*(1 - np.cos(2*np.pi*(x - a)/(b-a))), 0.)

def cosBell_0_05(x):
    "cosBell initial conditions with a=0.1, x=0.5"
    return cosBell(x, 0.1, 0.5)

def l2error(phi, phiA):
    '''
    Returns the l2 error norm of phi in comparison to exact solution phiA.
    Author: Hilary Weller <h.weller@reading.ac.uk>
    Parameters:
        phi (1darray): The numerical solution
        phiA (1darray): The analytic solution
    Returns
        l2 (float): The l2 error norm
    '''
    phiError = phi - phiA
    l2 = np.sqrt(np.sum(phiError**2)/np.sum(phiA**2))
    return l2

#need to add in other initial condition where phi is 1 for a<x<b and zero
# otherwise. will introduce discontinuities. try to include both on the same
# plot like has been done in the lecture notes
def initialcond2 (x,a,b):
    '''
    Returns a rectangular shape at height 1 between parameters a and b

    Parameters
    ----------
    x (1darray): the x loactions for the rectangle
    a (float): the start of the rectangle of height 1
    b (float): the end of the rectangle

    Returns
        phi : 1darray. A rectangle of height one between x=a and x=b

    '''
    return np.where((x > a) & (x < b), 
                    1, 0.)

def cond2(x):
    "rectangular initial conditions with a=0.1, x=0.5"
    return initialcond2(x, 0.1, 0.5)  

def sinWave(x):
    "periodic, sinusoidal intial conditions"
    return np.sin(2*np.pi*x)