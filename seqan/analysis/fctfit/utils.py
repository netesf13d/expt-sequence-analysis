# -*- coding: utf-8 -*-
"""
Utility functions for data fitting.
- <chk_bounds>, check that initial fitting parameter are within boundaries
- <estimate_p0>, estimate the initial parameters for fitting
- <get_bounds>, get the boundary values for the fitted parameters.

TODO gendoc
"""

import sys

import numpy as np
from numpy.fft import rfft

THRESHOLD = 0.7
FFT_PAD = 10


# =============================================================================
# UTILITY FUNCTION
# =============================================================================

def chk_bounds(p0: np.ndarray, bmin: np.ndarray, bmax: np.ndarray):
    """
    Check if the initial fitting parameters p0 are within the boundaries:
    bmin < p0 < bmax.

    Raises ValueError if bmin > p0 or bmax < p0.

    Parameters
    ----------
    p0 : 1D np.ndarray
        Initial fitting parameters.
    bmin, bmax : 1D np.ndarray
        Lower and upper boundaries. Must have same length as p0.

    """
    bmin = np.array(bmin)
    bmax = np.array(bmax)
    p0 = np.array(p0)
    if (i := np.union1d(np.nonzero(bmin>p0), np.nonzero(p0>bmax))).size > 0:
        raise ValueError(
            f"initial fitting parameters {p0} out of bounds at indices {i} ; "
            f"bounds are {bmin} and {bmax}"
            )

    if (i := np.nonzero(bmin == bmax)[0]).size > 0:
        raise ValueError(f"lower and upper bounds are equal at indices {i}")


# =============================================================================
# INITIAL PARAMETERS ESTIMATION
# =============================================================================

def estimate_p0(x: np.ndarray, y: np.ndarray)-> dict:
    """
    TODO finish implementation

    General function for initial parameters estimation. Should be used in the
    code in the same way as get_bounds:
    The individual <p0est_*> functions should wrap around this function
    to avoid redundancy in the code.


    Parameters
    ----------
    x : 1D np.ndarray
        Values at which y data is given.
    y : 2D np.ndarray
        Set of experimental data from which to evaluate the initial
        parameters.
        y[i, j] = dataset i at x[j]

    Returns
    -------
    dict
        Estimated parameters as 1D arrays.

    """
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y, axis=1)
    ymax = np.max(y, axis=1)
    ymax_abs = np.max(np.abs(y), axis=1)
    ymean = np.mean(y, axis=1)

    span = xmax - xmin
    step = span / (len(x)-1)

    n = FFT_PAD * len(x)
    ffty = rfft((y-ymean[:, None]), n=n, axis=1, norm='forward')
    kmax = np.argmax(np.abs(ffty), axis=1)


    # y0 : average of y at extremal scan points
    y0 = (y[:, 0] + y[:, -1]) / 2,
    # A : difference between the extremal y and y0
    A = np.where(y0-ymin > ymax-y0, ymin-y0, ymax-y0)
    # x0 : x at extremal value of y
    x0 = np.where(A < 0,
                  x[np.argmin(y, axis=1)],
                  x[np.argmax(y, axis=1)])
    # tau : the scan range
    tau = np.full_like(y0, span)
    # nu : argmax of the fft norm
    nu = kmax / (n*step)
    # width : nb of data point above threshold
    width = np.sum((y - y0[:, None])/A[:, None] > THRESHOLD, axis=1) * step
    # phi : phase of the max fft component
    # includes xmin correction, cosinus to sinus shift phi + Pi/2
    phi = np.angle(ffty[:, kmax]) - 2*np.pi * nu * xmin + np.pi/2
    phi -= 2*np.pi * (phi//(2*np.pi)) # reduction modulo 2*Pi


    # A_T1 : max difference data-ymean backpropagated to x=0 with exp
    A_T1 = np.maximum(ymax-ymean, ymean-ymin) * np.exp(xmin/tau)
    # A_T2 : max difference data-ymean backpropagated to x=0 with gauss
    A_T2 = np.maximum(ymax-ymean, ymean-ymin) * np.exp(xmin**2/(2*tau**2))


    p0 = {
        ## generic parameters
        'y0': y0,
        'A': A,
        ## common parameters
        'x0': x0,
        'tau': tau,
        'nu': nu,
        'width': width,
        'phi': phi,
        ## special parameters
        'x0_shift': width,
        'nu_shift': width,
        'A_T1': A_T1, # exp-scaled amplitude
        'A_T2': A_T2, # gauss-scaled amplitude
        }
    return p0


# =============================================================================
# BOUNDS EVALUATION
# =============================================================================

def get_bounds(x: np.ndarray, y: np.ndarray)-> tuple[dict, dict]:
    """
    Get fit bounds.

    The bounds are:
        - y0 : [0, ymax]
        - A : [-1.5*ymax, 1.5*ymax]
        - x0 : [xmin, xmax]
        - tau : [2*step, 10*span]
        - nu : [1/span, 1/(2*step)] (Nyquist frequency)
        - width : [0, span]
        - phi : [0, 2*Pi]
        - x0_shift : [-span, span]
        - nu_shift : [0, span]
        - A_T1 : A * exp(xmin * 2 * step)
        - A_T2 : A * exp(xmin^2 * 2 * step^2)

    Parameters
    ----------
    x : 1D np.ndarray
        x-values at which data is available.
    y : 1D np.ndarray
        y-values corresponding to measured values.

    Returns
    -------
    liminf : dict
        Inferior bounds for possible fitting parameters.
    limsup : dict
        Supperior bounds for possible fitting parameters.

    """
    xmin, xmax = np.min(x), np.max(x)
    span = xmax - xmin
    ymax = M if (M:=np.max(np.abs(y))) > 0 else 1.

    fMin = 1 / (10*span) # 1 / scan range / 10
    fMax = len(x) / (2 * span) # 1 / 2*step interval, Nyquist freq
    tMin = 1/fMax # 2*step interval
    tMax = 10 * span # 10 * scan range
    Amax = 1.5*ymax

    with np.errstate(over="ignore", under="ignore"):
        A_T1 = Amax*np.exp(xmin / tMin)
        A_T2 = Amax*np.exp(xmin**2 / tMin**2 / 2)


    liminf = {
        ## generic parameters
        'y0': -ymax, # offset
        'A': -Amax, # amplitude
        ## common parameters
        'x0': xmin, # peak position
        'tau': tMin, # characteristic decay time
        'nu': fMin, # freq of an oscillating signal
        'width': 0., # peak width
        'phi': 0., # phase
        ## special parameters
        'x0_shift': -span,
        'nu_shift': 0.,
        'A_T1': -A_T1, # exp-scaled amplitude
        'A_T2': -A_T2, # gauss-scaled amplitude
        }

    limsup = {
        ## generic parameters
        'y0': ymax, # offset
        'A': Amax, # amplitude
        ## common parameters
        'x0': xmax, # peak position
        'tau': tMax, # characteristic decay time
        'nu': fMax, # freq of an oscillating signal
        'width': span, # peak width
        'phi': 2*np.pi, # phase
        ## special parameters
        'x0_shift': span,
        'nu_shift': span,
        'A_T1': A_T1, # exp-scaled amplitude
        'A_T2': A_T2, # gauss-scaled amplitude
        }

    return liminf, limsup










