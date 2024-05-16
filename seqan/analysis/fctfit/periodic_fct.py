# -*- coding: utf-8 -*-
"""
Oscillating functions for data fitting.

For sequence data, the fits consist in three functions:
    - <fct>, the fitted function itself
    - <p0est_fct>, to make a fist guess of the fitted parameters from the
      data
    - <fit_fct>, to fit the data
The available fitting functions are:
    - <damped_sin>, exponentially damped sinus
    - <gauss_damped_sin>, gaussian damped sinus
    - <rabi_osc>, theoretical Rabi oscillation

TODO gendoc
"""

import sys

import numpy as np
from numpy import exp
from numpy.fft import rfft
from scipy.optimize import curve_fit
from scipy.integrate import quad_vec

from .utils import chk_bounds, get_bounds


# =============================================================================
# ========== Damped sinus ==========
# =============================================================================

def damped_sin(x: np.ndarray,
               y0: float,
               A: float,
               nu: float,
               tau: float,
               phi: float,)-> np.ndarray:
    """
    Damped sinus
    f(x) = A*sin(2pi*nu*x + phi)*exp(-x/tau) + y0

    Parameters
    ----------
    x : np.ndarray
        Values at which to evaluate the function.
    y0, A, nu, tau, phi : float
        Frequency, phase, decay time constant, ampplitude, offset.

    Returns
    -------
    np.ndarray
        Damped sin evaluated at x.

    """
    return A * np.sin(2*np.pi*nu*x + phi) * exp(-x/tau) + y0

damped_sin.pnames = ("y0", "A", "nu", "tau", "phi")
damped_sin.ptypes = ("y0", "A_T1", "nu", "tau", "phi")
damped_sin.latex = \
    r"$f(t) = y_0 + A e^{-\frac{t}{\tau}} \sin(2\pi \nu t + \phi)$"


def p0est_damped_sin(x: np.ndarray,
                     y: np.ndarray,
                     **kwargs)-> np.ndarray:
    """
    Estimate initial parameters for datasets y[i, :] evaluated at x:
        - y0 is the average of y
        - A0 is the maximum difference between y and y0
        - nu0 is the frequency of the largest non-zero component
          of the fft of y
        - tau0 is the inverse of the span
        - phi0 is the phase of the largest fft component

    Parameters
    ----------
    x : 1D np.ndarray
        Values at which y data is given.
    y : 2D np.ndarray
        Set of experimental data from which to evaluate the initial
        parameters.
        y[i, j] = dataset i at x[j]
    **kwargs : Any
        These are ignored.

    Returns
    -------
    2D np.ndarray
        Array of estimated parameters p0. The format is:
        p0 = [[y0_0, A0_0, nu0_0, tau0_0, phi0_0],
              [y0_1, A0_1, nu0_1, tau0_1, phi0_1],
              ...]

    """
    nbfit = len(y)
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y, axis=1), np.max(y, axis=1)
    span = xmax - xmin
    step = (xmax - xmin)/(len(x)-1)

    # y0 as the mean of the data
    y0 = np.mean(y, axis=1)
    # tau as the scan range
    tau0 = np.full(y0.shape, span)
    # A as the maximum difference between y0 and the data
    # backpropagated to x=0
    A0 = np.maximum(ymax-y0, y0-ymin) * exp(xmin/tau0)

    ## Get frequency and phase from the maximal component of the fft
    pad = 10 # pad the data with zeros to increase frequency resolution
    n = pad * len(x)
    y0_bc = np.reshape(y0, (nbfit, 1))
    ffty = rfft((y-y0_bc), n=n, axis=1, norm='forward')
    kmax = np.argmax(np.abs(ffty), axis=1)

    # Corresponding frequency
    nu0 = kmax / (n*step)

    # Corresponding phase, including the xmin offset.
    # FFT phase is the phase of the cosinus, so phi + Pi/2
    # because a sinus is fitted: cos(.) = sin(. + Pi/2)
    phi = np.angle(ffty[np.arange(nbfit), kmax]) \
          - 2*np.pi * nu0 * xmin + np.pi/2
    # reduce modulo 2*Pi
    phi0 = phi - 2*np.pi * (phi//(2*np.pi))

    return np.array([y0, A0, nu0, tau0, phi0]).transpose()


def fit_damped_sin(x: np.ndarray,
                   y: np.ndarray,
                   p0: np.ndarray,
                   y_sigma: np.ndarray = None,
                   **kwargs):
    """
    Fit damped sin using scipy.optimize.curve_fit.

    Bounds for the optimized parameters are computed from data:
        - -ymax <= y0 <= ymax
        - 0 <= A <= Amax
        - 1/span <= nu <= 1/(2*step) (Nyquist frequency)
        - 2*step <= tau <= 10*span
        - 0 <= phi <= 2*Pi

    Parameters
    ----------
    x : 1D np.ndarray
        x-values at which data is available.
    y : 1D np.ndarray
        y-values corresponding to measured values.
    p0 : np.ndarray
        Initial guess for the various parameters. The format is:
        p0[0] : offset y0
        p0[1] : amplitude A0
        p0[2] : frequency nu0
        p0[3] : decay tau0
        p0[4] : phase phi0
    y_sigma : None or 1D np.ndarray
        The uncertainity in y data (see curve_fit documentation).
        The default is None.
    **kwargs : Any
        These are ignored.

    Returns
    -------
    popt : 1D np.ndarray
        Optimized parameters. Format
        [y0, A, nu, tau, phi]
    pcov : 2D np.ndarray
        Covariance matrix of the optimized parameters.

    """
    linf, lsup = get_bounds(x, y)
    liminf = [linf['y0'], linf['A_T1'], linf['nu'], linf['tau'], linf['phi']]
    limsup = [lsup['y0'], lsup['A_T1'], lsup['nu'], lsup['tau'], lsup['phi']]
    bounds = (liminf, limsup)

    chk_bounds(p0, liminf, limsup)

    return curve_fit(damped_sin, x, y, p0, sigma=y_sigma, bounds=bounds)


# =============================================================================
# ========== Damped sinus ==========
# =============================================================================

def gauss_damped_sin(x: np.ndarray,
                     y0: float,
                     A: float,
                     nu: float,
                     tau: float,
                     phi: float)-> np.ndarray:
    """
    Damped sinus
    f(x) = A*sin(2pi*nu*x + phi)*exp(-x^2/(2 tau^2)) + y0

    Parameters
    ----------
    x : np.ndarray
        Values at which to evaluate the function.
    y0, A, nu, tau, phi : float
        Frequency, phase, decay time constant, ampplitude, offset.

    Returns
    -------
    np.ndarray
        Damped sin evaluated at x.

    """
    return A * np.sin(2*np.pi*nu*x + phi) * exp(-x**2/(2*tau**2)) + y0

gauss_damped_sin.pnames = ("y0", "A", "nu", "tau", "phi")
gauss_damped_sin.ptypes = ("y0", "A_T2", "nu", "tau", "phi")
gauss_damped_sin.latex = \
    r"$f(t) = y_0 + A e^{-\frac{t^2}{2\tau^2}} \sin(2\pi \nu t + \phi)$"


def p0est_gauss_damped_sin(x: np.ndarray,
                           y: np.ndarray,
                           **kwargs)-> np.ndarray:
    """
    Estimate initial parameters for datasets y[i, :] evaluated at x:
        - y0 is the average of y
        - A0 is the maximum difference between y and y0
        - tau0 is the inverse of the span
        - nu0 is the frequency of the largest non-zero component
          of the fft of y
        - phi0 is the phase of the largest fft component

    Parameters
    ----------
    x : 1D np.ndarray
        Values at which y data is given.
    y : 2D np.ndarray
        Set of experimental data from which to evaluate the initial
        parameters.
        y[i, j] = dataset i at x[j]
    **kwargs : Any
        These are ignored.

    Returns
    -------
    2D np.ndarray
        Array of estimated parameters p0. The format is:
        p0 = [[y0_0, A0_0, nu0_0, tau0_0, phi0_0],
              [y0_1, A0_1, nu0_1, tau0_1, phi0_1],
              ...]
    """
    nbfit = len(y)
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y, axis=1), np.max(y, axis=1)
    span = xmax - xmin
    step = (xmax - xmin)/(len(x)-1)

    # y0 as the mean of the data
    y0 = np.mean(y, axis=1)
    # tau as the scan range
    tau0 = np.full(y0.shape, span)
    # A as the maximum difference between y0 and the data
    # backpropagated to x=0
    A0 = np.maximum(ymax-y0, y0-ymin) * exp(xmin**2/(2*tau0**2))

    ## Get frequency and phase from the maximal component of the fft
    pad = 10 # pad the data with zeros to increase frequency resolution
    n = pad * len(x)
    y0_bc = np.reshape(y0, (nbfit, 1))
    ffty = rfft((y-y0_bc), n=n, axis=1, norm='forward')
    kmax = np.argmax(np.abs(ffty), axis=1)

    # Corresponding frequency
    nu0 = kmax / (n*step)

    # Corresponding phase, including the xmin offset.
    # FFT phase is the phase of the cosinus, so phi + Pi/2
    # because a sinus is fitted: cos(.) = sin(. + Pi/2)
    phi = np.angle(ffty[np.arange(nbfit), kmax]) \
          - 2*np.pi * nu0 * xmin + np.pi/2
    # reduce modulo 2*Pi
    phi0 = phi - 2*np.pi * (phi//(2*np.pi))

    return np.array([y0, A0, nu0, tau0, phi0]).transpose()


def fit_gauss_damped_sin(x: np.ndarray,
                         y: np.ndarray,
                         p0: np.ndarray,
                         y_sigma: np.ndarray = None,
                         **kwargs):
    """
    Fit damped sin using scipy.optimize.curve_fit.

    Bounds for the optimized parameters are computed from data:
        - -ymax <= y0 <= ymax
        - 0 <= A <= Amax
        - 1/span <= nu <= 1/(2*step) (Nyquist frequency)
        - 2*step <= tau <= 10*span
        - 0 <= phi <= 2*Pi

    Parameters
    ----------
    x : 1D np.ndarray
        x-values at which data is available.
    y : 1D np.ndarray
        y-values corresponding to measured values.
    p0 : np.ndarray
        Initial guess for the various parameters. The format is:
        p0[0] : offset y0
        p0[1] : amplitude A0
        p0[2] : frequency nu0
        p0[3] : decay tau0
        p0[4] : phase phi0
    y_sigma : None or 1D np.ndarray
        The uncertainity in y data (see curve_fit documentation).
        The default is None.
    **kwargs : Any
        These are ignored.

    Returns
    -------
    popt : 1D np.ndarray
        Optimized parameters. Format
        [y0, A, nu, tau, phi]
    pcov : 2D np.ndarray
        Covariance matrix of the optimized parameters.

    """
    linf, lsup = get_bounds(x, y)
    liminf = [linf['y0'], linf['A_T2'], linf['nu'], linf['tau'], linf['phi']]
    limsup = [lsup['y0'], lsup['A_T2'], lsup['nu'], lsup['tau'], lsup['phi']]
    bounds = (liminf, limsup)

    chk_bounds(p0, liminf, limsup)

    return curve_fit(gauss_damped_sin, x, y, p0, sigma=y_sigma, bounds=bounds)


# =============================================================================
# ========== Rabi oscillations ==========
# =============================================================================

def rabi_osc(x: np.ndarray,
             y0: float,
             A: float,
             nu: float,
             delta: float,
             phi: float)-> np.ndarray:
    """
    Rabi oscillations at Rabi frequency f = sqrt(nu^2 + delta^2)

    f(x) = y0 - A * nu/f * sin^2(Pi*f*x + phi)

    Returns
    -------
    np.ndarray
        Rabi oscillation evaluated at x.

    """
    f = np.sqrt(nu**2 + delta**2)
    return y0 - A * nu/f * np.sin(np.pi*f*x + phi)**2

rabi_osc.pnames = ("y0", "A", "nu", "delta", "phi")
rabi_osc.ptypes = ("y0", "A", "nu", "nu_shift", "phi")
rabi_osc.latex = \
    r"$f(t) = y_0 - A \frac{\nu}{\sqrt{\nu^2+\delta^2}}" \
    r"\sin^2 \left( \pi\sqrt{\nu^2+\delta^2} t + \phi \right)$"


def p0est_rabi_osc(x: np.ndarray,
                   y: np.ndarray,
                   **kwargs)-> np.ndarray:
    """
    Estimate initial parameters for datasets y[i, :] evaluated at x.

    The estimated parameters are:
        - y0 is the max of y
        - A0 is the maximum difference between y and y0
        - nu0 = A*f
        - delta0 = sqrt(1-A^2)*f
        - phi0 is half the phase of the largest fft component

    Parameters
    ----------
    x : 1D np.ndarray
        Values at which y data is given.
    y : 2D np.ndarray
        Set of experimental data from which to evaluate the initial
        parameters.
        y[i, j] = dataset i at x[j]
    **kwargs : Any
        These are ignored.

    Returns
    -------
    2D np.ndarray
        Array of estimated parameters p0. The format is:
        p0 = [[y0_0, A0_0, nu0_0, delta0_0, phi0_0],
              [y0_1, A0_1, nu0_1, delta0_1, phi0_1],
              ...]

    """
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y, axis=1), np.max(y, axis=1)
    step = (xmax - xmin)/(len(x)-1)

    y0 = ymax
    A0 = ymax - ymin
    A = A0 / y0

    ## Get frequency and phase from the maximal component of the fft
    pad = 10 # pad the data with zeros to increase frequency resolution
    n = pad * len(x)
    mean_y = np.reshape(np.mean(y, axis=1), (len(y), 1))
    ffty = rfft((y-mean_y), n=n, axis=1, norm='forward')
    kmax = np.argmax(np.abs(ffty), axis=1)

    # Corresponding frequency
    f = kmax / (n*step)
    # Corresponding phase, including the xmin offset
    phi = np.angle(ffty[np.arange(len(kmax)), kmax]) \
          + 2*np.pi * f * xmin
    # phi/2 because sin^2(. + phi/2) is like sin(. + phi)
    # and reduce modulo 2*Pi
    phi0 = phi/2 - 2*np.pi * (phi//(2*np.pi))

    nu0 = A*f
    delta0 = np.sqrt(1-A**2) * f

    return np.array([y0, A0, nu0, delta0, phi0]).transpose()


def fit_rabi_osc(x: np.ndarray,
                 y: np.ndarray,
                 p0: np.ndarray,
                 y_sigma: np.ndarray = None,
                 **kwargs):
    """
    Fit rabi oscillation using scipy.optimize.curve_fit

    Bounds for the optimized parameters are computed from data:
        - 0 <= y0 <= 2 * ymax
        - 0 <= A <= 2 * ymax
        - 1/span <= nu <= 1/(2*step) (Nyquist frequency)
        - 1/span <= delta <= 5 * Nyquist frequency
        - 0 <= phi <= 2 * Pi

    Rabi oscillation fitting is valid only for probabilities, ie
    y value such that 0 <= y <= 1.

    Parameters
    ----------
    x : 1D np.ndarray
        x-values at which data is available.
    y : 1D np.ndarray
        y-values corresponding to measured values.
    p0 : np.ndarray
        Initial guess for the various parameters. The format is:
        p0[0] : offset y0
        p0[1] : amplitude A0
        p0[2] : frequency nu0
        p0[3] : detuning delta0
        p0[3] : phase phi0
    y_sigma : None or 1D np.ndarray
        The uncertainity in y data (see curve_fit documentation).
        The default is None.
    **kwargs : Any
        These are ignored.

    Returns
    -------
    popt : 1D np.ndarray
        Optimized parameters. Format
        [nu0, phi0, tau0, A0, y0]
    pcov : 2D np.ndarray
        Covariance matrix of the optimized parameters.

    """
    linf, lsup = get_bounds(x, y)
    liminf = [linf['y0'], linf['A'], linf['nu'], linf['nu_shift'], linf['phi']]
    limsup = [lsup['y0'], lsup['A'], lsup['nu'], lsup['nu_shift'], lsup['phi']]
    bounds = (liminf, limsup)

    chk_bounds(p0, liminf, limsup)

    return curve_fit(rabi_osc, x, y, p0, sigma=y_sigma, bounds=bounds)

