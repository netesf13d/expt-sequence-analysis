# -*- coding: utf-8 -*-
"""
`Special` functions for data fitting.

For sequence data, the fits consist in three functions:
    - <fct>, the fitted function itself
    - <p0est_fct>, to make a fist guess of the fitted parameters from the
      data
    - <fit_fct>, to fit the data
The available fitting functions are:
    - <decay_exp>, decaying exponential
    - <decay_exp_0>, decaying exponential, no offset
    - <damped_arcsine_sin>, exponentially-damped sinus convolved with
        an acrsine distrib
    - <gauss_damped_arcsine_sin>, gaussian-damped sinus convolved with
        an acrsine distrib
    - <inverse_lorentz>, Lorentz peak convolved with an inverse distribution
    - <convol_lorentz>, lorentz peak convolved with a thermal distribution
        in an harmonic potential

TODO gendoc
"""

import sys

import numpy as np
from numpy import exp, sqrt, pi
from numpy.fft import rfft
from scipy.optimize import curve_fit
from scipy.integrate import quad, quad_vec

from .utils import chk_bounds, get_bounds

# =============================================================================
# ========== Exponential decay ==========
# =============================================================================

def decay_exp(x: np.ndarray,
              y0: float,
              A: float,
              tau: float)-> np.ndarray:
    """
    Exponential decay with offset y0.

    f(x) = y0 + A * exp(- x / tau)

    Parameters
    ----------
    x : np.ndarray
        Values at which the function is evaluated.
    y0, A, tau : float
        offset, amplitude, decay time constant.

    Returns
    -------
    np.ndarray
        Function evaluated at x.

    """
    return y0 + A * exp(- x / tau)

decay_exp.pnames = ("y0", "A", "tau")
decay_exp.ptypes = ("y0", "A_T1", "tau")
decay_exp.latex = r"$f(t) = y_0 + A \exp\left(-\frac{t}{\tau}\right)$"


def p0est_decay_exp(x: np.ndarray,
                    y: np.ndarray,
                    **kwargs)-> np.ndarray:
    """
    Estimate initial parameters for datasets y[i, :] evaluated at x:
        - y0 is the minimum of y
        - A0 is the maximum of y
        - tau0 is such that ymax/ymin = exp(-span/tau0)

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
        p0 = [[y0_0, A0_0, mu0_0, sigma0_0],
              [y0_1, A0_1, mu0_1, sigma0_1],
              ...]

    """
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y, axis=1)
    ymax = np.max(y, axis=1)
    step = (xmax-xmin) / (len(x)-1)

    # y0 as ymin
    y0 = ymin

    # tau as the span above a certain threshold
    THRESHOLD = 0.37
    y0_bc = y0.reshape((len(y0), 1))
    ymax_bc = ymax.reshape((len(ymax), 1))
    tau0 = np.sum((y - y0_bc)/(ymax_bc-y0_bc) > THRESHOLD, axis=1) * step

    # A0 as ymax backpropagated to x=0
    A0 = ymax * exp(xmin/tau0)

    return np.array([y0, A0, tau0]).transpose()


def fit_decay_exp(x: np.ndarray,
                  y: np.ndarray,
                  p0: np.ndarray,
                  y_sigma: np.ndarray = None,
                  **kwargs):
    """
    TODO update bounds
    Fit decay exponential using scipy.optimize.curve_fit

    Bounds for the optimized parameters are computed from data:
        - min(0, ymin) <= y0 <= 2* lowest non-zero y-value
        - 0 <= A <= 2* extrapolation of the data at 0
        - step <= tau <= 5*span

    Parameters
    ----------
    x : 1D np.ndarray
        x-values at which data is available.
    y : 1D np.ndarray
        y-values corresponding to measured values.
    p0 : 1D np.ndarray
        Initial guess for the various parameters. The format is:
        params[0] : offset y0
        params[1] : Amplitude A of the exponential
        params[2] : decay time constant
    y_sigma : 1D np.ndarray
        The uncertainity in y data (see curve_fit documentation), to be used
        for fitting. The default is None.
    **kwargs : Any
        These are ignored.

    Returns
    -------
    popt : 1D np.ndarray
        Optimized parameters. Format
        [y0, A, tau]
    pcov : 2D np.ndarray
        Covariance matrix of the optimized parameters.

    """
    linf, lsup = get_bounds(x, y)
    liminf = [linf['y0'], linf['A_T1'], linf['tau']]
    limsup = [lsup['y0'], lsup['A_T1'], lsup['tau']]
    bounds = (liminf, limsup)

    chk_bounds(p0, liminf, limsup)

    return curve_fit(decay_exp, x, y, p0, sigma=y_sigma, bounds=bounds)


# =============================================================================
# ========== Exponential decay ==========
# =============================================================================

def decay_exp_0(x: np.ndarray,
              A: float,
              tau: float)-> np.ndarray:
    """
    Exponential decay with offset y0.

    f(x) = y0 + A * exp(- x / tau)

    Parameters
    ----------
    x : np.ndarray
        Values at which the function is evaluated.
    y0, A, tau : float
        offset, amplitude, decay time constant.

    Returns
    -------
    np.ndarray
        Function evaluated at x.

    """
    return A * exp(- x / tau)

decay_exp.pnames = ("A", "tau")
decay_exp.ptypes = ("A_T1", "tau")
decay_exp.latex = r"$f(t) = A \exp\left(-\frac{t}{\tau}\right)$"


def p0est_decay_exp_0(x: np.ndarray,
                    y: np.ndarray,
                    **kwargs)-> np.ndarray:
    """
    Estimate initial parameters for datasets y[i, :] evaluated at x:
        - y0 is the minimum of y
        - A0 is the maximum of y
        - tau0 is such that ymax/ymin = exp(-span/tau0)

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
        p0 = [[y0_0, A0_0, mu0_0, sigma0_0],
              [y0_1, A0_1, mu0_1, sigma0_1],
              ...]

    """
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y, axis=1)
    ymax = np.max(y, axis=1)
    step = (xmax-xmin) / (len(x)-1)

    # y0 as ymin
    y0 = ymin

    # tau as the span above a certain threshold
    THRESHOLD = 0.37
    y0_bc = y0.reshape((len(y0), 1))
    ymax_bc = ymax.reshape((len(ymax), 1))
    tau0 = np.sum((y - y0_bc)/(ymax_bc-y0_bc) > THRESHOLD, axis=1) * step

    # A0 as ymax backpropagated to x=0
    A0 = ymax * exp(xmin/tau0)

    return np.array([A0, tau0]).transpose()


def fit_decay_exp_0(x: np.ndarray,
                  y: np.ndarray,
                  p0: np.ndarray,
                  y_sigma: np.ndarray = None,
                  **kwargs):
    """
    TODO update bounds
    Fit decay exponential using scipy.optimize.curve_fit

    Bounds for the optimized parameters are computed from data:
        - min(0, ymin) <= y0 <= 2* lowest non-zero y-value
        - 0 <= A <= 2* extrapolation of the data at 0
        - step <= tau <= 5*span

    Parameters
    ----------
    x : 1D np.ndarray
        x-values at which data is available.
    y : 1D np.ndarray
        y-values corresponding to measured values.
    p0 : 1D np.ndarray
        Initial guess for the various parameters. The format is:
        params[0] : offset y0
        params[1] : Amplitude A of the exponential
        params[2] : decay time constant
    y_sigma : 1D np.ndarray
        The uncertainity in y data (see curve_fit documentation), to be used
        for fitting. The default is None.
    **kwargs : Any
        These are ignored.

    Returns
    -------
    popt : 1D np.ndarray
        Optimized parameters. Format
        [y0, A, tau]
    pcov : 2D np.ndarray
        Covariance matrix of the optimized parameters.

    """
    xmin = np.min(x)
    xmax = np.max(x)
    ymax = np.max(y)

    span = xmax-xmin
    step = span/(len(x)-1)
    Amax = ymax * exp(xmin/step)

    # Boundaries
    # min(0, y0min) <= y0 <= 2*min(y>0) (the lowest value > 0)
    # 0 <= A <= 2*Amax (estimation from extrapolation at t=0)
    # step <= tau <= 5*span
    liminf = [0, step]
    limsup = [2*Amax, 5*span]
    bounds = (liminf, limsup)
    chk_bounds(p0, liminf, limsup)

    return curve_fit(decay_exp_0, x, y, p0, sigma=y_sigma, bounds=bounds)



# =============================================================================
# ========== Damped sinus with arcsine-distributed frequencies ==========
# =============================================================================

def damped_arcsine_sin(x: np.ndarray,
                        y0: float,
                        A: float,
                        nu: float,
                        tau: float,
                        phi: float,
                      dnu: float)-> np.ndarray:
    """
    TODO doc
    Damped sinus
    f(x) = y0 + A * exp(-x/tau)
            * int(sin(2pi * (nu + dnu*y) * x + phi) / (pi*sqrt(1-y^2)), y=-1..1)

    Parameters
    ----------
    x : np.ndarray
        Values at which to evaluate the function.
    y0, A, nu, tau, phi, dnu : float
        Frequency, phase, decay time constant, ampplitude, offset.

    Returns
    -------
    np.ndarray
        Damped sin evaluated at x.

    """
    f = lambda y: np.sin(2*pi*x* (nu+dnu*y) + phi) / sqrt(1-y**2)
    integral = quad_vec(f, -0.9999999, 0.9999999)[0]
    return A * integral / pi * exp(-x**2/(2*tau**2)) + y0

damped_arcsine_sin.pnames = ("y0", "A", "nu", "tau", "phi", "dnu")
damped_arcsine_sin.ptypes = ("y0", "A_T1", "nu", "tau", "phi", "nu_shift")
damped_arcsine_sin.latex = \
    r"$f(t) = y_0 + A e^{-\frac{t}{\tau}} \int_{-1}^{1} \mathrm{d}u" \
    r"\frac{\sin(2\pi (\nu + u\delta \nu) t + \phi)}{\pi \sqrt{1-u^2}} $"


def p0est_damped_arcsine_sin(
        x: np.ndarray,
        y: np.ndarray,
        **kwargs)-> np.ndarray:
    """
    TODO doc
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
        p0 = [[y0_0, A0_0, nu0_0, tau0_0, phi0_0, dnu0_0],
              [y0_1, A0_1, nu0_1, tau0_1, phi0_1, dnu0_1],
              ...]
    """
    nbfit = len(y)
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y, axis=1)
    ymax = np.max(y, axis=1)
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
    fftNorm = np.abs(ffty)

    # Corresponding frequency
    freqs = np.linspace(0, (k:=ffty.shape[-1]-1) / (n*step), k, endpoint=True)
    kmax = np.argmax(fftNorm, axis=1)
    nu0 = freqs[kmax]

    # Corresponding phase, including the xmin offset.
    # FFT phase is the phase of the cosinus, so phi + Pi/2
    # because a sinus is fitted: cos(.) = sin(. + Pi/2)
    phi = np.angle(ffty[np.arange(nbfit), kmax]) \
          - 2*pi * nu0 * xmin + pi/2
    # reduce modulo 2*Pi
    phi0 = phi - 2*pi * (phi//(2*pi))

    # estimate freq width
    pk = fftNorm > np.max(fftNorm, axis=1).reshape((-1, 1)) / sqrt(2)
    f1 = freqs[np.argmax(pk, axis=1)]
    f2 = freqs[len(freqs) - 1 - np.argmax(pk[:, -1::-1], axis=1)]
    dnu0 = abs((f2 - f1) / 2)

    return np.array([y0, A0, nu0, tau0, phi0, dnu0]).transpose()


def fit_damped_arcsine_sin(
        x: np.ndarray,
        y: np.ndarray,
        p0: np.ndarray,
        y_sigma: np.ndarray = None,
        **kwargs):
    """
    TODO doc
    Fit damped sin using scipy.optimize.curve_fit.

    Bounds for the optimized parameters are computed from data:
        - -ymax <= y0 <= ymax
        - 0 <= A <= Amax
        - 1/span <= nu <= 1/(2*step) (Nyquist frequency)
        - 2*step <= tau <= 10*span
        - 0 <= phi <= 2*Pi
        - 0 <= dnu <= 1/(2*step) (Nyquist frequency)

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
        p0[5] : dnu0
    y_sigma : None or 1D np.ndarray
        The uncertainity in y data (see curve_fit documentation).
        The default is None.
    **kwargs : Any
        These are ignored.

    Returns
    -------
    popt : 1D np.ndarray
        Optimized parameters. Format
        [y0, A, nu, tau, phi, dnu]
    pcov : 2D np.ndarray
        Covariance matrix of the optimized parameters.

    """
    linf, lsup = get_bounds(x, y)
    liminf = [linf['y0'], linf['A_T1'], linf['nu'],
              linf['tau'], linf['phi'], linf['nu_shift']]
    limsup = [lsup['y0'], lsup['A_T1'], lsup['nu'],
              lsup['tau'], lsup['phi'], lsup['nu_shift']]
    bounds = (liminf, limsup)

    chk_bounds(p0, liminf, limsup)

    return curve_fit(damped_arcsine_sin, x, y, p0,
                      sigma=y_sigma, bounds=bounds)


# =============================================================================
# ========== Gauss damped sinus with arcsine-distributed frequencies ==========
# =============================================================================

def gauss_damped_arcsine_sin(x: np.ndarray,
                       y0: float,
                       A: float,
                       nu: float,
                       tau: float,
                       phi: float,
                      dnu: float)-> np.ndarray:
    """
    TODO doc
    Damped sinus
    f(x) = y0 + A * exp(-x/tau)
           * int(sin(2pi * (nu + dnu*y) * x + phi) / (pi*sqrt(1-y^2)), y=-1..1)

    Parameters
    ----------
    x : np.ndarray
        Values at which to evaluate the function.
    y0, A, nu, tau, phi, dnu : float
        Frequency, phase, decay time constant, ampplitude, offset.

    Returns
    -------
    np.ndarray
        Damped sin evaluated at x.

    """
    f = lambda y: np.sin(2*pi*x* (nu+dnu*y) + phi) / sqrt(1-y**2)
    integral = quad_vec(f, -0.9999999, 0.9999999)[0]
    return A * integral / pi * exp(-x/tau) + y0

gauss_damped_arcsine_sin.pnames = ("y0", "A", "nu", "tau", "phi", "dnu")
gauss_damped_arcsine_sin.ptypes = ("y0", "A_T2", "nu", "tau", "phi", "nu_shift")
gauss_damped_arcsine_sin.latex = \
    r"$f(t) = y_0 + A e^{-\frac{t^2}{2 \tau^2}} \int_{-1}^{1} \mathrm{d}u" \
    r"\frac{\sin(2\pi (\nu + u\delta \nu) t + \phi)}{\pi \sqrt{1-u^2}} $"

def p0est_gauss_damped_arcsine_sin(
        x: np.ndarray,
        y: np.ndarray,
        **kwargs)-> np.ndarray:
    """
    TODO doc
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
        p0 = [[y0_0, A0_0, nu0_0, tau0_0, phi0_0, dnu0_0],
              [y0_1, A0_1, nu0_1, tau0_1, phi0_1, dnu0_1],
              ...]
    """
    nbfit = len(y)
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y, axis=1)
    ymax = np.max(y, axis=1)
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
    fftNorm = np.abs(ffty)

    # Corresponding frequency
    freqs = np.linspace(0, (k:=ffty.shape[-1]-1) / (n*step), k, endpoint=True)
    kmax = np.argmax(fftNorm, axis=1)
    nu0 = freqs[kmax]

    # Corresponding phase, including the xmin offset.
    # FFT phase is the phase of the cosinus, so phi + Pi/2
    # because a sinus is fitted: cos(.) = sin(. + Pi/2)
    phi = np.angle(ffty[np.arange(nbfit), kmax]) \
          - 2*pi * nu0 * xmin + pi/2
    # reduce modulo 2*Pi
    phi0 = phi - 2*pi * (phi//(2*pi))

    # estimate freq width
    pk = fftNorm > np.max(fftNorm, axis=1).reshape((-1, 1)) / sqrt(2)
    f1 = freqs[np.argmax(pk, axis=1)]
    f2 = freqs[len(freqs) - 1 - np.argmax(pk[:, -1::-1], axis=1)]
    dnu0 = abs((f2 - f1) / 2)

    return np.array([y0, A0, nu0, tau0, phi0, dnu0]).transpose()


def fit_gauss_damped_arcsine_sin(
        x: np.ndarray,
        y: np.ndarray,
        p0: np.ndarray,
        y_sigma: np.ndarray = None,
        **kwargs):
    """
    TODO doc
    Fit damped sin using scipy.optimize.curve_fit.

    Bounds for the optimized parameters are computed from data:
        - -ymax <= y0 <= ymax
        - 0 <= A <= Amax
        - 1/span <= nu <= 1/(2*step) (Nyquist frequency)
        - 2*step <= tau <= 10*span
        - 0 <= phi <= 2*Pi
        - 0 <= dnu <= 1/(2*step) (Nyquist frequency)

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
        p0[5] : dnu0
    y_sigma : None or 1D np.ndarray
        The uncertainity in y data (see curve_fit documentation).
        The default is None.
    **kwargs : Any
        These are ignored.

    Returns
    -------
    popt : 1D np.ndarray
        Optimized parameters. Format
        [y0, A, nu, tau, phi, dnu]
    pcov : 2D np.ndarray
        Covariance matrix of the optimized parameters.

    """
    linf, lsup = get_bounds(x, y)
    liminf = [linf['y0'], linf['A_T2'], linf['nu'],
              linf['tau'], linf['phi'], linf['nu_shift']]
    limsup = [lsup['y0'], lsup['A_T2'], lsup['nu'],
              lsup['tau'], lsup['phi'], lsup['nu_shift']]
    bounds = (liminf, limsup)

    chk_bounds(p0, liminf, limsup)

    return curve_fit(gauss_damped_arcsine_sin, x, y, p0,
                     sigma=y_sigma, bounds=bounds)


# =============================================================================
# ========== Lorentzian with inverse-distributed peak frequencies ==========
# =============================================================================

def inverse_lorentz(x: np.ndarray,
                    y0: float,
                    A: float,
                    mu: float,
                    gamma: float,
                    dmu: float,
                    kappa: float)-> np.ndarray:
    """
    TODO MANAGE POSITIVE AND NEGATIVE DMU
    Lorentz peak convolved with inverse distribution.
    TODO doc
    f(x) = y0 + A
           * int(sin(2pi * (nu + dnu*u) * x + phi) / (pi*sqrt(1-u^2)), u=-1..1)

    This is what you get when the peak frequency shifts over time with an
    exponential decay. For instance the transition frequency of a circular
    to elliptical transition when the electric field is settling with an
    exponential decay.

    Parameters
    ----------
    x : np.ndarray
        Values at which to evaluate the function.
    y0, A, mu, gamma, dmu : float
        Offset, amplitude, peak center, peak width, arcsine shift amplitude

    Returns
    -------
    np.ndarray
        Damped sin evaluated at x.

    """
    f = lambda y: 1 / ((1 + ((x - mu - y)/gamma)**2) * y)

    dmuMin, dmuMax = min(dmu*exp(-kappa), dmu), max(dmu*exp(-kappa), dmu)
    integral = quad_vec(f, dmuMin, dmuMax)[0]

    return A / kappa * integral + y0

inverse_lorentz.pnames = ("y0", "A", "mu", "gamma", "dmu", "kappa")
inverse_lorentz.ptypes = ("y0", "A", "x0", "width", "x0_shift", "nu_shift")
inverse_lorentz.latex = \
    r"$f(x) = y_0 + \frac{A\gamma}{\kappa}\int_{e^{-\kappa}d\mu}^{d\mu}" \
    r"\frac{\mathrm{d}u}{u} \frac{1}{\gamma ^2 + (x-\mu -u)^2}$"


def p0est_inverse_lorentz(x: np.ndarray,
                          y: np.ndarray,
                          **kwargs)-> np.ndarray:
    """
    TODO doc
    Estimate initial parameters for datasets y[i, :] evaluated at x:
        - y0 is the average of y values at scan edges
        - A0 is the maximum difference between y and y0
        - mu0 is the x value at extremal y value
        - gamma0 is proportional to the number of y values above
          a certain threshold

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
        p0 = [[y0_0, A0_0, mu0_0, gamma0_0],
              [y0_1, A0_1, mu0_1, gamma0_1],
              ...]

    """
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y, axis=1)
    ymax = np.max(y, axis=1)

    step = (xmax-xmin) / (len(x)-1)

    # y0 as the average of y at extremal scan points
    y0 = (y[:, 0] + y[:, -1]) / 2

    # A as the difference between the extremal y and y0
    A0 = np.where(y0-ymin > ymax-y0, ymin-y0, ymax-y0)

    # x0 at the extremal value of y
    mu0 = np.where(A0 < 0,
                   x[np.argmin(y, axis=1)],
                   x[np.argmax(y, axis=1)])

    # gamma as the width of the peak above a certain threshold
    THRESHOLD = 0.7 # 1/sqrt(2)
    y0_bc = y0.reshape((len(y0), 1))
    A_bc = A0.reshape((len(A0), 1))
    gamma0 = np.sum((y - y0_bc)/A_bc > THRESHOLD, axis=1) * step

    dmu0 = gamma0
    kappa0 = dmu0
    
    return np.array([y0, A0, mu0, gamma0, dmu0, kappa0]).transpose()


def fit_inverse_lorentz(x: np.ndarray,
                        y: np.ndarray,
                        p0: np.ndarray,
                        y_sigma: np.ndarray = None,
                        **kwargs):
    """
    TODO doc
    Fit single lorentzian peak using scipy.optimize.curve_fit.

    Bounds for the optimized parameters are computed from data:
        - 0 <= y0 <= ymax
        - -1.5*ymax <= A0 <= 1.5*ymax
        - xmin <= mu <= xmax
        - 0 <= gamma <= span

    Parameters
    ----------
    x : np.ndarray
        x-values at which data is available.
    y : np.ndarray
        y-values corresponding to measured values.
    p0 : np.ndarray
        Initial guess for the various parameters. The format is:
        params[0] : offset y0
        params[1] : Amplitude A of the peak
        params[2] : center mu of the peak
        params[3] : HWHM gamma of the peak
    y_sigma : np.ndarray
        The uncertainity in y data (see curve_fit documentation), to be used
        for fitting. The default is None.
    **kwargs : Any
        These are ignored.

    Returns
    -------
    popt : 1D np.ndarray
        Optimized parameters. Format
        [y0, A, mu, gamma]
    pcov : 2D np.ndarray
        Covariance matrix of the optimized parameters.

    """
    linf, lsup = get_bounds(x, y)
    liminf = [linf['y0'], linf['A'], linf['x0'],
              linf['width'], linf['x0_shift'], linf['nu_shift']]
    limsup = [lsup['y0'], lsup['A'], lsup['x0'],
              lsup['width'], lsup['x0_shift'], lsup['nu_shift']]
    bounds = (liminf, limsup)

    chk_bounds(p0, liminf, limsup)

    return curve_fit(inverse_lorentz, x, y, p0, sigma=y_sigma, bounds=bounds)


# =============================================================================
# ========== Lorentzian with arcsine-distributed peak frequencies ==========
# =============================================================================

def arcsine_lorentz(x: np.ndarray,
                    y0: float,
                    A: float,
                    mu: float,
                    gamma: float,
                    dmu: float)-> np.ndarray:
    """
    TODO doc
    Damped sinus
    f(x) = y0 + A * exp(-x/tau)
           * int(sin(2pi * (nu + dnu*y) * x + phi) / (pi*sqrt(1-y^2)), y=-1..1)

    Parameters
    ----------
    x : np.ndarray
        Values at which to evaluate the function.
    y0, A, mu, gamma, dmu : float
        Offset, amplitude, peak center, peak width, arcsine shift amplitude

    Returns
    -------
    np.ndarray
        Damped sin evaluated at x.

    """
    f = lambda y: 1 / ((1 + ((x - mu - dmu*y)/gamma)**2) * sqrt(1-y**2))
    integral = quad_vec(f, -0.9999999, 0.9999999)[0]
    return A * integral / pi + y0


# =============================================================================
# ========== Convolved lorentzian ==========
# =============================================================================

def convol_lorentz(x: np.ndarray,
                   y0: float,
                   A: float,
                   mu: float,
                   gamma: float,
                   muT: float)-> np.ndarray:
    """
    Lorentzian peak convolved with a thermal distribution of peak positions.
    f(x) = y0 + A
        * integral(sqrt(u) * exp(-u) / (1 + ((x - mu + u*muT)/gamma)**2),
                   u=0..inf)

    The probability of having a potential energy E in the harmonic
    approximation is:
        p[T](E) = 2/sqrt(Pi) * E^(1/2)/(k_B*T)^(3/2) * exp(-E/(k_B*T))
    The convolution reads:
        conv_lorentz(x) =
            integral(lorentz(x - (mu + mu'(E))) * p(E), E=0..inf)
    where the scaling and offset have been ommited for the sake of clarity.

    Defining nu := E/h, nu0 := k_B*T/h, u := nu/nu0, one can write
    p[nu0](u) = 2/sqrt(Pi) * 1/(h*nu0) * exp(-u),
    and the convolution reads, after the change of variable E -> u:
        conv_lorentz(x) ~
            integral(lorentz(x - (mu + muT*u)) * sqrt(u) * exp(-u), u=0..inf)
    where muT is the lightshift induced by a shift of energy h*nu0 in the
    trap (= LS_coef_5S5P / LS_coef_5S * nu0).

    Parameters
    ----------
    x : np.ndarray
        The positions at which the function is evaluated.
    y0, A, mu, gamma, muT : float
        Offset, amplitude, peak center, peak width, thermal lightshift

    Returns
    -------
    result : np.ndarray
        Lorentzian peaks evaluated at position(s) x.

    """
    integrand = lambda u: \
        sqrt(u) * exp(-u) / (1 + ((x - mu + u*muT)/gamma)**2)
    return y0 + A * quad_vec(integrand, 0, np.inf)[0]

convol_lorentz.pnames = ("y0", "A", "mu", "gamma", "muT")
convol_lorentz.ptypes = ("y0", "A", "x0", "width", "nu_shift")
convol_lorentz.latex = \
    r"$f(x) = y_0 + A * \int_0^{\infty}  \mathrm{d}u} \sqrt{u} e^{-u}" \
    r"\frac{\gamma}{\gamma ^2 + (x-\mu - u\mu_T)^2}$"


def p0est_convol_lorentz(x: np.ndarray,
                         y: np.ndarray,
                         **kwargs)-> np.ndarray:
    """
    Estimate initial parameters for datasets y[i, :] evaluated at x:
        - y0 is the average of y values at scan edges
        - A0 is the maximum difference between y and y0
        - mu0 is the x value at extremal y value
        - gamma0 is proportional to the number of y values above
          a certain threshold
        - mu_T0 is gamma0 / 10

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
        p0 = [[y0_0, A0_0, mu0_0, gamma0_0, muT0_0],
              [y0_1, A0_1, mu0_1, gamma0_1, muT0_1],
              ...]

    """
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y, axis=1)
    ymax = np.max(y, axis=1)

    step = (xmax-xmin) / (len(x)-1)

    # y0 as the average of y at extremal scan points
    y0 = (y[:, 0] + y[:, -1]) / 2

    # A as the difference between the extremal y and y0
    A0 = np.where(y0-ymin > ymax-y0, ymin-y0, ymax-y0)

    # x0 at the extremal value of y
    mu0 = np.where(A0 < 0,
                   x[np.argmin(y, axis=1)],
                   x[np.argmax(y, axis=1)])

    # gamma as the width of the peak above a certain threshold
    THRESHOLD = 0.8 # approx. exp(-1/8)
    y0_bc = y0.reshape((len(y0), 1))
    A_bc = A0.reshape((len(A0), 1))
    gamma0 = np.sum((y - y0_bc)/A_bc > THRESHOLD, axis=1) * step

    # mu0 as gamma0 / 10, j'ai la flemme.
    muT0 = gamma0 / 10

    return np.array([y0, A0, mu0, gamma0, muT0]).transpose()


def fit_convol_lorentz(x: np.ndarray,
                       y: np.ndarray,
                       p0: np.ndarray,
                       y_sigma: np.ndarray = None,
                       **kwargs):
    """
    Fit lorentzian convolved with thermal distribution of peak positions
    using scipy.optimize.curve_fit.

    Bounds for the optimized parameters are computed from data:
        - 0 <= y0 <= ymax
        - -4*ymax <= A0 <= 4*ymax
        - xmin <= mu <= xmax
        - 0 <= gamma <= span
        - 0 <= muT <= span

    Parameters
    ----------
    x : np.ndarray
        x-values at which data is available.
    y : np.ndarray
        y-values corresponding to measured values.
    p0 : np.ndarray
        Initial guess for the various parameters. The format is:
        params[0] : offset y0
        params[1] : Amplitude A of the peak
        params[2] : center mu of the peak
        params[3] : width gamma of the peak
        params[4] : thermal shift of the peak
    y_sigma : np.ndarray
        The uncertainity in y data (see curve_fit documentation), to be used
        for fitting. The default is None.
    **kwargs : Any
        These are ignored.

    Returns
    -------
    popt : 1D np.ndarray
        Optimized parameters. Format
        [y0, A, mu, gamma]
    pcov : 2D np.ndarray
        Covariance matrix of the optimized parameters.

    """
    linf, lsup = get_bounds(x, y)
    liminf = [linf['y0'], linf['A'], linf['x0'],
              linf['width'], linf['nu_shift']]
    limsup = [lsup['y0'], lsup['A'], lsup['x0'],
              lsup['width'], lsup['nu_shift']]
    bounds = (liminf, limsup)

    chk_bounds(p0, liminf, limsup)

    return curve_fit(convol_lorentz, x, y, p0, sigma=y_sigma, bounds=bounds)
