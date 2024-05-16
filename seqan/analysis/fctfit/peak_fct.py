# -*- coding: utf-8 -*-
"""
Peak functions for data fitting.

For sequence data, the fits consist in three functions:
    - <fct>, the fitted function itself
    - <p0est_fct>, to make a fist guess of the fitted parameters from the
      data
    - <fit_fct>, to fit the data
The available fitting functions are:
    - <lorentzian>, lorentzian peak
    - <gaussian>, gaussian peak
    - <mult_gaussian>, multiple gaussian peaks
    - <squared_sinc>, squared sinus cardinal (theoretical spectral lineshape)
    - <voigt>, voigt profile


TODO gendoc
"""

import sys
from functools import partial

import numpy as np
from numpy import exp
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
# from scipy.interpolate import interp1d

from .utils import chk_bounds, get_bounds

# =============================================================================
# ========== Lorentzian ==========
# =============================================================================

def lorentz(x: np.ndarray,
            y0: float,
            A: float,
            mu: float,
            gamma: float)-> np.ndarray:
    """
    Single lorentzian peak, with a global offset y0.
    f(x) = y0 + A / (1 + ((x-mu)/gamma)**2)

    FWHM = 2*gamma

    Parameters
    ----------
    x : np.ndarray
        The positions at which the function is evaluated.
    y0, A, mu, gamma : float
        Offset, amplitude, peak center, HWHM

    Returns
    -------
    result : np.ndarray
        Lorentzian peaks evaluated at position(s) x.

    """
    return y0 + A / (1 + ((x-mu)/gamma)**2)

lorentz.pnames = ("y0", "A", "mu", "gamma")
lorentz.ptypes = ("y0", "A", "x0", "width")
lorentz.latex = r"$f(x) = y_0 + \frac{A}{\gamma ^2 + (x-\mu)^2}$"


def p0est_lorentz(x: np.ndarray,
                  y: np.ndarray,
                  **kwargs)-> np.ndarray:
    """
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
    ymin = np.min(np.abs(y), axis=1)
    ymax = np.max(np.abs(y), axis=1)

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

    return np.array([y0, A0, mu0, gamma0]).transpose()


def fit_lorentz(x: np.ndarray,
                y: np.ndarray,
                p0: np.ndarray,
                y_sigma: np.ndarray = None,
                **kwargs):
    """
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
    liminf = [linf['y0'], linf['A'], linf['x0'], linf['width']]
    limsup = [lsup['y0'], lsup['A'], lsup['x0'], lsup['width']]
    bounds = (liminf, limsup)

    chk_bounds(p0, liminf, limsup)

    return curve_fit(lorentz, x, y, p0, sigma=y_sigma, bounds=bounds)


# =============================================================================
# ========== Single gaussian peak ==========
# =============================================================================

def gauss(x: np.ndarray,
          y0: float,
          A: float,
          mu: float,
          sigma: float)-> np.ndarray:
    """
    Gaussian peak, with a global offset.
    f(x) = y0 + A * exp(-(x-mu)**2/(2*sigma**2))

    FWHM = 2*sqrt(2*ln(2))*sigma ~= 2.355*sigma

    Parameters
    ----------
    x : np.ndarray,
        Positions at which the function is evaluated.
    y0, A, mu, sigma : float
        Offset, ampplitude, peak center, peak width (std dev)

    Returns
    -------
    np.ndarray
        Gaussian peaks evaluated at position(s) x.

    """
    return y0 + A * np.exp(-(x-mu)**2/(2*sigma**2))

gauss.pnames = ("y0", "A", "mu", "sigma")
gauss.ptypes = ("y0", "A", "x0", "width")
gauss.latex = r"$f(x) = y_0 + A e^{-\frac{(x-\mu)^2}{2 \sigma^2}}$"


def p0est_gauss(x: np.ndarray,
                y: np.ndarray,
                **kwargs)-> np.ndarray:
    """
    Estimate initial parameters for datasets y[i, :] evaluated at x:
        - y0 is the average of y values at scan edges
        - A0 is the maximum difference between y and y0
        - mu0 is the x value at extremal y value
        - sigma0 is proportional to the number of y values above
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
        p0 = [[y0_0, A0_0, mu0_0, sigma0_0],
              [y0_1, A0_1, mu0_1, sigma0_1],
              ...]

    """
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(np.abs(y), axis=1)
    ymax = np.max(np.abs(y), axis=1)

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
    sigma0 = np.sum((y - y0_bc)/A_bc > THRESHOLD, axis=1) * step

    return np.array([y0, A0, mu0, sigma0]).transpose()


def fit_gauss(x: np.ndarray,
              y: np.ndarray,
              p0: np.ndarray,
              y_sigma: np.ndarray = None,
              **kwargs):
    """
    Fit single gaussian peak using scipy.optimize.curve_fit

    Bounds for the optimized parameters are computed from data:
        - 0 <= y0 <= ymax
        - -1.5*ymax <= A0 <= 1.5*ymax
        - xmin <= mu <= xmax
        - 0 <= sigma <= span

    Parameters
    ----------
    x : np.ndarray
        x-values at which data is available.
    y : np.ndarray
        y-values corresponding to measured values.
    p0 : np.ndarray
        Initial guess for the various parameters. The format is:
        p0[0] : offset y0
        p0[1] : Amplitude A the peak
        p0[2] : center mu the peak
        p0[3] : width (std dev) sigma the gaussian
    y_sigma : np.ndarray
        The uncertainity in y data (see curve_fit documentation), to be used
        for fitting. The default is None.
    **kwargs : Any
        These are ignored.

    Returns
    -------
    popt : 1D np.ndarray
        Optimized parameters. Format
        [y0, A, mu, sigma]
    pcov : 2D np.ndarray
        Covariance matrix of the optimized parameters.

    """
    linf, lsup = get_bounds(x, y)
    liminf = [linf['y0'], linf['A'], linf['x0'], linf['width']]
    limsup = [lsup['y0'], lsup['A'], lsup['x0'], lsup['width']]
    bounds = (liminf, limsup)

    chk_bounds(p0, liminf, limsup)

    return curve_fit(gauss, x, y, p0, sigma=y_sigma, bounds=bounds)


# =============================================================================
# ========== Multiple gaussian peaks ==========
# =============================================================================
# !!! not functional

def mult_gauss(nbPeaks: int,
               x: np.ndarray,
               y0: float,
               *params)-> np.ndarray:
    """
    Multiple gaussian peaks, with a global offset.
    f(x) = y0 + sum(A_i * exp(-(x-mu_i)**2/(2*sigma_i**2)), i = 1..nbPeaks)

    The number of peaks is set by nbPeaks, the parameters of the various peaks
    is given by params: 3*nbPeaks parameters, respectively amplitude, center,
    width (std deviation).

    Parameters
    ----------
    nbPeaks : int
        Number of gaussian peaks.
    x : np.ndarray,
        The positions at which the function is evaluated.
    y0 : float
        Global offset.
    params : array-like
        Parameters of various peaks. The format is:
        params[3*i + 0] : Amplitude of i-th peak
        params[3*i + 1] : center of i-th peak
        params[3*i + 2] : width of i-th peak (std dev of the gaussian)

    Returns
    -------
    result : np.ndarray
        Gaussian peaks evaluated at position(s) x.

    """
    result = np.full(x.shape, y0, dtype=float)
    for i in range(nbPeaks):
        A = params[3*i]
        mu = params[3*i + 1]
        sigma = params[3*i + 2]
        result += A * exp(-(x-mu)**2/(2*sigma**2))
    return result

mult_gauss.pnames = ()
mult_gauss.latex = \
    r"$f(x) = y_0 + \sum_{i} A_i e^{-\frac{(x-\mu _i)^2}{2 \sigma _i^2}}$"


def p0est_mult_gauss(x: np.ndarray,
                     y: np.ndarray,
                     **kwargs):
    # TODO
    try:
        nbPeaks = kwargs['nbpeaks']
    except KeyError:
        raise KeyError("Fitting multiple gaussian peaks requires passing "
                       "`nbpeaks`: int > 0 as a fit metaparameter")
    if type(nbPeaks) is not int:
        raise TypeError("`nbpeaks` metaparameter must be of type int")
    if nbPeaks <= 0:
        raise ValueError("`nbpeaks` metaparameter must be > 0")
    return np.ones((y.shape[0], 1+3*nbPeaks))


def fit_mult_gauss(x: np.ndarray,
                   y: np.ndarray,
                   p0: np.ndarray,
                   y_sigma: np.ndarray = None,
                   **kwargs):
    """
    Fit multiple gaussian peaks using scipi.optimize.curve_fit.

    Bounds for the optimized parameters are computed from data:
        - 0 <= y0 <= ymax
        - -1.5*ymax <= A <= 1.5*ymax
        - xmin <= mu <= xmax
        - 0 <= sigma <= span

    Parameters
    ----------
    x : np.ndarray
        x-values at which data is available.
    y : np.ndarray
        y-values corresponding to measured values.
    p0 : np.ndarray
        Initial guess for the various parameters. The format is:
        p0[0] : offset y0
        p0[1 + 3*i + 0] : Amplitude A_i of i-th peak
        p0[1 + 3*i + 1] : center mu_i of i-th peak
        p0[1 + 3*i + 2] : width sigma_i of i-th peak (std dev of the gaussian)
    y_sigma : np.ndarray
        The uncertainity in y data (see curve_fit documentation), to be used
        for fitting. The default is None.
    **kwargs : Any
        'nbpeaks' must be specified as an int > 0.

    Returns
    -------
    popt : 1D np.ndarray
        Optimized parameters. Format
        [y0, A_0, mu_0, sigma_0, ..., A_i, mu_i, sigma_i, ...]
    pcov : 2D np.ndarray
        Covariance matrix of the optimized parameters.

    """
    try:
        nbPeaks = kwargs['nbpeaks']
    except KeyError:
        raise KeyError("Fitting multiple gaussian peaks requires passing "
                       "`nbpeaks`: int > 0 as a fit metaparameter")
    if type(nbPeaks) is not int:
        raise TypeError("`nbpeaks` metaparameter must be of type int")
    if nbPeaks <= 0:
        raise ValueError("`nbpeaks` metaparameter must be > 0")

    xmin = np.min(x)
    xmax = np.max(x)
    ymax = np.max(y)
    if ymax == 0.:
        ymax = 1.

    Amax = 1.5*ymax
    # Boundaries : adjust at will, but keep them tight
    # 0 <= offset <= ymax
    # -Amax <= A <= Amax
    # xmin <= mu <= xmax
    # 0 <= sigma <= xmax-xmin kHz
    liminf = [0] + [-Amax, xmin, 0]*nbPeaks
    limsup = [ymax] + [Amax, xmax, xmax - xmin]*nbPeaks
    bounds = (liminf, limsup)
    chk_bounds(p0, liminf, limsup)

    # def mult_gauss(x, offset, *params):
    #     return mult_gaussians(x, nbPeaks, offset, *params)

    return curve_fit(partial(mult_gauss, nbPeaks),
                     x, y, p0, sigma=y_sigma, bounds=bounds)


# =============================================================================
# ========== Squared sinc ==========
# =============================================================================

def squared_sinc(x: np.ndarray,
                 y0: float,
                 A: float,
                 x0: float,
                 gamma: float)-> np.ndarray:
    """
    Square of sinus cardinal with offset y0.

    f(x) = y0 + A * sin(Pi*(x-x0)/gamma)**2 / (Pi*(x-x0)/gamma)**2

    f(x) = 0 <=> x-x0 = n * Pi*gamma, n integer
    FWHM ~= 0.885*gamma

    Parameters
    ----------
    x : np.ndarray
        Values at which the function is evaluated.
    y0, A, x0, gamma : float
        Offset, amplitude, peak center, peak width.

    Returns
    -------
    np.ndarray
        Function evaluated at x.

    """
    return y0 + A * np.sinc((x - x0)/gamma)**2

squared_sinc.pnames = ("y0", "A", "x0", "gamma")
squared_sinc.ptypes = ("y0", "A", "x0", "width")
squared_sinc.latex = \
    r"$f(x) = y_0 + A \frac{\mathrm{sinc}(\pi /\gamma (x-x_0))}{\pi / \gamma}$"


def p0est_squared_sinc(x: np.ndarray,
                       y: np.ndarray,
                       **kwargs)-> np.ndarray:
    """
    Estimate initial parameters for datasets y[i, :] evaluated at x:
        - y0 is the average of y values at scan edges
        - A0 is the maximum difference between y and y0
        - x0 is the x value at extremal y value
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
        p0 = [[y0_0, A0_0, x0_0, gamma0_0],
              [y0_1, A0_1, x0_1, gamma0_1],
              ...]

    """
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(np.abs(y), axis=1)
    ymax = np.max(np.abs(y), axis=1)

    step = (xmax-xmin) / (len(x)-1)

    # y0 as the average of y at extremal scan points
    y0 = (y[:, 0] + y[:, -1]) / 2

    # A as the difference between the extremal y and y0
    A0 = np.where(y0-ymin > ymax-y0, ymin-y0, ymax-y0)

    # x0 at the extremal value of y
    x0 = np.where(A0 < 0,
                  x[np.argmin(y, axis=1)],
                  x[np.argmax(y, axis=1)])

    # gamma as the width of the peak above a certain threshold
    THRESHOLD = 0.5
    y0_bc = y0.reshape((len(y0), 1))
    A_bc = A0.reshape((len(A0), 1))
    gamma0 = np.sum((y - y0_bc)/A_bc > THRESHOLD, axis=1) * step

    return np.array([y0, A0, x0, gamma0]).transpose()


def fit_squared_sinc(x: np.ndarray,
                     y: np.ndarray,
                     p0: np.ndarray,
                     y_sigma: np.ndarray = None,
                     **kwargs):
    """
    Fit squared sinus cardinal peak using scipy.optimize.curve_fit

    Bounds for the optimized parameters are computed from data:
        - 0 <= y0 <= ymax
        - -1.5*ymax <= A <= 1.5*ymax
        - xmin <= x0 <= xmax
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
        params[2] : center x0 of the peak
        params[3] : width gamma of the peak
    y_sigma : np.ndarray
        The uncertainity in y data (see curve_fit documentation), to be used
        for fitting. The default is None.
    **kwargs : Any
        These are ignored.

    Returns
    -------
    popt : 1D np.ndarray
        Optimized parameters. Format
        [y0, A, x0, gamma]
    pcov : 2D np.ndarray
        Covariance matrix of the optimized parameters.

    """
    linf, lsup = get_bounds(x, y)
    liminf = [linf['y0'], linf['A'], linf['x0'], linf['width']]
    limsup = [lsup['y0'], lsup['A'], lsup['x0'], lsup['width']]
    bounds = (liminf, limsup)

    chk_bounds(p0, liminf, limsup)

    return curve_fit(squared_sinc, x, y, p0, sigma=y_sigma, bounds=bounds)


# =============================================================================
# ========== Voigt profile ==========
# =============================================================================

def voigt(
        x: np.ndarray,
        y0: float,
        A: float,
        mu: float,
        sigma: float,
        gamma: float)-> np.ndarray:
    """
    Voigt profile, with amplitude A and global offset y0.
    f(x) = y0 + A * Voigt(x - mu, sigma, gamma)

    FWHM ~= 0.5346*FWHM_L + sqrt(0.2166*FWHM_L^2 + FWHM_G^2)
         ~= 1.0692*gamma + sqrt(0.8664*gamma^2 + 5.5452*sigma^2)

    Parameters
    ----------
    x : np.ndarray
        The positions at which the function is evaluated.
    y0, A, mu : float
        Offset, amplitude, peak center
    sigma : float
        Standard deviation of the Normal distribution part
    gamma : float
        HWHM of the Cauchy distribution part

    Returns
    -------
    result : np.ndarray
        Voigt profile evaluated at position(s) x.

    """
    return y0 + A * voigt_profile(x - mu, sigma, gamma)

voigt.pnames = ("y0", "A", "mu", "sigma", "gamma")
voigt.ptypes = ("y0", "A", "x0", "width", "width")
voigt.latex = \
    r"$f(x) = y_0 + A \int_{-\infty}^{\infty} \mathrm{d}u " \
    r"\frac{e^{-\frac{(x-\mu)^2}{2 \sigma^2}}}{\gamma ^2 + (x-u-\mu)^2}$"


def p0est_voigt(
        x: np.ndarray,
        y: np.ndarray,
        **kwargs)-> np.ndarray:
    """
    Estimate initial parameters for datasets y[i, :] evaluated at x:
        - y0 is the average of y values at scan edges
        - A0 is the maximum difference between y and y0
        - mu0 is the x value at extremal y value
        - sigma0 is FWHM_0 / (2*sqrt(2*log(2)))
        - gamma0 is FWHM_0 / 2
    FWHM_0 = 2 / (sqrt(5) + 1) * FWHM_voigt, with FWHM_voigt estimated from
    the data. The gaussian and lorentzian FWHM are assumed equal and
    evaluated using the above formula (see wikipedia: Voigt profile)

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
        p0 = [[y0_0, A0_0, mu0_0, sigma0_0, gamma0_0],
              [y0_1, A0_1, mu0_1, sigma0_1, gamma0_1],
              ...]

    """
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(np.abs(y), axis=1)
    ymax = np.max(np.abs(y), axis=1)

    step = (xmax-xmin) / (len(x)-1)

    # y0 as the average of y at extremal scan points
    y0 = (y[:, 0] + y[:, -1]) / 2

    # A as the difference between the extremal y and y0
    A0 = np.where(y0-ymin > ymax-y0, ymin-y0, ymax-y0)
    # x0 at the extremal value of y
    mu0 = np.where(A0 < 0,
                   x[np.argmin(y, axis=1)],
                   x[np.argmax(y, axis=1)])
    # gamma = FWHM_lorentz / 2; sigma = FWHM_gauss / (2*sqrt(2*log(2)))
    # Assume FWHM_lorentz = FWHM_gauss = FWHM_0,
    # with FWHM_0 = 2 / (sqrt(5) + 1) * FWHM_voigt (see wiki: voigt profile)
    THRESHOLD = 0.5 # a
    y0_bc = y0.reshape((len(y0), 1))
    A_bc = A0.reshape((len(A0), 1))
    FWHM = np.sum((y - y0_bc)/A_bc > THRESHOLD, axis=1) * step
    FWHM_0 = 2 / (np.sqrt(5) + 1) * FWHM
    sigma0 = FWHM_0 / (2*np.sqrt(2*np.log(2)))
    gamma0 = FWHM_0 / 2
    A0 /= voigt_profile(0, sigma0, gamma0) # Need to rescale

    return np.array([y0, A0, mu0, sigma0, gamma0]).transpose()


def fit_voigt(
        x: np.ndarray,
        y: np.ndarray,
        p0: np.ndarray,
        y_sigma: np.ndarray = None,
        **kwargs):
    """
    Fit Voigt profile using scipy.optimize.curve_fit.

    Bounds for the optimized parameters are computed from data:
        - 0 <= y0 <= ymax
        - -1.5*ymax <= A0 <= 1.5*ymax
        - xmin <= mu <= xmax
        - 0 <= sigma <= span
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
        params[3] : width (std dev) sigma the gaussian
        params[4] : HWHM of the Cauchy distribution part
    y_sigma : np.ndarray
        The uncertainity in y data (see curve_fit documentation), to be used
        for fitting. The default is None.
    **kwargs : Any
        These are ignored.

    Returns
    -------
    popt : 1D np.ndarray
        Optimized parameters. Format
        [y0, A, mu, sigma, gamma]
    pcov : 2D np.ndarray
        Covariance matrix of the optimized parameters.

    """
    linf, lsup = get_bounds(x, y)
    liminf = [linf['y0'], linf['A'], linf['x0'], linf['width'], linf['width']]
    limsup = [lsup['y0'], lsup['A'], lsup['x0'], lsup['width'], lsup['width']]
    bounds = (liminf, limsup)

    chk_bounds(p0, liminf, limsup)

    return curve_fit(voigt, x, y, p0, sigma=y_sigma, bounds=bounds)



