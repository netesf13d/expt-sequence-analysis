# -*- coding: utf-8 -*-
"""
Polynomial functions for data fitting.

For sequence data, the fits consist in three functions:
    - <fct>, the fitted function itself
    - <p0est_fct>, to make a fist guess of the fitted parameters from the
      data
    - <fit_fct>, to fit the data
The available fitting functions are:
    - <parabola>, parabola f(x) = y0 + A * (x - x0)**2
    - <affine>, affine function f(x) = a*x + b


TODO gendoc
"""

import sys

import numpy as np
from scipy.optimize import curve_fit

from .utils import chk_bounds


# =============================================================================
# ========== Parabola ==========
# =============================================================================

def parabola(x: np.ndarray,
             y0: float,
             x0: float,
             A: float)-> np.ndarray:
    """
    Parabola.

    f(x) = y0 + A * (x - x0)**2

    Parameters
    ----------
    x : np.ndarray
        Values at which the function is evaluated.
    y0, x0, A : float
        offset, center, amplitude

    Returns
    -------
    np.ndarray
        Function evaluated at x.

    """
    return y0 + A * (x - x0)**2

parabola.pnames = ("y0", "x0", "A")
parabola.latex = r"$f(x) = y_0 + A \left(x - x_0\right)^2$"


def p0est_parabola(x: np.ndarray,
                   y: np.ndarray,
                   **kwargs)-> np.ndarray:
    """
    Estimate initial parameters for datasets y[i, :] evaluated at x:
        - y0 is the value at the parabola extremum
        - x0 is the abscissa of the parabola's extremum
        - A0 is the curvature (y_edge - y0) / (x_edge - x0)**2

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
        p0 = [[y0_0, x0_0, A0_0],
              [y0_1, x0_1, A0_1],
              ...]

    """
    nbstep = len(x)
    nbdata = len(y)
    curv = np.sign(y[:, 0] + y[:, -1] - 2 * y[:, nbstep//2])
    i0 = np.where(curv >= 0,
                  np.argmin(y, axis=1),
                  np.argmax(y, axis=1))

    y0 = y[np.arange(nbdata), i0]
    x0 = x[i0]

    iext = np.where(i0 > nbstep//2, 0, nbstep-1)

    A0 = (y[np.arange(nbdata), iext] - y0) / (x[iext] - x0)**2

    return np.array([y0, x0, A0]).transpose()


def fit_parabola(x: np.ndarray,
                 y: np.ndarray,
                 p0: np.ndarray,
                 y_sigma: np.ndarray = None,
                 **kwargs):
    """
    Fit parabola using scipy.optimize.curve_fit

    Bounds for the optimized parameters are computed from data:
        - ymin/2 <= y0 <= 1.5*ymax
        - xmin <= x0 <= xmax
        - -1.5*Amax <= A <= 1.5*Amax
    (Amax is the maximal curvature allowed by the data)

    Parameters
    ----------
    x : np.ndarray
        x-values at which data is available.
    y : np.ndarray
        y-values corresponding to measured values.
    p0 : np.ndarray
        Initial guess for the various parameters. The format is:
        params[0] : offset y0
        params[1] : center of he parabola
        params[2] : curvature
    y_sigma : np.ndarray
        The uncertainity in y data (see curve_fit documentation), to be used
        for fitting. The default is None.
    **kwargs : Any
        These are ignored.

    Returns
    -------
    popt : 1D np.ndarray
        Optimized parameters. Format
        [y0, x0, A]
    pcov : 2D np.ndarray
        Covariance matrix of the optimized parameters.

    """
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    yMax = np.max(np.abs(y))

    Amax = 1.5 * 4 * (ymax-ymin) / (xmax-xmin)**2
    # Boundaries
    # ymin/2 <= y0 <= 1.5*ymax
    # xmin <= x0 <= xmax
    # -Amax <= A <=Amax
    liminf = [-1.5*yMax, xmin, -Amax]
    limsup = [1.5*yMax, xmax, Amax]
    bounds = (liminf, limsup)
    chk_bounds(p0, liminf, limsup)

    return curve_fit(parabola, x, y, p0, sigma=y_sigma, bounds=bounds)


# =============================================================================
# ========== Affine ==========
# =============================================================================

def affine(x: np.ndarray,
           a: float,
           b: float)-> np.ndarray:
    """
    Affine function.

    f(x) = a*x + b

    Parameters
    ----------
    x : np.ndarray
        Values at which the function is evaluated.
    a, b : float
        slope, offset

    Returns
    -------
    np.ndarray
        Function evaluated at x.

    """
    return a*x + b

affine.pnames = ("a", "b")
affine.latex = r"$f(x) = ax + b$"


def p0est_affine(x: np.ndarray,
                 y: np.ndarray,
                 **kwargs)-> np.ndarray:
    """
    Estimate initial parameters for datasets y[i, :] evaluated at x:
        - a0 is the slope
        - b0 is the value at the origin

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
        p0 = [[a0_0, b0_0],
              [a0_1, b0_1],
              ...]

    """
    xmin = np.min(x)
    xmax = np.max(x)
    y0 = y[:, np.argmin(x)]
    y1 = y[:, np.argmax(x)]

    a0 = (y1 - y0) / (xmax - xmin)
    b0 = y0 - a0 * xmin

    return np.array([a0, b0]).transpose()


def fit_affine(x: np.ndarray,
               y: np.ndarray,
               p0: np.ndarray,
               y_sigma: np.ndarray = None,
               **kwargs):
    """
    Fit affine function using scipy.optimize.curve_fit

    Bounds for the optimized parameters are computed from data:
        - ymin/2 <= a <= 1.5*ymax
        - xmin <= b <= xmax

    Parameters
    ----------
    x : np.ndarray
        x-values at which data is available.
    y : np.ndarray
        y-values corresponding to measured values.
    p0 : np.ndarray
        Initial guess for the various parameters. The format is:
        params[0] : slope a
        params[1] : value at origin b
    y_sigma : np.ndarray
        The uncertainity in y data (see curve_fit documentation), to be used
        for fitting. The default is None.
    **kwargs : Any
        These are ignored.

    Returns
    -------
    popt : 1D np.ndarray
        Optimized parameters. Format
        [a, b]
    pcov : 2D np.ndarray
        Covariance matrix of the optimized parameters.

    """
    dy = (y - y.reshape((-1, 1))) \
        / np.where((a := x - x.reshape((-1, 1))) == 0, 1, a)
    amax = np.max(np.abs(dy))
    bmax = np.max(np.abs(y)) + amax * np.max(np.abs(x))

    # Boundaries
    # amin <= a <= amax
    # -bmin <= b <= amax
    liminf = [-amax, -bmax]
    limsup = [amax, bmax]
    bounds = (liminf, limsup)
    chk_bounds(p0, liminf, limsup)

    return curve_fit(affine, x, y, p0, sigma=y_sigma, bounds=bounds)
