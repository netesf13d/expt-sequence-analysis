# -*- coding: utf-8 -*-
"""
Monte-Carlo simulation routines to fit data.
The fitting functions proper are:
    - <sim_rr>, to fit the atom temperature from a release-recapture
      experiment
    - <sim_traposc>, fit trap waist from atomic oscillations in the trap
    - <sim_bobosc>, fit BoB trap power from oscillations in BoB


These functions make use of various sub-routines:
    - <fetch_calib_params>, fetch the experimental parameters required for
        fitting
    - <chisq>, compute the chi-squared distance between experimental and
        simulated data
    - <chisq_log>, compute the chi-squared distance between the log of the
       experimental and simulated data
    - <polyfit_chisq>, polynomial fit of the chi-squared to recover the
        optimal parameter and estimate its variance
    - <_scale_simfit>, fit the scaling parameters (amplitude and offset) of
        simulated data to the experimental data
    - <_core_simfit>, evaluate the values of the chi-squared by running
        simulations

TODO
- doc
- gendoc
- utility function to fetch parameters

"""

import sys
import warnings
from typing import Callable

import numpy as np
from numpy.fft import rfft
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit



from ..config.config import (
    # units
    TIME_UNIT, TEMP_UNIT, LENGTH_UNIT,
    # sim rr config
    NB_SAMPLES_RR_COARSE, NB_SAMPLES_RR,
    COARSE_TEMPERATURE_RANGE,
    SAMPLING_MODE_RR,
    #
    NB_SAMPLES_TO_COARSE, NB_SAMPLES_TO,
    COARSE_WAIST_RANGE,
    SAMPLING_MODE_TO,
    #
    NB_SAMPLES_BO,
    SAMPLING_MODE_BO,)

from ..config.config import SIMPACK_PATH
if not SIMPACK_PATH in sys.path:
    sys.path.append(SIMPACK_PATH)
try:
    from simpack import (
        build_gbeam_pot,
        build_gbeam_force,
        gbeam_sampling,
        normal_gbeam_sampler,
        Data_Grid,
        build_force,
        run_simu_rr,
        run_simu_gaussdyn,
        run_simu_bobdyn,)
except ModuleNotFoundError:
    warnings.warn("Atomic dynamics module `simpack` not found, Monte-Carlo "
                  "fitting is unavailable")
    pass

from ..calib.calib import (
    gauss_params,
    bob_params,
    LS_to_V0,
    LS_rep,
    lifetimes,
    atom_temp,
    recapture_threshold
    )


# =============================================================================
# Utilities
# =============================================================================

def fetch_calib_params(**kwargs):
    """
    TODO doc, update
    Fetch calibration parameters used for the simulations.

    Parameters
    ----------
    **kwargs : TYPE
        lifetime : callable, optional
            Lifetime of the atomic species to take into account as a
            reduction of recapture probability:
                precap(t) = precap(t) * lifetime(t)
            The default is np.inf, which corresponds to ground state atoms.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    dict
        DESCRIPTION.

    """
    # fetch repumper lightshift to get peak intensity
    try:
        LS = kwargs['lightshift']
    except KeyError:
        print("simulation: no light shift provided (key: `lightshift`). "
              f"Taking default value: {LS_rep/1e6:.4} MHz")
        LS = LS_rep
    V0 = LS * LS_to_V0
    # fetch gaussian tweezers waist
    try:
        w0 = kwargs['waist']
    except KeyError:
        print("simulation: no waist provided (key: `waist`). "
              f"Taking default value: {gauss_params['waist']*1e6:.4} um")
        w0 = gauss_params['waist'] # m
    # fetch gaussian tweezers Rayleigh range
    try:
        z_R = kwargs['z_R']
    except KeyError:
        # print("simulation: no Rayleigh range provided (key: `z_R`). "
        #       f"Taking default value: gauss_params['z_R']*1e6:.4} um")
        z_R = gauss_params['z_R'] # m
    # fetch gaussian tweezers `astigmatism parameter` dz
    try:
        dz = kwargs['dz']
    except KeyError:
        # print("simulation: no astigmatism parameter provided (key: `dz`). "
        #       "Taking default value: 0 um")
        dz = 0. # m
    # fetch bob_file path
    try:
        bob_f = kwargs['bob_file'] # Path | str
    except KeyError:
        bob_f = None
    # fetch recapture_threshold
    try:
        recap_thr = kwargs['recap_thr']
    except KeyError:
        # print("simulation: no recapture threshold provided (key: `recap_thr`)."
        #       f" Taking default value: {recapture_threshold:.4} J")
        recap_thr = recapture_threshold # J
    # fetch lifetime of the atomic state
    try:
        atomState = kwargs['atomstate']
    except KeyError:
        atomState = 'ground'
        print("rr simulation: no explicit atomic state provided "
              "(key: `atomstate`). Assuming atoms in ground state.")
    if atomState not in (keys := lifetimes.keys()):
        raise ValueError(
            f"`atomstate` metaparameter must be in {keys}, got {atomState}"
            )
    lifetime = lifetimes[atomState] # callable
    # fetch transfer efficiency
    try:
        transfer = kwargs['transfer']
    except KeyError:
        print("rr simulation: no transfer efficiency (to 52D for instance) "
              "provided (key: `transfer`). Taking default value: 1.")
        transfer = 1.
    # fetch atom temperature
    try:
        T0 = kwargs['temperature']
    except KeyError:
        print("simulation: no atom temperature provided (key: `temperature`). "
            f"Taking default value: {atom_temp*1e6:.4} uK")
        T0 = atom_temp # K
    # fetch atom temperature
    try:
        offset = kwargs['bob_offset']
    except KeyError:
        print("simulation: no BoB lateral offset provided (key: `bob_offset`). "
            f"Taking default value: {bob_params['offset']*1e6:.4} um")
        offset = bob_params['offset'] # m
    # fetch amplitude
    try:
        A = kwargs['A']
    except KeyError:
        print("simulation: no data y scale provided (key: `A`). "
              "Value will be fitted.")
        A = None
    # fetch offset
    try:
        y0 = kwargs['y0']
    except KeyError:
        print("simulation: no data y offset provided (key: `y0`). "
              "Value will be fitted.")
        y0 = None

    return {
        'V0': V0,
        'w0': w0,
        'z_R': z_R,
        'dz': dz,
        'bob_file': bob_f,
        'recap_thr': recap_thr,
        'lifetime': lifetime,
        'transfer': transfer,
        'T0': T0,
        'bob_offset': offset,
        'A': A,
        'y0': y0,
        }


# =============================================================================
# UTILITIES AND CORE FUNCTIONS
# =============================================================================

def chisq(y: np.ndarray,
          ysim: np.ndarray,
          y_sigma: np.ndarray | None = None,)-> float:
    """
    Compute the chi-squared distance between the experimental data y and
    the simulation ysim. The terms of the chi-sq are weighted by y_sigma
    if provided.

    chisq = sum((y - ysim)^2 / y_sigma^2)

    Parameters
    ----------
    y : 1D np.ndarray
        Experimental data.
    ysim : 1D np.ndarray
        Simulated data.
    y_sigma : Union[np.ndarray, None], optional
        Uncertainty in the experimental data. The default is None.

    Returns
    -------
    float
        sum((y - ysim)^2 / y_sigma^2)

    """
    if y_sigma is None:
        Chisq = np.sum((y - ysim)**2)
    else:
        Chisq = np.sum(((y - ysim) / y_sigma)**2)
    return Chisq


def chisq_log(y: np.ndarray,
              ysim: np.ndarray,
              y_sigma: np.ndarray | None = None,)-> float:
    """
    Compute the chi-squared distance between the logarithm of the
    experimental data y and the logarithm of the simulation ysim. The terms
    of the chi-sq are weighted by log(y_sigma) if provided.

    Note: this is not suitable for data that can have negative values
    but can be useful to increase the weight of the tails of probability
    distributions.

    Parameters
    ----------
    y : 1D np.ndarray
        Experimental data.
    ysim : 1D np.ndarray
        Simulated data.
    y_sigma : Union[np.ndarray, None], optional
        Uncertainty in the experimental data. The default is None.

    Returns
    -------
    float
        sum((log(y) - log(ysim))^2 / log(y_sigma)^2)

    """
    # Convert to log
    log_y = np.log10(y)
    log_ysim = np.log10(ysim)
    if y_sigma is None:
        log_y_sigma = np.ones(y.shape)
    else:
        log_y_sigma = 0.4343 * y_sigma / y
    return np.sum(((log_y - log_ysim) / log_y_sigma)**2)


def polyfit_chisq(val: np.ndarray,
                  chi2: np.ndarray,
                  deg: int = 4)-> tuple:
    """
    Fit the chi-squared with a polynomial to recover then minimum and
    curvature at the minimum.

    For the estimation of the variance, see
    - Seber, Nonlinear regression, 2003
      (theorem 2.1 pp.24, you'll have to do the math)
    - Bevington, Data Reduction and Error Analysis for the Physical Sciences
      (for the weak, no need to do the math: eq. 8.11 pp. 146)

    Parameters
    ----------
    val : 1D np.ndarray
        Values at which the chi-squared is evaluated.
    chi2 : np.ndarray
        Values of the chi-squared.
    deg : int, optional
        Degree of the polynomial fit. The default is 4.

    Returns
    -------
    vfit : float
        Value at which the polynomial fit is minimal.
    var_v : float
        Estimation of the variance of the fitted value from the curvature
        of the polynomial at the minimum.
    best_chi2 : float
        Minimum of the polynomial fit.

    """
    # Fit with polynomial
    poly = Polynomial.fit(val, chi2, deg=deg)
    dpoly = poly.deriv()
    ddpoly = dpoly.deriv()
    if np.any(np.isnan(dpoly.coef)): # fit failed: get T minimizing Chisq
        imin = np.argmin(chi2)
        best_val, best_chi2 = val[imin], chi2[imin]
        return best_val, np.nan, best_chi2
    else: # fit worked, compute derivative to get local extrema
        droots = dpoly.roots()

    # ## plot chisq, useful for debugging
    # import matplotlib.pyplot as plt
    # xx = np.linspace(np.min(val), np.max(val), 101)
    # plt.plot(val, chi2, marker="o", markersize=3)
    # plt.plot(xx, poly(xx))
    # plt.show()

    # Recover the correct minimum from the roots of the derivative
    vmin, vmax = np.min(val), np.max(val)
    filt = np.real([r for r in droots if not np.imag(r)]) # real roots
    filt = filt[np.nonzero((filt > vmin) & (filt < vmax))] # roots in range
    filt = filt[ddpoly(filt) > 0] # minima
    if filt.size > 1:
        warnings.warn(
            "Chisq polynomial fit has more than one minimum in range. "
            "Keeping first minimum.",
            RuntimeWarning)
    try:
        vfit = filt[0]
    except IndexError:
        vfit = np.nan
        warnings.warn("Chisq polynomial fit failed: no minimum in range",
                      RuntimeWarning)

    # Estimation of the variance
    ddv = ddpoly(vfit)
    best_chi2 = poly(vfit)
    var_v = 2 / ddv #

    return vfit, var_v, best_chi2, poly.coef


def _scale_simfit(x: np.ndarray,
                  y: np.ndarray,
                  y_sigma: np.ndarray,
                  ysim: np.ndarray,
                  p: tuple[float],
                  p0: tuple[float],
                  absolute_sigma: bool = False)-> tuple:
    """
    TODO doc

    Parameters
    ----------
    x, y, y_sigma : 1D np.ndarray
        Data to fit y evaluated at x with stdev y_sigma.
    ysim : np.ndarray
        DESCRIPTION.
    p : tuple[float] (y0, A)
        DESCRIPTION.
    p0 : tuple[float] (est_y0, est_A)
        DESCRIPTION.
    absolute_sigma : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    popt, pcov : np.ndarray
        DESCRIPTION.

    """

    y0, A = p
    est_y0, est_A = p0
    popt, pcov = np.zeros((2,), dtype=float), np.zeros((2, 2), dtype=float)

    if A is None and y0 is None:
        # set scale and bounds
        scale = lambda xval, offset, amplitude: amplitude * ysim + offset
        bounds = ([0., 0.], [np.inf, np.inf])
        # Curve fit
        popt, pcov = curve_fit(
            scale, x, y, p0=(est_y0, est_A), sigma=y_sigma,
            absolute_sigma=absolute_sigma, bounds=bounds)
        return popt, pcov
    elif A is None:
        scale = lambda xval, amplitude: amplitude * ysim + y0
        bounds = ([0.], [np.inf])
        [Aopt], [[Acov]] = curve_fit(
            scale, x, y, p0=(est_A,), sigma=y_sigma,
            absolute_sigma=absolute_sigma, bounds=bounds)
        popt[:] = y0, Aopt
        pcov[1, 1] = Acov
        return popt, pcov
    elif y0 is None:
        scale = lambda xval, offset: A * ysim + offset
        bounds = ([0.], [np.inf])
        [y0opt], [[y0cov]] = curve_fit(
            scale, x, y, p0=(est_y0,), sigma=y_sigma,
            absolute_sigma=absolute_sigma, bounds=bounds)
        popt[:] = y0opt, A
        pcov[0, 0] = y0cov
        return popt, pcov
    else:
        popt[:] = y0, A
        return popt, pcov


def _core_simfit(x: np.ndarray,
                 y: np.ndarray,
                 y_sigma: np.ndarray | None,
                 val: np.ndarray,
                 p: tuple,
                 sampler: Callable,
                 run_simu: Callable,
                 scaler: Callable)-> tuple:
    """
    TODO doc

    Parameters
    ----------
    x, y, y_sigma : 1D np.ndarray
        Data to fit y evaluated at x with stdev y_sigma.
    val : 1D np.ndarray
        Array of values of the Monte-Carlo parameter (temperature, ...)
        to fit.
    p : tuple (y0, A)
        DESCRIPTION.
    sampler : callable
        DESCRIPTION.
    run_simu : callable
        DESCRIPTION.
    scaler : callable
        DESCRIPTION.

    Returns
    -------
    chi2 : 1D np.ndarray
        chi-squared distance evaluated.
    popt : 1D np.ndarray, shape (2,)
        Fitted scaling parameters amplitude and offset.
    pcov : 2D np.ndarray, shape (2, 2)
        The covariance of the fitted scaling parameters.

    """
    chi2 = np.zeros_like(val)
    popt = np.zeros((len(val), 2), dtype=float)
    pcov = np.zeros((len(val), 2, 2), dtype=float)

    # import matplotlib.pyplot as plt
    # print(popt[i])
    # plt.plot(x, y); plt.plot(x, ysim_scaled); plt.show()

    for i, v in enumerate(val):
        spl = sampler(v)
        ysim = run_simu(v, spl)
        ysim_tofit = scaler(x, ysim, 0., 1.)
        popt[i, :], pcov[i, :, :] = _scale_simfit(
            x, y, y_sigma, ysim_tofit, p=p, p0=(0., 1.), absolute_sigma=True)
        ysim_scaled = scaler(x, ysim, popt[i, 0], popt[i, 1])
        chi2[i] = chisq(y, ysim_scaled, y_sigma)

    # ## plot chisq, useful for debugging
    # import matplotlib.pyplot as plt
    # plt.plot(val, chi2, marker="o", markersize=3)
    # plt.show()

    return chi2, popt, pcov


# =============================================================================
# Simulation fitting procedures
# =============================================================================

def sim_rr(x: np.ndarray,
           y: np.ndarray,
           y_sigma: np.ndarray = None,
           absolute_sigma: bool = False,
           **kwargs)-> tuple:
    """
    Fitting procedure for the atomic temperature from Monte-Carlo
    simulations of the recapture probability of a release-recapture
    experiment.

    Find the optimal temperature to fit experimental data in a range given
    by a list of temperatures.

    The best temperature is the one that minimizes the chi-squared
    distance between experimental recapture probability and simulated
    data. The minimum can be determined by two methods:
        - `best`, the temperature in temperatures that minimizes chi-sq.
          The variance is not estimated.
          Note that this method is sensitive to sampling noise.
        - `fit`, use a polynomial fit of chi-sq near its minimum to get
          the minimizing temperature and an estimation of the variance

    For each temperature, a set of samples is produced by the sampler,
    theoretical recapture probability data is computed for the given
    sequence and compared with the experimental probabilities.

    TODO doc, comment

    Parameters
    ----------
    x : 1D np.ndarray
        Release durations.
    y : 1D np.ndarray
        Recapture probabilities.
    y_sigma : 1D np.ndarray, optional
        Uncertainities on recapture probabilities. The default is None.
    **kwargs : Any
        Additional parameters for the fitting procedure:
        - recap_thr : float (SI unit: Joules)
          The recapture threshold for an atom in the trap.
        - lifetime : float (unit: config time unit)
          The lifetime of the atomic state subject to rr experiment
          (infinte for GS atoms, finite for Rydbergs)
        - transfer : float 0 < transfer < 1
          Fraction of the atoms transferred to the Rydberg state.
          Has no meaning if lifetime is not specified.
        - A : float
          The recapture efficiency at zero delay.

    Returns
    -------
    tuple
        DESCRIPTION.

    """
    scaled_x = x*TIME_UNIT
    # print(x, TIME_UNIT)
    ### Physical parameters initialization
    params = fetch_calib_params(**kwargs)
    recap_thr = params['recap_thr']
    lifetime = params['lifetime']
    transf = params['transfer']
    A = params['A']
    y0 = params['y0']

    # ydata -> reduced-y (lifetime, amplitude and offset corrected)
    # reduce = lambda x, y, y0, A: \
    #     (y - y0) / A * (transf / lifetime(x) + (1. - transf))
    # reduced-y -> ydata
    scale = lambda x, y, y0, A: \
        A * y * (transf * lifetime(x) + (1. - transf)) + y0

    ### Sampler, potential and simulation routine
    # build raw sample generator
    if SAMPLING_MODE_RR == "rejection":
        rsampler = lambda nbSamples, T: gbeam_sampling(nbSamples, T, **params)
    elif SAMPLING_MODE_RR == "normal":
        rsampler = normal_gbeam_sampler(**params)
    # setup simulation routine
    V = build_gbeam_pot(**params)
    def run_simu(temperature, samples):
        return run_simu_rr((scaled_x,), V, samples, recap_thr)

    ### Get quick estimation of the temperature
    T_array1 = np.copy(COARSE_TEMPERATURE_RANGE)
    sampler1 = lambda T: rsampler(NB_SAMPLES_RR_COARSE, T)
    chi2, _, _ = _core_simfit(
        scaled_x, y, y_sigma, T_array1, p=(y0, A),
        sampler=sampler1, run_simu=run_simu, scaler=scale)
    best_T = T_array1[np.argmin(chi2)]

    ### Try to get a more precise estimation of the temperature
    if best_T < np.max(T_array1):
        T_array2 = np.linspace(best_T - 5e-6,
                               best_T + 5e-6,
                               51, endpoint=True)
        T_array2 = T_array2[T_array2 > 0]
        sampler2 = lambda T: rsampler(NB_SAMPLES_RR, T)
        chi2, _, _ = _core_simfit(
            scaled_x, y, y_sigma, T_array2, p=(y0, A),
            sampler=sampler2, run_simu=run_simu, scaler=scale)
        Tfit, varT, _, _ = polyfit_chisq(T_array2, chi2)
    else:
        Tfit, varT = best_T, np.nan
        warnings.warn(
            "rr simulation: estimated temperature exceeds max estimable "
            f"temperature: {np.max(T_array1)*1e6:.4} uK.",
            RuntimeWarning)
    if np.isnan(varT):
        warnings.warn(
            "rr simulation: optimal temperature was not reached",
            RuntimeWarning)

    [Chi2], [spopt], [spcov] = _core_simfit(
        scaled_x, y, y_sigma, np.array([Tfit]), p=(y0, A),
        sampler=sampler2, run_simu=run_simu, scaler=scale)

    # Set-up popt, pcov
    popt = np.zeros((3,), dtype=float) # y0, A, T
    pcov = np.zeros((3, 3), dtype=float)
    popt[:2], popt[2] = spopt, Tfit
    pcov[:2, :2], pcov[2, 2] = spcov, varT
    if absolute_sigma is False:
        M, N = len(x), 3 # number of data points, number of parameters to fit
        if A is None:
            N -= 1
        if y0 is None:
            N -= 1
        pcov *= Chi2 / (M-N)

    # Fitted simulation
    sim_sampl = rsampler(NB_SAMPLES_RR, Tfit)
    xMax = np.ceil(np.max(x))
    xsim = np.linspace(0, xMax*TIME_UNIT, max(201, int(xMax+1)))
    ysim = run_simu_rr((xsim,), V, sim_sampl, recap_thr)
    ysim = scale(xsim, ysim, *popt[:2])

    # convert to temperature units
    popt[2] /= TEMP_UNIT
    pcov[2, 2] /= TEMP_UNIT**2
    return (popt, pcov, xsim/TIME_UNIT, ysim)


def sim_traposc(x: np.ndarray,
                y: np.ndarray,
                y_sigma: np.ndarray = None,
                absolute_sigma: bool = False,
                **kwargs)-> tuple:
    """
    Find the optimal waist to fit experimental data in a range given
    by a list of waists.

    The best waist minimizes the chi-squared distance between experimental
    recapture probability and simulated data. The minimum can be determined
    by two methods:
        - `best`, the waist in waists that minimizes chi-sq. The variance
          is not estimated.
          Note that this method is sensitive to sampling noise.
        - `fit`, use a polynomial fit of chi-sq near its minimum to get
          the minimizing waist and an estimation of the variance

    For each waist, a set of samples is produced by the sampler,
    theoretical recapture probability data is computed for the given
    sequence and compared with the experimental probabilities.
    TODO update, doc, comment

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION.
    y : np.ndarray
        DESCRIPTION.
    y_sigma : np.ndarray, optional
        DESCRIPTION. The default is None.
    **kwargs : TYPE
        DESCRIPTION.

    Raises
    ------
    KeyError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    xsim : TYPE
        DESCRIPTION.
    ysim : TYPE
        DESCRIPTION.

    """
    # set sequence parameters
    try: # first release duration
        T1 = kwargs['release_T1']
    except KeyError:
        raise KeyError(
            "Oscillations in gaussian tweezers: missing sequence info. "
            "1st release duration missing with key `release_T1`")
    try: # second release duration
        T2 = kwargs['release_T2']
    except KeyError:
        raise KeyError(
            "Oscillations in gaussian tweezers: missing sequence info. "
            "2nd release duration missing with key `release_T2`")
    seq = (T1, x*TIME_UNIT, T2)
    # Set trap parameters
    params = fetch_calib_params(**kwargs)
    T0 = params['T0']
    recap_thr = params['recap_thr']
    A = params['A']
    y0 = params['y0']
    var_params = lambda w: \
        params | {'waist': w, 'z_R': np.pi * w**2 / gauss_params['lambda_trap']}

    scale = lambda x, y, y0, A: A * y + y0

    ### Sampler, potential and simulation routine
    # sampler
    if SAMPLING_MODE_TO == "rejection":
        rsampler = lambda nbSamples, w: \
            gbeam_sampling(nbSamples, T0, **var_params(w))
    elif SAMPLING_MODE_TO == "normal":
        rsampler = lambda nbSamples, w: \
            normal_gbeam_sampler(**var_params(w))(nbSamples, T0)
    # Potential and force
    Vgen = lambda w: build_gbeam_pot(**var_params(w))
    Fgen = lambda w: build_gbeam_force(**var_params(w))
    # simulation routine
    substract_mean = True if y0 is None else False
    def run_simu(waist, samples):
        return run_simu_gaussdyn(seq, Vgen(waist), Fgen(waist),
                                 samples, recap_thr,
                                 substract_mean=substract_mean)

    ### Get quick estimation of the waist
    waists1 = np.copy(COARSE_WAIST_RANGE)
    sampler1 = lambda w: rsampler( NB_SAMPLES_TO_COARSE, w)
    chi2, _, _ = _core_simfit(
        x, y, y_sigma, waists1, p=(y0, A),
        sampler=sampler1, run_simu=run_simu, scaler=scale)
    best_w = waists1[np.argmin(chi2)]

    ### Try to get a more precise estimation of the waist
    if best_w < np.max(waists1):
        waists2 = np.linspace(best_w - 0.02e-6,
                              best_w + 0.02e-6,
                              31, endpoint=True)
        waists2 = waists2[waists2 > 0]
        sampler2 = lambda w: rsampler(NB_SAMPLES_TO, w)
        chi2, _, _ = _core_simfit(
            x, y, y_sigma, waists2, p=(y0, A),
            sampler=sampler2, run_simu=run_simu, scaler=scale)
        wfit, varw, _, _ = polyfit_chisq(waists2, chi2, deg=6)
    else:
        wfit, varw = best_w, np.nan
        warnings.warn(
            "trap osc simulation: estimated waist exceeds max estimable "
            f"waist: {np.max(waists1)*1e6:.4} um.",
            RuntimeWarning)
    if np.isnan(varw):
        warnings.warn(
            "trap osc simulation: optimal waist was not reached",
            RuntimeWarning)

    [Chi2], [spopt], [spcov] = _core_simfit(
        x, y, y_sigma, np.array([wfit]), p=(y0, A),
        sampler=sampler2, run_simu=run_simu, scaler=scale)

    popt = np.zeros((3,), dtype=float) # y0, A, T
    pcov = np.zeros((3, 3), dtype=float)
    popt[:2], popt[2] = spopt, wfit
    pcov[:2, :2], pcov[2, 2] = spcov, varw
    if absolute_sigma is False:
        M, N = len(x), 3 # number of data points, number of parameters to fit
        if A is None:
            N -= 1
        if y0 is None:
            N -= 1
        pcov *= Chi2 / (M-N)

    sim_sampl = sampler2(wfit)
    xMax = np.ceil(np.max(x))
    xsim = np.linspace(0, xMax*TIME_UNIT, int(4*xMax+1))
    ysim = run_simu_gaussdyn(
        (T1, xsim, T2), Vgen(wfit), Fgen(wfit), sim_sampl, recap_thr)
    ysim = scale(xsim, ysim, *popt[:2])

    # convert to length units
    popt[2] /= LENGTH_UNIT
    pcov[2, 2] /= LENGTH_UNIT**2
    return (popt, pcov, xsim/TIME_UNIT, ysim)


def sim_bobosc(x: np.ndarray,
               y: np.ndarray,
               y_sigma: np.ndarray,
               absolute_sigma: bool = False,
               **kwargs)-> tuple:
    """


    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION.
    y : np.ndarray
        DESCRIPTION.
    y_sigma : np.ndarray
        DESCRIPTION.
    absolute_sigma : bool, optional
        DESCRIPTION. The default is False.
    **kwargs : TYPE
        DESCRIPTION.

    Raises
    ------
    KeyError
        DESCRIPTION.

    Returns
    -------
    tuple
        DESCRIPTION.

    """
    ### set sequence parameters
    try: # first gauss-to-bob switch duration
        t1 = kwargs['switch_t1']
    except KeyError:
        raise KeyError(
            "Oscillations in BoB tweezers: missing sequence info. "
            "Gauss to BoB switching duration missing with key `switch_t1`")
    try: # release duration
        T = kwargs['release_T']
    except KeyError:
        raise KeyError(
            "Oscillations in BoB tweezers: missing sequence info. "
            "Release duration missing with key `release_T`")
    try: # release duration
        Ttot = kwargs['Ttot']
    except KeyError:
        raise KeyError(
            "Oscillations in BoB tweezers: missing sequence info. "
            "Total sequence duration missing with key `Ttot`")
    try: # second bob-to-gauss switch duration
        t2 = kwargs['switch_t2']
    except KeyError:
        t2 = t1
        warnings.warn(
            "No BoB to gauss switching duration provided (key `switch_t2`). "
            f"Using `switch_t1` ({t1*1e6} us) instead",
            UserWarning)
    scaled_x = x*TIME_UNIT
    # !!! this is the current implementation, that corresponds to the sequences
    delays = scaled_x - t1
    seq = (t1, delays, T, t2, Ttot)

    # Set parameters
    params = fetch_calib_params(**kwargs)
    bob_file = params['bob_file']
    T0 = params['T0']
    recap_thr = params['recap_thr']
    offset = params['bob_offset']
    A = params['A']
    y0 = params['y0']
    scale = lambda x, y, y0, A: A * y + y0

    ### build sampler
    if SAMPLING_MODE_BO == "rejection":
        sampler = lambda v: gbeam_sampling(NB_SAMPLES_BO, T0, **params)
    elif SAMPLING_MODE_BO == "normal":
        sampler = lambda v: normal_gbeam_sampler(**params)(NB_SAMPLES_BO, T0)

    ### setup simulation routine
    V = build_gbeam_pot(**params)
    with np.load(bob_file) as f:
        bob = Data_Grid(f['data'], f['pxsize'], f['loc'])
    F = lambda P: build_force(bob,
                              amplitude=P*bob_params['beta_ponder'],
                              offset=offset)
    substract_mean = True if y0 is None else False
    def run_simu(power, samples):
        return run_simu_bobdyn(seq, V, F(power), samples, recap_thr,
                               substract_mean=substract_mean)

    ### Estimate trap frequency from the fft
    pad = 5 # pad the data with zeros to increase frequency resolution
    n = pad * len(x)
    ffty = rfft((y-np.mean(y)), n=n, axis=0, norm='forward')
    kmax = np.argmax(np.abs(ffty))
    step = (np.max(scaled_x) - np.min(scaled_x))/(len(x)-1)
    est_freq = kmax / (n*step)
    P0 = (est_freq/2 / 110000)**2 # !!! 110 000 Hz/sqrt(W)
    P0 = 0.72 * P0 # empirical
    ### fit
    powers = np.linspace(P0*0.85, P0*1.15, 41, endpoint=True)
    chi2, _, _ = _core_simfit(
        scaled_x, y, y_sigma, powers, p=(y0, A),
        sampler=sampler, run_simu=run_simu, scaler=scale)

    Pfit, varP, _, _ = polyfit_chisq(powers, chi2)
    [Chi2], [spopt], [spcov] = _core_simfit(
        scaled_x, y, y_sigma, np.array([Pfit]), p=(y0, A),
        sampler=sampler, run_simu=run_simu, scaler=scale)

    popt = np.zeros((3,), dtype=float) # y0, A, T
    pcov = np.zeros((3, 3), dtype=float)
    popt[:2], popt[2] = spopt, Pfit
    pcov[:2, :2], pcov[2, 2] = spcov, varP
    if absolute_sigma is False:
        M, N = len(x), 3 # number of data points, number of parameters to fit
        if A is None:
            N -= 1
        if y0 is None:
            N -= 1
        pcov *= Chi2 / (M-N)

    sim_sampl = sampler(Pfit)
    xMin, xMax = np.floor(np.min(x)), np.ceil(np.max(x))
    xsim = np.linspace(xMin*TIME_UNIT, xMax*TIME_UNIT, int(2*(xMax-xMin)+1))
    sim_seq = (t1, xsim-t1, T, t2, Ttot)
    ysim = run_simu_bobdyn(sim_seq, V, F(Pfit), sim_sampl, recap_thr,
                           substract_mean=substract_mean)
    ysim = scale(xsim, ysim, *popt[:2])

    return (popt, pcov, xsim/TIME_UNIT, ysim)

