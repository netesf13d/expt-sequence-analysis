# -*- coding: utf-8 -*-
"""
Analysis routines.

These routines yield a result that is essentially different from the input
data (for instance it cannot be plotted together with the input data).
Moreover, the analysis can be carried on various types of data: fluorescence
signal, recapture probability, fit results, ...
The analysis routines are:
    - <fft_analysis>, compute the FFT of a set of time series
    - <moment_analysis>, compute mean, variance, skewness and kurtosis of
      a dataset
They are called by the wrapper:
    - <analysis>
Additionnal variables are defined:
    - analysis_keys, list of keys corresponding to available analysis routines


"""

import numpy as np
from numpy.fft import rfft


# =============================================================================
# Wrapper function to call analysis routines
# =============================================================================

analysis_keys = [
    "fft",
    "stats"
    ]


def analysis(data: tuple, routine: str)-> dict:
    """
    Wrapper function to call analysis routines.

    Parameters
    ----------
    data : tuple
        DESCRIPTION.
    routine : str
        Analysis routine to be called. Available are:
        - 'fft', calls fft_analysis
        - 'stats', calls moment_analysis

    Raises
    ------
    ValueError
        If the routine called is invalid.

    Returns
    -------
    dict
        The data returned by the corresponding analysis routine.

    """
    if routine == 'fft':
        return fft_analysis(*data)
    elif routine == 'stats':
        return moment_analysis(*data)
    else:
        raise ValueError(
            f"analysis routine {routine} does not exist, "
            f"must be in {analysis_keys}"
            )


# =============================================================================
# Analysis routines
# =============================================================================

def fft_analysis(x: np.ndarray,
                 y: np.ndarray)-> tuple:
    """
    TODO mod the way peaks indices and freqs are returned
    FFT analysis of time series y sampled at times x.

    Compute the 1D FFT of the time series y[i, :] to return:
        - The physical frequencies (unit: 1 / time unit)
        - The norm of the Fourier components
        - The phase of the Fourier components, suitably offset to account
          for nonzero xmin

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
        The keys are:
        - freqs : 1D np.ndarray
            The physical frequencies, unit: 1 / time unit
        - norm : 2D np.ndarray
            ffty_norm[i, j] = norm of j-th Fourier component of time series i.
        - phi : 2D np.ndarray
            phi[i, j] = phase of j-th Fourier component of time series i.
            The phase is offset to account for the nonzero xmin
        - power : 2D np.ndarray
            The square of the norm.
        - peak_indices : list[1D np.ndarray]
            The indices of maxima of the fft norm ordered in decreasing
            order. For instance, for any i and j,
            norm[peak_indices[i][j]] > norm[peak_indices[i][j+1]]
        - peak_freqs : list[1D np.ndarray]
            The corresponding frequencies of the maxima of the fft norm
            ordered in decreasing order.
            peak_freqs[i][j] = freqs[peak_indices[i][j]]

    """
    nbdata = len(y)
    xmin = np.min(x)
    xmax = np.max(x)
    step = (xmax - xmin)/(len(x)-1)

    ## Get frequency and phase from the maximal component of the fft
    pad = 10 # pad the data with zeros to increase frequency resolution
    n = pad * len(x)
    y0 = np.reshape(np.mean(y, axis=1), (nbdata, 1)) # the mean to be substracted
    ffty = rfft((y-y0), n=n, axis=1, norm='forward')

    ## compute array of physical frequencies
    k = ffty.shape[-1]
    freqs = np.linspace(0, (k-1) / (n*step), k, endpoint=True)

    ## Compute the norm of fft components
    ffty_norm = np.abs(ffty)
    ffty_pwr = ffty_norm**2

    ## Corresponding phase, including the xmin offset
    phi = np.angle(ffty) - 2*np.pi * freqs * xmin
    # reduce modulo 2*Pi
    phi = phi - 2*np.pi * (phi//(2*np.pi))

    ## Get the peaks positions
    peaks = (ffty_pwr > np.roll(ffty_pwr, 1, axis=1)) \
                & (ffty_pwr > np.roll(ffty_pwr, -1, axis=1))
    peaks[:, 0] = False
    peaks[:, -1] = False

    peak_indices = []
    peak_freqs = []
    for i in range(nbdata):
        temp, = np.nonzero(peaks[i, :])
        peak_indices.append(temp[np.argsort(ffty_pwr[i, temp])[-1::-1]])
        peak_freqs.append(freqs[peak_indices[i]])

    return {'freqs': freqs,
            'norm': ffty_norm,
            'phase': phi,
            'power': ffty_pwr,
            'peak_indices': peak_indices,
            'peak_freqs': peak_freqs}


def moment_analysis(val: np.ndarray)-> dict:
    """
    Compute the 4 lowest moments of the dataset:
        - mean, mu = E[X]
        - variance, sigma^2 = E[(X - mu)^2]
        - standard deviation, sigma = sqrt(E[(X - mu)^2])
        - skewness, E[(X - mu)^3] / sigma^3
        - kurtosis, E[(X - mu)^4] / sigma^4

    Parameters
    ----------
    val : 2D np.ndarray
        The set of values which moment are computed.

    Returns
    -------
    dict
        keys are:
        - mean
        - standard_deviation
        - variance
        - skewness
        - kurtosis

    """
    mean = np.mean(val, axis=1)
    mu = mean.reshape((-1, 1))
    stdev = np.std(val, axis=1)
    variance = np.var(val, axis=1)
    skewness = np.sum((val-mu)**3, axis=1) / (stdev**3 * val.shape[1])
    kurtosis = np.sum((val-mu)**4, axis=1) / (stdev**4 * val.shape[1])

    return {'mean': mean,
            'standard_deviation': stdev,
            'variance': variance,
            'skewness': skewness,
            'kurtosis': kurtosis}

