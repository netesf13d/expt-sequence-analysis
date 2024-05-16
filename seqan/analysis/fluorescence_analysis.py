# -*- coding: utf-8 -*-
"""
Fitting functions used to fit the histograms built from the atomic
fluorescence signal.

Two functions are available:
    - <gausson>, a poisson signal convolved with a gaussian background.
    - <double_gaussian>, two gaussian peaks, which are faster to fit.
These are associated to the fitting procedures:
    - <fit_gausson>
    - <fit_double_gaussian>


These two fitting functions are wrapped around by analysis procedures
which encompass histogram fitting and the determination of some
characteristics: loading probability, loading threshold, signal and
background amplitude, error rates, ...
The functions are:
    - <gg_histo_analysis> histo analysis by fitting two gaussian peaks
    - <gp_histo_analysis> histo analysis by fitting the convolution of a
      gaussian background and a Poisson signal.


"""


import numpy as np
from numpy import exp
from scipy.optimize import curve_fit
from scipy.special import erfc
from scipy.stats import poisson

from ..config.config import THRESHOLD_LVL

# =============================================================================
# FUNCTIONS - HISTOGRAM FIT
# =============================================================================

def gausson(x: np.ndarray,
            p: float,
            mu1: float,
            sigma1: float,
            mu2: float,
            A: float=1)-> np.ndarray:
    """
    Scaled probability distribution of a Poisson fluorescence signal,
    appearing with probability p, on top on a gaussian background signal.

    It corresponds to the (scaled) probability distribution of the
    following random variable.
    signal = (1-p)*X + p*(X + Y)
    with X ~ N[mu1, sigma1] ; Y ~ Poisson[mu2]

    More explicitly,
    gausson(k) = (1-p) * N[mu1, sigma1](k)
                 + p * sum(Poisson[mu2](j) * N[mu1, sigma1](k-j), j=0..k)

    Parameters
    ----------
    x : np.ndarray of int
        The value(s) at which the distribution is evaluated.
    p : float
        Weight of the Poisson fluorescence signal. Interpreted as the
        probability of loading of atoms in the trap.
    mu1 : float
        Mean of the gaussian background signal.
    sigma1 : float
        Standard deviation of the the gaussian background signal.
    mu2 : float
        The parameter of the Poisson fluorescence signal. Represents both
        the mean and the variance of the signal.
    A : float, optional
        Factor to rescale the disctribution, for instance for plotting.
        The default is 1, corresponding to a normalized probability
        distribution.

    Returns
    -------
    array-like
        The function evaluated at x.

    """
    x = np.array(x).astype(int)
    xmax = np.max(x)
    # Gaussian background without atoms
    background = (1-p) / (np.sqrt(2*np.pi) * sigma1) \
                 * exp(-(x-mu1)**2 / (2*sigma1**2))
    # Gaussian background on signal
    gsignal = np.array([exp(-(val - mu1)**2 / (2*sigma1**2))
                         for val in range(xmax+1)]) \
              / (np.sqrt(2*np.pi) * sigma1)
    # Poisson distributed Fluorescence signal
    psignal = poisson.pmf(np.arange(xmax+1), mu2)


    signal = p * np.array([np.dot(psignal[:val+1], gsignal[val:None:-1])
                           for val in x])

    return A * (background + signal)


def fit_gausson(histogram: tuple)-> tuple:
    """
    Fit a gaussian + Poisson peak from an histogram of the data using scipy
    curve_fit.

    The fitted function is :
    f(k) =
    A1 * exp(-(k-mu1)**2/(2*sigma1**2))
    + A2 * sum(exp(-(k-j-mu1)**2/(2*sigma1**2)) * exp(-mu2)*mu2**j/j!, j=0..k)

    Parameters
    ----------
    histogram : tuple
        Histogram of the data such as given by numpy.histogram.
        histogram[0] : frequencies in each bin
        histogram[1] : bin edges, bin[i] = [histogram[1][i], histogram[1][i+1]]

    Returns
    -------
    popt : 1D np.ndarray
        Optimized parameters. Format
        [Amp1, mu1, sigma1, Amp2, mu2]
    pcov : 2D np.ndarray
        Covariance matrix of the optimized parameters.

    """
    frequencies = histogram[0]
    bin_edges = histogram[1]
    bin_centers = [(bin_edges[i+1] + bin_edges[i])//2
                   for i in range(len(bin_edges)-1)]

    AMax = np.sum(frequencies)
    muMin = np.min(bin_centers)
    muMax = np.max(bin_centers)

    ## Clip extremes values to have robust initial guess
    CLIPPING_THRESHOLD = 5
    # muMin
    clipInd = 0
    clip = frequencies[0]
    while clip < CLIPPING_THRESHOLD:
        clipInd += 1
        clip += frequencies[clipInd]
    muMin2 = bin_centers[clipInd]
    # muMax
    clipInd = 0
    clip = frequencies[-1]
    while clip < CLIPPING_THRESHOLD:
        clipInd += 1
        clip += frequencies[-1 - clipInd]
    muMax2 = bin_centers[-1 - clipInd]

    # Initial guess
    p_0 = 0.5
    mu1_0 = 5*muMin2/6 + muMax2/6
    sigma1_0 = np.sqrt(mu1_0)
    mu2_0 = 4/6 * (muMax2 - muMin2)
    A_0 = np.max(frequencies) * sigma1_0 * np.sqrt(2*np.pi) / p_0

    p0 = np.array([p_0, mu1_0, sigma1_0, mu2_0, A_0])

    # Bounds
    # 0 < p <= 1
    # muMin <= mu1 <= muMax
    # 0 < sigma1 <= muMax
    # muMin <= mu2 <= muMax
    # 1 <= A <= 100*AMax
    liminf = [0, muMin, 0, 0, 1]
    limsup = [1, muMax, muMax, 2*AMax, 100*AMax]
    bounds = (liminf, limsup)

    return curve_fit(gausson, bin_centers, frequencies, p0, bounds=bounds)


def double_gaussian(x: np.ndarray,
                    A1: float, mu1: float, sigma1: float,
                    A2: float, mu2: float, sigma2: float)-> np.ndarray:
    """
    Two gaussian peaks, of respective amplitude/mean/std deviation
    A1/mu1/sigma1 and A2/mu2/sigma2, evaluated at x.
    """
    return A1 * exp(-(x-mu1)**2 / (2*sigma1**2)) \
              + A2 * exp(-(x-mu2)**2 / (2*sigma2**2))


def fit_double_gaussian(histogram: tuple)-> tuple:
    """
    Fit a double gaussian peak from an histogram of the data using scipy
    curve_fit.

    The fitted function is :
        f(x) = A1 * exp(-(x-mu1)**2/(2*sigma1**2))
               + A2 * exp(-(x-mu2)**2/(2*sigma2**2))

    Parameters
    ----------
    histogram : tuple
        Histogram of the data such as given by numpy.histogram.
        histogram[0] : frequencies in each bin
        histogram[1] : bin edges, bin[i] = [histogram[1][i], histogram[1][i+1]]

    Returns
    -------
    popt : 1D np.ndarray
        Optimized parameters. Format
        [Amp1, mu1, sigma1, Amp2, mu2, sigma2]
    pcov : 2D np.ndarray
        Covariance matrix of the optimized parameters.

    """
    frequencies = histogram[0]
    bin_edges = histogram[1]
    bin_centers = [np.float64(bin_edges[i+1] + bin_edges[i])/2.
                   for i in range(len(bin_edges)-1)]

    AMax = np.max(frequencies)
    muMin = np.min(bin_centers)
    muMax = np.max(bin_centers)

    ## Clip extremes values to have robust initial guess
    CLIPPING_THRESHOLD = 5
    # muMin
    clipInd = 0
    clip = frequencies[0]
    while clip < CLIPPING_THRESHOLD:
        clipInd += 1
        clip += frequencies[clipInd]
    muMin2 = bin_centers[clipInd]
    # muMax
    clipInd = 0
    clip = frequencies[-1]
    while clip < CLIPPING_THRESHOLD:
        clipInd += 1
        clip += frequencies[-1 - clipInd]
    muMax2 = bin_centers[-1 - clipInd]

    ## Initial guess
    A1_0 = np.max(frequencies)
    mu1_0 = 5*muMin2/6 + muMax2/6
    sigma1_0 = np.sqrt(mu1_0)

    A2_0 = A1_0
    mu2_0 = muMin2/6 + 5*muMax2/6
    sigma2_0 = np.sqrt(mu2_0)

    p0 = np.array([A1_0, mu1_0, sigma1_0, A2_0, mu2_0, sigma2_0])

    # Bounds
    # 0 < A1, A2 <= AMax
    # muMin <= mu1, mu2 <= muMax
    # 0 < sigma1, sigma2 <= muMax
    liminf = [0, muMin, 0, 0, muMin, 0]
    limsup = [2*AMax, muMax, muMax, 2*AMax, muMax, muMax]
    bounds = (liminf, limsup)

    return curve_fit(double_gaussian,
                     bin_centers,
                     frequencies,
                     p0,
                     bounds=bounds)


# =============================================================================
# FUNCTIONS - HISTOGRAM ANALYSIS
# =============================================================================


def gg_histo_analysis(data: np.ndarray, binning: int=15)-> dict:
    """
    Gauss-Gauss histogram analysis.
    Analysis of data by fitting histogram data with two gaussian peaks.
    (see <double_gaussian>)

    The relevant data relative to maximum likelihood estimation is stored
    in a dictionnary 'res' such that res[entry] is an array of values
    relative to 'entry' containing values for each dataset in data.
    For instance, res['mu1'][i] is the mean of background peak for i-th
    dataset data[i, :].

    Note that the result of the fitting procedure is dependent on the binning
    used to make histograms.

    Parameters
    ----------
    data : 2D array
        List of data to estimate the probability distribution from.
        data[i, j] corresponds to j-th data of i-th ROI.
    binning : int, optional
        Used in determining the number of bins for the histograms. Basically
        represents the average number of points per bin. Higher means less
        bins. The default is 15.

    Returns
    -------
    res : dict
        Dict containing the results of maxlikelihood estimation. Keys are:
        - 'mode' : str; Type of the fit data, set to 'dgauss'
        - 'A1' : 1D array; Amplitude of the background peak.
        - 'mu1' : 1D array; Mean of the background peak.
        - 'sigma1' : 1D array; Std dev of the background peak.
        - 'A2' : 1D array; Amplitude of the signal peak.
        - 'mu2' : 1D array; Mean of the signal peak.
        - 'sigma2' : 1D array; Std dev of the signal peak.
        - 'bins' : int; Number of bins of the histogram from which the
                   gauss peaks have been fitted.
        - 'p_th' : 1D array; The theoretical probability of atom loading,
                   computed from fitting.
        - 'p_exp' : 1D array; The experimental probability of atom loading,
                    computed as the fraction of data > threshold.
        - 'threshold' : 1D array; Estimation of the level of signal to
                        distinguish the presence of an atom.
        - 'threshold_safe' : 1D array; Safe estimation of the level of
                            signal to distinguish an atom. Reduces
                            false positives.
        - 'falseneg' : 1D array; Probability of false negative.
        - 'falsepos' : 1D array; Probability of false positive.
        - 'A1err' : 1D array; curve_fit squared error on the amp of the
                    background peak.
        - 'mu1err' : 1D array; curve_fit squared error on the mean of the
                     background peak.
        - 'sigma1err' : 1D array; curve_fit squared error on the std dev of
                        the background peak.
        - 'A2err' : 1D array; curve_fit squared error on the amplitude of
                    the signal peak.
        - 'mu2err' : 1D array; curve_fit squared error on the mean of the
                     signal peak.
        - 'sigma2err' : 1D array; curve_fit squared error on the std dev of
                        the signal peak.

    """
    nbROI = data.shape[0]
    nbData = data.shape[1]
    nbBins = min(50, nbData // binning)

    popt = np.empty((nbROI, 6))
    pcov = np.empty((nbROI, 6))
    thresholds = np.empty((nbROI,), dtype=int)

    for i in range(nbROI):
        histo = np.histogram(data[i], nbBins)
        temp_popt, temp_pcov = fit_double_gaussian(histo)
        popt[i, :] = temp_popt
        pcov[i, :] = np.diag(temp_pcov)

    res = {'mode' : 'dgauss', 'bins' : nbBins,
           'A1' : popt[:, 0], 'mu1' : popt[:, 1], 'sigma1' : popt[:, 2],
           'A1err' : pcov[:, 0], 'mu1err' : pcov[:, 1], 'sigma1err' : pcov[:, 2],
           'A2' : popt[:, 3], 'mu2' : popt[:, 4], 'sigma2' : popt[:, 5],
           'A2err' : pcov[:, 3], 'mu2err' : pcov[:, 4], 'sigma2err' : pcov[:, 5]}

    # Estimation of the threshold
    for i in range(nbROI):
        params = (res['A1'][i], res['mu1'][i], res['sigma1'][i],
                  res['A2'][i], res['mu2'][i], res['sigma2'][i])
        x = np.arange(int(res['mu1'][i]), int(res['mu2'][i]), 1)
        try:
            thresholds[i] = x[np.argmin(double_gaussian(x, *params))]
        except ValueError: # mu1 > mu2 because of shitty data
            thresholds[i] = 0 # NULL value
    res['threshold'] = thresholds
    res['threshold_safe'] = ((1-THRESHOLD_LVL)*res['mu1']
                             + THRESHOLD_LVL*res['mu2']).astype(int)

    # Set theoretical loading probability p = A2*sigma2/(A1*sigma1 + A2*sigma2)
    res['p_th'] = res['A2'] * res['sigma2'] \
                  / (res['A1'] * res['sigma1'] + res['A2'] * res['sigma2'])
    # Set experimental loading probability sum(data > threshold)
    res['p_exp'] = np.sum(data > thresholds.reshape((nbROI, 1)), axis=1) \
                   / nbData

    # Estimation of false positive and false negative probabilities
    res['falseneg'] = res['p_th'] / 2 \
                      * erfc((res['mu2']-res['threshold'])
                             / (np.sqrt(2)*res['sigma2']))
    res['falsepos'] = (1-res['p_th']) / 2 \
                      * erfc((res['threshold']-res['mu1'])
                             / (np.sqrt(2)*res['sigma1']))

    # Theoretical Poisson width and Poisson width form fit
    res['pwidth_th'] = np.sqrt(res['mu2'] - res['mu1'])
    res['pwidtherr_th'] = np.sqrt((res['mu1err'] + res['mu2err']))/2
    res['pwidth_fit'] = np.sqrt(res['sigma2']**2 - res['sigma1']**2)
    res['pwidtherr_fit'] = np.sqrt((res['sigma1err'] + res['sigma2err']))/2

    return res


def gp_histo_analysis(data: np.ndarray, binning: int=15)-> dict:
    """
    Gauss-Poisson histogram analysis.
    Analysis of data by fitting histogram data with a gaussian background
    convoluted with a Poisson signal. (see <gausson>)

    The relevant data relative to maximum likelihood estimation is stored
    in a dictionnary 'res' such that res[entry] is an array of values
    relative to 'entry' containing values for each dataset in data.
    For instance, res['mu1'][i] is the mean of background peak for i-th
    dataset data[i, :].

    Note that the result of the fitting procedure is dependent on the binning
    used to make histograms.

    Parameters
    ----------
    data : 2D array
        List of data to estimate the probability distribution from.
        data[i, j] corresponds to j-th data of i-th ROI.
    binning : int, optional
        Used in determining the number of bins for the histograms. Basically
        represents the average number of points per bin. Higher means less
        bins. The default is 15.

    Raises
    ------
    RuntimeError
        If the curve fit fails. Fitting is carried with different binnings
        for the histograms in case of failure, the error is raised if all
        tentatives fail.

    Returns
    -------
    res : dict
        Dict containing the results of maxlikelihood estimation. Keys are:
        - 'mode' : str; Type of the fit data, set to 'gausson'
        - 'p_th' : 1D array; The theoretical probability of atom loading,
                   computed from fitting.
        - 'p_exp' : 1D array; The experimental probability of atom loading,
                    computed as the fraction of data > threshold.
        - 'perr' : 1D array; curve_fit squared error on the atom loading
                   probability
        - 'mu1' : 1D array; Mean of the background peak.
        - 'mu1err' : 1D array; curve_fit squared error on the mean of the
                     background peak.
        - 'sigma1' : 1D array; Std dev of the background peak.
        - 'sigma1err' : 1D array; curve_fit squared error on the std dev of
                        the background peak.
        - 'mu2' : 1D array; Mean of the signal peak.
        - 'mu2err' : 1D array; curve_fit squared error on the mean of the
                     signal peak.
        - 'A' : 1D array; Amplitude of the fitted function (maps
                probabilities to counts)
        - 'Aerr' : 1D array; curve_fit squared error on the amplitude of
                   the fitted function
        - 'bins' : int; Number of bins of the histogram from which the
                   gauss peaks have been fitted.
        - 'A1' : 1D array; Amplitude of the background peak.
        - 'A2' : 1D array; Amplitude of the signal peak.
        - 'threshold' : 1D array; Estimation of the level of signal to
                        distinguish the presence of an atom.
        - 'threshold_safe' : 1D array; Safe estimation of the level of
                            signal to distinguish an atom. Reduces
                            false positives.
        - 'falseneg' : 1D array; Probability of false negative.
        - 'falsepos' : 1D array; Probability of false positive.

    """
    nbROI = data.shape[0]
    nbData = data.shape[1]
    nbBins = min(50, nbData // binning)

    popt = np.empty((nbROI, 5))
    pcov = np.empty((nbROI, 5))
    thresholds = np.empty((nbROI,), dtype=int)
    falseneg = np.empty((nbROI,), dtype=float)

    for i in range(nbROI):
        histo = np.histogram(data[i], nbBins)
        print('fitting...')
        temp_popt, temp_pcov = fit_gausson(histo)
        print('fit ok')
        popt[i, :] = temp_popt
        pcov[i, :] = np.diag(temp_pcov)

    res = {'mode' : 'gausson', 'bins' : nbBins,
           'p_th' : popt[:, 0], 'mu1' : popt[:, 1], 'sigma1' : popt[:, 2],
           'perr' : pcov[:, 0], 'mu1err' : pcov[:, 1], 'sigma1err' : pcov[:, 2],
           'mu2' : popt[:, 3], 'A' : popt[:, 4],
           'mu2err' : pcov[:, 3], 'Aerr' : pcov[:, 4]}

    # Amplitude of the two peaks
    res['A1'] = (1 - res['p_th']) * res['A']
    res['A2'] = res['p_th'] * res['A']

    # Estimation of the threshold
    for i in range(nbROI):
        params = (res['p_th'][i], res['mu1'][i], res['sigma1'][i],
                  res['mu2'][i], res['A'][i])
        x = np.arange(int(res['mu1'][i]),
                      int(res['mu1'][i] + res['mu2'][i]), 1)
        thresholds[i] = x[np.argmin(gausson(x, *params))]
    res['threshold'] = thresholds
    res['threshold_safe'] = (
        (1-THRESHOLD_LVL)*res['mu1'] + THRESHOLD_LVL*(res['mu2']+res['mu1'])
        ).astype(int)

    # Set experimental loading probability sum(data > threshold)
    res['p_exp'] = np.sum(data > thresholds.reshape((nbROI, 1)), axis=1) \
                   / nbData

    # Estimation of false positive and false negative probabilities
    for i in range(nbROI):
        # Cutoff at max values that poisson signal can take
        x = np.arange(0, 2*res['mu2'][i], 1)
        # Probability that background is < threshold - (mu + x)
        Fgauss = 1/2 * erfc(((res['mu1'][i] + x) - res['threshold'][i])
                            / (np.sqrt(2)*res['sigma1'][i]))
        # Probability that the signal is x photons
        Fpoisson = poisson.pmf(x, res['mu2'][i])
        # The total probability is the sum of different probabilities
        falseneg[i] = res['p_th'][i] * np.dot(Fgauss, Fpoisson)
    res['falseneg'] = falseneg

    res['falsepos'] = (1-res['p_th']) / 2 \
                      * erfc((res['threshold']-res['mu1'])
                             / (np.sqrt(2)*res['sigma1']))

    return res

