# -*- coding: utf-8 -*-
"""
Generic fit and utilities.

The utility functions are:
    - <pnames>, get the names of the fitting parameters given the model
    - <_fit_setup>, fetch fitting functions for a given model


Additionnal variables are defined:
    - fit_keys, list of keys corresponding to available fitting models


The Fit_Result class has methods:
    - <_check_data>, check that the input data has the correct format

    - <get_fct>, get the function associated to a given model
    - <fit>, fit the data according to a given model

    - <get_dict>, get the Fit_Result in dict format
    - <savebin>, save the Fit_Result as a npz archive
    - <savetxt>, save the fitted parameters in text format

    - <plot>, plot the data along with the fitted model
    - <mosaic_plot>, plot the given fitted parameter as a mosaic


The available analytical fitting functions are:
Oscillating functions:
    - damped_sin, exponentially damped sinus
    - gauss_damped_sin, gaussian damped sinus
    - rabi_osc, theoretical Rabi oscillation
Peak functions:
    - lorentz, lorentzian peak
    - gauss, gaussian peak
    - mult_gauss, multiple gaussian peaks, NOT FUNCTIONAL
    - squared_sinc, squared sinus cardinal (theoretical spectral lineshape)
    - voigt, a Voigt profile
Polynomials:
    - parabola, parabola A(x - x0)^2 + y0
    - affine, affine function a*x + b
Special functions:
    - decay_exp, decaying exponential
    - convol_lorentz, convolution of a lorentzian with a thermal
      distribution of the peak center
    - damped_arcsine_sin, exp damped sin convolved with arcsine distribution
    - inverse_lorentz, lorentzian with peak frequency convolved with an
      inverse distribution

The available simulation fits are:
    - sim_rr, to fit the atom temperature from a release-recapture
      experiment
    - sim_traposc, to fit the waist from a trap oscillation experiment
    - sim_bobosc, to fit BoB power from a BoB oscillation measurement


TODO
- gendoc
"""

import warnings
from pathlib import Path
from typing import Union

import numpy as np

from ..fileio import save_to_txt
from .fctfit import fctfit_pnames, fctfit_setup
from .fctfit import fctfitDict as fctDict
from .simfit import sim_rr, sim_traposc, sim_bobosc
from ..plots import Plot


# =============================================================================
#
# =============================================================================

fit_keys = {
    ##### Curve fit
    # Oscillating functions
    'damped_sin',
    'gausss_damped_sin',
    'rabi_osc',
    # Peak functions
    'lorentz',
    'gauss',
    'mult_gauss',
    'squared_sinc',
    'voigt',
    # Polynomials
    'parabola',
    'affine',
    # Special functions
    'decay_exp',
    'convol_lorentz',
    'damped_arcsine_sin',
    'inverse_lorentz',
    ##### simulations
    'sim_rr',
    'sim_traposc',
    'sim_bobosc',
    }


def pnames(model: str, **kwargs)-> tuple[str]:
    """
    TODO update
    Get a tuple of fitting parameters names for the specified model.
    Additionnal metaparameters may have to be provided.

    Parameters
    ----------
    model : str
        The model that is to be used for fitting data.
    **kwargs : Any
        Fitting metaparameters. For instance, in multiple gaussians
        fitting, the number of peaks must be specified as `nbpeaks`.
        'nbpeaks': int > 0.

    Raises
    ------
    ValueError
        If provided fitting model is not implemented.
    KeyError
        If a fitting metaparameter is missing.

    Returns
    -------
    tuple[str]
        Tuple (`p1`, `p2`, ..., `pk`) containing the names of the fitted
        parameters (same order as the fit results). The length gives the
        number of fitted parameters.

    """
    # Simulations
    if model == 'sim_rr':
        return ("y0", "A", "T",)
    elif model == 'sim_traposc':
        return ("y0", "A", "w",)
    elif model == 'sim_bobosc':
        return ("y0", "A", "P0",)
    # Function fitting
    elif model in fctDict.keys():
        return fctfit_pnames(model, **kwargs)
    else:
        raise ValueError(f"Fitting model `{model}` does not exist; available "
                         f"models are {fit_keys}")


# =============================================================================
#
# =============================================================================

def _fit_setup(model: str)-> tuple:
    """
    Fetch the functions necessary for data fitting of a given model.

    Parameters
    ----------
    model : str
        The model that is to be used for fitting data.

    Raises
    ------
    KeyError
        If the given model is invalid.

    Returns
    -------
    mode : {'function', 'simulation'}
        The type of fitting procedure.
    dict
        The necessary functions for data fitting:
        - 'fit' : the fitting function
        - 'p0est' : fuction for initial parameters estimation
                    (only for curve fitting)

    """
    # Simulation
    if model == 'sim_rr':
        return 'simulation', {'fit': sim_rr,}
    elif model == 'sim_traposc':
        return 'simulation', {'fit': sim_traposc,}
    elif model == 'sim_bobosc':
        return 'simulation', {'fit': sim_bobosc,}
    # Function fitting
    elif model in fctDict.keys():
        return 'function', fctfit_setup(model)
    else:
        raise ValueError(f"Fitting model `{model}` does not exist; available "
                         f"models are {fit_keys}")


# =============================================================================
# Fit_Result class
# =============================================================================

class Fit_Result():
    """
    Inspired from lmfit object ModelResult.
    It should extend it by encapsulating the results of Monte Carlo fits
    """
    default_info = {
        'name': "fitted data",
        'accessor': "unknown",
        'varunit': "none",
        'varname': "none",
        }


    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 y_sigma: np.ndarray = None,
                 info: dict = None,):
        """
        TODO doc

        Parameters
        ----------
        x : 1D np.ndarray
            x-values at which y data are given. The abscissa.
        y : 2D np.ndarray
            y-values, corresponding to sets of experimental data. The ordinate.
            y[i, j] = y-value at x[j] of dataset i
        y_sigma : 2D np.ndarray or None
            The uncertainity in y data (see curve_fit documentation).
            Same shape as y. The default is None.
        info : dict, optional
            Additional info on the data to be fitted
            (name, varname, varunit, ...). The default is None.

        Attributes
        ----------
        mode : {'function', 'simulation'},
            type of fitting procedure
        model : str, the name of the fitting model
        pnames : tuple of str, names of fitted parameters
        If mode is 'function':
        p0 : 2D np.ndarray, initial estimated parameters
            (might be useful for debugging)
        failedfit : 1D np.ndarray dtype bool,
            failedfit[i] True if fit of ROI i data failed.
        popt : 2D np.array dtype float (or np.nan)
            popt[i, j] j-th optimized parameter of i-th ROI fit,
            np.nan if fit failed or invalid ROI data.
        pcov : 3D np.ndarray dtype float (or np.nan)
            pcov[i, j, k] covariance between popt j and k of
            i-th ROI fit, np.nan if fit failed or invalid ROI data.
        stdev : 2D np.array dtype float (or np.nan)
            stdev[i, j] std deviation of j-th popt of i-th ROI fit,
            np.nan if fit failed or invalid ROI data.
        If mode is 'simulation':
        xsim: 1D np.ndarray, the x values at which the
            simulated data to plot are evaluated
        ysim: 2D np.ndarray, simulated data to plot
            ysim[i, j] simulated data for ROI i at xsim[j]

        """
        self._check_data(x, y, y_sigma)

        # Fitted data
        self.info = self.default_info | info
        self.x = x # 1D np.ndarray
        self.y = y # 2D np.ndarray
        if y_sigma is None:
            self.y_sigma = np.array([[]])
        else:
            self.y_sigma = y_sigma # 2D np.ndarray
        self.nbfit = len(y)

        # model
        self.model = "" # str
        self.mode = "" # {"function", "simulation"}
        self.pnames = () # tuple
        self.p0 = np.array([]) # 2D np.ndarray, initial parameters
        self.model_kw = {} # dict

        # fit results
        self.success = np.array([]) # 1D np.ndarray of bool
        self.popt = np.array([]) # 1D np.ndarray, fitted parameters
        self.pcov = np.array([]) # 2D np.ndarray, covariance matrix of fitted parameters
        self.stdev = np.array([]) # 1D np.ndarray, std deviation of fitted parameters
        self.xsim = np.array([]) #
        self.ysim = np.array([]) #


    def _check_data(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 y_sigma: np.ndarray,):
        """
        Check the sanity of data when intanciating a Fit_Result.
        """
        if x.size <= 1:
            raise ValueError("impossible to fit a single data point")
        if y.ndim != 2:
            raise ValueError(f"y should have dim 2, has dim {y.ndim}")
        if y_sigma is not None and (s:=y_sigma.shape) != (r:=y.shape):
            raise ValueError(
                f"y_sigma shape {s} must be equal to that of y, {r}")


    ########## Core fitting routine ##########

    def models(self):
        """
        A list of the available models
        """
        models_info = \
        """
===== Simulations =====
- `sim_rr` : Monte-Carlo simulation of release-recapture
           of a thermal distribution of atoms in gaussian
           tweezers.
- `sim_traposc` : Monte-Carlo simulation of a trap oscillation
                experiment for a thermal distribution of atoms
                in gaussian tweezers.
- `sim_bobosc` : Monte-Carlo simulation of a BoB oscillation experiment.
===== Function fitting =====
=== Oscillating functions ===
- `damped_sin` : Curve fit of an exponentially damped sinus
- `gauss_damped_sin` : Curve fit of a gauss-damped sinus
- `rabi_osc` : Curve fit of the theoretical rabi oscillation
=== peak functions ===
- `lorentz` : Curve fit of a lorentzian peak
- `gauss` : Curve fit of a gaussian peak
- `mult_gauss` : Curve fit of multiple gaussian peaks. NOT FUNCTIONAL
- `squared_sinc` : Curve fit of a squared sinus cardinal
- `voigt` : curve_fit of a voigt profile
=== Polynomials ===
- `parabola` : curve fit of a parabola y0 + A*(x - x0)^2
- `affine` : curve fit of an affine function a*x + b
=== Special functions ===
- `decay_exp` : curve fit of a decaying exponential with offset
- `decay_exp_0` : curve fit of a decaying exponential with zero offset
- `convol_lorentz` : Curve fit of the convolution of a lorentzian
  with a thermal distribution of the peak center
- `damped_arcsine_sin` : Curve fit of a damped sinus with frequency
  convolved with an arcsine distribution.
- `gauss_damped_arcsine_sin` : Curve fit of a damped sinus with frequency
  convolved with an arcsine distribution.
- `inverse_lorentz` : Curve fit of a lorentzian with frequency
  convolved with an inverse distribution.
        """
        return models_info


    def get_fct(self, model: str = None):
        """
        Get the function corresponding to a given model.
        """
        model = self.model if model is None else model
        return fctDict[model]


    def fit(self,
            model: str,
            p0: np.ndarray = None,
            model_kw: dict = None):
        """
        Use a given model to fit sets of experimental data.

        Fit sets of data y[i, :] evaluated at x[:] with the given
        model. The model can be either a function, that is fitted with
        scipy.optimize.curve_fit, or a simulation, in which case a custom
        algorithm is used.
        The results are given as a dictionary.

        If initial parameters p0 are given, they are used for curve
        fitting, otherwise a custom estimator is used.

        Parameters
        ----------
        model : str
            The model used for fitting. Call Fit_Results.models() for info.
        p0 : None or 1D array-like or 2D np.ndarray, optional
            Initial parameters for curve fit estimation.
            Can be provided either as 1D array-like (tuple, list, ..),
            in which case the parameters are used to initialize all
            curve fits, or as a 2D array (eg from previous fit results),
            in which case fit are initialized by each parameter set
            individually.
            If None is provided, a function is called to estimate initial
            parameters from experimental data. This is recommended as
            I spent lot of time building reliable estimators.
            The default is None.
            Note that p0 is not needed for simulations.
        model_kw : dict, optional
            Additional model parameters (e.g. the trap power for
            simulations). The default is None.

        """
        nbfit = self.nbfit

        # Initialize model fit
        if model_kw is None:
            model_kw = {}
        mode, fitSetup = _fit_setup(model)
        model_fit = fitSetup['fit']

        self.model = model
        self.mode = mode
        self.pnames = pnames(model, **model_kw)
        self.model_kw = model_kw
        nbParam = len(self.pnames)

        # Initial parameter management
        if mode == 'simulation': # no initial parameters for a simulation
            if p0 is not None:
                warnings.warn("Simulation fit: no initial parameters required")
            self.p0 = np.array([])
        if mode == 'function':
            # Setup initial parameters and check consistency
            if p0 is None:
                p0 = fitSetup['p0est'](self.x, self.y, **model_kw)
            else:
                p0 = np.broadcast_to(p0, (nbfit, nbParam))
            self.p0 = p0

        # init containers for common fit results
        success = np.ones((nbfit,), dtype=bool)
        popt = [None] * nbfit
        pcov = [None] * nbfit
        try:
            yerr = [np.where(self.y_sigma[i, :] == 0., 1, self.y_sigma[i, :])
                    for i in range(nbfit)]
        except IndexError:
            yerr = [None] * nbfit

        # Simulation fit
        if mode == 'simulation':
            xsim = None
            ysim = [None] * nbfit
            for i in range(nbfit):
                # Model fit
                popt[i], pcov[i], xsim, ysim[i] = \
                    model_fit(self.x, self.y[i, :], y_sigma=yerr[i], **model_kw)
                if np.any(np.isnan(pcov[i])):
                    success[i] = False
            self.xsim, self.ysim = np.array(xsim), np.array(ysim)

        # Function fit
        if mode == 'function':
            for i in range(nbfit):
                try: # Try to fit
                    popt[i], pcov[i] = \
                        model_fit(self.x, self.y[i, :], p0[i, :],
                                  y_sigma=yerr[i], **model_kw)
                except RuntimeError: # Oops, fit fucked up
                    print(f"ROI fit: fit {i} failed.")
                    success[i] = False
                    popt[i] = np.full((nbParam,), np.nan)
                    pcov[i] = np.full((nbParam, nbParam), np.nan)
            self.xsim, self.ysim = np.array([]), np.array([])

        # Compute standard deviation from covariance matrix
        stdev = [np.sqrt(np.diag(pcov[i])) for i in range(nbfit)]
        # Restructure fit results as an array
        self.success = success
        self.popt = np.array(popt)
        self.pcov = np.array(pcov)
        self.stdev = np.array(stdev)


    ########## Export data ##########

    def get_dict(self,):
        """
        Get the Fit_Result in dict format.
        """
        Fdict = {
            # Data
            'accessor': np.array(self.info['accessor']),
            'nbfit': np.array(self.nbfit),
            'xvalues': self.x,
            'varname': np.array(self.info['varname']),
            'varunit': np.array(self.info['varunit']),
            'y': self.y,
            'y_sigma': self.y_sigma,
            # Fit setup
            'model': np.array(self.model),
            'mode': np.array(self.mode),
            'pnames': np.array(self.pnames),
            'p0': self.p0,
            # Fit results
            'success': self.success,
            'popt': self.popt,
            'pcov': self.pcov,
            'stdev': self.stdev,
            'xsim': self.xsim,
            'ysim': self.ysim
            }
        model_kw = {'model_kw.' + k: np.array(v)
                    for k, v in self.model_kw.items()}
        # cast model kw from undesired dtype object to dtype str
        for k, v in model_kw.items():
            if v.dtype is np.dtype(object):
                model_kw[k] = v.astype(str)
        Fdict.update(model_kw)
        return Fdict


    def savebin(self, file: Union[str, Path]):
        """
        TODO update
        Export ROI fit results in a .npz compressed archive.

        The data is saved in a form analogous to a dict, with the following
        keys:
            - 'nbROI': np.array(nbROI), essentially an int,
                number of ROIs
            - 'shape': np.array(shape) or empty array,
                the shape of the tweezer array if defined
            - 'cframes': np.array((ref_cframe, delay_cframe)),
                the pair of frames used for statistics computation
            - 'raw_x_values': 1D np.ndarray, the raw scanned x_values
                (as set in Sequence_Setup)
            - 'varname': np.array(seqSetup.varname), essentially a str,
                the name of the scanned parameter
            - 'raw_varunit': np.array(seqSetup.varunit), essentially a str,
                the unit of the raw scanned parameter (set in Sequence_Setup)
            - 'calib': np.array(seqSetup.calib) or empty array,
                the calibration to convert units, if set
            - 'x_values': 1D np.ndarray,
                if a calibration is set, the scanned x values that were
                used for fitting
                otherwise, same as raw_x_values
            - 'varunit': np.array(seqSetup.varunit), essentially a str,
                if a calibration is set, the unit of x values used for
                fitting
                otherwise, same as raw_varunit
            - 'roifit.*', with * the keys of self.roiFit,
                the fit results given by <self.roi_fit>, for instance
                popt[i, j] = j-th optimized parameter of the fit of i-th ROI

        # The use of empty arrays avoids dtype=object arrays
        # So that np.savez_compressed does not use the unsafe pickle

        To load the data:

        with np.load(file, allow_pickle=False) as f:
            nbROI = f['nbROI']
            shape = f['shape']
            cframes = f['cframes']
            x_values = f['x_values']
            ...
            popt = f['roifit.popt']
            ...

        Raises
        ------
        AttributeError
            If no ROI fit is available to save.

        """
        if self.popt.size == 0:
            raise AttributeError("no fit to save")
        # Save as compressed archive
        np.savez_compressed(file, **self.get_dict())
        print(f"Fit results saved at: {file}")


    def savetxt(self, file: Union[str, Path]):
        """
        TODO update
        Save ROI fit data to a text file.

        Optimized values from the fitting of ROI stats are written
        to a text file. The format allows for easy import with origin.

        Raises
        ------
        AttributeError
            If no ROI fit results are available to save.

        Returns
        -------
        None.
            Fit results are saved to a text file.

        """
        if self.popt.size == 0:
            raise AttributeError("no fit to save")
        # ROI indices
        fit_idx = np.arange(self.nbfit)
        if 'position' in self.info.keys():
            fit_xloc = self.info['position'][:, 0]
            fit_yloc = self.info['position'][:, 1]
            fitInd = [fit_idx, fit_xloc, fit_yloc]
            fitNames = ['fit idx', 'x position', 'y position']
            fitUnits = ['None', 'um', 'um']
        else:
            fitInd = [fit_idx]
            fitNames = ['fit idx']
            fitUnits = ['None']

        # parameters
        pNames = []
        pUnits = []
        pData = []
        for i, pn in enumerate(self.pnames):
            pNames.extend([pn, pn + 'err'])
            pUnits.extend(['None', 'None']) # la flemme
            pData.append(self.popt[:, i])
            pData.append(self.stdev[:, i])

        names = fitNames + pNames
        units = fitUnits + pUnits
        exportData = np.array(fitInd + pData)
        save_to_txt(file, names, exportData, units)
        print(f"ROI fit results saved at: {file}")

    ########## plot ##########

    def plot(self,
             shape: tuple = None,
             roi_mapper: np.ndarray = None,
             show_p0: bool = False)-> Plot:
        """
        TODO doc

        Parameters
        ----------
        shape : tuple, optional
            Shape of the mosaic. It should match that of the tweezer array.
            The default (None) corresponds to the smallest rectangle in
            which the data can fit: ~(sqrt(nbfit), sqrt(nbfit))
        roi_mapper : 2D np.ndarray, optional
            roi_mapper[k, :] = [i, j], index coordinates of ROI dataset k
            in the array of plots.
            The default (None) is: i, j = divmod(k, shape[1]).
        show_p0 : bool, optional
            Show the function evaluated with the initial fitting paramters.
            This is useful for debugging a failed fit.
            The default is False.

        Returns
        -------
        Fplot : Plot
            The fit plot.

        """
        # initialize plot
        Fplot = Plot(self.nbfit, shape, roi_mapper)
        # plot data
        yerr = None if self.y_sigma.size == 0 else self.y_sigma
        Fplot.plot_data(self.x, self.y, yerr)
        # plot fit/simulation
        if self.mode == "function":
            fit_kw = {'label': "fit"}
            Fplot.plot_fct(self.popt, fctDict[self.model], fit_kw)
            if show_p0:
                p0_kw = {'marker': "", 'linestyle': "--",
                         'label': "initial parameters"}
                Fplot.plot_fct(self.p0, fctDict[self.model], p0_kw)
        elif self.mode == "simulation":
            sim_kw = {'marker': "", 'linestyle': "-",
                     'label': "best simulation"}
            Fplot.plot_data(self.xsim, self.ysim, plot_kw=sim_kw)
        # set title, x/y labels, numerotation
        Fplot.set_title(f"{self.info['name']}")
        Fplot.set_xlabel(f"{self.info['varname']} ({self.info['varunit']})")
        Fplot.set_ylabel(f"{self.info['accessor']}")
        Fplot.add_plot_indices()

        # set fit results
        suptext = []
        for i, pn in enumerate(self.pnames):
            p = np.mean(self.popt[:, i])
            if self.nbfit == 1:
                stdp = self.stdev[0, i]
            else:
                stdp = np.std(self.popt[:, i])
            suptext.append(f"${pn} = {p:.4} " r"\pm " f"{stdp:.4}$")

        suptext2 = [" ; ".join(suptext[3*i:3*(i+1)])
                    for i in range(len(self.pnames) // 3 + 1)]

        Fplot.fig_suptext("\n".join(suptext2), txt_kw={'fontsize': 10})

        return Fplot


    def mosaic_plot(self, param: str,
                    shape: tuple = None,
                    roi_mapper: np.ndarray = None,
                    mosaic_kw: dict = None)-> Plot:
        """
        TODO doc

        Parameters
        ----------
        param : str
            The fitted parameter to display.
            Format "stdev_*" to get the stdev of the parameter
        shape : tuple, optional
            Shape of the mosaic. It should match that of the tweezer array.
            The default (None) corresponds to the smallest rectangle in
            which the data can fit: ~(sqrt(nbfit), sqrt(nbfit))
        roi_mapper : 2D np.ndarray, optional
            roi_mapper[k, :] = [i, j], index coordinates of ROI dataset k
            in the mosaic.
            The default (None) is: i, j = divmod(k, shape[1]).
        mosaic_kw : dict, optional
            kwargs passed to the matplotlib.pcolormesh function.
            The default is None.

        Raises
        ------
        ValueError
            - Requested parameter not available to plot.
            - Incorrectly set shape or roi_mapper

        """
        p = param.split('_', 1)[-1]
        if p not in self.pnames:
            raise ValueError(f"parameter `{param}` not available to plot")
        i = self.pnames.index(p)
        if "stdev" in param:
            rdata = self.stdev[:, i]
        else:
            rdata = self.popt[:, i]

        nbfit = self.nbfit
        if shape is None:
            shape = (int(np.ceil(np.sqrt(nbfit))),
                     int(np.ceil(nbfit / np.ceil(np.sqrt(nbfit)))))
        elif len(shape) != 2 or np.prod(shape) < nbfit:
            raise ValueError("shape must be of length 2 with product > "
                             f"{nbfit}, got {shape}")
        if roi_mapper is None:
            roi_mapper = np.array([divmod(i, shape[1]) for i in range(nbfit)],
                              dtype=int)
        elif np.any(roi_mapper >= shape):
            raise ValueError("roi_mapper does not map within an array of shape "
                             f"{shape}")
        # Re-orient the roi_mapper: the origin of pcolormesh is lower left
        # while the rest of the plot set the origin upper left
        roi_mapper = np.copy(roi_mapper) # avoid changing the original object
        roi_mapper[:, 0] = shape[0]-1 - roi_mapper[:, 0]


        data = np.full(shape, np.nan, dtype=float)
        for k, rd in enumerate(rdata):
            data[roi_mapper[k, 0], roi_mapper[k, 1]] = rd

        #
        Mplot = Plot(nbplot=1)
        Mplot.plot_colormap(data, mosaic_kw)
        Mplot.set_title(f"{self.info['name']}")

        # set fit results
        suptext = \
            r"$\langle " f"{param}" r"\rangle = " f"{np.mean(data):.4}" \
            "\ ;\ " r"\mathrm{stdev}(" f"{param}) = {np.std(data):.4}" r"$"
        Mplot.fig_suptext(suptext, txt_kw={'fontsize': 12})

        return Mplot







