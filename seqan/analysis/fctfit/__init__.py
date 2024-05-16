# -*- coding: utf-8 -*-
"""
Data fitting functions.
"""

from .periodic_fct import (
    damped_sin, p0est_damped_sin, fit_damped_sin,
    gauss_damped_sin, p0est_gauss_damped_sin, fit_gauss_damped_sin,
    rabi_osc, p0est_rabi_osc, fit_rabi_osc,
    )
from .peak_fct import (
    lorentz, p0est_lorentz, fit_lorentz,
    gauss, p0est_gauss, fit_gauss,
    mult_gauss, p0est_mult_gauss, fit_mult_gauss,
    squared_sinc, p0est_squared_sinc, fit_squared_sinc,
    voigt, p0est_voigt, fit_voigt,
    )
from .polynomial_fct import (
    parabola, p0est_parabola, fit_parabola,
    affine, p0est_affine, fit_affine,
    )
from .special_fct import (
    decay_exp, p0est_decay_exp, fit_decay_exp,
    decay_exp_0, p0est_decay_exp_0, fit_decay_exp_0,
    convol_lorentz, p0est_convol_lorentz, fit_convol_lorentz,
    damped_arcsine_sin, p0est_damped_arcsine_sin, fit_damped_arcsine_sin,
    gauss_damped_arcsine_sin, p0est_gauss_damped_arcsine_sin, fit_gauss_damped_arcsine_sin,
    inverse_lorentz, p0est_inverse_lorentz, fit_inverse_lorentz,
    arcsine_lorentz,
    )


# =============================================================================
#
# =============================================================================

fctfitDict = {
    # Oscillating functions
    'damped_sin': damped_sin,
    'gauss_damped_sin': gauss_damped_sin,
    'rabi_osc': rabi_osc,
    # Peak functions
    'lorentz': lorentz,
    'gauss': gauss,
    'mult_gauss': mult_gauss,
    'squared_sinc': squared_sinc,
    'voigt': voigt,
    # polynomials
    'parabola': parabola,
    'affine': affine,
    # Special functions
    'decay_exp': decay_exp,
    'decay_exp_0': decay_exp_0,
    'convol_lorentz': convol_lorentz,
    'damped_arcsine_sin': damped_arcsine_sin,
    'gauss_damped_arcsine_sin': gauss_damped_arcsine_sin,
    'inverse_lorentz': inverse_lorentz,
    'arcsine_lorentz': arcsine_lorentz,
    }


def fctfit_pnames(model: str, **kwargs)-> tuple[str]:
    """
    TODO doc
    Get a tuple of fitting parameters names for the specified model.
    Additionnal metaparameters may have to be provided.

    Parameters
    ----------
    model : str
        The model that is to be used for fitting data. Currently
        available models are:
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
    # Oscillating functions
    if model == 'damped_sin':
        return ("y0", "A", "nu", "tau", "phi")
    elif model == 'gauss_damped_sin':
        return ("y0", "A", "nu", "tau", "phi")
    elif model == 'rabi_osc':
        return ("y0", "nu", "delta", "phi")
    # Peak functions
    elif model == 'lorentz':
        return ("y0", "A", "mu", "gamma")
    elif model == 'gauss':
        return ("y0", "A", "mu", "sigma")
    elif model == 'mult_gauss':
        try:
            nbPeaks = kwargs['nbpeaks']
        except KeyError:
            raise KeyError("Fitting multiple gaussian peaks requires passing "
                           "`nbpeaks`: int > 0 as a fit metaparameter")
        if type(nbPeaks) is not int:
            raise TypeError("`nbpeaks` metaparameter must be of type int")
        if nbPeaks <= 0:
            raise ValueError("`nbpeaks` metaparameter must be > 0")
        p = ["y0"]
        for i in range(nbPeaks):
            p.extend((f"A_{i}", f"mu_{i}", f"sigma_{i}"))
        return tuple(p)
    elif model == 'squared_sinc':
        return ("y0", "A", "x0", "gamma")
    elif model == 'voigt':
        return ("y0", "A", "mu", "sigma", "gamma")
    # Polynomials
    elif model == 'parabola':
        return ("y0", "x0", "A")
    elif model == 'affine':
        return ("a", "b")
    # Special functions
    elif model == 'decay_exp':
        return ("y0", "A", "tau")
    elif model == 'decay_exp_0':
        return ("A", "tau")
    elif model == 'convol_lorentz':
        return ("y0", "A", "mu", "gamma", "muT")
    elif model == 'damped_arcsine_sin':
        return ("y0", "A", "nu", "tau", "phi", "dnu")
    elif model == 'gauss_damped_arcsine_sin':
        return ("y0", "A", "nu", "tau", "phi", "dnu")
    elif model == 'inverse_lorentz':
        return ("y0", "A", "mu", "gamma", "dmu", "kappa")
    else:
        raise ValueError(f"Fitting model `{model}` does not exist.")


def fctfit_setup(model: str)-> tuple:
    """
    TODO doc
    Fetch the functions necessary for data fitting of a given model.

    Parameters
    ----------
    model : str
        The model that is to be used for fitting data. Details are given
        in function <pnames> documentation.

    Raises
    ------
    KeyError
        If the given model is invalid.

    Returns
    -------
    dict
        The necessary functions for data fitting:
        - 'fit' : the fitting function
        - 'p0est' : fuction for initial parameters estimation
                    (only for curve fitting)

    """
    # Oscillating functions
    if model == 'damped_sin':
        return {'p0est': p0est_damped_sin,
                'fit': fit_damped_sin}
    elif model == 'gauss_damped_sin':
        return {'p0est': p0est_gauss_damped_sin,
                'fit': fit_gauss_damped_sin}
    elif model == 'rabi_osc':
        return {'p0est': p0est_rabi_osc,
                'fit': fit_rabi_osc}
    # Peak functions
    elif model == 'lorentz':
        return {'p0est': p0est_lorentz,
                'fit': fit_lorentz}
    elif model == 'gauss':
        return {'p0est': p0est_gauss,
                'fit': fit_gauss}
    elif model == 'mult_gauss':
        return {'p0est': p0est_mult_gauss,
                'fit': fit_mult_gauss}
    elif model == 'squared_sinc':
        return {'p0est': p0est_squared_sinc,
                'fit': fit_squared_sinc}
    elif model == 'voigt':
        return {'p0est': p0est_voigt,
                'fit': fit_voigt}
    # Polynomials
    elif model == 'parabola':
        return {'p0est': p0est_parabola,
                'fit': fit_parabola}
    elif model == 'affine':
        return {'p0est': p0est_affine,
                'fit': fit_affine}
    # Special functions
    elif model == 'decay_exp':
        return {'p0est': p0est_decay_exp,
                'fit': fit_decay_exp}
    elif model == 'decay_exp_0':
        return {'p0est': p0est_decay_exp_0,
                'fit': fit_decay_exp_0}
    elif model == 'convol_lorentz':
        return {'p0est': p0est_convol_lorentz,
                'fit': fit_convol_lorentz}
    elif model == 'damped_arcsine_sin':
        return {'p0est': p0est_damped_arcsine_sin,
                'fit': fit_damped_arcsine_sin}
    elif model == 'gauss_damped_arcsine_sin':
        return {'p0est': p0est_gauss_damped_arcsine_sin,
                'fit': fit_gauss_damped_arcsine_sin}
    elif model == 'inverse_lorentz':
        return {'p0est': p0est_inverse_lorentz,
                'fit': fit_inverse_lorentz}
    else:
        raise ValueError(f"Fitting model `{model}` does not exist.")
