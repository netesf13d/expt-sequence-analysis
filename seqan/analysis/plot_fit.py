# -*- coding: utf-8 -*-
"""
Fit results displayed on plots.
The functions are:
    - <result_txt>, get the result text to be displayed on plots
Additionnal variables are defined:
    - fctLatex : LaTeX expression of the fitting functions

TODO
- gendoc
- finish implementation of voigt
"""

import sys

import numpy as np


# =============================================================================
#
# =============================================================================


fctLatex = {
    # oscillating functions
    'damped_sin':
        r"$f(t) = y_0 + A e^{-\frac{t}{\tau}} \sin(2\pi \nu t + \phi)$",
    'rabi_osc':
        r"$f(t) = y_0 - \frac{\nu}{\sqrt{\nu^2+\delta^2}}"
        r"\sin^2 \left( \pi\sqrt{\nu^2+\delta^2} t + \phi \right)$",
    # Peak functions
    'lorentz':
        r"$f(x) = y_0 + \frac{A}{\gamma ^2 + (x-\mu)^2}$",
    'gauss':
        r"$f(x) = y_0 + A e^{-\frac{(x-\mu)^2}{2 \sigma^2}}$",
    'mult_gauss':
        r"$f(x) = y_0 + \sum_{i} A_i e^{-\frac{(x-\mu _i)^2}{2 \sigma _i^2}}$",
    'squared_sinc':
        r"$f(x) = y_0 + A \frac{\mathrm{sinc}(\pi / \gamma (x-x_0))}{\pi / \gamma}$",
    # Polynomials
    'parabola':
        r"$f(x) = y_0 + A \left(x - x_0\right)^2$",
    'affine':
        r"$f(x) = ax + b$",
    # Special functions
    'decay_exp':
        r"$f(t) = y_0 + A \exp\left(-\frac{t}{\tau}\right)$",
    'convol_lorentz':
        r"$f(x) = y_0 + \frac{A\gamma}{\gamma ^2 + (x-\mu)^2}$",
    'damped_arcsine_sin':
        r"$f(t) = y_0 + A e^{-\frac{t}{\tau}} \int_{-1}^{1} \mathrm{d}y"
        r"\frac{\sin(2\pi (\nu + y\delta \nu) t + \phi)}{\pi \sqrt{1-y^2}} $",
    'inverse_lorentz':
        r"$f(x) = y_0 + \frac{A\gamma}{\kappa}\int_{e^{-\kappa}d\mu}^{d\mu}"
        r"\frac{1}{\gamma ^2 + (x-\mu -y)^2} \frac{\mathrm{d}y}{y}$",
    }


def result_txt(model: str, fitResults: np.ndarray, plotMode: str)-> np.ndarray:
    """
    Dirty function to get fit results displayed on the plots

    Parameters
    ----------
    model : str
        The model used for fitting. See <fit_setup> documentation
        for details.
    fitResults : np.ndarray
        DESCRIPTION.
    plotMode : {'one', 'many'}
        The kind of plot on which the results are displayed. There is more
        space on single plots, hence more details on fit results.

    Raises
    ------
    ValueError
        If provided fitting model is not implemented.

    Returns
    -------
    np.ndarray
        Structured result text, fed to plots.plot_one or plots.plot_many.

    """
    nbResults = len(fitResults)
    if model == 'histo' and plotMode == 'one':
        restxt = [(r"p = " + f"{pexp:.2f}", 0.57, 0.85, {'size': 10})
                  for pexp in fitResults]
        return np.array(restxt, dtype=object)[:, np.newaxis, :]
    if model == 'histo' and plotMode == 'many':
        restxt = [(r"p = " + f"{pexp:.2f}", 0.57, 0.85, {'size': 10})
                  for pexp in fitResults]
        return np.array(restxt, dtype=object)[:, np.newaxis, :]

    elif model == 'sim_rr' and plotMode == 'one':
        res = fitResults
        restxt = [(r"$T$ = {:.2f} $\pm$ {:.2f} uK"\
                   .format(res[0], res[1]),
                   0.1, 0.1, {'size': 14})]
        return np.array(restxt, dtype=object)[:, np.newaxis, :]
    elif model == 'sim_rr' and plotMode == 'many':
        restxt = [(r"T = " + f"{res[0]:.2f} +- {res[1]:.2f}",
                   0.6, 0.85, {'size': 8})
                  for res in fitResults]
        return np.array(restxt, dtype=object)[:, np.newaxis, :]

    elif model == 'sim_traposc' and plotMode == 'one':
        res = fitResults
        restxt = [(r"$w$ = {:.3f} $\pm$ {:.3f} um"\
                   .format(res[0], res[1]),
                   0.1, 0.1, {'size': 14})]
        return np.array(restxt, dtype=object)[:, np.newaxis, :]
    elif model == 'sim_traposc' and plotMode == 'many':
        restxt = [(r"w = " + f"{res[0]:.2f} +- {res[1]:.2f}",
                   0.6, 0.85, {'size': 8})
                  for res in fitResults]
        return np.array(restxt, dtype=object)[:, np.newaxis, :]


    elif model == 'damped_sin' and plotMode =='one':
        res = fitResults
        restxt = [(r"$\nu$ = {:.2f} $\pm$ {:.2f} kHz"\
                   .format(1e3*res[0], 1e3*res[5]) + "\n"
                    + r"$\phi$ = {:.2f} $\pm$ {:.2f} rad"\
                      .format(res[1], res[6]) + "\n"
                    + r"$\tau$ = {:.2f} $\pm$ {:.2f}"\
                      .format(res[2], res[7]) + r"$\mathrm{\mu s}$" + "\n"
                   + r"$A$ = {:.3f} $\pm$ {:.3f}"\
                     .format(res[3], res[8]) + "\n"
                    + r"$y_0$ = {:.3f} $\pm$ {:.3f}"\
                      .format(res[4], res[9])
                   ,0.65, 0.05, {})]
        restxt = np.array(restxt, dtype=object)[:, np.newaxis, :]
        fctTex = [(fctLatex[model], 0.05, 0.9, {})]
        fctTex = np.array(fctTex, dtype=object)[:, np.newaxis, :]
        return np.concatenate((fctTex, restxt), axis=1)
    elif model == 'damped_sin' and plotMode =='many':
        return np.empty((nbResults, 0))

    elif model == 'rabi_osc' and plotMode =='one':
        res = fitResults
        restxt = [(r"$\nu$ = {:.2f} $\pm$ {:.2f} MHz"\
                   .format(res[0], res[4]) + "\n"
                   + r"$\delta$ = {:.2f} $\pm$ {:.2f} MHz"\
                     .format(res[1], res[5]) + "\n"
                   + r"$\phi$ = {:.2f} $\pm$ {:.2f} rad"\
                     .format(res[2], res[6]) + "\n"
                   + r"$y_0$ = {:.2f} $\pm$ {:.2f}"\
                     .format(res[3], res[7]),
                   0.6, 0.55, {})]
        restxt = np.array(restxt, dtype=object)[:, np.newaxis, :]
        fctTex = [(fctLatex[model], 0.2, 0.9, {})]
        fctTex = np.array(fctTex, dtype=object)[:, np.newaxis, :]
        return np.concatenate((fctTex, restxt), axis=1)
    elif model == 'rabi_osc' and plotMode =='many':
        return np.empty((nbResults, 0))

    elif model == 'lorentz' and plotMode =='one':
        res = fitResults
        restxt = [(r"$\mu$ = {:.2f} $\pm$ {:.2f} MHz"\
                   .format(res[2], res[6]) + "\n"
                   + r"$\gamma$ = {:.2f} $\pm$ {:.2f} MHz"\
                     .format(res[3], res[7]),
                   0.6, 0.55, {})]
        restxt = np.array(restxt, dtype=object)[:, np.newaxis, :]
        fctTex = [(fctLatex[model], 0.2, 0.9, {})]
        fctTex = np.array(fctTex, dtype=object)[:, np.newaxis, :]
        return np.concatenate((fctTex, restxt), axis=1)
    elif model == 'lorentz' and plotMode =='many':
        return np.empty((nbResults, 0))

    elif model == 'gauss' and plotMode == 'one':
        res = fitResults
        restxt = [(r"$\mu$ = {:.2f} $\pm$ {:.2f} MHz"\
                   .format(res[2], res[6]) + "\n"
                   + r"$\sigma$ = {:.2f} $\pm$ {:.2f} MHz"\
                     .format(res[3], res[7]),
                   0.6, 0.55, {})]
        restxt = np.array(restxt, dtype=object)[:, np.newaxis, :]
        fctTex = [(fctLatex[model], 0.2, 0.9, {})]
        fctTex = np.array(fctTex, dtype=object)[:, np.newaxis, :]
        return np.concatenate((fctTex, restxt), axis=1)
    elif model == 'gauss' and plotMode =='many':
        return np.empty((nbResults, 0))

    elif model == 'mult_gauss' and plotMode == 'one':
        return np.empty((nbResults, 0))
    elif model == 'mult_gauss' and plotMode =='many':
        return np.empty((nbResults, 0))

    elif model == 'convol_lorentz' and plotMode =='one':
        res = fitResults
        restxt = [(r"$\mu$ = {:.2f} $\pm$ {:.2f} MHz"\
                   .format(res[2], res[7]) + "\n"
                   + r"$\gamma$ = {:.2f} $\pm$ {:.2f} MHz"\
                     .format(res[3], res[8]) + "\n"
                   + r"$\mu_T$ = {:.2f} $\pm$ {:.2f} MHz"\
                     .format(res[4], res[9]),
                   0.6, 0.55, {})]
        restxt = np.array(restxt, dtype=object)[:, np.newaxis, :]
        fctTex = [(fctLatex[model], 0.2, 0.9, {})]
        fctTex = np.array(fctTex, dtype=object)[:, np.newaxis, :]
        return np.concatenate((fctTex, restxt), axis=1)
    elif model == 'convol_lorentz' and plotMode =='many':
        return np.empty((nbResults, 0))

    elif model == 'squared_sinc' and plotMode == 'one':
        res = fitResults
        restxt = [(r"$x_0$ = {:.2f} $\pm$ {:.2f} MHz"\
                   .format(res[2], res[6]) + "\n"
                   + r"$\gamma$ = {:.2f} $\pm$ {:.2f} MHz"\
                     .format(res[3], res[7]) + "\n"
                   + r"$A$ = {:.2f} $\pm$ {:.2f}"\
                     .format(res[1], res[5]) + "\n"
                   + r"$y_0$ = {:.2f} $\pm$ {:.2f}"\
                     .format(res[0], res[4]),
                   0.65, 0.05, {})]
        restxt = np.array(restxt, dtype=object)[:, np.newaxis, :]
        fctTex = [(fctLatex[model], 0.2, 0.9, {})]
        fctTex = np.array(fctTex, dtype=object)[:, np.newaxis, :]
        return np.concatenate((fctTex, restxt), axis=1)
    elif model == 'squared_sinc' and plotMode =='many':
        return np.empty((nbResults, 0))

    elif model == 'decay_exp' and plotMode == 'one':
        res = fitResults
        restxt = [(r"$y_0$ = {:.3f} $\pm$ {:.3f}"\
                     .format(res[0], res[3]) + "\n"
                   r"$A$ = {:.2f} $\pm$ {:.2f}"\
                     .format(res[1], res[4]) + "\n"
                   r"$\tau$ = {:.2f} $\pm$ {:.2f}"\
                     .format(res[2], res[5]) + r"$\mathrm{\mu s}$",
                   0.6, 0.55, {})]
        restxt = np.array(restxt, dtype=object)[:, np.newaxis, :]
        fctTex = [(fctLatex[model], 0.2, 0.9, {})]
        fctTex = np.array(fctTex, dtype=object)[:, np.newaxis, :]
        return np.concatenate((fctTex, restxt), axis=1)
    elif model == 'decay_exp' and plotMode =='many':
        restxt = [(r"$\tau$ = " + f"{res[2]:.2f}", 0.7, 0.85, {'size': 10})
                  for res in fitResults]
        return np.array(restxt, dtype=object)[:, np.newaxis, :]

    elif model == 'damped_arcsine_sin' and plotMode =='one':
        res = fitResults
        restxt = [(r"$\nu$ = {:.2f} $\pm$ {:.2f} MHz"\
                   .format(1e3*res[0], 1e3*res[6]) + "\n"
                    + r"$\delta \nu$ = {:.2f} $\pm$ {:.2f} MHz"\
                      .format(res[1], res[7]) + "\n"
                    + r"$\phi$ = {:.2f} $\pm$ {:.2f} rad"\
                      .format(res[2], res[8]) + "\n"
                    + r"$\tau$ = {:.2f} $\pm$ {:.2f}"\
                      .format(res[3], res[9]) + r"$\mathrm{\mu s}$" + "\n"
                   + r"$A$ = {:.3f} $\pm$ {:.3f}"\
                     .format(res[4], res[10]) + "\n"
                    + r"$y_0$ = {:.3f} $\pm$ {:.3f}"\
                      .format(res[5], res[11])
                   ,0.65, 0.05, {})]
        restxt = np.array(restxt, dtype=object)[:, np.newaxis, :]
        fctTex = [(fctLatex[model], 0.05, 0.9, {})]
        fctTex = np.array(fctTex, dtype=object)[:, np.newaxis, :]
        return np.concatenate((fctTex, restxt), axis=1)
    elif model == 'damped_arcsine_sin' and plotMode =='many':
        return np.empty((nbResults, 0))

    elif model == 'inverse_lorentz' and plotMode =='one':
        res = fitResults
        restxt = [(r"$\mu$ = {:.2f} $\pm$ {:.2f} MHz"\
                   .format(res[2], res[8]) + "\n"
                   + r"$d \mu$ = {:.2f} $\pm$ {:.2f} MHz"\
                     .format(res[4], res[10])
                   ,0.65, 0.05, {})]
        restxt = np.array(restxt, dtype=object)[:, np.newaxis, :]
        fctTex = [(fctLatex[model], 0.05, 0.9, {})]
        fctTex = np.array(fctTex, dtype=object)[:, np.newaxis, :]
        return np.concatenate((fctTex, restxt), axis=1)
    elif model == 'inverse_lorentz' and plotMode =='many':
        return np.empty((nbResults, 0))

    elif model == 'parabola' and plotMode =='one':
        res = fitResults
        restxt = [(r"$y_0$ = {:.2f} $\pm$ {:.2f}"\
                     .format(res[0], res[3]) + "\n"
                   r"$x_0$ = {:.2f} $\pm$ {:.2f}"\
                     .format(res[1], res[4]) + "\n"
                   r"$A$ = {:.2f} $\pm$ {:.2f}"\
                     .format(res[2], res[5]),
                   0.02, 0.5, {})]
        restxt = np.array(restxt, dtype=object)[:, np.newaxis, :]
        fctTex = [(fctLatex[model], 0.02, 0.75, {})]
        fctTex = np.array(fctTex, dtype=object)[:, np.newaxis, :]
        return np.concatenate((fctTex, restxt), axis=1)
    elif model == 'parabola' and plotMode =='many':
        return np.empty((nbResults, 0))

    elif model == 'affine' and plotMode =='one':
        res = fitResults
        restxt = [(r"$a$ = {:.2f} $\pm$ {:.2f}"\
                     .format(res[0], res[2]) + "\n"
                   r"$b$ = {:.2f} $\pm$ {:.2f}"\
                     .format(res[1], res[3]),
                   0.02, 0.5, {})]
        restxt = np.array(restxt, dtype=object)[:, np.newaxis, :]
        fctTex = [(fctLatex[model], 0.02, 0.75, {})]
        fctTex = np.array(fctTex, dtype=object)[:, np.newaxis, :]
        return np.concatenate((fctTex, restxt), axis=1)
    elif model == 'affine' and plotMode =='many':
        return np.empty((nbResults, 0))

    else:
        raise ValueError(f"Fitting model `{model}` does not exist.")
