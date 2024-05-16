# -*- coding: utf-8 -*-
"""
This module implements the Plot class, that wraps around matplotlib to
display data.

A Plot instance is essentially a container for a matplotlib Figure and an
array of Axes.
Utility methods are:
    - <_set_plot_type>, set the plot type: `graph`, `histogram` or `colormap`
        set automatically when calling a plot method
    - <_build_figures>, build the Figure and the array of Axes
    - <_set_axis_ticks>, set axis ticks automatically (gives a different
        result that matplotlib default)
    - <_broadcast_data>, broadcast data to plot for proper dispatching to
        the different Axes
    - <_check_data>, check the data to plot
Getters are:
    - <get_figure>, get the matplotlib Figure instance
    - <get_axes>, get the array of matplotlib Axes instances
Data plotting methods:
    - <plot_data>, plot a graph (x, y)
        sets plot type to `graph`
    - <plot_fct>, plot a function
        possible only with plot types `graph` or `histogram`
    - <plot_hist>, plot a histogram
        sets plot type to `histogram`
    - <plot_colormap>, plot a colormap
        sets plot type to `colormap`
Plot scaling:
    - <set_yscale>, set y scale (log, linear, ...)
    - <set_xlim>, set x limits
    - <set_yim>,set y limits
Add stuff on the plot
    - <axes_text>, add text on each Axes
    - <add_plot_indices>, add an index on each Axes
    - <set_xlabel>, set x label
    - <set_ylabel>, set y label
    - <set_title>, set figure title
    - <fig_text>, add text on the figure
    - <legend>, add a legend to the plot (NOT IMPLEMENTED)
Plot display and saving:
    - <show>, display the plot
    - <save>, save the plot


In addition, numerous dict are defined that contain the default parameters
(size, fontsize, position, ...) for the figure itself and the various
texts.

A dict dkeyTxt converts pair stats accessors into a nice y label. It is not
used anymore.


TODO
- doc
- gendoc
"""

from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .config import DPI



# =============================================================================
# Defaults
# =============================================================================

# default figure parameters
default_fig_params = {
    'figsize': lambda nx, ny: (6 + (nx-1), 4 + 0.7*(ny-1)), # fig size in inches
    'left_margin': 0.65, # inches
    'right_margin': 0.45, # inches
    'bottom_margin': 0.5, # inches
    'top_margin': 0.8, # inches
    }


# Plot default kw
default_graph_kw = {
    'marker': "o",
    'linestyle': "",
    'markersize': 4,
    }
default_errorbar_kw = {
    'elinewidth': 1,
    }

# pcolormesh default kw
default_pcm_kw = {
    'cmap': "plasma",
    'edgecolors': "black",
    'linewidth': 0.5}


# Text default kw
default_txt_kw = {
    'ha': "center",
    'va': "center",
    'fontsize': 9,
    }

# Title default kw
default_title_kw = {
    'ha': "center",
    'va': "center",
    'x': 0.5,
    'fontsize': 14,
    'fontweight': "bold",
    }
default_title_settings = { # Those parameters that are not keywords
    'y_in_inches': 0.12,}

# Subtitle default kw
default_suptxt_kw = {
    'ha': "center",
    'va': "center",
    'fontsize': 11,
    'fontweight': "normal",
    }
default_suptxt_settings = { # Those parameters that are not keywords
    'x': 0.5, # x position in fig coord
    'y_in_inches': 0.5, # y position from the top of the figure
    }

# Xlabel default kw
default_xlabel_kw = {
    'ha': "center",
    'va': "center",
    'x': 0.5,
    'fontsize': 11,
    }
default_xlabel_settings = { # Those parameters that are not keywords
    'y_in_inches': 0.12,}

# Ylabel default kw
default_ylabel_kw = {
    'ha': "center",
    'va': "center",
    'y': 0.5,
    'fontsize': 11,
    }
default_ylabel_settings = { # Those parameters that are not keywords
    'x_in_inches': 0.12,}


# =============================================================================
#
# =============================================================================

class Plot():

    plot_types = {"graph", "histogram", "colormap"}
    title_fontsize = 12
    txt_fontsize = 9

    def __init__(self,
                 nbplot: int,
                 shape: tuple | None = None,
                 roi_mapper: np.ndarray | None = None):
        """
        Instanciate a custom plot using Matplotlib.

        Parameters
        ----------
        nbplot : int
            Number of subplots in the plot.
        shape : tuple, optional
            Arrangement of the subplots on a rectangular array.
            The default is None.
        roi_mapper : 2D np.ndarray, optional
            roi_mapper[k, :] = [i, j], index coordinates of dataset k in the
            array of plots. The default is None.

        Raises
        ------
        ValueError
            If the shape provided for the suplot array cannot hold all the
            plots, ie, shape[0]*shape[1] < nbplot
        """
        # Type of plot
        self.plot_type = None # str {"graph", "histogram", "colormap"}

        # Plot structure parameters
        self.nbplot = nbplot
        self.shape = (0, 0)
        self.roi_mapper = None

        # Matplotlib objects
        self.fig = None
        self.ax = None

        #
        self.xlim = None


        if shape is None:
            self.shape = (int(np.ceil(np.sqrt(nbplot))),
                          int(np.ceil(nbplot / np.ceil(np.sqrt(nbplot)))))
        else:
            if len(shape) != 2 or np.prod(shape) < nbplot:
                raise ValueError("shape must be of length 2 with product > "
                                 f"{nbplot}, got {shape}")
            self.shape = shape

        if roi_mapper is None:
            self.roi_mapper = np.array([(i // self.shape[1], i % self.shape[1])
                                    for i in range(self.nbplot)],
                                   dtype=int)
        else:
            self.roi_mapper = roi_mapper

        self._build_figure()


    def _set_plot_type(self, tgt_ptype: str):
        if tgt_ptype not in self.plot_types:
            raise ValueError(f"{tgt_ptype} must be in {self.plot_types}")
        if self.plot_type is None:
            self.plot_type = tgt_ptype
        elif tgt_ptype == self.plot_type:
            return
        else:
            raise ValueError(f"target plot type, {tgt_ptype}, incompatible "
                             f"with current plot type, {self.plot_type}")


    def _build_figure(self,):
        """
        Initialize the matplotlib figure and axes.
        """
        sh = self.shape
        figsize = default_fig_params['figsize'](sh[1], sh[0])
        l = default_fig_params['left_margin']/figsize[0]
        r = 1. - default_fig_params['right_margin']/figsize[0]
        b = default_fig_params['bottom_margin']/figsize[1]
        t = 1. - default_fig_params['top_margin']/figsize[1]

        fig = plt.figure(figsize=figsize,
                         dpi=DPI,
                         clear=True)
        gs = fig.add_gridspec(nrows=sh[0], ncols=sh[1],
                              wspace=0, hspace=0,
                              left=l, bottom=b, right=r, top=t,
                              figure=fig)
        axs = gs.subplots(sharex=True, sharey=True, squeeze=False)

        for i, j in np.ndindex(*sh):
            # Set the grid
            axs[i, j].grid(which='minor', alpha=0.2)
            axs[i, j].grid(which='major', alpha=0.5)
            # Remove ticks of interior axes
            if i > 0 and i < sh[0]-1: # vertical bulk ticks params
                axs[i, j].tick_params(axis='both', direction='inout',
                                      top=True, labeltop=False,
                                      bottom=True, labelbottom=False)
            elif j > 0 and j < sh[1]-1: # horizontal bulk ticks params
                axs[i, j].tick_params(axis='both', direction='inout',
                                      labelleft=False, labelright=False)
            if i == 0 and i != sh[0]-1: # upper edge ticks params
                axs[i, j].tick_params('x', direction='in',
                                      top=True, labeltop=False,
                                      bottom=True, labelbottom=False)
            if i == sh[0]-1 and i != 0: # lower edge ticks params
                axs[i, j].tick_params('x', direction='inout',
                                      top=True, labeltop=False)
            if j == 0 and j != sh[1]-1: # left edge ticks params
                axs[i, j].tick_params('y', direction='inout',
                                      left=True, labelleft=True)
            if j == sh[1]-1 and j != 0: # right edge ticks params
                axs[i, j].tick_params('y', direction='inout',
                                      right=True, labelright=True,
                                      left=True, labelleft=False)
        
        self.fig, self.ax = fig, axs


    def _set_axis_ticks(self,):
        """

        """
        STEP_X = 6
        STEP_Y = 4

        ax = self.ax[0, 0]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xticks = []
        xrange = xlim[1] - xlim[0]
        xtickStep = ((xrange * 100) // STEP_X) / 100

        if 0 >= (xlim[0]-xtickStep) and 0 <= (xlim[1] - xtickStep/2.1):
            xticks.append(0)
            for i in range(1, STEP_X+1):
                if xlim[0] <= -i*xtickStep:
                    xticks.append(-i*xtickStep)
                if i*xtickStep <= (xlim[1] - xtickStep/2.1):
                    xticks.append(i*xtickStep)
            xticks.sort()
        else:
            xticks.append(xlim[0])
            for i in range(1, STEP_X+1):
                if (x := xlim[0] + i*xtickStep) <= (xlim[1] - xtickStep/2.1):
                    xticks.append(x)
        ax.set_xticks(xticks)

        yticks = []
        yrange = ylim[1] - ylim[0]
        ytickStep = ((yrange * 100) // STEP_Y) / 100
        if 0 >= ylim[0] and 0 <= (ylim[1] - ytickStep/2.1):
            yticks.append(0)
            for i in range(1, STEP_Y+1):
                if ylim[0] <= -i*ytickStep:
                    yticks.append(-i*ytickStep)
                if i*ytickStep <= (ylim[1] - ytickStep/2.1):
                    yticks.append(i*ytickStep)
            yticks.sort()
        else:
            yticks.append(ylim[0])
            for i in range(1, STEP_Y+1):
                if (y := ylim[0] + i*ytickStep) <= (ylim[1] - ytickStep/2.1):
                    yticks.append(y)
        ax.set_yticks(yticks)


    def _broadcast_data(self,
                        arr: np.ndarray | None,
                        )-> np.ndarray | None:
        """
        TODO

        Parameters
        ----------
        arr : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if arr is None:
            return None
        if arr.ndim == 1:
            return np.broadcast_to(arr, (self.nbplot, 1, arr.shape[-1]))
        elif arr.ndim == 2:
            if self.nbplot > 1 and (l:=len(arr)) != self.nbplot:
                raise ValueError(
                    f"number of data to plot/display, {l}, does not match "
                    f"the number of plots, {self.nbplot}")
            return arr.reshape((self.nbplot, 1, arr.shape[-1]))
        elif arr.ndim == 3:
            if (l:=len(arr)) != self.nbplot:
                raise ValueError(
                    f"number of data to plot/display, {l}, does not match "
                    f"the number of plots, {self.nbplot}")
            return arr
        else:
            raise ValueError(f"data dim must be 1, 2 or 3, not {arr.ndim}")


    def _check_data(self, *args):
        """
        Check that the number of data to plot, text to display, etc.
        matches the number of plots nbplot.
        """
        for arg in args:
            if self.nbplot > 1 and (l:=len(arg)) != self.nbplot:
                raise ValueError(
                    f"number of data to plot/display, {l}, does not match "
                    f"the number of plots, {self.nbplot}")


    def get_figure(self,):
        return self.fig


    def get_axes(self,):
        return self.ax


    def plot_data(self,
                  x: np.ndarray,
                  y: np.ndarray,
                  yerr: np.ndarray= None,
                  plot_kw: dict = None):
        """


        Parameters
        ----------
        x : 1D np.ndarray of shape (M,)
            The x values.
        y : 1, 2 or 3D np.ndarray
            - If ndim = 1, shape (M,)
                The same graph (x, y) is plotted on all subplots.
            - If ndim = 2, shape (nbplot, M)
                The graph (x, y[i, :]) is plotted on subplot i.
            - If ndim = 3, shape (nbplot, N, M)
                The graphs (x, y[i, k, :]) for k = 0..N-1 are plotted on
                subplot i.
        yerr : 1, 2 or 3D np.ndarray, optional
            Errorbars associated to the data. Same shape as y.
            The default is None.
        plot_kw : dict, optional
            DESCRIPTION. The default is None.

        """
        # checks
        self._set_plot_type("graph")
        y = self._broadcast_data(y)
        yerr = self._broadcast_data(yerr)
        #
        plot_kw = {} if plot_kw is None else plot_kw
        kw = default_graph_kw | plot_kw
        xMin, xMax = np.min(x), np.max(x)
        if xMin != xMax:
            self.set_xlim((xMin, xMax), override=False)

        for k, (i, j) in enumerate(self.roi_mapper):
            for p, yk in enumerate(y[k, :, :]):
                # Plot data
                if yerr is not None:
                    self.ax[i, j].errorbar(x, yk, yerr=yerr[k, p],
                                           **(default_errorbar_kw | kw))
                else:
                    self.ax[i, j].plot(x, yk, **kw)


    def plot_fct(self,
                 params: np.ndarray,
                 fct: Callable,
                 plot_kw: dict | None = None):
        """


        Parameters
        ----------
        popt : np.ndarray
            DESCRIPTION.
        fct : callable
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # checks
        self._check_data(params)
        if self.plot_type == "colormap":
            raise ValueError("Cannot plot function on colormap")
        #
        plot_kw = {} if plot_kw is None else plot_kw
        x = np.linspace(self.xlim[0], self.xlim[1], 201, endpoint=True)
        for k, (i, j) in enumerate(self.roi_mapper):
            self.ax[i, j].plot(x, fct(x, *params[k]), **plot_kw)


    def plot_hist(self, x: np.ndarray,
                  nbBins: int = None,
                  thresholds: np.ndarray = None,
                  hist_kw: dict = None):
        """


        Parameters
        ----------
        x : np.ndarray
            DESCRIPTION.
        nbBins : int, optional
            DESCRIPTION. The default is None.
        thresholds : np.ndarray, optional
            DESCRIPTION. The default is None.
        plot_kw : dict, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if nbBins is None:
            nbBins = min(50, x.shape[-1] // 15)
        # histogram keywords
        hist_kw = {} if hist_kw is None else hist_kw

        xMin, xMax = np.min(x), np.max(x)
        if xMin != xMax:
            self.set_xlim((xMin, xMax), override=False)
        # self.set_ylim((0., None))

        for k, (i, j) in enumerate(self.roi_mapper):
            # Plot hist
            self.ax[i, j].hist(x[k], bins=nbBins, **hist_kw)
            # Plot thresholds
            if thresholds is not None:
                self.ax[i, j].axvline(thresholds[k], color='g')


    def plot_colormap(self,
                      data: np.ndarray,
                      pcm_kw: dict | None = None):
        """


        Parameters
        ----------
        data : np.ndarray
            DESCRIPTION.
        pcm_kw : dict, optional
            pcolormesh keywords. The default is None.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # checks
        self._set_plot_type("colormap")
        if self.nbplot > 1:
            raise ValueError("only single mosaic plot is implemented")
        
        # Hide the grid
        for ax in self.ax.flat:
            ax.grid(visible=False)
        
        # set pcolormesh keywords
        pcm_kw = {} if pcm_kw is None else pcm_kw
        pcm_kw = default_pcm_kw | pcm_kw

        ax = self.ax[0, 0]
        ax.set_aspect('equal')
        ax.tick_params(
            axis='both', direction='inout',
            top=False, labeltop=False,
            bottom=False, labelbottom=False,
            left=False, labelleft=False,
            right=False, labelright=False
            )

        divider = make_axes_locatable(ax)
        ax_cb = divider.append_axes("right", size="7%", pad="5%")
        self.fig.add_axes(ax_cb)

        mosaic = ax.pcolormesh(data, **pcm_kw)
        cb = self.fig.colorbar(
            mosaic, cax=ax_cb, orientation="vertical",
            ticklocation="right")
        cb.ax.tick_params(
            axis='both', direction='out',
            labelsize=10, pad=2,)


    def axes_text(self,
                xpos: float,
                ypos: float,
                txt: list[str],
                txt_kw: dict | None = None):
        """


        Parameters
        ----------
        xpos : float
            DESCRIPTION.
        ypos : float
            DESCRIPTION.
        txt : list[str]
            DESCRIPTION.
        txt_kw : dict, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        txt_kw = {} if txt_kw is None else txt_kw
        kw = default_txt_kw | txt_kw
        # checks
        self._check_data(txt)
        #
        for k, (i, j) in enumerate(self.roi_mapper):
            self.ax[i, j].text(xpos, ypos, txt[k], **kw,
                               transform=self.ax[i, j].transAxes)


    def add_plot_indices(self,):
        """

        """
        xpos, ypos = 0.98, 0.96
        txt = [f"{k}" for k in range(self.nbplot)]
        self.axes_text(xpos, ypos, txt,
                       {'fontsize': 11, 'ha': "right", 'va': "top"})


    def set_yscale(self, scale: str):
        """


        Parameters
        ----------
        scale : str {"linear", "log", "symlog", "logit", ...}
            See matplotlib.axes.Axes.set_yscale documentation.

        """
        for i, j in np.ndindex(*self.shape):
            self.ax[i, j].set_yscale(scale)


    def set_xlim(self, xlim: tuple, override: bool = True):
        if self.xlim is None or override:
            self.ax[0, 0].set_xlim(xlim)
            self.xlim = xlim
        else:
            self.xlim = min(xlim[0], self.xlim[0]), max(xlim[1], self.xlim[1])
            self.ax[0, 0].set_xlim(self.xlim)


    def set_ylim(self, ylim: tuple):
        self.ax[0, 0].set_ylim(bottom=ylim[0], top=ylim[1])


    def set_xlabel(self, xlabel: str, label_kw: dict = None):
        ysz = self.fig.get_size_inches()[1]
        yabs = default_xlabel_settings['y_in_inches']

        label_kw = {} if label_kw is None else label_kw
        kw = {'y': yabs/ysz} | default_xlabel_kw | label_kw
        self.fig.supxlabel(xlabel, **kw)


    def set_ylabel(self, ylabel: str, label_kw: dict = None):
        xsz = self.fig.get_size_inches()[0]
        xabs = default_ylabel_settings['x_in_inches']

        label_kw = {} if label_kw is None else label_kw
        kw = {'x': xabs/xsz} | default_ylabel_kw | label_kw
        self.fig.supylabel(ylabel, **kw)


    def set_title(self, title: str, title_kw: dict = None):
        ysz = self.fig.get_size_inches()[1]
        yabs = default_title_settings['y_in_inches']

        title_kw = {} if title_kw is None else title_kw
        kw = {'y': 1-yabs/ysz} | default_title_kw | title_kw
        self.fig.suptitle(title, **kw)


    def fig_suptext(self, suptxt: str, txt_kw: dict = None):
        x = default_suptxt_settings['x']
        ysz = self.fig.get_size_inches()[1]
        yabs = default_suptxt_settings['y_in_inches']
        y = 1 - yabs/ysz

        txt_kw = {} if txt_kw is None else txt_kw
        kw = default_suptxt_kw | txt_kw
        self.fig.text(x, y, suptxt, **kw)


    def fig_text(self,
                xpos: float,
                ypos: float,
                txt: str,
                txt_kw: dict | None = None):
        """


        Parameters
        ----------
        xpos : float
            DESCRIPTION.
        ypos : float
            DESCRIPTION.
        txt : str
            DESCRIPTION.
        txt_kw : dict, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        txt_kw = {} if txt_kw is None else txt_kw
        kw = default_txt_kw | txt_kw
        #
        self.fig.text(xpos, ypos, txt, **kw)


    def legend(self, legend_kw: dict = None):
        kw = {} if legend_kw is None else legend_kw
        self.ax[0, 0].legend(**kw)


    def show(self,):
        self._set_axis_ticks
        self.fig.show


    def save(self, file: str | Path):
        self.fig.savefig(file)


# =============================================================================
#
# =============================================================================
dkeyTxt = {
    'nbref': "number of atoms loaded",
    'nbdelay': "number of atoms in delay frame",
    'nbrecap': "number of recaptured atoms",
    'nbloss': "number of lost atoms",
    'nbgen': "number of appearing atoms",
    'pref': "loading probability",
    'pdelay': "probability of presence in delay frame",
    'precap': "recapture probability",
    'ploss': "loss probability",
    'pgen': "atom apparition probability",
    'stdev_pref': "std deviation of loading probability",
    'stdev_pdelay': "std deviation of delay frame presence probability",
    'stdev_precap': "std deviation of recapture probability",
    'stdev_ploss': "std deviation of loss probability",
    'stdev_pgen': "std deviation of atom apparition probability",
    'stdev_nbref': "std deviation of number of atoms loaded",
    'stdev_nbdelay': "std deviation of number of atoms in delay frame",
    'stdev_nbrecap': "std deviation of number of recaptured atoms",
    'stdev_nbloss': "std deviation of number of lost atoms",
    'stdev_nbgen': "std deviation of number of appearing atoms",}



# =============================================================================
#
# =============================================================================




