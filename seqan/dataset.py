# -*- coding: utf-8 -*-
"""
This module implements the Data_Set class.

A Data_Set wraps around processed experimental data and provides functionality
such as plotting, data import/export, and fitting.

The methods are:
    - <_load>, intanciate the Data_Set from a dict
    - <set_parameters>, set other parameters pertaining to the Data_Set
    - <set_path>, set the file path of the Data_Set, NOT USED
    - <get_dict>, get the Data_Set in dict format
    - <savebin>, save the Data_Set as a npz archive
    - <savetxt>, save the selected data in txt format

    - <accessors>, get the accessors shortcuts for the available data
    - <get_data>, get a given data using an accessor
    - <get_stdev>, get the stdev of a data using an accessor
    - <get_xvalues>, get the scanned parameter values

    - <fit>, fit data according to a given model
    - <get_fit>, get the results of a fit (class analysis.fit.Fit_Result)
    - <save_fit>, save the results of a fit
    - <analyze>, analyze data (fft, ...)
    - <get_analysis>, get the analysis results (a dict)
    - <savebin_analysis>, save analysis results in binary format

    - <plot>, to plot data


Additional functions are provided:
    - <merge_datasets>, to merge two Data_Set into one


TODO doc

"""

import json
import warnings
from numbers import Integral
from pathlib import Path

import numpy as np

from .utils import merge_xvalues, cast_2d
from .calib.calib import calibDict
from .stats import (Atom_Stats, merge_stats)
from .analysis import (Fit_Result, analysis)
from .fileio import save_to_txt, Sequence_Manager_Data
from .plots import Plot


DataTypes = dict | Sequence_Manager_Data | str | Path

# =============================================================================
# FUNCTIONS -
# =============================================================================

def merge_datasets(datasets: tuple,
                   newname: str = "merged datasets",
                   tol: float = 1e-3)-> dict:
    """
    TODO doc

    Parameters
    ----------
    datasets : tuple
        DESCRIPTION.
    newname : str, optional
        DESCRIPTION. The default is "merged datasets".

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    dict
        DESCRIPTION.

    """
    if len(datasets) == 0:
        raise ValueError("must provide at least one Data_Set")
    # compatibility checks
    ds0 = datasets[0]
    for ds in datasets:
        if (ds0._data_type, ds._data_type) != ("atom_stats",)*2:
            raise ValueError("merge_with not implemented for data type "
                             "other than `atom_stats`")
    if (vn0:=ds0.varname) != (vn1:=ds.varname):
        warnings.warn(f"varname are different: {vn0} and {vn1}",
                      UserWarning)
    if (vu0:=ds0.varunit) != (vu1:=ds.varunit):
        raise ValueError(f"incompatible varunit: {vu0} and {vu1}")

    # Initialize output data dict
    DSdict = {
        'data_type': np.array("atom_stats"),
        'name': np.array(newname),
        'varname': np.array(ds0.varname),
        'raw_varunit': np.array(ds0.varunit),
        }
    # merged x-values
    new_xvals, idx = merge_xvalues([ds.xvalues for ds in datasets], tol=tol)
    DSdict['raw_xvalues'] = new_xvals
    # merged data
    merged_data = merge_stats([ds.data for ds in datasets], idx)
    for k, v in merged_data.items():
        DSdict['data.' + k] = np.array(v)
    # merged parameters
    merged_params = {}
    for ds in datasets:
        merged_params |= ds.parameters
    DSdict['nb_parameters'] = len(merged_params)
    for k, v in merged_params.items():
        DSdict['parameters.' + k] = np.array(v)
    
    return Data_Set(source=DSdict)


# =============================================================================
#
# =============================================================================

class Data_Set():
    """
    A class to hold individual datasets
    """

    DATA_SOURCES = {
        "sequence_manager",
        "atom_stats"}

    def __init__(self, *,
                 source: DataTypes,
                 parameters: dict | str | Path = None):
        """
        TODO DOC

        Parameters
        ----------
        sequenceSetup : Sequence_Data
            DESCRIPTION.
        rawData : 2D array
            Raw data for atomic signal. The format is:
            rawData[i, j] = signal of i-th ROI in j-th frame
            ROI indexation begins at 0.
            NOTE : Care must be taken to map rawData indices to ROI indices.
        thresholds : list, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self._data_type = "" # {"sequence_manager", "atom_stats"}

        self.name = None # str
        self.path = None # str

        self.parameters = {} # dict

        self.varname = "" # str
        self.varunit = "" # str
        self.xvalues = None # 1D np.ndarray
        self.data = None # dict or Atom_Stats

        self.fits = [] # list[Fit_Result]
        self.analyses = [] # list[dict]

        # load data
        if isinstance(source, dict):
            self._load_data(source)
        elif isinstance(source, Sequence_Manager_Data):
            src = source.get_dict()
            self._load_data(src)
        elif isinstance(source, (str, Path)):
            with np.load(source, allow_pickle=False) as f:
                self._load_data(f)
        else:
            raise TypeError("incorrect type to instanciate Data_Set, "
                            f"{type(source)}; valid types are `dict`, `str`, "
                            "`Path`, `Sequence_Manager_Data`")
        # Load parameters
        if parameters is not None:
            self.set_parameters(parameters)


    def _load_data(self, source: dict):
        """
        TODO doc
        This function will load parameters.

        Parameters
        ----------
        source : dict
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self._data_type = source['data_type']
        self.name = source['name']
        self.varname = source['varname']
        self.xvalues = source['raw_xvalues']
        self.varunit = source['raw_varunit']

        # Verify that xvalues are unique, problems might arise otherwise
        if not (x:=self.xvalues).size == np.unique(x).size:
            warnings.warn(
                "xvalues are not distinct, this might result in "
                "unexpected behavior; you should sanitize your input")

        # Set data
        dataDict = {k.split('.', 1)[-1]: v for k, v in source.items()
                    if k.split('.', 1)[0] == "data"}
        if self._data_type == "sequence_manager":
            self.data = dataDict
        elif self._data_type == "atom_stats":
            self.data = Atom_Stats(source=dataDict)
        else:
            raise ImportError("wrong source to instanciate Data_Set")
        
        # set parameters
        try:
            nb_params = source['nb_parameters']
        except KeyError: # Backward compatibility
            nb_params = 0
        if nb_params > 0:
            self.set_parameters(source)
            
            


    def set_parameters(self, parameters: dict | str | Path):
        """
        TODO doc

        Parameters
        ----------
        parameters : Union[dict, str, Path]
            The parameters to be set.

        """
        # Load from file or dict
        if isinstance(parameters, (str, Path)):
            try:
                with np.load(parameters, allow_pickle=False) as f:
                    params = dict(f)
            except (ValueError, OSError):
                with open(parameters, 'rt') as f:
                    params = json.load(f)
        elif isinstance(parameters, dict):
            params = parameters
        else:
            raise TypeError(
                f"parameters must be str, Path or dict, not {type(parameters)}")

        if self.parameters: # oops, erasing existing parameters
            print("set_parameters: erasing current parameters")
        ## Set the parameters
        # Parameters from a Sequence_Setup
        if ("parameters" in params.keys()
            and isinstance(params['parameters'], dict)):
            self.parameters = params['parameters']
        # parameters from a Data_Set dict (possibly loaded from a file)
        elif any(['parameters.' in k for k in params]):
            self.parameters = {k.split('.', 1)[-1]: v
                               for k, v in params.items()
                               if k.split('.', 1)[0] == "parameters"}
        else:
            self.parameters = params


    def set_path(self, path: str):
        """
        Set the path of the Data_Set. The path is currently not used.
        """
        self.path = path


    def get_dict(self)-> dict:
        """
        TODO doc

        Returns
        -------
        DSdict : TYPE
            DESCRIPTION.

        """
        # Initialize output data dict
        DSdict = {
            'data_type': np.array(self._data_type),
            'name': np.array(self.name),
            'varname': np.array(self.varname),
            'raw_xvalues': self.xvalues,
            'raw_varunit': np.array(self.varunit),
            'nb_parameters': np.array(len(self.parameters))
            }
        for k, v in self.parameters.items():
            DSdict['parameters.' + k] = np.array(v)
        if self._data_type == "atom_stats":
            data = self.data.get_dict()
        elif self._data_type == "sequence_manager":
            data = self.data
        for k, v in data.items():
            DSdict['data.' + k] = np.array(v)
        return DSdict


    def savebin(self,
                file: str | Path,
                calib: str | None = None,
                conv_to: str | None = None,
                conv_from: str | None = None):
        """
        TODO update
        Export ROI stats data in a .npz compressed archive.

        The data is saved in a form analogous to a dict, with the following
        keys:
            - 'raw_x_values': 1D np.ndarray, the raw scanned x_values
                (as set in Sequence_Setup)
            - 'varname': np.array(seqSetup.varname), essentially a str,
                the name of the scanned parameter
            - 'raw_varunit': np.array(seqSetup.varunit), essentially a str,
                the unit of the raw scanned parameter (set in Sequence_Setup)
            - 'calib': np.array(seqSetup.calib) or empty array,
                the calibration to convert units, if set
            - 'x_values': 1D np.ndarray,
                if a calibration is set, the scanned x values converted
                to `calibunit`
                otherwise, same as raw_x_values
            - 'varunit': np.array(seqSetup.varunit), essentially a str,
                `calibunit` if a calibration is set
                otherwise, same as raw_varunit
            - 'empty_rois.*', with * the keys of self.emptyROIs: 1D np.ndarray,
                empty ROIs given by <stats.get_shitty_roi>
            - 'roistats.*', with * the keys of self.roiStats: 2D np.ndarray,
                the statistics given by <stats.compute_pstat>, for instance
                precap[i, j] = recapture probability of ROI i at scan point j

        To load the data:

        with np.load(file, allow_pickle=False) as f:
            nbROI = f['nbROI']
            shape = f['shape']
            cframes = f['cframes']
            x_values = f['x_values']
            ...
            precap = f['roistats.precap']
            ...


        # The use of empty arrays avoids dtype=object arrays
        # So that np.savez_compressed does not use the unsafe pickle

        Parameters
        ----------
        calibunit : str, optional
            Select a unit among those available for conversion if a
            calibration is set in the sequence setup.
            The default is the empty string: the preset unit.

        Raises
        ------
        AttributeError
            If no ROI stats are available to save.

        """
        data_out = self.get_dict()
        data_out['xvalues'] = self.get_xvalues(calib, conv_to, conv_from)
        if conv_to is None:
            data_out['varunit'] = np.array(self.varunit)
        else:
            data_out['varunit'] = np.array(conv_to)
        np.savez_compressed(file, **data_out)
        print(f"Data_Set `{self.name}` saved at: {file}")


    def savetxt(self,
                accessor: str | tuple,
                file: str | Path,
                calib: str | None = None,
                conv_to: str | None = None,
                conv_from: str | None = None):
        """
        TODO update
        Save ROI data to a text file.

        Scan values corresponding to entry dkey of selected ROIs are
        written to a text file. The format allows for easy import with
        origin.

        Parameters
        ----------
        file : str
            Filename to which the data is saved.
        dkey : str
            Key corresponding to the data to save in ROI statistics
            dictionnary. Its values can be:
                - 'nbref', 'nbdelay', 'nbrecap', 'nbloss', 'nbgen'
                - 'pref', 'pdelay', 'precap', 'ploss', 'pgen'
                - 'stdev_*'
        roiSelect : int or slice or list or 1D array or None
            Selector of the ROIs to be averaged. Basically, the
            selected ROI indices are ROIs[roiSelect].
            The default is None, which selects all the data.
        calibunit : str, optional
            Select a unit among those available for conversion if a
            calibration is set in the sequence setup.
            The default is the empty string.

        Raises
        ------
        AttributeError
            If no ROI statistics are available to save.

        Returns
        -------
        None.
            Data is saved to a text file.

        """
        # Data to save
        x = self.get_xvalues(calib, conv_to, conv_from)
        y = cast_2d(self.get_data(accessor))
        y_sigma = cast_2d(self.get_stdev(accessor))
        if (d:=y.ndim) >= 3:
            raise ValueError(f"selected data must have dim 1 or 2, not {d}")

        names = [str(self.varname)]
        units = [str(conv_to)]
        comments = ["None"]
        exportData = [x]

        nbdata = len(y)
        str_access = str(accessor)
        if y_sigma is not None:
            names += [str_access, "stdev_" + str_access] * nbdata
            units += ['', ''] * nbdata
            for i in range(nbdata):
                comments.extend([f'data{i}', f'data{i}'])
                exportData.extend([y[i, :], y_sigma[i, :]])
        else: # y_sigma is None
            names += [str_access,] * nbdata
            units += ['',] * nbdata
            for i in range(nbdata):
                comments.extend([f'data{i}',])
                exportData.append(y[i, :])

        exportData = np.array(exportData)
        save_to_txt(file, names, exportData, units=units, comments=comments)
        print(f"Data {str_access} saved at: {file}")


    ########## Get data, xvalues, ... ##########

    def accessors(self)-> dict[str, set]:
        """
        Returns the accessors to fetch the data.
        """
        if self._data_type == "atom_stats":
            return self.data.accessors() # keys, shortcuts
        elif self._data_type == "sequence_manager":
            return {'shortcuts': set(self.data.keys())} # shortcuts


    def get_data(self, accessor: str | tuple)-> np.ndarray:
        """
        TODO doc

        Parameters
        ----------
        accessor : Union[str, tuple]
            DESCRIPTION.

        Raises
        ------
        KeyError
            If the accessor is not recognized.

        Returns
        -------
        1D or 2D np.ndarray
            The fetched data.

        """
        if self._data_type == "atom_stats":
            try:
                return np.copy(self.data.get_stat(accessor))
            except KeyError:
                raise KeyError(f"accessor key `{accessor[0]}` not recognized; "
                               f"valid keys are {self.accessors()['keys']}")
        elif self._data_type == "sequence_manager":
            try:
                return np.copy(self.data[accessor])
            except KeyError:
                msg = f"valid accessors are {self.accessors()['shortcuts']}"
                raise KeyError(f"accessor `{accessor}` not recognized; {msg}")


    def get_stdev(self, accessor: str | tuple)-> np.ndarray:
        """
        TODO doc

        Parameters
        ----------
        accessor : Union[str, tuple]
            DESCRIPTION.

        """
        if isinstance(accessor, str):
            stdev_accessor = "stdev_" + accessor
        elif isinstance(accessor, tuple):
            stdev_accessor = ("stdev_" + accessor[0], accessor[1])
        try: # Set uncertainity if available
            return self.get_data(stdev_accessor)
        except (ValueError, KeyError): # Uncertainity not available
            return None


    def get_xvalues(self, calib: str | None = None,
                    conv_to: str | None = None,
                    conv_from: str | None = None)-> np.ndarray:
        """
        Get the values of the scanned parameter converted to the desired
        unit.

        Parameters
        ----------
        calibunit : str, optional
            Select a unit among those available for conversion if a
            calibration is set in the sequence setup.
            Default is the empty string: the raw unit set in sequence setup.

        """
        if calib is None or conv_to is None:
            return self.xvalues
        if conv_from is None:
            conv_from = self.varunit
        return calibDict[calib](self.xvalues,
                                conv_to=conv_to,
                                conv_from=conv_from)


    ########## Analysis and fitting ##########

    def fit(self,
            accessor: str | tuple,
            model: str,
            *,
            dslice: slice = slice(None),
            model_kw: dict | None = None,
            p0: np.ndarray | None = None,
            calib: str | None = None,
            conv_to: str | None = None,
            conv_from: str | None = None)-> Fit_Result:
        """
        TODO doc

        Parameters
        ----------
        accessor : Union[str, tuple]
            Accessor to fetch the data to be fitted.
        model : str
            Fitting model.
        dslice : slice, optional
            Data slice. Fitted data corresponds to xvalues[dslice].
            The default is slice(None), which takes all the data.
        model_kw : dict, optional
            Model keywords. Additional parameters for the fit.
            The default is None.
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
        calib : str, optional
            Calibration to convert xvalues before fitting.
            The default is None (no xvalues conversion).
        conv_to : str, optional
            If calib is specified, convert to given unit.
            The default is None.
        conv_from : str, optional
            If calib and conv_to are specified, convert from given unit.
            This is most useful when varunit is not recognized.
            The default is None, corresponding to conv_from = varunit.

        Returns
        -------
        fit : Fit_Result
            DESCRIPTION.

        """
        dataSel = np.zeros_like(self.xvalues, dtype=bool)
        dataSel[dslice] = True

        info = {'name': self.name,
                'accessor': str(accessor),
                'data_select': dataSel,
                'varunit': self.varunit if conv_to is None else conv_to,
                'varname': self.varname,}

        # Initialize model kwargs
        model_kw = {} if model_kw is None else model_kw
        mkw = {}
        mkw.update(self.parameters | model_kw)

        # Initialize data to fit
        x = self.get_xvalues(calib, conv_to, conv_from)[dataSel]
        y = cast_2d(self.get_data(accessor)[..., dataSel])
        y_sigma = cast_2d(self.get_stdev(accessor)[..., dataSel])
        if isinstance(y_sigma, np.ndarray): # zero uncertainity is pathologic
            y_sigma = np.where(y_sigma == 0., 1., y_sigma)

        fit = Fit_Result(x, y, y_sigma, info)
        fit.fit(model, p0=p0, model_kw=mkw)

        self.fits.append(fit)
        return fit


    def get_fit(self, fitptr: int = -1):
        """
        Get the fitptr-th fit of the list. Defaults to the last one.
        """
        return self.fits[fitptr]


    def save_fit(self,
                 file: str | Path,
                 savemode: str = "text",
                 fitptr: int = -1):
        """
        Wrapper around Fit_Result.save*.

        Parameters
        ----------
        file : Union[str, Path]
            DESCRIPTION.
        savemode : str, {"text", "binary"}
            DESCRIPTION. The default is "text".
        fitptr : int, optional
            Fit selection. The default is -1 (last fit).

        """
        fit = self.get_fit(fitptr)
        if savemode == "text":
            fit.savetxt(file)
        elif savemode == "binary":
            fit.savebin(file)
        else:
            raise ValueError(
                f"savemode must be 'text' or 'binary', not {savemode}")


    def analyze(self,
                accessor: str | tuple,
                routine: str,)-> dict:
        """
        TODO test, doc, comment

        Parameters
        ----------
        dataSelector : Union[str, tuple]
            === If provided as a tuple: ===
            The data to be analyzed packed in a tuple.
            - (x, y) for `fft` analysis
            - (y,) for `stats` analysis
            === If provided as a str: ===
            Data to fetch expressed as a path. The format is:
            stem.leaf, with
            - stem : str, {"data", "thresholds", "roiPop",
                           "roiStats", "avgStats",
                           "roiFit", "avgroiFit"}
                Represents the category of data to be fetched
            - leaf : str, int
                Represent the subcategory of the data to be analyzed.
                For instance the cframe index of the fluorescence data,
                or a key in case the stem is tructured as a dict.
                It is not always necessary, and is ignored in this case.
        routine : {"fft", "stats"}
            DESCRIPTION.

        Raises
        ------
        ValueError
            If the analysis routine incorrectly specified.
        TypeError
            If the dataSelector is incorrectly specified.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        y = cast_2d(self.get_data(accessor))
        if routine == 'fft':
            x = self.get_xvalues()
            res = analysis((x, y), "fft")
        elif routine == "stats":
            res = analysis((y,), "stats")
        else:
            raise ValueError(
                f"routine must be 'fft' or 'stats', not {routine}")

        res.update(
            {'data_accessor': str(accessor),
             'routine': routine})
        self.analyses.append(res)
        return res


    def get_analysis(self, anptr: int = -1):
        """
        Get the anptr-th analysis of the list. Defaults to the last one.
        """
        return self.analyses[anptr]


    def savebin_analysis(self,
                         file: str | Path,
                         anptr: int = -1):
        """
        Export analysis results in a .npz compressed archive.

        The data is saved in a form analogous to a dict, with the following
        keys:
            - 'raw_xvalues': 1D np.ndarray, the scanned xvalues
            - 'varname': np.array(seqSetup.varname), essentially a str,
                the name of the scanned parameter
            - 'raw_varunit': np.array(seqSetup.varunit), essentially a str,
                the unit of the raw scanned parameter (set in Sequence_Setup)
            - 'analysis.*', with * the keys of self.analyses[anptr],
                the analysis results given by <self.analyze>

        To load the data:

        with np.load(file, allow_pickle=False) as f:
            xvalues = f['raw_xvalues']
            varname = f['varname']
            ...
            freqs = f['analysis.freqs']
            ...

        The use of empty arrays avoids dtype=object arrays
        So that np.savez_compressed does not use the unsafe pickle

        Parameters
        ----------
        file : Union[str, Path]
            DESCRIPTION.
        anptr : int, optional
            Analysis selection. The default is -1 (last analysis).
        """
        # Initialize output data dict
        data_out = {
            'raw_xvalues': self.get_xvalues(),
            'varname': np.array(self.varname),
            'raw_varunit': np.array(self.varunit)
            }

        an = self.get_analysis(anptr)
        for key in an.keys():
            if isinstance(an[key], (list, tuple)) and len(an[key]) > 1:
                for i, data in enumerate(an[key]):
                    data_out["analysis." + key + f".{i}"] = np.array(data)
            else:
                data_out["analysis." + key] = np.array(an[key])

        # Save as compressed archive
        np.savez_compressed(file, **data_out)
        print(f"Analysis results saved at: {file}")


    ########## plot ##########

    def plot(self,
             accessors: list[str | tuple],
             fitIdx: list[int] | int | None = None,
             shape: tuple[int, ...] = None,
             roi_mapper: np.ndarray = None,
             calib: str = None,
             conv_to: str = None,
             conv_from: str = None)-> Plot:
        """


        Parameters
        ----------
        accessors : list
            DESCRIPTION.
        fitIdx : Union[list, int], optional
            DESCRIPTION. The default is None.
        shape : tuple, optional
            DESCRIPTION. The default is None.
        roi_mapper : np.ndarray, optional
            DESCRIPTION. The default is None.
        calib : str, optional
            DESCRIPTION. The default is None.
        conv_to : str, optional
            DESCRIPTION. The default is None.
        conv_from : str, optional
            DESCRIPTION. The default is None.

        Raises
        ------
        ValueError
            DESCRIPTION.
        IndexError
            DESCRIPTION.

        Returns
        -------
        Plot
            DESCRIPTION.

        """
        if not isinstance(accessors, list):
            raise TypeError(
                f"accessors must be specified as a list, not {type(accessors)}")
        # Fetch data
        data = [self.get_data(acc) for acc in accessors]
        stdev_data = [self.get_stdev(acc) for acc in accessors]
        for i in range(len(data)):
            data[i] = cast_2d(data[i])
            stdev_data[i] = cast_2d(stdev_data[i])
        # check consistency of number of data
        nbdata = [len(d) for d in data]
        if np.unique(nbdata).size != 1:
            raise ValueError("all the data to be plotted must have the same "
                             "number of sub-datasets to plot")
        nbdata = nbdata[0]
        # set xvalues, var unit
        x = self.get_xvalues(calib, conv_to, conv_from)
        vunit = self.varunit if conv_to is None else conv_to

        # initialize Data_Set plot
        DSplot = Plot(nbdata, shape, roi_mapper)
        # plot data
        for acc, y, yerr in zip(accessors, data, stdev_data):
            DSplot.plot_data(x, y, yerr, plot_kw={'label': str(acc)})
        # Check and get fit indices
        fits = []
        if fitIdx is not None:
            fits = [fitIdx] if isinstance(fitIdx, Integral) else fitIdx
            if max(fits) >= (l:=len(self.fits)):
                raise IndexError(
                    f"asking for a fit out of range (only {l} fits available)")
        # plot fit/simulation
        for i in fits:
            # check consistent var unit
            fit = self.fits[i]
            if (vu:=fit.info['varunit']) != vunit:
                raise ValueError("incompatible varunits between data and fit "
                                 f"{i}: `{vunit}` and `{vu}` respectively")
            # plot
            if fit.mode == "function":
                fit_kw = {'label': f"fit {i}"}
                DSplot.plot_fct(fit.popt, fit.get_fct(), fit_kw)
            elif fit.mode == "simulation":
                sim_kw = {'marker': "", 'linestyle': "-",
                          'label': f"simu {i}"}
                DSplot.plot_data(fit.xsim, fit.ysim, plot_kw=sim_kw)

        # set title, x/y labels, numerotation
        DSplot.set_title(self.name)
        DSplot.set_xlabel(f"{self.varname} ({vunit})")
        DSplot.set_ylabel("utterly relevant quantity")
        DSplot.add_plot_indices()
        #
        return DSplot






