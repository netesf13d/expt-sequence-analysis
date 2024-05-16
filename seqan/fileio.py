# -*- coding: utf-8 -*-
"""
Utility functions pertaining to data saving and loading on disk.

The functions are:
    - <save_to_txt>, to save data and labels in a text file

A class is implemented:
    - <Solis_ROI_Data>, class implementing method to deal with data from
      Solis csv files.

"""

import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import SelectTypes, data_selector

# =============================================================================
#
# =============================================================================


def save_to_txt(filename: str,
                names: list,
                data: np.ndarray,
                units: list = None,
                comments: list = None):
    """
    TODO use numpy savetxt
    Save data to a file. The file is formatted so that it is conveniently
    loaded on Origin.

    For instance, for
    names = ['name1', 'name2', 'name3', ...]
    units = ['unit1', 'unit2', 'unit3', ...]
    comments = ['com1', 'com2', 'com3', ...]
    data = [[data1_1 data1_2 data1_3 ...]
            [data2_1 data2_2 data2_3 ...]
            [data3_1 data3_2 data3_3 ...]
            ....]

    The output file has the form:

    name1  , name2  , name3  , ...
    unit1  , unit2  , unit3  , ...
    com1   , com2   , com3   ,
    data1_1, data2_1, data3_1, ...
    data1_2, data2_2, data3_2, ...
    data1_3, data2_3, data3_3, ...
    ...    , ...    , ...    , ...

    Note that data[i, :] is the i-th column of data in output text file.

    Parameters
    ----------
    filename : str
        Name of the file to be saved, including path to directory.
    names : list[str]
        Names corresponding to the various entries of the data.
    data : np.ndarray
        The data that is to be exported.
    units : list[str], optional
        Units of the various data. The default is None.
    comments : list[str], optional
        Comments. The default is None.

    Returns
    -------
    None.

    """
    if (ln := len(names)) != (ld := len(data)):
        raise ValueError("Data and names must have the same length. But "
                         f"len(names) = {ln}, len(data) = {ld}")

    if units is not None:
        assert (lu := len(units)) == (ld := len(data)), \
            "Data and units must have the same length. But " \
            f"len(units) = {lu}, len(data) = {ld}"

    if comments is not None:
        assert (lc := len(comments)) == (ld := len(data)), \
            "Data and units must have the same length. But " \
            f"len(comments) = {lc}, len(data) = {ld}"

    with open(filename, 'wt') as file:
        file.write(','.join(names))
        if units and comments:
            file.write('\n' + ','.join(units))
            file.write('\n' + ','.join(comments))
        elif units:
            file.write('\n' + ','.join(units))
        elif comments:
            file.write('\n' + ','*(len(names)-1))
            file.write('\n' + ','.join(comments))
        else:
            pass

        for i in range(data.shape[1]):
            file.write('\n' + ','.join(data[:, i].astype(str)))


# =============================================================================
# Solis csv reader class
# =============================================================================


class Solis_ROI_Data():

    dataEntries = ("Value    \    Scan",
                   "Time",
                   "Standard Deviation",
                   "Mean",
                   "Maximum",
                   "Total Counts",
                   "Signal To Noise")
    nb_entries = len(dataEntries)


    def __init__(self, fname: str):
        """
        Instantiate ROI data by loading a csv file such as output by solis.

        Parameters
        ----------
        fname : str
            The file to be loaded (including path to dir)

        Returns
        -------
        None.

        """

        self.fname = fname

        self.nb_rois = 0
        self.roi_loc = None
        self.nbFrames = None
        self.data = {}

        self.load_data(fname)


    def load_data(self, fname: str):
        """
        Load ROI data exported by Solis software in csv format.
        The structure of the input file is:

        # Acquisition ,                      , ,,,,,...
        "ROI 1"       ,                      , ,,,,,...
                      , "Value    \    Scan" , data...
                      , "Time"               , data...
                      , "Standard Deviation" , data...
                      , "Mean"               , data...
                      , "Maximum"            , data...
                      , "Total Counts"       , data...
                      , "Signal To Noise"    , data...
                      ,                      , ,,,,,...
                      , "ROI 2"              , ,,,,,...
                      , "Value/Scan"         , data...
                      , "Time"               , data...
                      ...
                      ...
                      , ROI, Left, Bottom, Width, Height,
                      , 1  , xxx , xxx   , xxx  , xxx   ,
                      , 2  , ...
                      ...

        The data is stored in attribute <data> as a dict of dict such that:
        data["ROIx"][entry] = np.array([data], dtype=float)
        with:
            - x an integer representing the ROI index (usually 1 .. 9)
            - entry a str in dataEntries
        The array is of shape (nbFrames,)

        Additionally, the data relative to ROI localization is stored in
        attribute roi_loc in the form of a pandas DataFrame.

        Note that the ROI indexation starts at 1.

        Parameters
        ----------
        fname : str
            csv file to be loaded, path and file name.

        Returns
        -------
        None.

        """

        with open(fname,'rt')as file:

            data = csv.reader(file)
            loc_roi = False
            roi_localization = []

            # go over the rows of the file to initialize the data arrays
            for row in data:
                # Indicates that ROI localization data comes next
                if row[1] == 'ROI':
                    loc_roi = True # Switch to filling ROI localization info

                # Find a line that defines a new ROI dataset instantiate an ROI
                if row[0].startswith('ROI '):
                    # Add ROI
                    self.nb_rois += 1
                    ROI = 'ROI' + str(self.nb_rois)
                    self.data[ROI] = {}
                    # entry_counter = 0

                # Fill with entry data
                if row[1] and not loc_roi and not row[0]:
                    self.data[ROI][row[1]] = \
                        np.asarray([float(str_.replace(',', '.'))
                                    for str_ in row[2:]])
                    # entry_counter += 1

                # Set ROI localization data
                if loc_roi:
                    if row[1] == 'ROI':
                        roi_localization.append(
                            [str_ for str_ in row if str_])
                    elif row[1]:
                        roi_localization.append(
                            [int(str_) for str_ in row if str_])

        # Use last imported data to retrieve number of frames
        ## TODO update
        if loc_roi:
            pass
            # self.roi_loc = pd.DataFrame(np.asarray(roi_localization[1:]),
            #                             columns=roi_localization[0])
        self.nbFrames = np.shape(self.data[ROI]["Value    \    Scan"])[0]


    def get_data(self, ROIs: list, entry, dslice=slice(None))-> np.ndarray:
        """
        Get the data relative to "entry" of given ROI.
        Set dslice as a slice to select a subset of the data.
        """
        ROIkeys = ['ROI' + str(ROI) for ROI in ROIs]
        return np.array([self.data[ROIkey][entry][dslice]
                         for ROIkey in ROIkeys])


    def get_histogram(self, ROIkey: str, entry: str,
                      bins=None, dslice: slice=slice(None)):
        """
        Get the histogram data (frequencies, bin_edges) relative to "entry"
        of given ROI.
        Set dslice as a slice to select a subset of the data.
        """
        if bins == None:
            bins = self.nbFrames // 10

        return np.histogram(self.data[ROIkey][entry][dslice], bins)


# =============================================================================
# Solis csv reader class
# =============================================================================


class Sequence_Manager_Data():

    DATA_SUFFIX = ".dat"


    def __init__(self, file: str):
        """
        Instantiate Sequence manager data by loading a txt file such as
        output by solis.

        Parameters
        ----------
        fname : str
            The file to be loaded (including path to dir)

        Returns
        -------
        None.

        """
        self.file = Path(file)
        self.fpath = self.file.parents[0]
        self.fname = self.file.stem

        self.nbstep = 0 # int, number of scan points
        self.nbscan = 0 # number of scans
        self.xvalues = None # 1D np.ndarray
        self.varname = "" # str
        self.varunit = "" # str

        self.rawData = {} # dict
        self.data = {} # dict
        self.parseData = {} # dict

        loadok = self._load()
        if loadok == 0:
            self._parse_data()


    def _load(self)-> int:
        """
        TODO READ ARRIVAL TIMES
        Load data from the given file.

        Raises
        ------
        ValueError
            If there is no data to load from the file (eg aborted sequence).

        Returns
        -------
        0 if things got well
        -1 if there was no useful data to be found

        """
        # load raw data
        for x in self.file.iterdir():
            if x.is_file() and x.suffix == self.DATA_SUFFIX:
                self.rawData[x.stem] = pd.read_table(
                    x, sep='\t', engine="python", encoding_errors="replace")

        if not self.rawData:
            raise ValueError(
                f"file `{self.file}` has no data to load")

        # remove aborted or laser-unlocked sweeps
        has_sweeps = True # no sweeps RUN0, RUN1, .. is there was only one
        td = next(iter(self.rawData.values())).to_numpy()[:, 3:]
        if td.size == 0:
            has_sweeps = False
            td = next(iter(self.rawData.values())).to_numpy()[:, 1:3]
        test = np.all(np.logical_or(td == 0., np.isnan(td)), axis=0)
        # All the data is empty: raise a warning
        if np.all(test):
            warnings.warn(
                f"load Sequence Manager Data: data `{self.file}` is empty",
                RuntimeWarning
                )
            return -1

        # Keep data that is not empty
        if has_sweeps:
            self.nbscan = np.sum(~test)
            keep_runs = np.concatenate((np.ones((3,), dtype=bool), ~test))
        else:
            self.nbscan = 1
            keep_runs = np.ones((3,), dtype=bool)

        for key in self.rawData.keys():
            newkey = key.split('_', 1)[-1]
            rdata = self.rawData[key]
            self.data[newkey] = rdata.iloc[:, keep_runs]
        return 0


    def _parse_data(self,):
        for k, v in self.data.items():
            self.parseData[k] = v.iloc[:, 1].to_numpy()
            self.parseData['stdev_' + k] = v.iloc[:, 2].to_numpy()
        self.varname = v.columns[0]
        self.varunit = self._infer_unit(self.varname)
        self.xvalues = v.iloc[:, 0].to_numpy()
        self.nbscan = len(v.columns) - 3
        self.nbstep = len(self.xvalues)


    def _infer_unit(self, varname: str)-> str:
        # TODO
        varunit = ""
        if varname == "duration":
            varunit = "us"
        elif varname == "frequency":
            varunit = "GHz"
        elif varname == "startTime":
            varunit = "us"
        return varunit


    def get_dict(self):
        SMDdict = {
            'data_type': np.array("sequence_manager"),
            'name': np.array(self.fname),
            'raw_xvalues': np.array(self.xvalues),
            'raw_varunit': np.array(self.varunit),
            'xvalues': np.array(self.xvalues),
            'varunit': np.array(self.varunit),
            'varname': np.array(self.varname),
            'nb_parameters': np.array(0),
            # 'nbroi': np.array(1),
            # 'nbscan': np.array(self.nbscan),
            # 'nbstep': np.array(self.nbstep),
            }
        for k, v in self.parseData.items():
            SMDdict['data.' + k] = v
        return SMDdict


    def get_datakeys(self):
        return self.data.keys()


    def get_data(self, key: str,
                 to_numpy: bool = True)-> list:
        """
        Get the data pertaining to key.
        to_numpy converts the result from pandas DataFrame to numpy array
        """
        if to_numpy:
            return self.data[key].iloc[:, :3].to_numpy()
        else:
            return self.data[key].iloc[:, :3]


    def get_scan_data(self, key: str,
                       scans: SelectTypes = None,
                       to_numpy: bool = True)-> list:
        """
        Get selected scan data pertaining to key
        to_numpy converts the result from pandas DataFrame to numpy array
        """
        sel = np.concatenate(
            (np.zeros((3,), dtype=bool), data_selector(self.nbscan, scans)))
        if to_numpy:
            return self.data[key].iloc[:, sel].to_numpy()
        else:
            return self.data[key].iloc[:, sel]


