# -*- coding: utf-8 -*-
"""
This module implements the Sequence_Setup class.
An instance of Sequence_Data holds all the information relevant to an
experimental sequence: how to structure the data, the type of scan,
location of the data, along with some metadata.


A sequence is decomposed as follows:

    aabbccdd aabbccdd ... aabbccdd

- The sequence is the whole acquisition
- Each subsequence is called a scan (eg `aabbccdd`)
- A scan point is a series of identical letters (eg `aa`, `bb`, etc)
- Each letter represents a run
- The number of repetitions is the number of runs at a given scan point
  (eg the scan point `aa` has 2 runs, hence 2 repetitions)
- Due to software idiosyncrasies, some triggers may be missed at the end of a
  scan point, resulting in irrelevant frames that must be removed

For instance the sequence
    aaaxbbbxcccxdddx aaaxbbbxcccxdddx
has:
    - trigmiss = 1
    - nbscan = 2
    - repperstep = 3
    - nbrep = 2*3 = 6

Inside each run, multiple image acquisitions are done, yielding frames.
For instance, a run [--f1---f2--] may consist in the acquisition of two
frames f1 and f2.

The collection of frames f1 for each run is called a congruent frame set,
that is, the first frames of the runs are said to be congruent, and
called cframes in the code.


The class implements the following useful methods:
Private methods are:
    - <_check_consistency>, check the consistency of the parameters
    - <_update_settings>, update settings when parameters are set
Useful methods are:
    - <get_dict>, get the Sequence_Setup in dict format
    - <get_path>, get the file path pointing to the data location
    - <set_path>, set the file path pointing to the data location
    - <set_param>, to set a Sequence_Data attribute
    - <save_sequence> to save the Sequence_Data in json format
    - <filename> to get a filename stem, useful to build other filenames
      (eg figures to save)
    - <cframe_slices> to split a list of frames in sets of congruent frames
    - <deletion_slices> to remove rubbish data (eg the laser unlocked)
    - <scanpt_selectors> to split sets of congruent frames in subsets that
      correspond to the same scan point
    - <scanned_values> to get the values of the scanned parameter

"""

import datetime
import json
import warnings
from numbers import Integral, Real
from pathlib import Path

import numpy as np


# =============================================================================
#
# =============================================================================

## Allowed settings of a Sequence setup
settings = {
    'name': str, # the name of the file eg CA001
    'date': str, # date and time at which the data was saved
    'nbroi': Integral, # number of ROIs corresponding to tweezers, NOT the ones for the background or whatever
    'nbframes': Integral, # number of frames
    'frameperrun': Integral, # number of frames per run of the sequence. Usually 1 or 2
    'nbrun': Integral, # number of congruent frames, after removal of unwanted data (deletion slices)
    'deletionslice': tuple, # contains tuples from which slices are built to discard irrelevant data
    'trigmiss': Integral, # number of triggers missed when switching to the next scan point
    'nbscan': Integral, # number of scans (ie number of times scanning through the step)
    'repperstep': Integral, # number of repetitions per step
    'nbstep': Integral, # number of steps for the scanned parameter
    'nbrep': Integral, # number of repetitions per step (= rep per scan * nb scan)
    'step': Real, # step between successive values of the parameter
    'vari': Real, # initial value of the scanned parameter
    'varf': Real, # final value of the scanned parameter
    'varname': str, # name of the scanned parameter
    'varunit': str, # unit of the scanned parameter
    'parameters': dict, # additional parameters relevant to the sequence (eg power of a laser, position of a beam, ...)
    'info': str, # information relative to the sequence
    }

## Old names for the sequence attributes
## To be sought for in case the current ones fail (for old sequence setups)
aliases = {
    'nbroi': ['nbROI', 'nb_ROI',],
    'nbframes': ['nb_frames',],
    'frameperrun': ['frames_per_rep', 'fpr'],
    'nbrun': ['nbcframes',],
    'trigmiss': ['trig_miss',],
    'nbscan': ['nbsweep', 'nb_runs',],
    'repperstep': ['reppersweep', 'rep_per_step', 'reps_per_step'],
    'nbstep': ['nb_step', 'nb_steps',],
    'nbrep': ['nb_rep', 'nb_reps',],
    'vari': ['start',],
    'varf': ['end',],
    'varname': ['scan_variable',],
    'varunit': ['unit',],
    }

def get_from_alias(key: str):
    for k, v in aliases.items():
        if key in v:
            return k
    raise KeyError(key)


# =============================================================================
#
# =============================================================================

class Sequence_Setup():
    """
    Class to hold settings relative to scan sequences.
    """

    def __init__(self, file: str | Path | None = None, **kwargs):
        """
        Initialization of a Sequence_Data instance, either from a file or
        by specifying the scan settings. The possible settings are given
        with some description in the dict settings above.

        The following is an example of acquisition parameters dict:

        acqDict = {
            'name': "test",
            'date': "",
            'nbroi': 0,
            'nbframes': 1020,
            'frameperrun': 2,
            'nbrun': 0,
            'deletionslice': (),
            'trigmiss': 0,
            'nbscan': 2,
            'repperstep': 25,
            'nbstep': 10,
            'nbrep': 50,
            'step': 1.,
            'vari': 0.,
            'varf': 9,
            'varname': "test",
            'varunit': "None",
            'parameters': {},
            'info': "Test sequence",
            }

        Parameters
        ----------
        file : [Path, str], optional
            Path to a file containing sequence data to be loaded.
            (such files end with '_seq.json'). In case a file is passed
            as argument, keyword arguments are ignored.
            The default is None.
        **kwargs : Any
            The settings given at the creation of the instance. Valid
            settings are:
                - 'name': str
                - 'date': str
                - 'nbroi': int
             (!)- 'nbframes': int
             (!)- 'frameperrun': int
                - 'nbrun': int
                - 'deletionslice': tuple of slice
             (!)- 'trigmiss': int
             (!)- 'nbscan': int
             (!)- 'repperstep': int
             (!)- 'nbstep': int
             (!)- 'nbrep': int
             (!)- 'step': int
                - 'vari': float
                - 'varf': float
                - 'varname': str
                - 'varunit': str
                - 'parameters': dict
                - 'info': str
            The settings preceded by (!) are necessary, an exception
            will be raised otherwise.

        Raises
        ------
        TypeError
            If one of the specified sequence parameters has incorrect type.

        Returns
        -------
        None.

        """

        # Initialize the dictionnary of metadata to default values
        self._settings = {
            'name': "",
            'date': "",
            'nbroi': 0,
            'nbframes': 0,
            'frameperrun': 0,
            'nbrun': 0,
            'deletionslice': (),
            'trigmiss': 0,
            'nbscan': 0,
            'repperstep': 0,
            'nbstep': 0,
            'nbrep': 0,
            'step': None,
            'vari': None,
            'varf': None,
            'varname': "",
            'varunit': "",
            'parameters': {},
            'info': "",
            }

        if file is None:
            self._path = None
        else:
            self._path = Path(file).resolve().absolute().parent # pathlib.Path

        # boolean to indicate whether the scanning parameters are set
        self._scanSet = False

        # Load settings from file
        if file:
            with open(file, 'rt') as f:
                self._settings = json.load(f)
        # Settings from kwargs
        else:
            for k, v in kwargs.items():
                try:
                    vtype = settings[k]
                except KeyError: # invalid key
                    warnings.warn(
                        f"Sequence_Setup: {k} is not a valid parameter, will "
                        "be ignored.",
                        UserWarning)
                    continue
                # check that the input parameter has correct type
                if v is not None and not isinstance(v, vtype):
                    raise TypeError(
                        f"wrong type for parameter {k}: {type(v)}, "
                        f"expected {vtype}")
                self._settings[k] = v

        self._update_settings()


    def __getattr__(self, key):
        """
        Access the items of _settings as if they were attributes.
        """
        try: # try the current name
            return self._settings[key]
        except KeyError: # try old names for the attribute
            for k in aliases[key]:
                try:
                    return self._settings[k]
                except KeyError:
                    pass
        raise KeyError(key)


    def __repr__(self):
        return str(self._settings)


    def _check_consistency(self):
        """
        Check consistency of set parameters
        """
        nbf = self.nbframes
        fpr = self.frameperrun
        nbsc = self.nbscan
        nbstp = self.nbstep
        rps = self.repperstep
        tmiss = self.trigmiss

        tempfr = np.arange(nbf)
        for slc in self.deletion_slices():
            tempfr = np.delete(tempfr, slc, axis=0)

        nbf2 = len(tempfr)

        if nbf2 != (exf := fpr * nbsc * nbstp * (tmiss+rps)):
            raise ValueError(
                f"Inconsistent number of data frames: got {nbf2} but "
                f"expected {exf} from scan setup")

        if nbsc * rps != self.nbrep:
            raise ValueError(
                f"Inconsistency between nbrep: {self.nbrep}, nbscan: "
                f"{nbsc}, repperstep: {rps}.")


    def _update_settings(self)-> None:
        """
        Construct missing entries from the ones already provided. The
        relevant entries are set.
        """
        self._check_consistency()
        # set date
        if not self._settings['date']:
            self._settings['date'] = str(datetime.datetime.now())

        # Set number of congruent frames
        frames = np.arange(self.nbframes)
        for delSlice in self.deletion_slices():
            frames = np.delete(frames, delSlice)
        self._settings['nbrun'] = len(frames) // self.frameperrun

        # complete the scanning parameters
        vari = self.vari
        varf = self.varf
        step = self.step
        nbstep = self.nbstep
        # Complete from starting point, step, number of steps
        if vari != None and step != None and nbstep:
            try:
                self._settings['varf'] = vari + step*(nbstep-1)
            except ZeroDivisionError:
                self._settings['varf'] = vari
            self._scanSet = True
        # Complete from initial value, final value and number of steps
        elif vari != None and varf != None and nbstep:
            try:
                self._settings['step'] = (varf-vari) / (nbstep-1)
            except ZeroDivisionError:
                self._settings['step'] = 0.
            self._scanSet = True

        # Update old parameter names for new parameter names
        for key in aliases.keys():
            self._settings[key] = self.__getattr__(key)
        # Remove parameters not in settings
        for key in set(self._settings.keys()).difference(settings.keys()):
            del self._settings[key]


    def get_dict(self,):
        self._update_settings()
        return self._settings


    def get_path(self,):
        return self._path


    def set_path(self, path: str | Path):
        path = Path(path).resolve().absolute()
        if self._path:
            warnings.warn(f"replacing path {self._path} with {path}",
                          UserWarning)
        self._path = path


    def set_param(self, **kwargs)-> None:
        """
        Set a parameter.
        In the case of attribute 'parameters', the existing parameters
        are updated rather than overwritten.

        Use: self.set_attribute(nbroi=9, roishape=(3, 3)) to set
        parameters nbroi and roishape for instance.

        Parameters
        ----------
        **kwargs : Any
            The parameters that are to be set.

        Raises
        ------
        TypeError
            If one of the specified sequence parameters has incorrect type.

        """
        for k, v in kwargs.items():
            if k in settings.keys():
                # check that the input parameter has correct type
                if v is not None and not isinstance(v, settings[k]):
                    raise TypeError(
                        f"wrong type for setting parameter {k}: {type(v)}, "
                        f"expected {settings[k]}")
                # set the value if valid
                elif k == 'parameters': # the dict of parameters is updated rather than erased
                    self._settings[k].update(v)
                else:
                    self._settings[k] = v
            else: # invalid key: warn and ignore
                warnings.warn(
                    f"Sequence_Setup: {k} is not a valid parameter, will be "
                    "ignored.",
                    UserWarning)
        self._update_settings()


    def save_sequence(self, path: str | Path | None = None)-> None:
        """
        Save the settings in standardized json format, allowing for
        easy peasy loading.
        """
        self._update_settings()

        if path is None:
            if self._path is None:
                raise ValueError("path is not set: use <set_path>")
        else:
            self.set_path(path)
        self._path.mkdir(parents=True, exist_ok=True)

        fout = self._path / (self.name + "_seq.json")
        if not fout.exists():
            with open(fout, 'wt') as file:
                file.write(json.dumps(self._settings, indent=4))
            print(f"Sequence data saved at {fout}")
        else:
            msg = (f"sequence setup file already exists with the name: '{fout}'"
                   f"- If you are reloading the data, call Sequence_Setup(file='{fout}')."
                   "- If you actually want to erase the existing file, delete it manually."
                   "I know it's a pain to not repeatedly do the same thing, blindly, over and"
                   "over, but it prevents you from erasing other sequence setups or some of"
                   "their parameters. I hope you understand.")
            raise PermissionError(msg)


    def filename(self, extension: str)-> str:
        """
        Get the filename path + name + extension.

        This is useful to get filenames for loading or saving data, e.g.
        path + name + ".csv" to load camera data,
        path + name + "_1st_img_histo.pdf" to save an histogram

        Parameters
        ----------
        extension : str
            Extension of the file that contains the data, including the dot.
            (eg '.csv')
        """
        return str(self._path / (self.name + extension))


    def cframe_slices(self)-> list[slice]:
        """
        Get the slices to decompose the frames into sets of congruent
        frames.

        Basically, for n frames per run, frames k*n + i belong to
        cframe i.

        Raises
        ------
        ValueError
            If the number of frames per run has not been set.

        """
        fpr = self.frameperrun
        if fpr == 0:
            raise ValueError("Number of frames per run is not set.")

        return [slice(i, None, fpr) for i in range(fpr)]


    def deletion_slices(self)-> list[slice]:
        """
        Build the list of deletion slices from the data contained
        in attribute deletionslice.

        Returns
        -------
        list of slices
            The list of slices built from deletionslice.

        """
        return [slice(*slc) for slc in self.deletionslice]


    def scanpt_selectors(self,
                         scanSelector: np.ndarray | None = None)-> np.ndarray:
        """
        Compute selectors for the data relative to each scan point. The
        selectors apply to each congruent set of frames.

        The scans from which the selectors are built are set with
        scanSelector.

        For a given scan point, the selector takes the form of an array of
        bool.
        selector[j, k] is True if run k belongs to scan point j and scan
        number i is selected by scanSelector

        Parameters
        ----------
        scanSelector : 1D np.ndarray of bool, optional
            scanSelector[i] = True if scan number i is to be selected
                              False otherwise
            The default is None, in which case all scans are selected.

        Raises
        ------
        AttributeError
            If the necessary acquisition parameters are not set.
        ValueError
            If scanSelector is improperly set.

        Returns
        -------
        2D np.ndarray of bool
            shape (nbscan, nbrun)
            selector[i, j] is True if cframe j belongs to scan point i.

        """
        self._update_settings()
        nbscan = self.nbscan
        nbstep = self.nbstep
        rps = self.repperstep
        trigmiss = self.trigmiss

        # AttributeError if the necesssary acquisition parameters are not set
        if not (nbstep and rps and nbscan):
            raise AttributeError(
                "acquisition parameters nb of step or nb of scans or "
                "repetition per scant are not set.")

        if scanSelector is None:
            scanSelector = np.ones((nbscan,), dtype=bool)
        # ValueError if scanSelector is incorrectly specified
        if (sh:=scanSelector.shape) != (nbscan,):
            raise ValueError(
                "`runSelector` should be specified as an array of shape "
                f"{(nbscan,)}, not {sh}")

        def scanpt_selector(x, scanIndex):
            scanPtLen = (rps + trigmiss)
            scanLen = nbstep * scanPtLen
            xrun = scanSelector[(x // scanLen)]
            x2 = (xrun-1) + xrun * (x % scanLen)
            return (x2 // scanPtLen == scanIndex) \
                    * (x2 % scanPtLen < rps)

        return np.array([scanpt_selector(np.arange(self.nbrun), i)
                         for i in range(nbstep)],
                        dtype=bool)


    def scanned_values(self)-> np.ndarray:
        """
        Get the values array([x1, x2, ..., xk]) of the scanned parameter
        at each scan point.

        Raises
        ------
        AttributeError
            If the scan parameters (initial value, final value, step,
            number of steps) are not set.

        """
        if not self._scanSet:
            raise AttributeError("Scan parameters are not set.")

        vari = self.vari
        nbstep = self.nbstep
        step = self.step

        rawscan = vari + np.arange(0, nbstep)*step

        return rawscan
