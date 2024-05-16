# -*- coding: utf-8 -*-
"""
Data processing and analysis package developed during my PhD.

This package is dedicated to analyze data obtained from atomic physics
experiments carried in the single-atom regime. The specificity of these
experiments is that we essentially detect whether an atom is still present
after an experimental sequence or not. Thus, in addition to the standard data
fitting and plotting functionality, some tools are implemented to analyze the
raw atomic signal and convert it to the desired statistical quantity (eg the
recapture probability).

More detailed explanations about the data processing and analysis process can
be found in my thesis, more specifically in chapter 3.

The package provides:
* Analysis of atomic florescence in regions of interest (ROIs) and atom
  detection.
* Determination of various events (eg loss/recapture of a trapped atom), and
  computation of derived quantities (eg probability of recapture from many
  repeated experiments, average over ROIs, etc).
* Analysis and plotting of processed data.

The code was written while I was rushing to finish writting my thesis.
Although it was extensively tested and is provided with some examples, the 
code is somewhat lacking documentation and could be improved.
"""

from .utils import General_Accessor

from .fileio import (
    Solis_ROI_Data,
    Sequence_Manager_Data
    )

from .roi_manager import ROI_Manager

from .seqsetup import Sequence_Setup

from .stats import Atom_Stats, merge_stats

from .trapanalysis import Trap_Analyzer

from .dataset import Data_Set, merge_datasets

from .data_analysis import Data_Analyzer

from .analysis import (Fit_Result, analysis, fctDict)

from .calib.calib import *
