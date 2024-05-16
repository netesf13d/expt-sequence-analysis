# -*- coding: utf-8 -*-
"""
This module implements the Data_Analyzer class.

The idea was to have a container holding many Data-Set
"""


import sys

import numpy as np

from .utils import SelectTypes, data_selector
from .stats import Atom_Stats, merge_stats
from .dataset import Data_Set

from .fileio import save_to_txt
from .plots import Plot


# =============================================================================
#
# =============================================================================

class Data_Analyzer():
    """
    TODO
    """

    def __init__(self, *args):
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
        self.datasets = []
        self.analyses = None
        # Plot info and data
        self.figures = None

        for ds in args:
            if not isinstance(ds, Data_Set):
                raise TypeError(
                    f"input data must currently be Data_Set, got {type(ds)}")
            self.datasets.append(ds)



    def _check_compatibility(self, ):
        # TODO
        pass


    def pop_data(self, dataIdx: int):
        return self.datasets.pop(dataIdx)


    def get_data(self, dataIdx: int):
        return self.datasets[dataIdx]


    def merge_data(self, dataSelect: list[int])-> Data_Set:

        pass








