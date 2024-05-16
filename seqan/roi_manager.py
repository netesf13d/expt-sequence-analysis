# -*- coding: utf-8 -*-
"""
This module implements the ROI_Manager class.

Initially thought to handle the mapping of physical trapping sites to abstract
ROIs.

The methods are:
    - <_check_init>,
    - <_load_from_dict>,
    - <_load_calib>,

    - <set_tgt_rois>,
    - <set_clusters>,
    - <save_clusters>,
    - <get_roi_mapper>,


The utility functions are:
    - <get_item>,


TODO doc

"""

import json
from numbers import Integral
from pathlib import Path

import numpy as np



# =============================================================================
# FUNCTIONS -
# =============================================================================

## Allowed settings of a ROI_Manager
settings = {
    # 'name': str, # the name of the file eg CA001
    'nbroi': Integral, # number of ROIs corresponding to tweezers, NOT the ones for the background or whatever
    'position': (tuple, list, np.ndarray), # localization of the ROIs
    'roishape': (tuple, list, np.ndarray), # shape of the ROI array, set to None if not rectangular
    'tgtrois': (tuple, list, np.ndarray), # target ROIs, to be filled with the rearranger
    'clusters': (tuple, list, np.ndarray), # clusters
    'info': str, # information relative to the sequence
    }

## Aliases for the ROI_Manager attributes
## To be sought for in case the current ones fail (for old sequence setups)
aliases = {
    'nbroi': ['nbROI', 'nb_ROI',],
    'position': [],
    'roishape': ['ROIshape', 'ROI_shape',],
    'roi_mapper': ['ROI_mapper',],
    'tgt_rois': ['target_rois', 'atom_target'],
    'clusters': [],
    }

def get_item(source: dict, key: str):
    try:
        return source[key]
    except KeyError:
        for k in aliases[key]:
            try:
                return source[k]
            except KeyError:
                pass
    return None


# =============================================================================
#
# =============================================================================


class ROI_Manager():
    """
    A class to manage ROIs
    """

    def __init__(self, *,
                 source: dict | str | Path = None,
                 calib: dict | str | Path = None,
                 **kwargs,):
        """


        Parameters
        ----------
        source : Union[dict, str, Path], optional
            DESCRIPTION. The default is None.
        calib : Union[dict, str, Path], optional
            DESCRIPTION. The default is None.

        """
        self.calib = None #

        self.nbroi = 0
        self.position = None # 2D np.ndarray
        self.roishape = None # tuple
        self.roi_mapper = None # 2D np.ndarray

        self.tgt_rois = None # 1D np.ndarray
        self.nbtgtroi = 0 # int

        self.clusters = None # 2D np.ndarray
        self.nbcluster = 0 # int
        self.csize = 0 # int

        # Load from file or dict
        if isinstance(source, (str, Path)):
            try:
                with np.load(source, allow_pickle=False) as f:
                    self._load_from_dict(f)
            except (ValueError, OSError):
                with open(source, 'rt') as f:
                    self._load_from_dict(json.load(f))
        elif isinstance(source, dict):
            self._load_from_dict(source)
        else:
            self._load_from_dict(kwargs)
        # final check
        self._check_init()


    def _check_init(self,):
        """
        Check that the instantiation is correct.
        The number of ROIs must be an int > 0.
        """
        if self.nbroi is None or self.nbroi <= 0:
            raise ValueError(f"nbroi must be an int > 0, is {self.nbroi}")


    def _load_from_dict(self, source: dict):
        """


        Parameters
        ----------
        source : dict
            DESCRIPTION.

        """
        self.nbroi = get_item(source, 'nbroi')
        self.position = get_item(source, 'position')
        self.roishape = get_item(source, 'roishape')
        # Load target ROIs
        tgt_rois = get_item(source, 'tgt_rois')
        if tgt_rois is not None:
            self.set_tgt_rois(tgt_rois)
        # Load clusters
        clusters = get_item(source, 'clusters')
        if clusters is not None:
            self.set_clusters(clusters)


    def _load_calib(self, calib):
        pass


    def set_tgt_rois(self, tgt_rois: np.ndarray):
        """
        TODO doc

        Parameters
        ----------
        tgt_rois : 1D np.ndarray of int
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        """
        tgt_rois = np.array(tgt_rois, dtype=int)
        if tgt_rois.size == 0:
            return
        if (d:=tgt_rois.ndim) != 1:
            raise ValueError(
                f"tgt_rois dimension must be equal to 1, is {d}")
        if (m:=np.min(tgt_rois)) < 0 or (M:=np.max(tgt_rois)) > self.nbroi:
            raise ValueError(
                f"tgt_rois must involve ROIs in the range [0, {self.nbroi}], "
                f"got a range [{m}, {M}]")
        self.tgt_rois = tgt_rois
        self.nbtgtroi = len(tgt_rois)


    def set_clusters(self, clusters: np.ndarray):
        """
        TODO doc

        Parameters
        ----------
        clusters : 2D np.ndarray of int
            Array representing the clusters.
            Shape (nb_clusters, cluster_size).
            The format is:
                clusters[i, :] = [j1, j2, ..., jn]
                i is the cluster index,
                j1, j2, ..., jn are the ROI indices that constitute the
                cluster

        Raises
        ------
        ValueError
            DESCRIPTION.
        TypeError
            DESCRIPTION.

        """
        clusters = np.array(clusters, dtype=int)
        if (d:=clusters.ndim) not in {2, 3}:
            raise ValueError(
                f"clusters dimension must be equal to 2 or 3, is {d}")
        if (m:=np.min(clusters)) < 0 or (M:=np.max(clusters)) > self.nbroi:
            raise ValueError(
                f"clusters must involve ROIs in the range [0, {self.nbroi}], "
                f"got a range [{m}, {M}]")
        self.clusters = clusters
        self.nbcluster = self.clusters.shape[0]
        self.csize = self.clusters.shape[1]


    def save_clusters(self, file: str):
        """
        Save thresholds to file using numpy.save.
        """
        if self.clusters is None:
            print("No clusters set, no clusters saved. Peace.")
        else:
            np.save(file, self.clusters)


    def get_roi_mapper(self,):
        """
        TODO doc

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self.roi_mapper is not None:
            return np.copy(self.roi_mapper)

        print("No ROI mapper available, switching to default")
        if self.roishape is None:
            raise ValueError("ROI array shape (`roishape`) is not defined")
        rois, sh = np.arange(self.nbroi), self.roishape
        mapper = np.transpose(np.array(
            [sh[0]-1 - rois % sh[0], rois // sh[0]]))
        return mapper












