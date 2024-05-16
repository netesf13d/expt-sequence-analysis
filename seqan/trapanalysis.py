# -*- coding: utf-8 -*-
"""
This module implements the Trap_Analyzer class.
A Trap_Analyzer instance takes a set of frames with photon counts for ROIs
to perform analysis of the data at different levels.

Data are first structured into sequences at init, then the analysis is
implemented by methods that can be categorized as follows:

Utility methods.
    - <_discard_data>, remove undesired data using the `deletion slices` set
        in the Sequence_Setup input
    - <_check_consistency>, check compatibility between the data and the
        sequence setup
ROI management
    - <get_roi_manager>, get the ROI_Manager attribute
    - <set_clusters>, set ROI clusters (wrapper around ROI_Manager.set_clusters)
    - <save_clusters>, save ROI clusters (wrapper around ROI_Manager.save_clusters)
    - <set_tgt_rois>, set target ROIs (wrapper around ROI_Manager.set_tgt_rois)
Threshold management. Thresholds are used to binarize the fluorescence
signal, can be set manually, saved, or determined by histogram analysis:
    - <set_thresholds>, set the detection thresholds manually
    - <save_thresholds>, save thresholds in a file (using numpy.save)
    - <compute_thresholds>, compute thresholds by fitting histograms
    - <savebin_histo_data>, save fluorescence data and histogram analysis
    - <savebin_histo_fit>, save histogram analysis (not the fluo data)
Frame analysis, atom detection and population computation. Some tools to
analyze the fluorescence signal:
    - <get_frames>, get frame indices that are above or below a given
        threshold
    - <atom_detection>, binarize the signal, detect the presence of atoms
        in individual ROIs using the thresholds
    - <cluster_pop>, determine the clusters configurations using the binarized
        populations
    - <target_pop>, determine the target ROI configurations using the
        binarized populations
    - <get_dselect>, get the runs which validate a given condition on the
        ROIs
Statistics:
    - <_get_stats>, core function that instanciates and returns an
        Atom_Stats object as required by the caller
    - <get_pair_stats>, compute pair statistics and returns a Data_Set object
        for further analysis
    - <get_cluster_stats>, compute cluster statistics and returns a Data_Set
        object for further analysis
    - <get_cp_stats>, compute cluster pair statistics and returns a Data_Set
        object for further analysis
    - <get_target_stats>, compute target ROI statistics and returns a
        Data_Set object for further analysis
    - <get_tp_stats>, compute target ROI pair statistics and returns a
        Data_Set object for further analysis
Plot:
    - <plot_histogram>, plot fluorescence signal histograms


An utility is implemented:
    - <fetch_thresholds>, build the thresholds in standard format from
        heterogeneous inputs


A sequence is decomposed as the following:

    aabbccdd aabbccdd ... aabbccdd

- The sequence is the whole acquisition
- Each subsequence aabbccdd is called a sweep (ie aabbccdd)
- Each letter represents a run
- The number of repetitions is the total number of runs at a given scan point

Inside each run, multiple image acquisitions are done, yielding frames.
For instance, a run [--f1---f2--] may consist in the acquisition of two
frames f1 and f2.

The collection of frames f1 for each run is called a congruent frame set,
that is, the first frames of the runs are said to be congruent, and
called cframes in the code.


TODO
- doc
- gendoc

"""

import warnings
from pathlib import Path

import numpy as np

from .config import SAFE_THRESHOLD
from .utils import SelectTypes, data_selector
from .roi_manager import ROI_Manager
from .seqsetup import Sequence_Setup
from .stats import Atom_Stats
from .dataset import Data_Set
from .analysis import (double_gaussian, gg_histo_analysis,
                       gausson, gp_histo_analysis)
from .plots import Plot


warnings.filterwarnings("default", category=np.VisibleDeprecationWarning)

# =============================================================================
# FUNCTIONS -
# =============================================================================

def fetch_thresholds(thresholds: str | int | float | list | np.ndarray):
    """
    Fetch thresholds.

    The thresholds can be specified:
    - as a filename, then the thresholds are loaded from the file and
      set as the following.
    - as a number, then the same threshold is applied to all ROIs and
      all cframes
    - as a 1D list or np.ndarray of length = nbROIs, then the list of
      thresholds is broadcasted to all cframes
    - as a 2D list or np.ndarray of shape (nb_cframes, nbROIs), then
      the thresholds are set the obvious way

    Parameters
    ----------
    thresholds : str, int, float, list or np.ndarray
        - str (eg "/path/to/thresholds/super_thresholds.npz")
        - number (eg same_threshold_for_all = 120)
        - 1D list or array [thr_0, thr_1, ..., thr_(nbROI-1)] of
          thresholds specified for each ROI, set common to all cframes
        - 2D list or array thr[i, j] = threshold for ROI j of cframe i

    Raises
    ------
    TypeError
        If thresholds were specified as the incorrect type.

    """
    # Load from file
    if isinstance(thresholds, (str, Path)):
        with np.load(thresholds, allow_pickle=False) as f:
            thr = f
    # Convert number/list/array to array
    elif isinstance(thresholds, (int, float, list, np.ndarray)):
        thr = np.array(thresholds)
    else:
        raise TypeError(f"wrong type for thresholds: {type(thresholds)}")
    return thr


# =============================================================================
# Trap_Analyzer class
# =============================================================================

class Trap_Analyzer():
    """
     doc
    """

    def __init__(self, *,
                 sequenceSetup: Sequence_Setup,
                 roiManager: ROI_Manager,
                 rawData: np.ndarray,
                 thresholds: str | Path | int | list | np.ndarray | None = None,):
        """
        TODO DOC

        Parameters
        ----------
        sequenceSetup : Sequence_Data
            Sequence data parameters.
        rawData : 2D array
            Raw data for atomic signal. The format is:
            rawData[i, j] = signal of i-th ROI in j-th frame
            ROI indexation begins at 0.
            NOTE : Care must be taken to map rawData indices to ROI indices.
        thresholds : str, int, float, list or np.ndarray
            - str (eg "/path/to/thresholds/super_thresholds.npz")
            - number (eg same_threshold_for_all = 120)
            - 1D list or array [thr_0, thr_1, ..., thr_(nbROI-1)] of
              thresholds specified for each ROI, set common to all cframes
            - 2D list or array thr[i, j] = threshold for ROI j of cframe i
            If thresholds are specified, <set_thresholds> is called.
            The default is None, in which case no thresholds are specified.

        """
        ##### Initializtion
        # Sequence info contained in an seqmanager.Sequence_Data object
        self.seqSetup = sequenceSetup
        self.roiManager = roiManager
        self.name = self.seqSetup.date.split(' ')[0]+" - "+self.seqSetup.name

        self._fpr = self.seqSetup.frameperrun

        # Number of ROIs for the dataset, a priori different from the total
        # number of ROIs in the sequence (which might include background
        # ROI, etc)
        self.nbroi = len(rawData) # int

        ## First step of structuration of the data:
        # discard rubbish data (eg laser unlocked during acquisition)
        # set in seqSetup.deletion_slices()
        temp = self._discard_data(rawData)
        # Split the data in sets pertaining to different cframes
        # shape = (frameperrun, nbroi, nbcframes)
        self.data = np.array( # 3D np.ndarray
            [temp[:, slc] for slc in self.seqSetup.cframe_slices()])

        # Attibutes to hold histogram fits and thresholds
        self.histofit = [None] * self._fpr # list[dict]
        self.thresholds = None # 2D np.ndarray
        if thresholds is not None:
            self.set_thresholds(thresholds)

        # ROI and cluster populations
        self.pop = None # 3D np.ndarray, ROI population
        self.clusters = self.roiManager.clusters # 2D np.ndarray
        self.nbcluster = self.roiManager.nbcluster # int
        self.cpop = None # 4D np.ndarray, cluster population
        self.scpop = None # 4D np.ndarray, symmetrized cluster population
        self.tgt_rois = self.roiManager.tgt_rois # 1D np.ndarray
        self.tpop = None # 3D np.ndarray


    def _discard_data(self, rawData: np.ndarray):
        """
        Discard data that is irrelevant, determined by the deletion slices
        set in the sequence setup.

        Usually, it will remove the last frames (with slice(n, None, 1))
        that correspond to an improperly set number of frames or a
        sequence that had a problem at some point.
        """
        tempData = rawData
        for delSlice in self.seqSetup.deletion_slices():
            tempData = np.delete(tempData, delSlice, axis=1)
        return tempData


    def _check_consistency(self,):
        # check compatibility between data and sequence setup
        if self.data.shape[2] != self.seqSetup.nbcframes:
            raise ValueError(
                f"incompatible number of cframes in data ({self.data.shape[2]})"
                f" and sequence setup ({self.seqSetup.nbcframes})"
                )


    ########## ROI management ##########

    def get_roi_manager(self,):
        return self.roiManager


    def set_clusters(self, clusters: np.ndarray):
        """
        Wrapper for ROI_Manager.set_clusters.
        clusters : 2D np.ndarray of int
        """
        if (clusters.ndim == 3
            and (nc:=clusters.shape[-1]) != (cf:=self.seqSetup.nbcframes)):
            raise ValueError(
                f"nb of cluster sets {nc} must match nb of cframes {cf}")

        self.roiManager.set_clusters(clusters)
        self.clusters = self.roiManager.clusters
        self.nbcluster = self.roiManager.nbcluster

        ## reinitialize cluster-dependent quantities
        self.cpop = None
        self.scpop = None


    def save_clusters(self, file: str):
        """
        Wrapper for ROI_Manager.save_clusters.
        """
        self.roiManager.save_clusters(file)


    def set_tgt_rois(self, tgt_rois: np.ndarray):
        """
        Wrapper for ROI_Manager.set_tgt_rois.
        tgt_rois : 1D np.ndarray of int
        """
        self.roiManager.set_tgt_rois(tgt_rois)
        self.tgt_rois = self.roiManager.tgt_rois

        ## reinitialize cluster-dependent quantities
        self.tpop = None


    ########## Threshold management ##########

    def set_thresholds(self,
                       thresholds: str | Path | int | float | list | np.ndarray):
        """
        Set the thresholds.

        The thresholds can be specified:
        - as a filename, then the thresholds are loaded from the file and
          set as the following.
        - as a number, then the same threshold is applied to all ROIs and
          all cframes
        - as a 1D list or np.ndarray of length = nbROIs, then the list of
          thresholds is broadcasted to all cframes
        - as a 2D list or np.ndarray of shape (nb_cframes, nbROIs), then
          the thresholds are set the obvious way

        Parameters
        ----------
        thresholds : str, int, float, list or np.ndarray
            - str or Path (eg "/path/to/thresholds/super_thresholds.npz")
            - number (eg same_threshold_for_all = 120)
            - 1D list or array [thr_0, thr_1, ..., thr_(nbROI-1)] of
              thresholds specified for each ROI, set common to all cframes
            - 2D list or array thr[i, j] = threshold for ROI j of cframe i

        Raises
        ------
        TypeError
            If thresholds were specified as the incorrect type.

        """
        thr = fetch_thresholds(thresholds)
        self.thresholds = np.broadcast_to(thr, (self._fpr, self.nbroi))


    def save_thresholds(self, file: str):
        """
        Save thresholds to file using numpy.save.
        """
        if self.thresholds is None:
            print("No thresholds set, no thresholds saved. Peace.")
        else:
            np.save(file, self.thresholds)
            print(f"Detection thresholds saved at {file}")


    def compute_thresholds(self,
                           cframeIndex: int = 0,
                           binning: int = 15,
                           mode: str = 'gauss'):
        """
        Compute the thresholds by fitting histograms corresponding
        to the given frame data.
        The fitting functions can be either <stats.curvefit_analysis>
        or <stats.gcurvefit_analysis>.

        Parameters
        ----------
        cframeIndex : int
            Index of the congruent set of frames used for threshold
            determination. Usually, it will be the first frame of
            the run, corresponding to index 0.
            The default is (therefore) 0.
        binning : int, optional
            Used in determining the number of bins for the histograms.
            Basically represents the average number of points per bin.
            Higher means less bins. The default is 15.
        mode : {'gauss', 'poisson'}, optional
            'poisson': fit the histograms with a Poissonnian signal
                convoluted with a gaussian background.
            'gauss': fit the histograms with a gaussian signal plus
                a gaussian background. This one is faster.
            The default is 'gauss'.

        Raises
        ------
        ValueError
            If a fitting mode different from {'gauss', 'poisson'}
            is passed as argument.

        """
        modes = ['gauss', 'poisson']
        if mode not in modes:
            raise ValueError(
                f"threshold computation mode must be in {modes}, but {mode} "
                "was passed as argument."
                )

        if SAFE_THRESHOLD: # config variable to select the type of threshold
            thresKey = 'threshold_safe'
        else:
            thresKey = 'threshold'

        # Fit histograms
        if mode == 'gauss':
            res = gg_histo_analysis(self.data[cframeIndex], binning)
            self.histofit[cframeIndex] = res
        elif mode == 'poisson':
            res = gp_histo_analysis(self.data[cframeIndex], binning)
            self.histofit[cframeIndex] = res

        # Set thresholds
        if self.thresholds is None:
            print("compute_thresholds: thresholds not yet set, broadcasting "
                  "computed thresholds to all cframes.")
            self.thresholds = \
                np.broadcast_to(res[thresKey], (self._fpr, self.nbroi))
        else:
            self.thresholds[cframeIndex] = res[thresKey]


    def savebin_histo_data(self,
                           file: str,
                           cframeIndex: int = 0):
        """
        TODO update doc
        Export histogram data relative to given cframe in a .npz compressed
        archive.

        The data is saved in a form analogous to a dict, with the following
        keys:
            - 'cframeIndex': np.array(cframeIndex), essentially an int
            - 'nbROI': np.array(nbROI), essentially an int,
                number of ROIs
            - 'shape': np.array(shape) or empty array,
                the shape of the tweezer array if defined
            - 'data': 2D np.ndarray, the fluorescence data
                data[i, j] = fluorescence signal of ROI i in frame j
            - 'thresholds': 1D np.ndarray of empty array, the thresholds
                thresholds[i] = detection threshold of ROI i
            If histogram analysis data is available:
            - 'histofit.*', with * the keys of self.histofit[cframeIndex]:
                the histogram analysis results (see
                <analysis.fluorescence_analysis.gg_histo_analysis> or
                <analysis.fluorescence_analysis.gp_histo_analysis>)
            If no histogram analysis data is available:
            - 'histofit': the empty array

        To load the data:

        with np.load(file, allow_pickle=False) as f:
            cframeIndex = f['cframeIndex']
            nbroi = f['nbroi']
            shape = f['shape']
            thresholds = f['thresholds']
            histodata = f['data']
            ...
            p_exp = f['histofit.p_exp']
            ...

        The use of empty arrays avoids dtype=object arrays
        So that np.savez_compressed does not use the unsafe pickle

        Parameters
        ----------
        cframeIndex : int, optional
            Index corresponding to the cframe data to export.
            The default is 0.

        """
        # Initialize output data dict
        data_out = {
            'name': np.array(self.name),
            'cframeIndex': np.array(cframeIndex),
            'nbroi': np.array(self.nbroi),
            'data': self.data[cframeIndex]
            }

        # Set shape if it is defined
        if self.roiManager.roishape is not None:
            data_out['shape'] = np.array(self.roiManager.roishape)
        else: # Empty array if shape is not defined
            data_out['shape'] = np.array([])

        # Set thresholds if available
        if self.thresholds[cframeIndex] is not None:
            data_out['thresholds'] = self.thresholds[cframeIndex]
        else:# Empty array if no thresholds are available
            data_out['thresholds'] = np.array([])

        # Set histogram analysis results if available
        if self.histofit[cframeIndex] is not None:
            for k, v in self.histofit[cframeIndex].items():
                data_out['histofit.' + k] = np.array(v)
        else: # Empty array if no histogram analysis is available
            data_out['histofit'] = np.array([])

        # Save as compressed archive
        np.savez_compressed(file, **data_out)
        print(f"cframe {cframeIndex} histogram data saved at: {file}")


    def savebin_histo_fit(self,
                           file: str,
                           cframeIndex: int = 0):
        """
        TODO update doc
        Export histogram data relative to given cframe in a .npz compressed
        archive.

        The data is saved in a form analogous to a dict, with the following
        keys:
            - 'cframeIndex': np.array(cframeIndex), essentially an int
            - 'nbROI': np.array(nbROI), essentially an int,
                number of ROIs
            - 'shape': np.array(shape) or empty array,
                the shape of the tweezer array if defined
            - 'data': 2D np.ndarray, the fluorescence data
                data[i, j] = fluorescence signal of ROI i in frame j
            - 'thresholds': 1D np.ndarray of empty array, the thresholds
                thresholds[i] = detection threshold of ROI i
            If histogram analysis data is available:
            - 'histofit.*', with * the keys of self.histofit[cframeIndex]:
                the histogram analysis results (see
                <analysis.fluorescence_analysis.gg_histo_analysis> or
                <analysis.fluorescence_analysis.gp_histo_analysis>)
            If no histogram analysis data is available:
            - 'histofit': the empty array

        To load the data:

        with np.load(file, allow_pickle=False) as f:
            cframeIndex = f['cframeIndex']
            nbroi = f['nbroi']
            shape = f['shape']
            thresholds = f['thresholds']
            histodata = f['data']
            ...
            p_exp = f['histofit.p_exp']
            ...

        The use of empty arrays avoids dtype=object arrays
        So that np.savez_compressed does not use the unsafe pickle

        Parameters
        ----------
        cframeIndex : int, optional
            Index corresponding to the cframe data to export.
            The default is 0.

        """
        # Initialize output data dict
        data_out = {
            'name': np.array(self.name),
            'cframeIndex': np.array(cframeIndex),
            'nbroi': np.array(self.nbroi),
            }

        # Set histogram analysis results if available
        if self.histofit[cframeIndex] is not None:
            for k, v in self.histofit[cframeIndex].items():
                data_out[k] = np.array(v)
        else: # Error if no histogram analysis is available
            raise ValueError(f"no histo fit at cframe index {cframeIndex}")

        # Save as compressed archive
        np.savez_compressed(file, **data_out)
        print(f"cframe {cframeIndex} histogram fit saved at: {file}")


    ########## Frame analysis, atom detection and cluster population ##########

    def get_frames(self,
                   thresholds: str | int | float | list | np.ndarray,
                   roiSelect: SelectTypes | None = None,
                   out_struct: str = "struct",
                   above_or_below: str = "above"):
        """
        Get frames indices that yield a signal above or below a certain
        threshold.

        Only those frames which have a signal above or below threshold
        for the selected ROIs are taken into account.

        The retuned frames indices are returned as an array of indices.
        The result can be either structured as a single array of frame
        indices ("raw") or as a list of cframe indices ("struct").

        Parameters
        ----------
        thresholds : str, int, float, list, np.ndarray
            Same format as <set_thresholds>.
        roiSelect : Union[int, slice, list, np.ndarray] or None, optional
            Selector of the ROIs to get frames from. Basically, the
            selected ROI indices are ROIs[roiSelect].
            The default is None, which selects all the data.
        out_struct : {"struct", "raw"}, optional
            The structure of the returned frames indices.
            The default is "struct".
        above_or_below : {"above", "below"}, optional
            Get frames indice above or below the given thresholds.
            The default is "above".

        Raises
        ------
        TypeError
            If thresholds were specified as the incorrect type.
        ValueError
            If out_struct or above_or_below are not correctly set.

        Returns
        ------
        list or np.ndarray
            If out_struct == "struct", the format is
              [array([i1, i2, ..., ik0]),
               array([j1, j2, ..., jk1]),
               ...]
              where i1, i2, ... are the retrived frames indices (not cframe
              indices!) in 1st cframe set,
              j1, j2, ... are the retrieved frame indices (not cframe
              indices!) in 2nd cframe set, etc
            If out_struct == "raw", the format is
              array([i1, i2, i3, ..., ik])
              where i1, i2, i3, ... are frame indices (not cframe indices!)

        """
        # Prepare the thresholds
        thr = fetch_thresholds(thresholds)
        thr = np.broadcast_to(thr, (self._fpr, self.nbroi))

        # Manage ROI selector
        roiSel = data_selector(self.nbroi, roiSelect)

        # Get the frame indices (in structured form by default)
        frames = []
        for i in range(self._fpr):
            rthr = thr[i, :].reshape((self.nbroi, 1))
            if above_or_below == "above":
                frames.append(np.unique(np.nonzero(
                    self.data[i, roiSel, :] > rthr[roiSel, :]
                    )[1]))
            elif above_or_below == "below":
                frames.append(np.unique(np.nonzero(
                    self.data[i, roiSel, :] < rthr[roiSel, :]
                    )[1]))
            else:
                raise ValueError(
                    "above_or_below must be `above` or `below`, "
                    f"got {above_or_below}"
                    )

        # Convert cframes indices to frame indices
        frames = [i + self._fpr * frames[i] for i in range(self._fpr)]
        # Convert to output format
        if out_struct == "struct":
            return frames
        elif out_struct == "raw":
            return np.concatenate(frames, axis=0)
        else:
            raise ValueError("mode must be `struct` or `raw`, "
                             f"got {out_struct}")


    def atom_detection(self)-> np.ndarray:
        """
        TODO doc
        From the fluorescence signal, discriminate the presence of an atom in
        each ROI using the thresholds.

        The output array format is the following:
            pop[i, j, 1, k] = True if an atom is present in
                               i-th cframe set
                               j-th ROI
                               k-th cframe
                           False otherwise
            pop[i, j, 0, k] is logical_not(pop[i, j, 1, k])
        The reason to have an addtional axis with redundancy is to have
        a shared structure with cluster populations (ndim = 4) so that both
        can be managed with the same code.
        An ROI is essentially seen as a cluster of size 1.

        Input attributes
        ----------------
        data : 3D np.ndarray of numbers
            Fluorescence data, which conditions the presence or not of an
            atom on the given frame.
            The format is:
                data[i, j, k] = fluorescence signal in
                                    i-th cframe set
                                    j-th ROI
                                    k-th run
        thresholds : 2D array of numbers
            Detection thresholds pertaining to the data for each cframe set
            and ROI.
            thresholds[i, j] = detection threshold for
                                   i-th cframe set
                                   j-th ROI

        Raises
        ------
        AttributeError
            If thresholds are not set.

        Set attribute
        -------------
        pop : 4D np.ndarray of bool
            Atomic population pop, with same shape as data.
            pop[i, j, 1, k] = True if data[i, j, k] > threshold[i, j]
                              False otherwise
            pop[i, j, 0, k] = ~ pop[i, j, 1, k]
        """
        if self.thresholds is None:
            raise AttributeError("thresholds are not set")

        fpr, nbroi, nbcf = self.data.shape
        pop = np.empty((fpr, nbroi, 2, nbcf), dtype=bool)
        # Atom detection
        pop[:, :, 1, :] = self.data > self.thresholds[..., np.newaxis]
        pop[:, :, 0, :] = np.logical_not(pop[:, :, 1, :])
        self.pop = pop
        return pop


    def cluster_pop(self)-> tuple[np.ndarray]:
        """
        TODO doc, test
        Build the cluster populations from the individual ROI populations.

        Input attributes
        ----------------
        pop : 3D np.ndarray of bool
            Atomic population. Format is that returned by <atom_detection>.
        clusters : 2D np.ndarray of int
            Array representing the clusters. Shape (nb_clusters, cluster_size).
            The format is:
                clusters[i, :] = [j1, j2, ..., jn]
                i is the cluster index,
                j1, j2, ..., jn are the ROI indices that constitute the cluster

        Raises
        ------
        AttributeError
            If clusters are not set or atomic populations have not been
            computed (<detect_atoms>).

        Set attribute
        -------------
        cpop : 4D np.ndarray of bool
            Cluster populations in each frame.
            cpop[i, j, p, k] = True if atoms are present in
                                   i-th cframe set
                                   j-th cluster
                                   p-th population index
                                   k-th run
                               False otherwise
            p = sum(c_n * 2**n, n=0..cluster_size-1)
            c_n is the population of cluster atom n (1 if occupied else 0)

        """
        if self.clusters is None:
            raise AttributeError("clusters are not set")
        if self.pop is None:
            raise AttributeError("atomic populations have not been computed")

        nbc, csize, *_ = self.clusters.shape
        fpr, _, _, nbcf = self.pop.shape

        if self.clusters.ndim == 2:
            cfidx = slice(None)
        elif self.clusters.ndim == 3:
            cfidx = np.arange(nbcf)

        cpop = np.zeros((fpr, nbc, 2**csize, nbcf), dtype=bool)
        for p, i in np.ndenumerate(np.arange(2**csize).reshape((2,)*csize)):
            p = np.reshape(p, (1, -1, 1))
            for j, c in enumerate(self.clusters):
                cpop[:, j, i, :] = np.all(
                    self.pop[:, c, 1, cfidx] == p, axis=1)
        self.cpop = cpop

        scpop = np.zeros((fpr, nbc, csize+1, nbcf))
        cpop_count = np.sum(self.pop[:, self.clusters, 1, :], axis=2)
        for p in range(csize+1):
            scpop[:, :, p, :] = (cpop_count == p)
        self.scpop = scpop
        return cpop, scpop


    def target_pop(self)-> np.ndarray:
        """
        TODO doc, test
        Build the target ROI populations from love.

        Input attributes
        ----------------
        pop : 3D np.ndarray of bool
            Atomic population. Format is that returned by <atom_detection>.
        tgt_rois : 1D np.ndarray of int
            Array representing the clusters. Shape (nb_clusters, cluster_size).
            The format is:
                clusters[i, :] = [j1, j2, ..., jn]
                i is the cluster index,
                j1, j2, ..., jn are the ROI indices that constitute the cluster

        Raises
        ------
        AttributeError
            If clusters are not set or atomic populations have not been
            computed (<detect_atoms>).

        Set attribute
        -------------
        cpop : 4D np.ndarray of bool
            Cluster populations in each cframe.
            cpop[i, j, p, k] = True if atoms are present in
                                   i-th cframe set
                                   j-th cluster
                                   p-th population index
                                   k-th run
                               False otherwise
            p = sum(c_n * 2**n, n=0..cluster_size-1)
            c_n is the population of cluster atom n (1 if occupied else 0)

        """
        if self.tgt_rois is None:
            raise AttributeError("target ROIs are not set")
        if self.pop is None:
            raise AttributeError("atomic populations have not been computed")

        nbt, = self.tgt_rois.shape
        fpr, _, _, nbcf = self.pop.shape

        tpop = np.zeros((fpr, 1, nbt+1, nbcf), dtype=bool)
        pop_count = np.sum(self.pop[:, self.tgt_rois, 1, :], axis=1)
        for p in range(nbt+1):
            tpop[:, 0, p, :] = (pop_count == p)
        self.tpop = tpop
        return tpop


    def get_dselect(self,
                    cf_idx: int = 0,
                    condition=None)-> np.ndarray:
        """
        Compute a data selector that matches the given condition in each run
        for the selected cframe set.

        The main use is for data post-selection in large atomic configurations.
        For instance, when performing quantum simulation, you may want to
        keep only the data for which the rearrangement was perfect.

        To do that set cf_idx as the cframe index after rearrangement, and
        condition as the list of ROIs which must be filled (with other ROIs
        empty). The method will yield a 1D array of bool suitable for data
        post-selection (argument `dselect` in <get_*_stats> methods).


        Parameters
        ----------
        cf_idx : int, optional
            Index of the cframe set on which the contition is tested.
            The default is 0.
        condition : None, (list, slice, np.ndarray) or int, optional
            The condition from which the data post-selector is determined.
            - None
                No condition, select all the data.
                The default is None, which is essentially useless.
            - (list, slice, np.ndarray)
                Data is selected if the populated ROIs match the condition
                pattern.
                If condition = [1, 3, 5], will select the data in which
                only ROIs 1, 3, 5 are populated in selected cframe set.
            - int
                Data is selected if the number of ROIs populated match the
                value given.

        Raises
        ------
        AttributeError
            If atomic populations have not been computed (<detect_atoms>).
        TypeError
            If the condition does not have the correct type.

        Returns
        -------
        dselect : 1D np.ndarray of bool, shape (nb_cframes,)
            dselect[k] = True if condition is validated in run k
                         False otherwise

        """
        if self.pop is None:
            raise AttributeError("atomic populations have not been computed")

        pop = self.pop[cf_idx, :, 1, :]
        if condition is None:
            return np.ones((self.seqSetup.nbrun,), dtype=bool)
        elif isinstance(condition, (slice, list, np.ndarray)):
            sel = np.zeros((self.nbroi), dtype=bool)
            sel[condition] = True
            return np.prod(pop == sel[:, np.newaxis], axis=0)
        elif isinstance(condition, int):
            return np.sum(pop, axis=0) == condition
        else:
            raise TypeError("`condition` must be None, slice, list, "
                            f"np.ndarray or int, got {type(condition)}")


    ########## Statistics ##########

    def _get_stats(self,
                   rpop: np.ndarray,
                   dpop: np.ndarray | None = None,
                   dselect: np.ndarray = True,
                   scanSelect: SelectTypes | None = None,
                   aggregator: list[SelectTypes] | None = None,
                   symmetrized: bool = False,)-> dict:
        """
        TODO doc

        Parameters
        ----------
        ptype : str {"roi", }
            DESCRIPTION.
        rpop : np.ndarray
            DESCRIPTION.
        dpop : Union[np.ndarray, None], optional
            DESCRIPTION. The default is None.
        scanSelect : SelectTypes, optional
            DESCRIPTION. The default is None.
        aggregator : list[SelectTypes], optional
            DESCRIPTION. The default is None.
        symmetrized : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        rstatsDict = { # raw stats dict
            #
            'data_type': np.array("atom_stats"),
            'name': np.array(self.name),
            'raw_xvalues': self.seqSetup.scanned_values(),
            'varname': np.array(self.seqSetup.varname),
            'raw_varunit': np.array(self.seqSetup.varunit),
            'nb_parameters': np.array(0),
            }

        # Set selectors, instanciate Atom_Stats
        scanSel = data_selector(self.seqSetup.nbscan, scanSelect)
        raw_scanptSel = self.seqSetup.scanpt_selectors(scanSel)
        scanptSel = raw_scanptSel * dselect

        stats = Atom_Stats(
            rpop=rpop, dpop=dpop,
            scanSelect=scanptSel,
            symmetrized=symmetrized)

        # Set aggregator and average data if required
        nbcluster = rpop.shape[-3]
        if aggregator is None:
            agg = np.eye(nbcluster, dtype=bool)
            rstatsDict.update({"data." + k: v for k, v
                                in stats.get_dict(complete=False).items()})
        else:
            agg = np.array([data_selector(nbcluster, ag)
                            for ag in aggregator])
            rstatsDict.update({"data." + k: v for k, v
                                in stats.average(agg).items()})

        # Info pertaining to the specific stats
        rstatsDict['info.data_select'] = dselect
        rstatsDict['info.scan_select'] = scanSel
        rstatsDict['info.rc_select'] = agg

        return rstatsDict


    def get_pair_stats(self,
                       refFrame: int = 0,
                       delayFrame: int = 0,
                       *,
                       dselect: np.ndarray = True,
                       scanSelect: SelectTypes | None = None,
                       roiAggreg: list[SelectTypes] | None = None,)-> Data_Set:
        """
        TODO update doc
        Compute pair-of-frames ROI statistics.

        This consists in:
            - Computing population statistics in each ROI for each
              scan point, done by <stats.pair_stats>
            - Detecting ROIs for which the data is not good enough
              (which will yield errors when attempting to fit),
              done by <stats.get_shitty_roi>

        Parameters
        ----------
        refFrame : int, optional
            Index of the set of congruent frames that are considered
            as reference frames. Usually, it will be the first frame
            of the run, corresponding to index 0.
            The default is (therefore) 0.
        delayFrame : int, optional
            Index of the set of congruent frames that are considered
            as delay frames, which atomic population is compared to
            the reference frame. Usually, it will be the second frame
            of the run, corresponding to index 1.
            The default is (however) 0.
        scanSelect : SelectTypes, optional
            Selector of the scans for stats computation. Basically,
            the selected scan indices are np.arange(nbscan)[roiSelect].
            The default is None, which selects all scans.

        Raises
        ------
        AttributeError
            If atomic populations have not been computed (<detect_atoms>).

        Returns
        -------
        None.
            Set attributes
            - pstats, dict returned by <stats.pair_stats>
                Statistics in individual ROIs, for each scan point.
            - pstats_md, dict
                Metadata pertaining to the stats computation. Contains
                `ref_frame`: refFrame
                `delay_frame`: delayFrame
                `scan_select`: scanSelector in standard selector format
                `empty`: empty ROIs in standard selector format

        """
        if self.pop is None:
            raise AttributeError("atomic populations have not been computed")

        refpop = self.pop[refFrame]
        delaypop = self.pop[delayFrame]

        pstatsDict = self._get_stats(refpop, delaypop, dselect, scanSelect,
                                     roiAggreg, symmetrized=False)

        pstatsDict['info.ref_frame'] = np.array(refFrame)
        pstatsDict['info.delay_frame'] = np.array(delayFrame)

        return Data_Set(source=pstatsDict,
                        parameters=self.seqSetup.get_dict())


    def get_cluster_stats(self,
                          *,
                          dselect: np.ndarray = True,
                          scanSelect: SelectTypes | None = None,
                          clusterAggreg: list[SelectTypes] | None = None,
                          symmetrize: bool = False)-> Data_Set:
        """
        TODO doc
        Compute single frame cluster statistics.

        Raises
        ------
        AttributeError
            If cluster populations have not been computed
            (<compute_cluster_pop>)
        for p in range(nbt):
            tpop[:, 0, p, :] = (pop_count == p).

        Returns
        -------
        None.
            Set attributes
            - cstats, dict returned by <stats.cluster_stats>
                Statistics in individual ROIs, for each scan point.
            - pstats_md, dict
                Metadata pertaining to the stats computation. Contains
                `ref_frame`: refFrame
                `delay_frame`: delayFrame
                `scan_select`: scanSelector in standard selector format
                `empty`: empty ROIs in standard selector format

        """
        if self.cpop is None:
            raise AttributeError("cluster populations have not been computed")

        if symmetrize: # symmetrize pop
            pop = self.scpop
        else:
            pop = self.cpop

        cstatsDict = self._get_stats(pop, None, dselect, scanSelect,
                                     clusterAggreg, symmetrized=symmetrize)

        cstatsDict['info.ref_frame'] = np.array([])
        cstatsDict['info.delay_frame'] = np.array([])

        return Data_Set(source=cstatsDict,
                        parameters=self.seqSetup.get_dict())


    def get_cp_stats(self,
                     refFrame: int = 0,
                     delayFrame: int = 0,
                     *,
                     dselect: np.ndarray = True,
                     scanSelect: SelectTypes | None = None,
                     clusterAggreg: list[SelectTypes] | None = None,
                     symmetrize: bool = False,)-> Data_Set:
        """
        TODO doc, comment

        Parameters
        ----------
        refFrame : int, optional
            Index of reference set of cframes. The default is 0.
        delayFrame : int, optional
            Index of delay set of cframes. The default is 0.
        scanSelect : SelectTypes, optional
            Selector of the scans for stats computation.
            The default is None, which selects all scans.

        Raises
        ------
        AttributeError
            DESCRIPTION.

        Returns
        -------
        None.
            Set attributes
            - cpstats, dict returned by <stats.cluster_stats>
                Statistics in individual ROIs, for each scan point.

        """
        if self.cpop is None:
            raise AttributeError("cluster populations have not been computed")

        if symmetrize: # symmetrize pop
            rpop = self.scpop[refFrame]
            dpop = self.scpop[delayFrame]
        else:
            rpop = self.cpop[refFrame]
            dpop = self.cpop[delayFrame]

        cpstatsDict = self._get_stats(rpop, dpop, dselect, scanSelect,
                                      clusterAggreg, symmetrized=symmetrize)

        cpstatsDict['info.ref_frame'] = np.array(refFrame)
        cpstatsDict['info.delay_frame'] = np.array(delayFrame)

        return Data_Set(source=cpstatsDict,
                        parameters=self.seqSetup.get_dict())


    def get_target_stats(self,
                         *,
                         dselect: np.ndarray = True,
                         scanSelect: SelectTypes | None = None,)-> Data_Set:
        """
        TODO doc

        Parameters
        ----------
        scanSelect : SelectTypes, optional
            DESCRIPTION. The default is None.

        Raises
        ------
        AttributeError
            DESCRIPTION.

        Returns
        -------
        tstatsDict : TYPE
            DESCRIPTION.

        """
        if self.tpop is None:
            raise AttributeError("target ROI populations have not been computed")

        tstatsDict = self._get_stats(self.tpop, None, dselect, scanSelect,
                                     symmetrized=True)

        tstatsDict['info.ref_frame'] = np.array([])
        tstatsDict['info.delay_frame'] = np.array([])

        return Data_Set(source=tstatsDict,
                        parameters=self.seqSetup.get_dict())


    def get_tp_stats(self,
                     refFrame: int = 0,
                     delayFrame: int = 0,
                     *,
                     dselect: np.ndarray = True,
                     scanSelect: SelectTypes | None = None)-> Data_Set:
        """
        TODO doc

        Parameters
        ----------
        refFrame : int, optional
            DESCRIPTION. The default is 0.
        delayFrame : int, optional
            DESCRIPTION. The default is 0.
        scanSelect : SelectTypes, optional
            DESCRIPTION. The default is None.

        Raises
        ------
        AttributeError
            DESCRIPTION.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        if self.tpop is None:
            raise AttributeError("target ROI populations have not been computed")

        rpop = self.tpop[refFrame]
        dpop = self.tpop[delayFrame]

        tpstatsDict = self._get_stats(rpop, dpop, dselect, scanSelect,
                                      symmetrized=True)

        tpstatsDict['info.ref_frame'] = np.array(refFrame)
        tpstatsDict['info.delay_frame'] = np.array(delayFrame)

        return Data_Set(source=tpstatsDict,
                        parameters=self.seqSetup.get_dict())


    ########## Plot histogram ##########

    def plot_histogram(self,
                       cframeIndex: int,
                       shape: tuple[int, ...] | None = None,
                       roi_mapper: np.ndarray | None = None):
        """


        Parameters
        ----------
        cframeIndex : int
            DESCRIPTION.
        roi_mapper : np.ndarray, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        histoPlot : TYPE
            DESCRIPTION.

        """
        p_exp, p_load, log_perr = "--", "--", "--"
        model, popt = None, None
        if self.thresholds is None:
            print("Histogram: no thresholds to plot.")
            thr = None
        elif (hfit := self.histofit[cframeIndex]) is None:
            print("Histogram: no fit to plot.")
            thr = self.thresholds[cframeIndex]
            p_exp = np.sum(self.data[cframeIndex] > thr[:, None], axis=1) \
                    / self.data[cframeIndex].shape[1]
            p_load = np.mean(p_exp)
        else:
            thr = self.thresholds[cframeIndex]
            p_exp = hfit['p_exp']
            p_load = np.mean(p_exp)
            log_perr = np.log10(
                np.mean(hfit['falsepos'][:] + hfit['falseneg'][:]))
            # Histogram fit stuff
            if hfit['mode'] == 'dgauss':
                model = double_gaussian
                popt = np.transpose(np.array(
                    [hfit['A1'], hfit['mu1'], hfit['sigma1'],
                     hfit['A2'], hfit['mu2'], hfit['sigma2']]))
            elif hfit['mode'] == 'gausson':
                model = gausson
                popt = np.transpose(np.array(
                    [hfit['p_th'], hfit['mu1'], hfit['sigma1'],
                     hfit['mu2'], hfit['A']]))

        suptext = r"$p_{load}$" + f" = {p_load:.3}" \
            + r"  ;  $\log (p_{err})$" + f" = {log_perr:.3}"

        shape = shape if shape is not None else self.roiManager.roishape
        roi_mapper = roi_mapper if roi_mapper is not None \
                 else self.roiManager.get_roi_mapper()
        histoPlot = Plot(self.nbroi, shape, roi_mapper)
        histoPlot.plot_hist(self.data[cframeIndex],
                            thresholds=thr)

        # set title, x/y labels, numerotation
        histoPlot.set_title(self.seqSetup.date[:11] + self.seqSetup.name)
        histoPlot.set_xlabel("Photon count")
        histoPlot.set_ylabel("Count frequency")
        histoPlot.add_plot_indices()
        histoPlot.fig_suptext(suptext, txt_kw={'fontsize': 12})
        if p_exp is not None:
            p_txt = [f"p = {p:.3}" for p in p_exp]
            histoPlot.axes_text(0.6, 0.88, p_txt)

        # Plot fit
        if model is not None:
            histoPlot.plot_fct(popt, model)

        return histoPlot







