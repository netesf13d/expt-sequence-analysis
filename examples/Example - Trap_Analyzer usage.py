# -*- coding: utf-8 -*-
"""
========== Processing of raw data with a Trap_Analyzer ==========

This example illustrates the use of the `Trap_Analyzer` class to preprocess
raw atomic fluorescence data and produce datasets for further analysis.

The workflow is as follows:
* Load the raw data and experimental sequence information, instanciate the
  Trap_Analyzer.
* Commpute atom detection thresholds for each trapping site.
* Binarize the signal based on the thresholds.
* Compute the statistics relevant to various events and return them as a
  `Data_Set` instance.


A sequence can be represented as follows:

    aabbccdd aabbccdd ... aabbccdd

- The sequence is the whole acquisition
- Each subsequence is called a scan (eg `aabbccdd`)
- A scan point is a series of identical letters (eg `aa`, `bb`, etc)
- Each letter represents a run
- The number of repetitions is the number of runs at a given scan point
  (eg the scan point `aa` has 2 runs, hence 2 repetitions)

Inside each run, multiple image acquisitions are done, yielding frames.
For instance, a run [--f1---f2--] may consist in the acquisition of two
frames f1 and f2 (this is the case in all our examples).

The collection of frames f1 for each run is called a congruent frame set,
that is, the first frames of the runs are said to be congruent, and
called cframes in the code.
"""

import sys
from pathlib import Path

import numpy as np

seqand = Path(__file__).resolve().parents[1] # seqan directory
if not str(seqand) in sys.path:
    sys.path.append(str(seqand))
import seqan as sa


# =============================================================================
# ########## Prepare the Trap_Analyzer ##########
# =============================================================================

fpath = "./Rabi/"
"""
The Sequence_Setup encapsulates information about the experimental sequence:
frames per run, number of repetitions, number of scans, number of scan steps,
the values of the scanned parameter, etc.

These are used to structure the raw list of frames into ones that correspond to
a given value of the scanned parameter, a given scan number, etc.
"""
## Information about the experimental sequence
seq = sa.Sequence_Setup(file=fpath + "Rabi_seq.json")

"""
The ROI_Manager encapsulates information about the physical position of the
trapping sites and links it to the abstract ROI indices.

The mapping of ROI indices to positions on a 2D array for displaying data
is done by an ROI mapper, a 2D array of shape (nbroi, 2): ROI-index -> 2D-index

Here the geometry of the traps is a (3, 6) array with a non-trivial indexation.
"""
roiM = sa.ROI_Manager(source=fpath + "Rabi_seq.json")
## Set the ROI mapper manually
nbroi, sh = 18, (3, 6)
mapper = np.transpose(np.array(
    [sh[0]-1 - np.arange(nbroi) % sh[0], np.arange(nbroi) // sh[0]]))
roiM.roi_mapper = mapper


## Load raw data
with np.load(fpath + "Rabi.npz", allow_pickle=False) as f:
    data = f['data'] # 2D array of int
## Instanciate the Trap_Analyzer
ta = sa.Trap_Analyzer(sequenceSetup=seq,
                      roiManager=roiM,
                      rawData=data)


# =============================================================================
# ########## Atom detection threshold computation ##########
# =============================================================================

"""
The distribution of the fluorescence signal presents two peaks, one at low
photon count that corresponds to an empty trap, and one at higher photon
counts that corresponds to a single trapped atom.
The detection threshold is computed (roughly) as the average of the modes
for each ROI.
"""
ta.compute_thresholds(cframeIndex=0, # use the first frames of each run
                      mode='gauss') # fit with two gaussian peaks

# ta.save_thresholds(file=seq.filename("_thr.npy")) # can save that
# ta.savebin_histo_data(file=seq.filename("_histodata.npz")) # and also that

## Create a plot of the histogram
histoPlot = ta.plot_histogram(cframeIndex=0) # <seqan.plots.Plot> object
histoPlot.show()
# histoPlot.save(seq.filename("_histoplot.png")) # saving is also possible


# =============================================================================
# ########## Atom detection and cluster population ##########
# =============================================================================

## Atom detection. Binarize the signal: atom present (1) or absent (0)
ta.atom_detection()

"""
Various kinds of statistics can be computed from the atomic population in each
ROI/frame.

They can be divided in two broad categories:
* Single frame stats.
  Here the various atomic configurations are counted for each cframe set. This
  is only available for 'target ROIs' and 'clusters' since it degenerates to
  the atomic population for single ROIs.
* Pair-of-frames stats.
  Here the pair of configurations in two frames are counted.
  For instance for a single ROI, it computes the number of events in which an
  atom appears, disappears or stays in the trap between two frames.

Furthermore, these statistics can be computed on various objects
* Single ROI stats.
  Each trapping site is considered independently. This is relevant to
  non-interacting atoms.
* Cluster ROI stats.
  Clusters are sets of (possibly overlapping) ROIs. They must have the same
  number of ROIs (cluster size). This is relevant to interacting atoms.
  For instance one can consider pairs of traps and observe the population
  transfer from one trap to the other using clusters.
  Clusters keep track of the atomic configuration, which scales as 2^nb_ROIs,
  hence the cluster size should be kept low.
* Target ROI stats.
  Target ROIs are a single (and possibly large) set of ROIs. Compared to
  clusters, only the counts (not the configuration) are computed.
  This can be used to determine the efficiency of atom rearrangement for
  instance.

There is nothing to be done for single ROI stats, but clusters and target ROIs
can be set manually. These are set in the ROI_Manager internally, and can be
set at instantiation of the latter (eg from a file).
"""

## Set clusters
clusters = np.arange(18).reshape((-1, 3)) # vertical lines of the trap array
ta.set_clusters(clusters)
ta.cluster_pop()

## Set target ROIs
tgt_rois = np.arange(9) # ROIs 0 to 8
ta.set_tgt_rois(tgt_rois)
ta.target_pop()


# =============================================================================
# 
# =============================================================================

## data selection according to a condition (see doc for details)
"""
It is possible to select data according to a condition on the atomic population
in a given cframe. This can be used for instance to post-select those events in
which all the trapping sites were occupied.

Here we select those runs where 10/18 traps were occupied in the first cframe.
"""
ds = ta.get_dselect(cf_idx=0, condition=10)


"""
Single ROI pair stats are relevant to non-interacting atoms and are thus the
most frequently used (at least during my PhD).

It compares the atomic population in a latter frame `delayFrame` with that of
an initial frame `refFrame`. For single ROIs, it amounts to counting the events
appearance (0 -> 1), loss (1 -> 0), recapture (1 -> 1), and void (0 -> 0).

In addition to the data selection mentioned above, it is also possible to
select the scans. This can be useful for instance if a problem occured during
the course of the sequence and part of the data should be discarded.

More importantly, it is possible to aggregate the data from multiple ROIs,
which consists in summing the event counts of the selected ROIs as if there was
one ROI subject to a sequence with more repetitions.
This is used extensively to get the 'average ROI' effect, and improves the
signal to noise ratio.
"""
## Pair stats computation
pstats_roi = ta.get_pair_stats(
    refFrame=0, # pair stats from cframe 0...
    delayFrame=1, # ...to cframe 1
    dselect=True, # select all data
    scanSelect=None, # select all scans
    roiAggreg=None) # no ROI aggregation
pstats_avg = ta.get_pair_stats(
    refFrame=0,
    delayFrame=1, # default is 0!
    roiAggreg=[np.arange(18)]) # aggregate all ROIs

np.savez_compressed(fpath + "ds_roi.npz", **pstats_roi.get_dict())
np.savez_compressed(fpath + "ds_avg.npz", **pstats_avg.get_dict())


"""
Cluster stats have essentially the same interface as single ROIs (an ROI is
essentially a cluster of size 1). Data selection, scan selection and cluster
aggregation is therefore also possible with clusters.

However, in computing cluster stats, it is also possible to compute the atom
counts in place of the detailed configuration. This is done by setting
`symmetrize` to True.
"""
## Cluster stats computation
cstats = ta.get_cluster_stats(
    dselect=ds, # data selection enabled
    clusterAggreg=[ # aggregate clusters into 4 'effective' clusters
        np.arange(5),      # (0) aggregate [0, 1, 2, 3, 4]
        slice(0, None, 2), # (1) aggregate [0, 2, 4]
        [0, 5],            # (2) aggregate 0 and 5
        3],                # (3) select 3
    symmetrize=False)
## Cluster pair stats computation
cpstats = ta.get_cp_stats(
    refFrame=0,
    delayFrame=1,
    scanSelect=[0], # select scan 0
    clusterAggreg=None, # no cluster aggregation
    symmetrize=True) # record only atom count in clusters, not configuration


"""
With target stats, only data and scan selection is possible.
"""
## Target stats computation
tstats = ta.get_target_stats()
## Target pair stats computation
tpstats = ta.get_tp_stats(refFrame=0, delayFrame=1)
