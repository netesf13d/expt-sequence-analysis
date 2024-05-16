# -*- coding: utf-8 -*-
"""
This module implements functions pertaining to statistics computation on
binarized fluorescence data. It is centered around the class Atom_Stats,
which provide user-level methods to manage stats (load, save, compute,
access...). Helper functions are implemented that perform actual stat
computations.

The core stats


The Atom_Stats class methods are:
    - <get_dict>, to get the core attributes as a dict
    - <_load>, to properly instantiate from precomputed stats
    - <save>, to save to a file in binary format (.npz)
    - <symmetrize>,
    - <average>,
    - <_compute_full_stats>,
    - <accessors>,
    - <get_stat>,


The helper functions operate on the `low-level` stats and are not meant to
be called at user level. They are wrapped around by other functions/methods
elsewhere in the code.
The helper functions are:
    - <_parse_input>, to check whether the input data used for instantiation
      is valid
    - <pair_stats>, to compute pair-of-frames stats
    - <cluster_stats>, to compute cluster stats
    - <symmetrize_pstats>, to symmetrize pair-of-frames stats
    - <symmetrize_cstats>, to symmetrize cluster stats
    - <full_pstats>,
    - <full_cstats>,
    - <merge_stats>, to merge sets of stats. Used by <dataset.merge_datasets>



TODO
- gendoc
- doc

"""

import warnings
from pathlib import Path

import numpy as np

from .stat_accessors import access_dispatcher


# =============================================================================
#
# =============================================================================

def _parse_input(rpop: np.ndarray,
                 dpop: np.ndarray | None,
                 scanSelect: np.ndarray):
    """
    TODO doc, comment

    Parameters
    ----------
    rpop : np.ndarray
        DESCRIPTION.
    dpop : Union[np.ndarray, None]
        DESCRIPTION.
    scanSelect : np.ndarray
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if rpop is None or scanSelect is None:
        raise TypeError("`ref_pop` and `scanSelect` cannot be None")

    if (d:=rpop.ndim) not in [3, 4]:
        raise ValueError(f"`ref_pop` dimension must be 3 or 4, got {d}")

    if d == 3 and dpop is None:
        raise ValueError("`delay_pop` must not be None for pair stats")

    if d == 4 and dpop is not None:
        warnings.warn("ignoring `delay_pop`...", UserWarning)

    if dpop is not None and (dsh:=dpop.shape) != (rsh:=rpop.shape):
        raise ValueError("incompatible shapes between ref and delay "
                         f"populations: {rsh} and {dsh}")

    if (ssh:=scanSelect.shape[1]) != (rsh:=rpop.shape[-1]):
        raise ValueError(f"`scanSelect` shape, {ssh}, does not match that of "
                         f"the populations, {rsh}")



def pair_stats(ref_cpop: np.ndarray,
               delay_cpop: np.ndarray,
               scanSelect: np.ndarray)-> np.ndarray:
    """
    TODO doc


    === Note ===
    The number of occurences follows a multinomial distribution
    (see https://en.wikipedia.org/wiki/Multinomial_distribution ).
    The corresponding probabilities follow a categorical distribution
    (see https://en.wikipedia.org/wiki/Categorical_distribution ).
    The computation of second order moments is reduced to that of the
    variance, not the full covariance matrix.

    Parameters
    ----------
    ref_cpop : 3D np.ndarray
        Cluster populations in reference cframe set.
        cpop[i_ref] with cpop returned by <cluster_pop>.
    delay_cpop : 3D np.ndarray
        Cluster populations in delay cframe set.
        cpop[i_delay] with cpop returned by <cluster_pop>.
    scanSelect : 2D np.ndarray of bool
        Data selectors, selecting the data relative to scan points.
        (see for instance <seqsetup.Sequence_Setup.scanpt_selectors>)
        scanSelect[i, j] = True if frame j belongs to scan point i
                           False otherwise

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    nbx : 4D np.ndarray
        nbx[j, p0, p1, k] = stat relative to
                              j-th cluster
                              p0-th ref population index
                              p1-th delay population index
                              k-th run
        p = sum(c_n * 2**n, n=0..cluster_size-1)
        c_n is the population of cluster atom n (1 if occupied else 0)

    """
    _parse_input(ref_cpop, delay_cpop, scanSelect)

    nbc, nbp, nbs = ref_cpop.shape
    # cross populations
    xpop = np.zeros((nbc, nbp, nbp, nbs), dtype=bool)
    for p in range(nbp):
        xpop[:, :, p, :] = ref_cpop * delay_cpop[:, [p], :]
    # Number of occurrences
    nbx = np.array([np.sum(xpop[..., scanSlice], axis=-1)
                    for scanSlice in scanSelect])
    nbx = np.moveaxis(nbx, 0, -1) # cross occurrences, dim 4

    return nbx


def cluster_stats(cpop: np.ndarray,
                  scanSelect: np.ndarray)-> np.ndarray:
    """
    TODO doc

    === Note ===
    The number of occurences follows a multinomial distribution
    (see https://en.wikipedia.org/wiki/Multinomial_distribution ).
    The corresponding probabilities follow a categorical distribution
    (see https://en.wikipedia.org/wiki/Categorical_distribution ).
    The computation of second order moments is reduced to that of the
    variance, not the full covariance matrix.

    Parameters
    ----------
    cpop : 4D np.ndarray
        Cluster populations in each cframe set. Returned by <cluster_pop>.
    scanSelect : 2D np.ndarray of bool
        Data selectors, selecting the data relative to scan points.
        (see for instance <seqsetup.Sequence_Setup.scanpt_selectors>)
        scanSelect[i, j] = True if frame j belongs to scan point i
                           False otherwise

    Returns
    -------
    nb : 4D np.ndarray
        nb[i, j, p, k] = stat relative to
                            i-th cframe set
                            j-th cluster
                            p-th population index
                            k-th run
        p = sum(c_n * 2**n, n=0..cluster_size-1)
        c_n is the population of cluster atom n (1 if occupied else 0)

    """
    _parse_input(cpop, None, scanSelect)
    ## Statistics
    nb = np.array([np.sum(cpop[..., scanSlice], axis=-1)
                   for scanSlice in scanSelect])
    nb = np.moveaxis(nb, 0, -1) # config occurrences, dim 4
    return nb


########## Symmetrize ##########

def symmetrize_pstats(nbx: np.ndarray)-> np.ndarray:
    """
    TODO  doc

    Parameters
    ----------
    nbx : np.ndarray
        DESCRIPTION.

    Returns
    -------
    snbx : TYPE
        DESCRIPTION.

    """
    nbc, cs, _, nbstep = nbx.shape
    csize = int(np.log2(cs))

    # compute the symmetric number of occurences
    snbx = np.zeros((nbc, csize+1, csize+1, nbstep), dtype=int)
    for p0, i0 in np.ndenumerate(np.arange(2**csize).reshape((2,)*csize)):
        n0 = sum(p0)
        for p1, i1 in np.ndenumerate(np.arange(2**csize).reshape((2,)*csize)):
            n1 = sum(p1)
            snbx[:, n0, n1, :] += nbx[:, i0, i1, :]

    return snbx


def symmetrize_cstats(nb: np.ndarray)-> np.ndarray:
    """
    TODO doc

    Parameters
    ----------
    cstats : dict
        Cluster stats returned by <cluster_stats>.

    Returns
    -------
    dict
        DESCRIPTION.

    """
    # get number of occurrences stat
    fpr, nbc, cs, nbstep = nb.shape
    csize = int(np.log2(cs))

    # compute the symmetric number of occurences
    snb = np.zeros((fpr, nbc, csize+1, nbstep), dtype=int)
    for p, i in np.ndenumerate(np.arange(2**csize).reshape((2,)*csize)):
        n = sum(p)
        snb[:, :, n, :] += nb[:, :, i, :]

    return snb


########## Full stats ##########

@np.errstate(invalid="ignore") # ignore 0/0 that may occur in the np.where calls
def full_pstats(nbx: np.ndarray,
                nbrep: np.ndarray)-> dict:
    """
    TODO doc


    Parameters
    ----------
    nbx : np.ndarray
        DESCRIPTION.
    nbrep : np.ndarray
        DESCRIPTION.

    Returns
    -------
    cpstatDict : dict
        Dictionnary containing various statistical quantities pertaining
        to individual ROIs. The categories are:
        - reference stats. Stats pertaining to the reference frame:
            'nbref', 'stdev_nbref' for number of occurrences
            'pref', 'stdev_pref' for related probabilities
        - delay stats. Stats pertaining to the delay frame:
            'nbdelay', 'stdev_nbdelay' for number of occurrences
            'pdelay', 'stdev_pdelay' for related probabilities
        - cross configuration stats. Stats pertaining to config in ref frame
          and delay frame:
            'nbx', 'stdev_nbx' for number of occurrences
            'xprob', 'stdev_xprob' for related probabilities
        - conditional probability stats. Conditional cross config stats:
            'pcond', 'stdev_pcond'

        reference and delay stats are identical to the stats returned by
        <cluster_stats> with i = i_ref, i_delay respectively.

        For the other keys, cpstatDict[key] is a 4D array, which format is:
        cstatDict[key][j, p0, p1, k] = stat relative to
                                         j-th cluster
                                         p0-th ref population index
                                         p1-th delay population index
                                         k-th run
        p = sum(c_n * 2**n, n=0..cluster_size-1)
        c_n is the population of cluster atom n (1 if occupied else 0)

    """
    nbrep3 = nbrep[:, np.newaxis, :]
    nbrep4 = nbrep[:, np.newaxis, np.newaxis, :]

    nbref = np.sum(nbx, axis=2) # ref occurrences, dim 3
    nbdelay = np.sum(nbx, axis=1) # delay occurrences, dim 3

    ########## Probabilities ##########
    pref = np.where(nbref == 0, 0., nbref / nbrep3) # ref proba
    pdelay = np.where(nbdelay == 0, 0., nbdelay / nbrep3) # delay proba
    xprob = np.where(nbx == 0, 0., nbx / nbrep4) # cross proba
    pcond = np.where(nbx == 0, 0., nbx / nbref[:, :, np.newaxis, :]) # conditional proba

    ########## Standard deviations of probabilities ##########
    stdev_pref = np.where(
        pref*(1-pref) == 0., 0., np.sqrt(pref*(1-pref) / nbrep3))
    stdev_pdelay = np.where(
        pdelay*(1-pdelay) == 0., 0., np.sqrt(pdelay*(1-pdelay) / nbrep3))
    stdev_xprob = np.where(
        xprob*(1-xprob) == 0., 0.,
        np.sqrt(xprob*(1-xprob) / nbrep4))
    stdev_pcond = np.where(
        pcond*(1-pcond) == 0., 0.,
        np.sqrt(pcond*(1-pcond) / nbref[:, :, np.newaxis, :]))

    ########## Standard deviations of occurrence numbers ##########
    stdev_nbref = nbref * stdev_pref
    stdev_nbdelay = nbdelay * stdev_pdelay
    stdev_nbx = nbx * stdev_xprob

    pstatDict =  {
        'nbref': nbref, 'stdev_nbref': stdev_nbref, # dim3
        'nbdelay': nbdelay, 'stdev_nbdelay': stdev_nbdelay, # dim 3
        'nbx': nbx, 'stdev_nbx': stdev_nbx, # dim 4
        'pref': pref, 'stdev_pref': stdev_pref, # dim 3
        'pdelay': pdelay, 'stdev_pdelay': stdev_pdelay, # dim 3
        'xprob': xprob, 'stdev_xprob': stdev_xprob, # dim 4
        'pcond': pcond, 'stdev_pcond': stdev_pcond, # dim 4
        }

    return pstatDict


@np.errstate(invalid="ignore") # ignore 0/0 that may occur in the np.where calls
def full_cstats(nb: np.ndarray,
                nbrep: np.ndarray)-> dict:
    """
    TODO doc

    === Note ===
    The number of occurences follows a multinomial distribution
    (see https://en.wikipedia.org/wiki/Multinomial_distribution ).
    The corresponding probabilities follow a categorical distribution
    (see https://en.wikipedia.org/wiki/Categorical_distribution ).
    The computation of second order moments is reduced to that of the
    variance, not the full covariance matrix.

    Parameters
    ----------
    nb : np.ndarray
        DESCRIPTION.
    nbrep : np.ndarray
        DESCRIPTION.

    Returns
    -------
    cstatDict : dict
        Dictionnary containing various statistical quantities pertaining
        to individual ROIs. The keys are:
        - 'nb': Number of occurences of different cluster config.
        - 'stdev_nb': Standard deviation of the number of occurences
        - 'prob': Probabilities of each cluster config.
        - 'stdev_nb': Standard deviation of the probabilities

        For each key, cstatDict[key] is a 4D array, which format is:
        cstatDict[key][i, j, p, k] = stat relative to
                                          i-th cframe set
                                          j-th cluster
                                          p-th population index
                                          k-th run
        p = sum(c_n * 2**n, n=0..cluster_size-1)
        c_n is the population of cluster atom n (1 if occupied else 0)

    """
    nbrep4 = nbrep[np.newaxis, :, np.newaxis, :]
    ## Statistics
    prob = np.where(nb == 0, 0., nb / nbrep4)
    stdev_prob = np.where(
        prob*(1-prob) == 0., 0., np.sqrt(prob*(1-prob) / nbrep4))
    stdev_nb = nb * stdev_prob

    cstatDict = {
        'nb': nb, 'stdev_nb': stdev_nb,
        'prob': prob, 'stdev_prob': stdev_prob}
    return cstatDict


# =============================================================================
#
# =============================================================================

class Atom_Stats():

    stat_types = {"cluster", "pair"}
    modes = {"raw", "symmetric"}

    def __init__(self, *,
                 source: dict | str | Path | None = None,
                 rpop: np.ndarray | None = None,
                 dpop: np.ndarray | None = None,
                 scanSelect: np.ndarray | None = None,
                 symmetrized: bool = False,):
        """
        TODO doc

        Parameters
        ----------
        source : Union[dict, str, Path, None], optional
            File of dict to load atom stats from. The default is None.
        rpop : np.ndarray, optional
            DESCRIPTION. The default is None.
        dpop : np.ndarray, optional
            DESCRIPTION. The default is None.
        scanSelect : 2D np.ndarray of bool, optional
            Selector for the scan points, including data post-selection.
            scanSelect[i, j] = True if run j corresponds to scan point i
                                    and data is post-selected
                               False otherwise
            The default is None.
        symmetrized : bool, optional
            If True, the populations are assumed to be symmetrized, ie
            they pertain to the number of atoms in the cluster. If False,
            they are assumed to give the detailed atomic configuration.
            The default is False, whichis your only option in case of large
            clusters, such as the target ROIs.

        Raises
        ------
        ImportError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self._stat_type = "" # {"cluster", "pair"}
        self._mode = "" # {"raw", "symmetric"}
        self.nbrep = np.array([[0]], dtype=int) # 2D array, broadcastable to shape (nbj, nbk)
        self.csize = 0 # cluster size
        self.exshape = np.zeros((5,), dtype=int) # nbi, nbj, nbp0, nbp1, nbk
        self.mstat = None # 3D or 4D np.ndarray

        self.statDict = {}

        ## Instanciation proper
        if isinstance(source, dict): # init from dict
            if 'stat_type' in source.keys():
                self._load(source)
            else:
                raise ValueError(
                    "wrong source dict to instanciate Atom_Stats")
        elif isinstance(source, (str, Path)): # load from file
            with np.load(source, allow_pickle=False) as f:
                if 'stat_type' in f.keys():
                    self._load(f)
                else:
                    raise ValueError(
                        "wrong source file to instanciate Atom_Stats")
        else: # compute from population arrays
            _parse_input(rpop, dpop, scanSelect)
            # compute master stat
            if rpop.ndim == 3: # pair stats
                self._stat_type = "pair"
                self.mstat = pair_stats(rpop, dpop, scanSelect)
                self.exshape[[1, 2, 3, 4]] = self.mstat.shape
                self.nbrep = np.sum(self.mstat, axis=(1, 2))
            elif rpop.ndim == 4: # cluster stats
                self._stat_type = "cluster"
                self.mstat = cluster_stats(rpop, scanSelect)
                self.exshape[[0, 1, 2, 4]] = self.mstat.shape
                self.nbrep = np.sum(self.mstat[0, ...], axis=1)
            # set stats mode and cluster size
            if symmetrized:
                self._mode = "symmetric"
                self.csize = rpop.shape[-2]
            else:
                self._mode = "raw"
                self.csize = int(np.log2(rpop.shape[-2]))


    def get_dict(self, complete: bool = False)-> dict:
        """
        TODO doc

        Parameters
        ----------
        complete : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        ASdict = { # Cluster Atom Stats dict
            'stat_type': np.array(self._stat_type),
            'mode': np.array(self._mode),
            'nbrep': self.nbrep,
            'csize': np.array(self.csize),
            'exshape': self.exshape,
            'mstat': self.mstat,
            }
        if complete:
            if not self.statDict:
                self._compute_full_stats()
            for k, v in self.statDict.items():
                ASdict["stat." + k] = v
        return ASdict


    def _load(self, source: dict):
        self._stat_type = source['stat_type']
        self._mode = source['mode']
        self.nbrep = source['nbrep']
        self.csize = source['csize']
        self.exshape = source['exshape']
        self.mstat = source['mstat']


    def save(self, file: str | Path, complete: bool = False):
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        data_out = self.get_dict(complete)
        np.savez_compressed(file, **data_out)
        print(f"Atom_Stats saved at: {str(file)}")


    def symmetrize(self,)-> dict:
        """
        Symmetrize `raw` atom stats.


        Returns
        -------
        dict
            DESCRIPTION.

        """
        if self._mode == "symmetric":
            warnings.warn("Atom_Stats already symmetric", UserWarning)
            return self
        else:
            exshape = np.zeros((5,), dtype=int)
            if self._stat_type == "pair":
                smstat = symmetrize_pstats(self.mstat)
                exshape[[1, 2, 3, 4]] = smstat.shape
            elif self._stat_type == "cluster":
                smstat = symmetrize_cstats(self.mstat)
                exshape[[0, 1, 2, 4]] = smstat.shape
            symASdict = {
                'stat_type': np.array(self._stat_type),
                'mode': np.array("symmetric"),
                'nbrep': np.array(self.nbrep),
                'csize': np.array(self.csize),
                'exshape': exshape,
                'mstat': smstat,
                }
            return symASdict


    def average(self, aggreg: np.ndarray)-> dict:
        """
        TODO doc
        Average stat as an effective average ROI.

        The effective average ROI is obtained by pooling all individual ROI
        data. The number of atoms in different conditions (reference, delay,
        recapture, ...) are summed, and probabilities are computed anew
        from these values.

        This is equivalent to weighting the average of probabilities by the
        amount of reference data for each ROI. In other words, probabilities
        from robust data are given more importance than those from shitty data.

        Parameters
        ----------
        aggreg : int or slice or list or 1D array
            Aggregator of the ROIs to be averaged.

        Returns
        -------
        avgStatDict : dict
            Dictionnary containing various statistical quantities pertaining
            to the "averaged" ROI. The keys are the same as those of statDict
            (see <compute_stats> documentation), exxcept for one additional
            key:
            - 'roiselect': 1D np.ndarray of bool, length nbROI
              roiselect[i] is True if ROI i was involved in the computation
              of the average statistics, False otherwise

            For each key, statDict[key] is a 1D array, which format is:
            avgStatDict[key][j] = statistic relative at scan point j.

        """
        exshape = np.zeros((5,), dtype=int)
        if self._stat_type == "pair":
            nb = self.mstat
            avgmstat = np.array([np.sum(nb*ag.reshape((-1, 1, 1, 1)), axis=0)
                                 for ag in aggreg])
            nbrep = np.sum(avgmstat, axis=(1, 2))
            exshape[[1, 2, 3, 4]] = avgmstat.shape
        elif self._stat_type == "cluster":
            nbx = self.mstat
            avgmstat = np.swapaxes(
                np.array([np.sum(nbx*ag.reshape((1, -1, 1, 1)), axis=1)
                          for ag in aggreg]),
                0, 1)
            nbrep = np.sum(avgmstat[0], axis=1)
            exshape[[0, 1, 2, 4]] = avgmstat.shape

        avgASdict = {
            'stat_type': np.array(self._stat_type),
            'mode': np.array(self._mode),
            'nbrep': nbrep,
            'csize': np.array(self.csize),
            'exshape': exshape,
            'mstat': avgmstat,
            }
        return avgASdict


    def _compute_full_stats(self,):
        """
        TODO doc
        """
        if self._stat_type == "pair":
            self.statDict = full_pstats(self.mstat, self.nbrep)
        elif self._stat_type == "cluster":
            self.statDict = full_cstats(self.mstat, self.nbrep)


    def accessors(self)-> dict[str, set]:
        """


        Returns
        -------
        keys : TYPE
            DESCRIPTION.

        """
        if not self.statDict:
            self._compute_full_stats()
        keys = self.statDict.keys()

        access = access_dispatcher(self._stat_type,
                                   self._mode,
                                   self.exshape[1],
                                   self.csize)

        return {'keys': set(keys), 'shortcuts': access.shortcuts}


    def get_stat(self, accessor)-> np.ndarray:
        """

        """
        if not self.statDict:
            self._compute_full_stats()

        access = access_dispatcher(self._stat_type,
                                   self._mode,
                                   self.exshape[1],
                                   self.csize)
        key, idx = access(accessor)
        return self.statDict[key][idx]



# =============================================================================
# FUNCTIONS -
# =============================================================================

def merge_stats(atomstats: tuple[Atom_Stats, ...],
                indexation: np.ndarray | None = None)-> dict:
    """
    TODO doc
    Merge stats

    Parameters
    ----------
    atomstats : tuple[Atom_Stats]
        The Atom_Stats instances which stats are to be merged.
    indexation : 2D np.ndarray of bool, optional
        Dispatcher for the new stats, shape (len(atomstats), new_nbsteps).
        Basically, the stats of n-th Atom_Stat pertaining to k-th scan pt
        goes to new_k = index of k-th True value of indexation[n, :].
        This is subject to the constraint
        sum(indexation[n, :]) == nbsteps of atomstats[n].
        The default is None, which is equivalent to
        np.ones((len(atomstats, nbk), dtype=bool)

    Raises
    ------
    ValueError
        - atomstats is empty
        - len(indexation) != len(atomsats)
        - The Atom_Stats instances are incompatible
        - the constaint sum(indexation[n, :]) == nbsteps of atomstats[n]
          is not satisfied

    Returns
    -------
    mergedASDict : dict
        DESCRIPTION.

    """
    if (l_AS:=len(atomstats)) == 0:
        raise ValueError("need at least one Atom_Stat to merge")
    if indexation is None:
        indexation = np.full((l_AS, atomstats[0].exshape[4]),
                             True, dtype=bool)
    # check nb of Atom_Stats match number of indexation arrays
    if l_AS != (l_idx:=len(indexation)):
        raise ValueError(
            f"number of stats to merge, {l_AS}, and indexations, {l_idx}, do "
            "not match")
    # check stat type (`pair` or `cluster`) homogeneity
    statTypes = np.array([AS._stat_type for AS in atomstats])
    if (ierr:=np.nonzero(statTypes != statTypes[0])[0]).size > 0:
        raise ValueError(
            f"incompatible stat types with the first one at indices {ierr}")
    # check stat mode (`raw` or `symmetric`) homogeneity
    statModes = np.array([AS._mode for AS in atomstats])
    if (ierr:=np.nonzero(statModes != statModes[0])[0]).size > 0:
        raise ValueError(
            f"incompatible stat modes with the first one at indices {ierr}")
    # check indexation is ok
    nbsteps = np.array([AS.exshape[4] for AS in atomstats])
    if (ierr:=np.nonzero(nbsteps != np.sum(indexation, axis=1))[0]).size > 0:
        raise ValueError(
            f"numbers of steps and indexation do not match at indices {ierr}")
    # Check sizes homogeneity
    sh = np.array([AS.exshape[:4] for AS in atomstats])
    if (ierr:=np.nonzero(sh != sh[0])[0]).size > 0:
        raise ValueError(
            f"incompatible stat sizes with the first one at indices {ierr}")

    # initialize merged stats containers
    exshape = np.concatenate((sh[0], [indexation.shape[-1]]), dtype=int)
    merged_sh = exshape[np.nonzero(exshape)]
    merged_mstat = np.zeros(merged_sh, dtype=int)
    merged_nbrep = np.zeros((exshape[1], exshape[4]), dtype=int)
    # Merge stats
    for AS, idx in zip(atomstats, indexation):
        merged_nbrep[:, idx] += AS.nbrep
        merged_mstat[:, :, :, idx] += AS.mstat

    mergedASDict = {
        'stat_type': np.array(statTypes[0]),
        'mode': np.array(statModes[0]),
        'nbrep': merged_nbrep,
        'csize': np.array(atomstats[0].csize),
        'exshape': exshape,
        'mstat': merged_mstat,
        }
    return mergedASDict

