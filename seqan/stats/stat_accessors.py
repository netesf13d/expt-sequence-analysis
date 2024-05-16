# -*- coding: utf-8 -*-
"""
This module implements the Atom_Stats accessor utilities.

Two core utilities are:
    - <access_dispatcher>,
    - <_access>,

The custom accessors are defined in dedicated functions:
    - <pstat_access>, pair-stats accessors
        those correspond to the ones defined in the previous code
    - <cstat_access>, cluster stats accessors
    - <scstat_access>, symmetric cluster stats accessors
    - <cpstat_access>, cluster pair stats accessors
    - <scpstat_access>, symmetric cluster pair stats accessors
    - <tstat_access>, target ROIs stats accessors
    - <tpstat_access>, target ROIs pair stats accessors

TODO
- gendoc
- doc
- merge
- get
- accessors

"""

from typing import Callable


All = slice(None)

# =============================================================================
# ACCESSOR FUNCTIONS
# =============================================================================

def _access(accessor: str | tuple,
            access_str: dict,
            shortcuts: set)-> tuple:
    """


    Parameters
    ----------
    accessor : Union[str, tuple]
        DESCRIPTION.
    access_str : dict
        DESCRIPTION.
    shortcuts : set
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    tuple
        DESCRIPTION.

    """
    if isinstance(accessor, tuple):
        if len(accessor) != 2:
            raise ValueError(
                f"tuple accessor must have len 2, not {len(accessor)}")
        return accessor
    elif isinstance(accessor, str):
        if accessor not in shortcuts:
            raise ValueError(
                f"Accessor shortcut `{accessor}` is not recognized; "
                f"valid shortcuts are {shortcuts}. Shortcuts can be set in "
                "stats.stat_accessors.py.")
        acc = accessor.split('_', 1)
        if acc[0] == "stdev":
            key, idx = access_str[acc[-1]]
            key = "stdev_" + key
        else:
            key, idx = access_str[accessor]
        return key, idx
    else:
        raise ValueError(f"accessor must be tuple or str, is {type(accessor)}")


# =============================================================================
# Custom accessors definition
# =============================================================================


def pstat_access(accessor: str | tuple)-> tuple:
    """
    TODO unify !
    Compute pair-of-frames statistics.
    From the populations determined by <detect_atom>, compute statistics
    pertaining to each scan point.

    The output array format is the following:
        stat[i, j] = statistic relative to ROI i at scan point j.

    === Note ===
    The number of occurences follows a binomial distribution
    (see https://en.wikipedia.org/wiki/Binomial_distribution ).
    The corresponding probabilities follow a Bernoulli distribution
    (see https://en.wikipedia.org/wiki/Bernoulli_distribution ).

    Parameters
    ----------
    popDict : dict
        Population dictionnary, as returned by <detect_atoms>.
    scanSelect : 2D np.ndarray of bool
        Data selectors, selecting the data relative to scan points.
        (see for instance <seqsetup.Sequence_Setup.scanpt_selectors>)
        scanSelect[i, j] = True if frame j belongs to scan point i
                            False otherwise

    Returns
    -------
    pstatDict : dict
        Dictionnary containing various statistical quantities pertaining
        to individual ROIs. The keys are:
        Raw quantities:
        - 'nbref': Number of atoms in reference frames.
        - 'nbdelay': Number of atoms in delay frames.
        - 'nbrecap': Number of recaptured atoms.
        - 'nbloss': Number of lost atoms (= nbref - nbrecap)
        - 'nbgen': Number of atoms that appeared in delay frames.
        Probabilities (Bernoulli estimator):
        - 'pref': Loading probability.
        - 'pdelay': probability to have an atom in delay frame.
        - 'precap': Probability of recapture.
        - 'ploss': Probability of loss (= 1 - precap)
        - 'pgen': Probability to have an atom appearing at delay frame.
        Standard deviations of the Bernoulli estimator (= sqrt(p(1-p)/N)).
        - 'stdev_pref': Standard deviation of pref
        - 'stdev_pdelay': Standard deviation of pdelay
        - 'stdev_precap': Standard deviation of precap
        - 'stdev_ploss': Standard deviation of ploss
        - 'stdev_pgen': Standard deviation of pgen
        Standard deviations of raw counts, stdev_nb = nb * stdev_p
        - 'stdev_nbref': Standard deviation of nbref
        - 'stdev_nbdelay': Standard deviation of npdelay
        - 'stdev_nbrecap': Standard deviation of nbrecap
        - 'stdev_nbloss': Standard deviation of nbloss
        - 'stdev_nbgen': Standard deviation of nbgen

        For each key, statDict[key] is a 2D array, which format is:
        statDict[key][i, j] = statistic relative to ROI i at scan point j.

    """
    return _access(accessor, pstat_access.access_str, pstat_access.shortcuts)

pstat_access.access_str = {
    'nbref': ('nbref', (All, 1, All)),
    'nbdelay': ('nbdelay', (All, 1, All)),
    'nbrecap': ('nbx', (All, 1, 1, All)),
    'nbloss': ('nbx', (All, 1, 0, All)),
    'nbgen': ('nbx', (All, 0, 1, All)),
    'pref': ('pref', (All, 1, All)),
    'pdelay': ('pdelay', (All, 1, All)),
    'precap': ('pcond', (All, 1, 1, All)),
    'ploss': ('pcond', (All, 1, 0, All)),
    'pgen': ('pcond', (All, 0, 1, All)),
    }
pstat_access.shortcuts = {
    'nbref', 'stdev_nbref', 'nbdelay', 'stdev_nbdelay',
    'nbrecap', 'stdev_nbrecap', 'nbloss', 'stdev_nbloss',
    'nbgen', 'stdev_nbgen', 'pref', 'stdev_pref',
    'pdelay', 'stdev_pdelay', 'precap', 'stdev_precap',
    'ploss', 'stdev_ploss', 'pgen', 'stdev_pgen',
    }


def cstat_access(accessor)-> tuple:
    return _access(accessor,
                   cstat_access.access_str,
                   cstat_access.shortcuts)

cstat_access.access_str = {}
cstat_access.shortcuts = set()


def scstat_access(accessor):
    return _access(accessor,
                   scstat_access.access_str,
                   scstat_access.shortcuts)

scstat_access.access_str = {
    }
scstat_access.shortcuts = set()


def cpstat_access(accessor)-> tuple:
    return _access(accessor,
                   cpstat_access.access_str,
                   cpstat_access.shortcuts)

cpstat_access.access_str = {
    }
cpstat_access.shortcuts = set()


def scpstat_access(accessor)-> tuple:
    return _access(accessor,
                   scpstat_access.access_str,
                   scpstat_access.shortcuts)

scpstat_access.access_str = {
    'p_full-to-0': ('pcond', (All, -1, 0, All)),
    'p_full-to-1': ('pcond', (All, -1, 1, All)),
    'p_full-to-2': ('pcond', (All, -1, 2, All)),
    }
scpstat_access.shortcuts = {
    'p_full-to-0', 'stdev_p_full-to-0',
    'p_full-to-1', 'stdev_p_full-to-1',
    'p_full-to-2', 'stdev_p_full-to-2'}


def tstat_access(accessor)-> tuple:
    return _access(accessor,
                   tstat_access.access_str,
                   tstat_access.shortcuts)

tstat_access.access_str = {
    'loading-nb': ('nb', (All, 0, All, All)),
    }
tstat_access.shortcuts = {
    'loading-nb', 'stdev_loading-nb',
    }


def tpstat_access(accessor)-> tuple:
    return _access(accessor,
                   tpstat_access.access_str,
                   tpstat_access.shortcuts)

tpstat_access.access_str = {
    }
tpstat_access.shortcuts = set()


# =============================================================================
#
# =============================================================================

def access_dispatcher(stype: str, mode: str, nbc: int, csize: int)-> Callable:
    """
    Dispatch the data fetching to the correct accessor routine depending
    on the type of data.



    Parameters
    ----------
    stype : str {"cluster", "pair"}
        The type of statistic.
    mode : str {"raw", "symmetric"}
        The mode
    nbc : int
        Number of clusters.
    csize : int
        Size of the clusters.

    Returns
    -------
    callable
        The appropriate function to manage accessors, and in particular
        the shortcuts.

    """
    # Pair stats
    if stype == "pair" and csize == 1:
        return pstat_access
    # Cluster stats
    if (stype, mode) == ("cluster", "raw"):
        return cstat_access
    # Cluster pair stats
    elif (stype, mode) == ("pair", "raw") and csize > 1:
        return cpstat_access
    # Target ROI stats
    elif (stype, mode) == ("cluster", "symmetric") and nbc == 1:
        return tstat_access
    # Target ROI pair stats
    elif (stype, mode) == ("pair", "symmetric") and nbc == 1:
        # you would expect tpstat_access but it is actually the same as symmetric pair stats
        # so I choose to return scpstat_access. You can change this behavior if you want
        return scpstat_access
    # Symmetric cluster stats
    elif (stype, mode) == ("cluster", "symmetric") and nbc > 1:
        return scstat_access
    # Symmetric pair stats
    elif (stype, mode) == ("pair", "symmetric") and nbc > 1:
        return scpstat_access

