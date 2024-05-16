# -*- coding: utf-8 -*-
"""
This module implements seqan utilities.

The functions are:
    - <data_selector>,
    - <merge_xvalues>,
    - <cast_2d>,


The General_Accessor class implements the following methods:
    - <_parse>,
    - <_parse_indexation>
    - <_unparse>,
    - <_unparse_indexation>,


TODO
- doc
- gendoc

"""

import warnings
from ast import literal_eval
from numbers import Integral

import numpy as np

warnings.filterwarnings("default", category=np.VisibleDeprecationWarning)

SelectTypes = int | slice | list | np.ndarray

# =============================================================================
# FUNCTIONS -
# =============================================================================

def data_selector(ntot: int,
                  selector: SelectTypes | None = None)-> np.ndarray:
    """
    Return data selector `sel` in standard selector format.
    The standard selector format is:
    sel : 1D np.ndarray of bool
        self[i] = True if index is selected
                  False otherwise
    """
    if selector is None:
        return np.ones((ntot,), dtype=bool)
    elif isinstance(selector, (int, slice, list, np.ndarray)):
        sel = np.zeros((ntot,), dtype=bool)
        sel[selector] = True
        return sel
    else:
        raise TypeError(f"incorrect type for selector: {type(selector)}")


def merge_xvalues(xvals: tuple[np.ndarray, ...],
                  tol: float = 1e-3)-> tuple:
    """
    TODO doc

    Parameters
    ----------
    xvals : tuple[np.ndarray] (xval1, xval2, ...)
        The xvalues to be merged.
    tol : float, optional
        Tolerance on xvalues. The default is 1e-3.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    new_xval : TYPE
        DESCRIPTION.
    indexation : TYPE
        DESCRIPTION.

    """
    for i, xval in enumerate(xvals):
        if not isinstance(xval, np.ndarray) or xval.ndim != 1:
            raise ValueError(f"xvalues set {i} is not a 1D np.ndarray")
        if xval.size != np.unique(xval).size:
            raise ValueError(f"xvalues set {i} has two identical elements")

    # concatenate xvalues
    xconcat = np.concatenate(xvals)
    # map to xconcat indices to xvals index
    xvalIdx = np.concatenate([np.full_like(x, i, dtype=int)
                               for i, x in enumerate(xvals)],
                              dtype=int)
    # sort the concatenated xvalues
    xsort, xsortarg = np.sort(xconcat), np.argsort(xconcat)
    # set tolerance and determine those xvalues considered identical
    atol = np.max(np.abs(xconcat)) * tol
    xdiff = np.abs(xsort - np.roll(xsort, 1)) < atol
    newxIdx = np.arange(len(xdiff)) - np.cumsum(xdiff, dtype=int)
    # Compute new xvalues and indexation
    nbx = newxIdx[-1] + 1
    new_xval = np.zeros((nbx,), dtype=float)
    indexation = np.zeros((len(xvals), nbx), dtype=bool)
    for i in range(nbx):
        new_xval[i] = np.mean(xsort[newxIdx == i])
        concatIdx = xsortarg[np.nonzero(newxIdx == i)]
        indexation[xvalIdx[concatIdx], i] = True

    return new_xval, indexation


def cast_2d(arr: np.ndarray):
    """
    TODO doc

    Parameters
    ----------
    arr : np.ndarray
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if arr is None:
        return None
    if arr.ndim == 1:
        return arr.reshape((1, -1))
    elif arr.ndim == 2:
        return arr
    else:
        raise ValueError(f"{arr.ndim}-dim array cannot be cast to 2D")




# =============================================================================
# CLASSES
# =============================================================================

class General_Accessor():

    def __init__(self, access: str | tuple):

        self.access_str = ""
        self.access_tuple = ()

        if isinstance(access, str):
            self.access_str = access
            self.access_tuple = self._parse(access)
        elif isinstance(access, tuple):
            self.access_str = self._unparse(access)
            self.access_tuple = access


    def __repr__(self,):
        return str(self.access_tuple)


    def _parse(self, saccess: str):
        taccess = saccess.split('.')
        for i, ta in enumerate(taccess):
            try:
                taccess[i] = literal_eval(ta)
            except ValueError:
                pass
        for i, ta in enumerate(taccess):
            if not isinstance(ta, (Integral, str, tuple, list)):
                raise TypeError(f"incorrect type for accessor, {type(ta)}")
            if isinstance(ta, list):
                taccess[i] = np.array(ta, dtype=int)
            if isinstance(ta, tuple):
                taccess[i] = self._parse_indexation(ta)

        return tuple(taccess)


    def _parse_indexation(self, raw_index: tuple):
        index = []
        for ridx in raw_index:
            if isinstance(ridx, tuple):
                index.append(slice(*ridx))
            elif isinstance(ridx, list):
                index.append(np.array(ridx, dtype=int))
            elif isinstance(ridx, Integral):
                index.append(ridx)
            else:
                raise ValueError(f"incorrect type for index: {type(ridx)}")
        return tuple(index)


    def _unparse(self, taccess: tuple):
        saccess = []
        for ta in taccess:
            if isinstance(ta, (Integral, list, str)):
                saccess.append(str(ta))
            elif isinstance(ta, np.ndarray):
                saccess.append(str(ta.tolist()))
            elif isinstance(ta, tuple):
                saccess.append(self._unparse_indexation(ta))
        return '.'.join(saccess)


    def _unparse_indexation(self, raw_index: tuple):
        index = []
        for ridx in raw_index:
            if isinstance(ridx, (Integral, list)):
                index.append(ridx)
            elif isinstance(ridx, np.ndarray):
                index.append(ridx.tolist())
            elif isinstance(ridx, slice):
                index.append((ridx.start, ridx.stop, ridx.step))
            else:
                raise TypeError(f"incorrect type for index: {type(ridx)}")
        return(str(tuple(index)))