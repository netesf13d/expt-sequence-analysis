# -*- coding: utf-8 -*-
"""
A few tests for the various components of seqan.
"""

import os
import sys
from pathlib import Path

import numpy as np
from numpy.random import default_rng

seqand = Path(__file__).resolve().parents[1] # seqan directory
if not str(seqand) in sys.path:
    sys.path.append(str(seqand))
import seqan as sa


rng = default_rng()
fpath = "./data/"


# =============================================================================
# Test Sequence_Setup
# =============================================================================

acqDict = {
    'name': "test",
    'date': "",
    'nbroi': 0,
    'nbframes': 352,
    'nbrun': 0,
    'frameperrun': 2,
    'deletionslice': (),
    'trigmiss': 1,
    'nbscan': 4,
    'repperstep': 10,
    'nbrep': 40,
    'nbstep': 4,
    'step': 1.,
    'vari': 0.,
    'varf': 9,
    'varname': "test",
    'varunit': "None",
    'info': "Test sequence",
    }

def test_Sequence_Setup(acq_dict: dict):
    acqTest = sa.Sequence_Setup(**acq_dict)
    
    print(acqTest)
    print("\nTest sequence setup...")
    acqTest.set_path(".")
    print(bool(acqTest.get_path()))
    print(acqTest.get_path())
    print("Deletion slices:", acqTest.deletion_slices())
    scanSel = acqTest.scanpt_selectors(np.array([True, True, False, True]))
    print("Scan slices:", np.sum(scanSel[0]))
    print("Scanned values:", acqTest.scanned_values())
    print("Selected scans:", scanSel.shape)
    
    acqTest.save_sequence(fpath)
    acqTest.set_param(parameters={'lasser_power': 0.3})
    print("Scanned values:\n", acqTest.scanned_values())
    
    acqTest2 = sa.Sequence_Setup(fpath + "test_seq.json")
    print(acqTest2)
    try:
        os.remove(fpath + "test_seq.json")
    except FileNotFoundError:
        pass
    print("...done")


def test_General_Accessor():
    print("\nTest General_Accessor...")
    access = "1.key1.key2.((0, None, 2), 1, (0, None, 2))"
    GA = sa.General_Accessor(access)
    print(GA.access_tuple)
    GA2 = sa.General_Accessor(GA.access_tuple)
    print(GA2.access_str)
    GA3 = sa.General_Accessor(GA2.access_str)
    print(GA3.access_tuple)
    
    a = np.arange(40).reshape((5, 2, 4))
    print(a[GA.access_tuple[-1]])
    print("...done")


def test_ROI_Manager():
    print("\nTest ROI_Manager...")
    roiM = sa.ROI_Manager(source=fpath + "Rabi_seq.json")
    
    clusters = np.arange(roiM.nbroi).reshape((-1, 3))
    roiM.set_clusters(clusters)
    
    tgt_rois = np.arange(9)
    roiM.set_tgt_rois(tgt_rois)
    print("...done")


def test_atom_stats():
    print("\nTest Atom_Stats...")
    nbroi = 18 # number of ROIs
    fpr = 2 # frames per run
    k0 = 11 # nb scan points
    nbrep = 50 # nb repetitions per scan point
    pload = 0.6 # loading probability
    
    sel = np.zeros((k0, nbrep*k0), dtype=bool)
    for i in range(k0):
        sel[i, i*nbrep:(i+1)*nbrep] = True
    
    pop = np.zeros((fpr, nbroi, 2, k0*nbrep), dtype=bool)
    pop[0, :, 1, :] = rng.binomial(1, pload, size=(nbroi, k0*nbrep))
    pop[1, :, 1, :] = rng.binomial(1, pload, size=(nbroi, k0*nbrep))
    pop[:, :, 0, :] = np.logical_not(pop[:, :, 1, :])
    
    ########## pair stats ##########
    rpop, dpop = pop[0, ...], pop[1, ...]
    pairAS = sa.Atom_Stats(rpop=rpop, dpop=dpop, scanSelect=sel)
    
    ## save stats
    pairAS.save(fpath+"pairAS.npz")
    pairAS = sa.Atom_Stats(source=fpath+"pairAS.npz")
    try:
        os.remove(fpath + "pairAS.npz")
    except FileNotFoundError:
        pass
    
    ## symmetrize
    sym_pairAS = pairAS.symmetrize()
    
    ## merge stats
    indexation = np.zeros((3, 2*k0), dtype=bool)
    for i in range(3):
        indexation[i, rng.choice(2*k0, k0, replace=False)] = True
    merged_pairAS = sa.merge_stats((pairAS,)*3, indexation)
    print(indexation)
    
    ## average stats
    roiSelect = np.zeros((3, nbroi), dtype=bool)
    idx = rng.choice(3*nbroi, nbroi, replace=False)
    idx = (idx // nbroi, idx % nbroi)
    roiSelect[idx] = True
    avg_pairAS = pairAS.average(roiSelect)
    
    ## full stats
    avg_pairAS = sa.Atom_Stats(source=avg_pairAS)
    avg_pairAS._compute_full_stats()
    
    
    ########## cluster stats ##########
    nbc, csize = 6, 3
    clusters = np.arange(nbroi).reshape((nbc, csize))
    cpop = np.zeros((fpr, nbc, 2**csize, k0*nbrep), dtype=bool)
    for p, i in np.ndenumerate(np.arange(2**csize).reshape((2,)*csize)):
        p = np.reshape(p, (1, -1, 1))
        for j, c in enumerate(clusters):
            cpop[:, j, i, :] = np.all(pop[:, c, 1, :] == p, axis=1)
    clusterAS = sa.Atom_Stats(rpop=cpop, scanSelect=sel)
    
    ## save stats
    clusterAS.save(fpath + "clusterAS.npz")
    clusterAS = sa.Atom_Stats(source=fpath+"clusterAS.npz")
    try:
        os.remove(fpath + "clusterAS.npz")
    except FileNotFoundError:
        pass
    
    ## symmetrize
    sym_clusterAS = clusterAS.symmetrize()
    
    ## merge stats
    merged_clusterAS = sa.merge_stats((clusterAS,)*3, indexation)
    
    ## average stats
    cSelect = np.zeros((3, nbc), dtype=bool)
    idx = rng.choice(3*nbc, nbc, replace=False)
    idx = (idx // nbc, idx % nbc)
    cSelect[idx] = True
    avg_clusterAS = clusterAS.average(cSelect)
    
    ## full stats
    avg_clusterAS = sa.Atom_Stats(source=avg_clusterAS)
    avg_clusterAS._compute_full_stats()
    
    print("...done")


# =============================================================================
# 
# =============================================================================

if __name__ == '__main__':
    test_Sequence_Setup(acqDict)
    test_General_Accessor()
    test_ROI_Manager()
    test_atom_stats()
    




