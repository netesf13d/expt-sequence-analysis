# -*- coding: utf-8 -*-
"""
========== Data analysis of other data sources with a Data_Set ==========

This example illustrates the use of the <seqan.dataset.Data_Set> class to
analyze and display data from external sources (eg files). It shows the
workflow to load, analyze and display data from external files.

The script describes:
* The creation of a `Sequence_Manager_Data` object to load and preprocess
  data files in a directory.
* The creation of a `Data_Set` object directly from a `Sequence_Manager_Data`
* The access to the data available.
* A typical workflow for data treatment.
"""

import sys
from pathlib import Path

seqand = Path(__file__).resolve().parents[1] # seqan directory
if not str(seqand) in sys.path:
    sys.path.append(str(seqand))
import seqan as sa


fpath = "./Spectro/"


# =============================================================================
# Data_Set from other sources
# =============================================================================

"""
When dealing with data from external sources, the Data_Set must be created
from a dict.

Here, the data is produced as text files by a software (`Sequence Manager`).
The files are first loaded and preprocessed with a <Sequence_Manager_Data>
object before being fet to the Data_Set.

Note that the Sequence_Manager_Data deals only with this specific file format
output by the software (and not all of them).
"""
## Load and preprocess data files through a Sequence_Manager_Data
smd = sa.Sequence_Manager_Data(fpath)
## Instanciate the Data_Set from the Sequence_Manager_Data object
ds = sa.Data_Set(source=smd)
## Parameters must be set externally
ds.set_parameters({'pulse_duration': 50.}) # 50 us MW pulse

# ds.plot(["Transfert 52C-50C"])
"""
Data access differs from `Atom_Stats`-derived data. Here, the data is stored
directly as a dict, thus only shortcut access is possible.
"""
print(ds.accessors())
### Only shortcuts are allowed to fetch data
transfer = ds.get_data("Transfert 52C-50C")
stdev_transfer = ds.get_stdev("Transfert 52C-50C")
stdev_transfer = ds.get_data("stdev_Transfert 52C-50C") # equuivalent but less elegant


"""
The remaining of the workflow is similar to the `Atom-stats`-based data
analysis.

Here we fit with a lorentz peak convolved with an exponential relaxation of the
transition frequency. The resulting fit is plotted, the fitted parameters and
the plot are saved.
"""
## Fit and save
fit = ds.fit("Transfert 52C-50C", model="inverse_lorentz")
# fit.savetxt(fpath + "Transfert_52C-50C_fit_results.txt")
## Plot and save
fit_plot = fit.plot(show_p0=True)
# fit_plot.save(fpath + "Transfer_52C-50C_fit_plot.png")

