# -*- coding: utf-8 -*-
"""
========== Data analysis of Atom_Stats with a Data_Set ==========

This example illustrates the use of the <seqan.dataset.Data_Set> class to
analyze and display preprocessed data. It also shows the use of
<seqan.analysis.fit.Fit> and <seqan.plots.Plot> classes that encapsulate
analysis results and plots, respectively.

The script describes:
* The creation and management of a Data_Set object through a Trap_Analyzer or
  a file.
* The access to the various statistics available.
* Data analysis, fitting with a model
* Plotting data and fit results
* Features specific to multiple ROI datasets (eg mosaic plot)
"""

import sys
from pathlib import Path

import numpy as np

seqand = Path(__file__).resolve().parents[1] # seqan directory
if not str(seqand) in sys.path:
    sys.path.append(str(seqand))
import seqan as sa


fpath = "./Rabi/"


# =============================================================================
# Average ROI data analysis
# =============================================================================

"""
A Data_Set instance is directly returned by a Trap_Analyzer when calling
<Trap_Analyzer.get_*_stats>. This is the most natural way of doing.

A Data_Set can be converted to a dict, and in therefore also instanciated from
a dict. This dict is saved to disk when calling <Data_Set.savebin>.
"""
## Data_Set from a Trap_Analyzer
with np.load(fpath + "Rabi.npz", allow_pickle=False) as f:
    data = f['data']
ta = sa.Trap_Analyzer(sequenceSetup=sa.Sequence_Setup(fpath + "Rabi_seq.json"),
                      roiManager=sa.ROI_Manager(source=fpath + "Rabi_seq.json"),
                      rawData=data)
ta.compute_thresholds()
ta.atom_detection()
ds_avg = ta.get_pair_stats(delayFrame=1,  # returns a Data_Set instance
                           roiAggreg=[np.arange(18)])

## Convert to dict, save this dict
ds_dict = ds_avg.get_dict()
ds_avg.savebin(file=fpath + "ds_avg.npz")
np.savez_compressed(fpath + "ds_avg.npz", **ds_dict) # equivalent to above line

## Load from file or dict
# ds_avg = sa.Data_Set(source=fpath + "ds_avg.npz")
# ds_avg = sa.Data_Set(source=ds_dict)


"""
Additional experimental parameters (for instance laser powers) can be stored in
the `parameters` attribute.
Currently, these are used by the Monte-Carlo simulation fits which require
trap depth, etc.
"""
## Set from dict ...
ds_avg.set_parameters({'test': 42})
print(ds_avg.parameters)
## ...or file
ds_avg.set_parameters(fpath + "Rabi_seq.json") # reset and replace parameters
print(ds_avg.parameters)


"""
The data is actually stored in an `Atom_Stats` object that holds a dict
statDict: stat_type -> np.ndarray for the various statistics available.

Accessing data in the Data_Set can be done in two ways
* Raw data access.
  The accessor is a tuple (key, indexing). The key selects the statistics
  array, and the indexing selects the subarray/elements.
* Shortcut access.
  The assessor is a str, which is mapped to a raw accessor before fetching the
  data. This method is much simpler and less error prone.
  Shortcuts can be defined in the file seqan.stats.stats_accessors.
"""
print(type(ds_avg.data)) # Atom_Stats
print(ds_avg.accessors()) # Show raw accessor keys and accessor shortcuts

## Fetch recapture probability and its standard deviation
## Using shortcuts
precap = ds_avg.get_data("precap")
stdev_precap = ds_avg.get_stdev("precap") # eqivalent to ds_avg.get_data("stdev_precap")
## Using raw accessors
precap = ds_avg.get_data(('pcond', (slice(None), 1, 1, slice(None))))
stdev_precap = ds_avg.get_data(('stdev_pcond', (slice(None), 1, 1, slice(None))))

## Fetch the scanned values
xval = ds_avg.get_xvalues()


"""
The main data analysis procedure consists in fitting the data with a model
function. This returns a <seqan.analysis.fit.Fit> object that encapsulates the
fitted data along with the fit results and provides data saving and plotting
functionality.
Some models require additional parameters, that can be set manually and are
passed along with the Data_Set.parameters dict. The initial values for the
model parameters can be set manually. This, however, is not recommended since
efficient routines are implemented to estimate them by default. Unit conversion
for the x-values (eg voltage to frequency for a VCO) can be done by selecting
a calibration and the desired conversion.
Fit results are stored as a list in the attribut `Data_Set.fits`.

Other analysis procedures are available: moment computation and FFT analysis.
These return a dictionary containing the results.
Analysis results are stored as a list in the attribut `Data_Set.analyses`.
"""
## Fitting with a damped sinus
avg_fit = ds_avg.fit(
    "precap", # fit the recapture probability
    model="damped_sin", # damped sinus model
    dslice=slice(None), # slice for x-values selection
    model_kw=None, # kwargs passed to the fitting routine in addition to Data_Set.parameters
    p0=None, # Set initial parameters for the fit
    calib=None, # select calibration
    conv_to=None, # conversion to units `conv_to`
    conv_from=None) # conversion from units `conv_from`

avg_fit = ds_avg.get_fit(-1) # fit results are stored internally

## FFT analysis
fft_analysis = ds_avg.analyze("precap", routine="fft")
fft_analysis = ds_avg.get_analysis(-1)
# ds_avg.savebin_analysis(fpath + "Spectro_fft_analysis.npz") # saving


"""
Plots can be prepared by calling the <plot> method of either a Data_Set or
Fit object. This returns a <seqan.plots.Plot> object, which encapsulates a
matplotlib figure, allowing for further processing before displaying/saving.
"""
## Create plot displaying recapture and loss probabilities
plot = ds_avg.plot(["precap", "ploss"])
plot.legend() # add a legend
## manually display text on the figure (can be done directly with <Plot.fig_text>)
fig = plot.get_figure()
fig.text(0.5, 0.9, "Nice Rabi", **{'ha': "center"}) # add text
## manually set ylim (can be done directly with <Plot.set_ylim>)
ax = plot.get_axes()[0, 0]
ax.set_ylim(0, 1)
## Show plot
plot.show()

## Create plot from the fit result
fit_plot = avg_fit.plot(show_p0=True) # show the initial fit parameters
fit_plot.set_ylim((0, 1))
fit_plot.legend()
fit_plot.show()


"""
Data can be saved at each level, usually both in text an binary format
* <Data_Set.savebin> saves the whole Data_Set as seen before
* <Data_Set.savetxt> saves the selected entry in text format
* <Fit.savebin> saves the Fit_Result as a npz archive
* <Fit.savetxt> saves the fitted parameters in text format
* <Plot.save> saves the plot
"""
## Save in text format (useful for data analysis with other software)
# ds_avg.savetxt("precap", fpath + "avg_precap.txt") # will also save stdev_precap
## Save fitted parameters to text (for further analysis with other software)
# avg_fit.savetxt(fpath + "avg_precap_damped_sin_fit.txt")
## save a plot
# fit_plot.save(fpath + "avg_fit_plot.png")


"""
Datasets pertaining to the same type of experiments can be merged.
A new Data_Set is created with scanned values corresponding to the distinct
values of the combined datasets. At identical scanned points, the event counts
are summed to improve the signal to noise ratio.
"""
merged_ds = sa.merge_datasets([ds_avg, ds_avg, ds_avg], newname="merged stuff")



# =============================================================================
# Multiple ROI data analysis
# =============================================================================

"""
Working with multiple ROIs is essentially the same as working with one.
"""

ds_roi = sa.Data_Set(source=fpath + "ds_roi.npz")


"""
When fitting or carrying an analysis with multiple ROIs, each ROI data is
analyzed independently.
"""
roi_fit = ds_roi.fit("precap", model="damped_sin")
print(roi_fit.popt.shape) # 18 ROIs, 5 parameters

"""
For multiple ROI plots, the position of each individual plot can be set using
an ROI mapper.
Any fitted parameter can be plotted as a mosaic. This is useful to appreciate
a spatial dependency.
"""
## Setup an ROI mapper
nbroi, sh = 18, (3, 6)
mapper = np.transpose(np.array(
    [sh[0]-1 - np.arange(nbroi) % sh[0], np.arange(nbroi) // sh[0]]))

## Multiple ROI plot
roi_plot = roi_fit.plot(shape=sh, roi_mapper=mapper)

## Mosaic plot of the oscillation frequency
mosaic_plot = roi_fit.mosaic_plot("nu", shape=sh, roi_mapper=mapper)

