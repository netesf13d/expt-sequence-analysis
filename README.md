# expt-sequence-analysis

Data processing and analysis package developed during my PhD.

This package is dedicated to analyze data obtained from atomic physics experiments carried in the single-atom regime. The specificity of these experiments is that we essentially detect whether an atom is still present after an experimental sequence or not. Thus, in addition to the standard data fitting and plotting functionality, some tools are implemented to analyze the raw atomic signal and convert it to the desired statistical quantity (eg the recapture probability).

More detailed explanations about the data processing and analysis process can be found in [my thesis](https://theses.hal.science/tel-04551702), more specifically in chapter 3.

The package provides:
* Analysis of atomic florescence in regions of interest (ROIs) and atom
  detection.
* Determination of various events (eg loss/recapture of a trapped atom), and
  computation of derived quantities (eg probability of recapture from many
  repeated experiments, average over ROIs, etc).
* Analysis and plotting of processed data.

The code was written while I was rushing to finish writting my thesis. Although it was extensively tested and is provided with some examples, the  code is somewhat lacking documentation and could be improved.

I do not maintain this code anymore.


## Dependencies

This package requires Python > 3.10 and the following packages:
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/) (special functions and data fitting)
- [matplotlib](https://matplotlib.org/)
- [pandas](https://pandas.pydata.org/) (file loading)


## Notes

The typing annotations in the code are by no means rigorous. They are made to facilitate the understanding of the nature of various parameters.
