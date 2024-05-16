# -*- coding: utf-8 -*-
"""
Configuration file.
"""

from pathlib import Path

import numpy as np


# =============================================================================
# PATHS
# =============================================================================

# atomic dynamics `simpack` simulation package location
SIMPACK_PATH = str(Path(__file__).resolve().parents[3])


# =============================================================================
# PLOTS CONFIG
# =============================================================================

DPI = 100 # Dots per inch, resolution of plots


# =============================================================================
# FLUORESCENCE THRESHOLD STUFF
# =============================================================================

# Enable safe threshold
# The usual threshold is set by minimizing the error rate assuming the data
# comes from the fitted model. However, this is not exactly the case and
# data can often be found in between the atom fluorescence and background
# signal peaks. SAFE_THRESHOLD enables the use of a manually set threshold
# level determined by THRESHOLD_LVL.
SAFE_THRESHOLD = True

# Safe threshold level
# If SAFE_THRESHOLD is True, enable the use of a threshold computed as:
# threshold = (1-t)*mu1 + t*mu2
# with given THRESHOLD_LVL t, mu1 the mean background signal, mu2 the mean
# atom fluorescence signal.
# Higher values reduce the false positives at the expense of false negatives.
# 0.45 is found to be a good value.
THRESHOLD_LVL = 0.45


# =============================================================================
# FIT CONFIG
# =============================================================================




# =============================================================================
# SIMULATIONS CONFIG
# =============================================================================

# Number of samples drawn for simulations
# Release-recapture
NB_SAMPLES_RR_COARSE = 20000 # used for coarse estimation
NB_SAMPLES_RR = 100000
# Oscillation in gauss trap
NB_SAMPLES_TO_COARSE = 2000 # used for coarse estimation
NB_SAMPLES_TO = 10000 # used for precise estimation
# Oscillation in BoB trap
NB_SAMPLES_BO = 5000

# Time unit for the simulations
TIME_UNIT = 1e-6 # second --> 1 us

# Frequency unit for the simulations
FREQUENCY_UNIT = 1/TIME_UNIT # Hz --> MHz

# Temperature unit
TEMP_UNIT = 1e-6 # kelvin --> 1 uK

# Length unit
LENGTH_UNIT = 1e-6 # meter --> 1 um

########## Release-Recapture simulation specific parameters ##########

### Coarse temperature estimation range
# The values of temperatures tested for coarse estimation
# in analysis.simfit.sim_rr
COARSE_TEMPERATURE_RANGE = np.linspace(4e-6, 80e-6, 39)

### Sampling method
# "rejection" or "normal"
# - "rejection" uses rejection sampling of atomic positions.
#   in this case a warning will be issued if T_MAX > 150e-6 K
# - "normal" uses normal sampling, ie assumes the trap is harmonic
# Normal sampling is much faster (approx 20 times) but imprecise at
# high temperatures (> 100 uK)
SAMPLING_MODE_RR = "normal" # recommended is "normal"


########## Trap oscillation simulation specific parameters ##########

### Coarse waist estimation range
# The values of the waist tested for coarse estimation
# in analysis.simfit.sim_traposc
COARSE_WAIST_RANGE = np.linspace(1.1e-6, 1.3e-6, 21)

### Sampling method
# "rejection" or "normal"
# - "rejection" uses rejection sampling of atomic positions.
#   in this case a warning will be issued if T_MAX > 150e-6 K
# - "normal" uses normal sampling, ie assumes the trap is harmonic
# Normal sampling is much faster (approx 20 times) but imprecise at
# high temperatures (> 100 uK)
SAMPLING_MODE_TO = "rejection" # recommended is "normal"


########## BoB trap oscillation simulation specific parameters ##########

### Coarse BoB power estimation range
# The values of the BoB power tested for coarse estimation
# in analysis.simfit.sim_bobosc
# !!! currently not used
COARSE_POWER_RANGE = np.linspace(0.01, 0.02, 21)

### Sampling method
# "rejection" or "normal"
# - "rejection" uses rejection sampling of atomic positions.
#   in this case a warning will be issued if T_MAX > 150e-6 K
# - "normal" uses normal sampling, ie assumes the trap is harmonic
# Normal sampling is much faster (approx 20 times) but imprecise at
# high temperatures (> 100 uK)
SAMPLING_MODE_BO = "rejection" # recommended is "normal"


# =============================================================================
# Physical parameters
# =============================================================================

if SAMPLING_MODE_RR not in {"normal", "rejection"}:
    raise ValueError(
        "release-recapture sampling mode (config file) must be `rejection` "
        f"or `normal`, got {SAMPLING_MODE_RR}")

if SAMPLING_MODE_TO not in {"normal", "rejection"}:
    raise ValueError(
        "trap oscillation sampling mode (config file) must be `rejection` "
        f"or `normal`, got {SAMPLING_MODE_TO}")


