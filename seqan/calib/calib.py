# -*- coding: utf-8 -*-
"""
TODO
- gen doc
- fct doc
-

"""

from numbers import Number
from functools import partial
from pathlib import Path

calib_path = Path(__file__).resolve().parents[0]

import numpy as np
from numpy import pi
from scipy.interpolate import interp1d

from scipy.constants import physical_constants
c = physical_constants['speed of light in vacuum'][0]
h = physical_constants['Planck constant'][0] # J.s.rad
# hbar = physical_constants['reduced Planck constant'][0] # Js
# e = physical_constants['elementary charge'][0]
alpha = physical_constants['fine-structure constant'][0]
# epsilon_0 = physical_constants['vacuum electric permittivity'][0]
k_B = physical_constants['Boltzmann constant'][0] # J/K
m_e = physical_constants['electron mass'][0]

# Inerpolation method for experimental calibration
INTERP_KIND = 'linear'

# =============================================================================
# PHYSICAL PARAMETERS CALIBRATION
# =============================================================================

########## Physical constants ##########

# light-shift coefficient for F=1,2 level
LScoef_5S = -0.01816429616015425 # Hz/(W/m^2)
# light-shift coefficient for the F=1 to F'=2 repumper transistion
LScoef_5S5P = 0.023876855585657605 # Hz/(W/m^2)
# Conversion from repumper lightshift to trap depth
LS_to_V0 = - LScoef_5S / LScoef_5S5P * h # J.s


lifetimes = {
    'ground': lambda x: 1.,
    }


########## Empirical parameters ##########

# (Re)capture threshold. Throughout the calculations, an atom is considered
# trapped if its mechanical energy is below the threshold.
# Care to set it negative !
recapture_threshold = - 0. * k_B


########## Experimental parameters ##########

# Trap wavelength
lambda_trap = 821 * 1e-9 # m

# Repumper spectro light shift
LS_rep = 24 * 1e6 # Hz

# Atom temperature
atom_temp = 16 * 1e-6 # K

# Gaussian beam parameters
gauss_params = {
    'wavelength': lambda_trap, # m, laser wavelength
    'waist': (w:=1.2 * 1e-6), # m
    'z_R': np.pi * w**2 / lambda_trap, # m,
    'lightshift': 27 * 1e6, # Hz
    'dz': 0., # m, advanced stuff: you dont care
    'beta_ponder': alpha * h /(m_e * (2*np.pi * c / lambda_trap)**2),
    }

# BoB parameters
bob_params = {
    'wavelength': lambda_trap, # m, laser wavelength
    'offset': 0.3 * 1e-6, # m
    'power': 3.14 * 1e-3, # W
    'beta_ponder': alpha * h /(m_e * (2*np.pi * c / lambda_trap)**2),
    }



# =============================================================================
#
# =============================================================================

def zero(x):
    if isinstance(x, Number):
        return 0
    elif isinstance(x, np.ndarray):
        return np.zeros_like(x)
    else:
        raise TypeError("x must be either a number or an array")

def identity(x):
    return x


# =============================================================================
# HARDWARE CALIBRATIONS
# =============================================================================


# =============================================================================
# ========================= CALIBRATIONS - VALUES =============================
# Mind to adapt the calibration builders if you want to add new units for
# conversion.
# =============================================================================

# ========== MOPA frequency ==========

MOPA_freq = {
    'units': ('V_dac', 'MHz_VCO', 'MHz_laser_abs', 'MHz_laser_rel'),
    # 2021-12-20 book 37 p.66
    'Vdac_to_VCO':
        (np.array( # Vdac
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
         np.array( # VCO freq
             [128.46, 141.77, 152.97, 162.65, 172.13, 181.54, 191.77, 202.52,
              213.45, 225.58, 236.86])),
    # Difference between VCO frequency and beatnote frequency
    'lock_offset': +9, # MHz
    # Beatlock frequency at which the MOT is resonant with F=2 -> F'=3
    'resonance': 0, # MHz
    }


# ========== Probe frequency ==========

probe_freq = {
    'units': ('V_dac', 'MHz_aom_abs', 'MHz_aom_rel',
              'MHz_laser_abs', 'MHz_laser_rel'),
    'Vdac_to_VCO':
        (np.array( # Vdac
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
         np.array( # VCO freq
             [43.74, 53.24, 60.22, 66.85, 73.91, 81.29, 88.80, 96.68, 97.76,
              97.77, 97.79])),
    # OK
    'VCO_to_laser': 2, # double pass on order +1
    'offsetVdac': 4.27,
    }


# ========== Repumper frequency ==========

repumper_freq = {
    'units': ('V_dac', 'MHz_aom_abs', 'MHz_aom_rel',
              'MHz_laser_abs', 'MHz_laser_rel'),
    'Vdac_to_VCO':
        (np.array( # Vdac
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
         np.array( # VCO freq
             [62.02, 69.19, 75.82, 81.94, 88.30, 94.69, 100.71, 106.58,
              112.35, 118.07, 123.67])),
    # OK
    'VCO_to_laser': -2, # double pass on order -1
    
    'offsetVdac': 5.443,
    }


# ========== Magnetic fields ==========

Bx = {
    'units': ('V_dac', 'A', 'Gauss'),
    # OK
    'Vdac_to_A': 3/5,
    # TODO
    'A_to_Gauss': 2.17,
    }

By = {
    'units': ('V_dac', 'A', 'Gauss'),
    # OK
    'Vdac_to_A': 3/10,
    # TODO
    'A_to_Gauss': 5/3,
    }

Bz = {
    'units': ('V_dac', 'A', 'Gauss'),
    # OK
    'Vdac_to_A': 3/10,
    # TODO
    'A_to_Gauss': 10/3,
    }


# =============================================================================
# ========================= CALIBRATIONS - BUILDERS ===========================
# =============================================================================

# ========== MOPA frequency ==========

MOPA_freq_conv = np.full((len(MOPA_freq['units']), len(MOPA_freq['units'])),
                          zero, dtype=object)
np.fill_diagonal(MOPA_freq_conv, identity)

calx = MOPA_freq['Vdac_to_VCO'][0]
caly = MOPA_freq['Vdac_to_VCO'][1]
MOPA_lockoff = MOPA_freq['lock_offset']
MOPA_res = MOPA_freq['resonance']

MOPA_freq_conv[0, 1] = interp1d(calx, caly, kind=INTERP_KIND)
MOPA_freq_conv[0, 2] = \
    lambda x: MOPA_freq_conv[0, 1](x) + MOPA_lockoff
MOPA_freq_conv[0, 3] = \
    lambda x: MOPA_freq_conv[0, 1](x) + MOPA_lockoff - MOPA_res
MOPA_freq_conv[1, 2] = \
    lambda x: x + MOPA_lockoff
MOPA_freq_conv[1, 3] = \
    lambda x: x + MOPA_lockoff - MOPA_res
MOPA_freq_conv[2, 3] = \
    lambda x: x - MOPA_res

MOPA_freq_conv[1, 0] = interp1d(caly, calx, kind=INTERP_KIND)
MOPA_freq_conv[2, 0] = \
    lambda x: MOPA_freq_conv[1, 0](x - MOPA_lockoff)
MOPA_freq_conv[3, 0] = \
    lambda x: MOPA_freq_conv[1, 0](x - MOPA_lockoff + MOPA_res)
MOPA_freq_conv[2, 1] = \
    lambda x: x - MOPA_lockoff
MOPA_freq_conv[3, 1] = \
    lambda x: x - MOPA_lockoff + MOPA_res
MOPA_freq_conv[3, 2] = \
    lambda x: x + MOPA_res

del calx, caly


# ========== Probe Frequency ==========

probe_freq_conv = np.full((len(probe_freq['units']), len(probe_freq['units'])),
                          zero, dtype=object)
np.fill_diagonal(probe_freq_conv, identity)

calx = probe_freq['Vdac_to_VCO'][0]
caly = probe_freq['Vdac_to_VCO'][1]
prb_f0 = interp1d(calx, caly, kind=INTERP_KIND)(probe_freq['offsetVdac'])
prb_k = probe_freq['VCO_to_laser']

probe_freq_conv[0, 1] = \
    interp1d(calx, caly, kind=INTERP_KIND)
probe_freq_conv[0, 2] = \
    lambda x: probe_freq_conv[0, 1](x) - prb_f0
probe_freq_conv[0, 3] = \
    lambda x: prb_k * probe_freq_conv[0, 1](x)
probe_freq_conv[0, 4] = \
    lambda x: prb_k * (probe_freq_conv[0, 1](x) - prb_f0)
probe_freq_conv[1, 2] = \
    lambda x: x - prb_f0
probe_freq_conv[1, 3] = \
    lambda x: prb_k * x
probe_freq_conv[1, 4] = \
    lambda x: prb_k * (x - prb_f0)
probe_freq_conv[2, 3] = \
    lambda x: prb_k * (x + prb_f0)
probe_freq_conv[2, 4] = \
    lambda x: prb_k * x
probe_freq_conv[3, 4] = \
    lambda x: x - prb_k * prb_f0

probe_freq_conv[1, 0] = \
    interp1d(caly, calx, kind=INTERP_KIND)
probe_freq_conv[2, 0] = \
    lambda x: probe_freq_conv[1, 0](x + prb_f0)
probe_freq_conv[3, 0] = \
    lambda x: probe_freq_conv[1, 0](x / prb_k)
probe_freq_conv[4, 0] = \
    lambda x: probe_freq_conv[1, 0](x / prb_k + prb_f0)
probe_freq_conv[2, 1] = \
    lambda x: x + prb_f0
probe_freq_conv[3, 1] = \
    lambda x: x / prb_k
probe_freq_conv[4, 1] = \
    lambda x: x/prb_k + prb_f0
probe_freq_conv[3, 2] = \
    lambda x: x/prb_k - prb_f0
probe_freq_conv[4, 2] = \
    lambda x: x / prb_k
probe_freq_conv[4, 3] = \
    lambda x: x + prb_k * prb_f0

del calx, caly


# ========== Repumper Frequency ==========

repumper_freq_conv = np.full((len(repumper_freq['units']),
                              len(repumper_freq['units'])),
                             zero, dtype=object)
np.fill_diagonal(repumper_freq_conv, identity)

calx = repumper_freq['Vdac_to_VCO'][0]
caly = repumper_freq['Vdac_to_VCO'][1]
rep_f0 = interp1d(calx, caly, kind=INTERP_KIND)(repumper_freq['offsetVdac'])
rep_k = repumper_freq['VCO_to_laser']

repumper_freq_conv[0, 1] = \
    interp1d(calx, caly, kind=INTERP_KIND)
repumper_freq_conv[0, 2] = \
    lambda x: repumper_freq_conv[0, 1](x) - rep_f0
repumper_freq_conv[0, 3] = \
    lambda x: rep_k * repumper_freq_conv[0, 1](x)
repumper_freq_conv[0, 4] = \
    lambda x: rep_k * (repumper_freq_conv[0, 1](x) - rep_f0)
repumper_freq_conv[1, 2] = \
    lambda x: x - rep_f0
repumper_freq_conv[1, 3] = \
    lambda x: rep_k * x
repumper_freq_conv[1, 4] = \
    lambda x: rep_k * (x - rep_f0)
repumper_freq_conv[2, 3] = \
    lambda x: rep_k * (x + rep_f0)
repumper_freq_conv[2, 4] = \
    lambda x: rep_k * x
repumper_freq_conv[3, 4] = \
    lambda x: x - rep_k * rep_f0

repumper_freq_conv[1, 0] = \
    interp1d(caly, calx, kind=INTERP_KIND)
repumper_freq_conv[2, 0] = \
    lambda x: repumper_freq_conv[1, 0](x + rep_f0)
repumper_freq_conv[3, 0] = \
    lambda x: repumper_freq_conv[1, 0](x / rep_k)
repumper_freq_conv[4, 0] = \
    lambda x: repumper_freq_conv[1, 0](x / rep_k + rep_f0)
repumper_freq_conv[2, 1] = \
    lambda x: x + rep_f0
repumper_freq_conv[3, 1] = \
    lambda x: x / rep_k
repumper_freq_conv[4, 1] = \
    lambda x: x/rep_k + rep_f0
repumper_freq_conv[3, 2] = \
    lambda x: x/rep_k - rep_f0
repumper_freq_conv[4, 2] = \
    lambda x: x / rep_k
repumper_freq_conv[4, 3] = \
    lambda x: x + rep_k * rep_f0

del calx, caly

# ========== Magnetic fields ==========

## Bx
Bx_conv = np.full((len(Bx['units']), len(Bx['units'])),
                  zero, dtype=object)
np.fill_diagonal(Bx_conv, identity)
Bx_conv[0, 1] = \
    lambda x : x * Bx['Vdac_to_A']
Bx_conv[1, 2] = \
    lambda x : x * Bx['A_to_Gauss']
Bx_conv[0, 2] = \
    lambda x : x * Bx_conv[0, 1] * Bx_conv[1, 2]

Bx_conv[1, 0] = \
    lambda x : x / Bx['Vdac_to_A']
Bx_conv[2, 1] = \
    lambda x : x / Bx['A_to_Gauss']
Bx_conv[2, 0] = \
    lambda x : x / (Bx_conv[0, 1] * Bx_conv[1, 2])

## By
By_conv = np.full((len(By['units']), len(By['units'])),
                  zero, dtype=object)
np.fill_diagonal(By_conv, identity)
By_conv[0, 1] = \
    lambda x : x * By['Vdac_to_A']
By_conv[1, 2] = \
    lambda x : x * By['A_to_Gauss']
By_conv[0, 2] = \
    lambda x : x * By_conv[0, 1] * By_conv[1, 2]

By_conv[1, 0] = \
    lambda x : x / By['Vdac_to_A']
By_conv[2, 1] = \
    lambda x : x / By['A_to_Gauss']
By_conv[2, 0] = \
    lambda x : x / (By_conv[0, 1] * By_conv[1, 2])

## Bz
Bz_conv = np.full((len(Bz['units']), len(Bz['units'])),
                  zero, dtype=object)
np.fill_diagonal(Bz_conv, identity)
Bz_conv[0, 1] = \
    lambda x : x * Bz['Vdac_to_A']
Bz_conv[1, 2] = \
    lambda x : x * Bz['A_to_Gauss']
Bz_conv[0, 2] = \
    lambda x : x * Bz_conv[0, 1] * Bz_conv[1, 2]

Bz_conv[1, 0] = \
    lambda x : x / Bz['Vdac_to_A']
Bz_conv[2, 1] = \
    lambda x : x / Bz['A_to_Gauss']
Bz_conv[2, 0] = \
    lambda x : x / (Bz_conv[0, 1] * Bz_conv[1, 2])

# =============================================================================
# ========================= CALIBRATIONS - FUNCTIONS ==========================
#
# =============================================================================

def convert(calib: str, x, conv_to: str = None, conv_from: str = None):
    """
    TODO doc

    Parameters
    ----------
    calib : str
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    conv_to : str, optional
        DESCRIPTION.
    conv_from : str, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if calib == 'MOPA_freq':
        units = MOPA_freq['units']
        conv = MOPA_freq_conv
    elif calib == 'probe_freq':
        units = probe_freq['units']
        conv = probe_freq_conv
    elif calib == 'repumper_freq':
        units = repumper_freq['units']
        conv = repumper_freq_conv
    elif calib == 'Bx':
        units = Bx['units']
        conv = Bx_conv
    elif calib == 'By':
        units = By['units']
        conv = By_conv
    elif calib == 'Bz':
        units = Bz['units']
        conv = Bz_conv
    else:
        raise ValueError(f"calibration `{calib}` not available")

    if not conv_to:
        conv_to = units[0]
    if not conv_from:
        conv_from = units[0]
    if conv_to not in units:
        raise ValueError(f"can only convert to {units}, not {conv_to}")
    if conv_from and (conv_from not in units):
        raise ValueError(f"can only convert from {units}, not {conv_from}")
    return conv[units.index(conv_from), units.index(conv_to)](x)

# =============================================================================
#
# =============================================================================
cal = ['MOPA_freq',
       'probe_freq',
       'repumper_freq',
       'Bx',
       'By',
       'Bz']

calibDict = {conv: partial(convert, conv) for conv in cal}