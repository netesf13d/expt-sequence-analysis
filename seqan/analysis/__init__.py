# -*- coding: utf-8 -*-
"""
Data analysis submodule.
"""

from .analysis import (
    analysis,
    analysis_keys,
    )

from .fit import (
    fctDict,
    Fit_Result
    )

from .plot_fit import (
    result_txt,)

from .fluorescence_analysis import (
    double_gaussian,
    gg_histo_analysis,
    gausson,
    gp_histo_analysis,
    )