"""
TrainSelPy: Python implementation of the TrainSel package for optimal training set selection
"""

from trainselpy.core import (
    make_data,
    train_sel,
    train_sel_control,
    set_control_default,
    time_estimation
)

from trainselpy.optimization_criteria import (
    dopt,
    maximin_opt,
    pev_opt,
    cdmean_opt,
    cdmean_opt_target,
    fun_opt_prop,
    aopt,
    eopt,
    coverage_opt
)

from trainselpy.utils import (
    r_data_to_python,
    create_distance_matrix,
    calculate_relationship_matrix,
    create_mixed_model_data,
    plot_optimization_progress,
    plot_pareto_front
)

# Import data module
from trainselpy.data import wheat_data

__version__ = "0.1.1"