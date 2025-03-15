# TrainSelPy

A pure Python implementation of the TrainSel R package for optimal selection of training populations.

## Overview

TrainSelPy provides tools for optimizing the selection of training populations, primarily for genomic selection and experimental design. It implements genetic algorithms and simulated annealing to select optimal subsets from candidate sets based on various criteria.

## Installation

### From PyPI

```bash
pip install trainselpy
```

### From Source

```bash
git clone https://github.com/yourusername/trainselpy.git
cd trainselpy
pip install -e .
```

## Key Features

- **Multiple Selection Types**:
  - Unordered Set (UOS): Selection where order doesn't matter
  - Ordered Set (OS): Selection where order matters
  - Multi-sets (UOMS, OMS): Selections with repeated elements
  - Boolean (BOOL): Binary selection
  - Continuous (DBL): Numeric variable optimization

- **Built-in Optimization Criteria**:
  - CDMean: For mixed models, optimizing prediction accuracy
  - D-optimality: Maximizing the determinant of the information matrix
  - PEV: Minimizing prediction error variance
  - Maximin: Maximizing the minimum distance between selected samples

- **Advanced Optimization Algorithms**:
  - Genetic Algorithm (GA): Population-based optimization
  - Simulated Annealing (SANN): Fine-tuning solutions
  - Island Model: Multiple populations evolving in parallel
  - Multi-objective optimization with diverse Pareto front solutions

- **Parallelization**:
  - Support for parallel computing to speed up optimization
  - Different parallelization strategies for different optimization scenarios

## Basic Usage

```python
import numpy as np
from trainselpy import make_data, train_sel, set_control_default

# Load example data
from trainselpy.data import wheat_data

# Create the TrainSel data object
ts_data = make_data(M=wheat_data["M"])

# Set control parameters
control = set_control_default()
control["niterations"] = 10

# Run the selection algorithm
result = train_sel(
    data=ts_data,
    candidates=[list(range(200))],  # Select from first 200 lines
    setsizes=[50],                  # Select 50 lines
    settypes=["UOS"],              # Unordered set
    stat=None,                     # Use default CDMean
    control=control
)

print("Selected indices:", result.selected_indices)
print("Fitness value:", result.fitness)
```

## Examples

### D-optimality Criterion

```python
from trainselpy import dopt

# Add feature matrix to data
ts_data["FeatureMat"] = wheat_data["M"]

# Run with D-optimality criterion
result = train_sel(
    data=ts_data,
    candidates=[list(range(200))],
    setsizes=[50],
    settypes=["UOS"],
    stat=dopt,  # Use D-optimality
    control=control
)
```

### Parallel Processing

```python
# Set control parameters for parallel processing
control = train_sel_control(
    size="demo",
    niterations=50,
    npop=200,
    nislands=4,      # Use 4 islands
    parallelizable=True,
    mc_cores=4       # Use 4 cores
)

# Run with parallel processing
result = train_sel(
    data=ts_data,
    candidates=[list(range(200))],
    setsizes=[50],
    settypes=["UOS"],
    stat=None,     # Use default CDMean
    control=control,
    n_jobs=4       # Use 4 parallel jobs
)
```

### Multi-objective Optimization with Diverse Solutions

```python
from trainselpy import cdmean_opt, dopt

# Define a multi-objective function
def multi_objective(solution, data):
    cdmean_value = cdmean_opt(solution, data)
    dopt_value = dopt(solution, data)
    return [cdmean_value, dopt_value]

# Run with multi-objective optimization and ensure diverse solutions
control = train_sel_control(
    niterations=100,
    npop=500,
    solution_diversity=True  # Ensure unique solutions on Pareto front
)

result = train_sel(
    data=ts_data,
    candidates=[list(range(200))],
    setsizes=[50],
    settypes=["UOS"],
    stat=multi_objective,
    n_stat=2,       # 2 objectives
    control=control
)

# Plot the Pareto front
from trainselpy.utils import plot_pareto_front
plot_pareto_front(
    result.pareto_front,
    obj_names=["CDMean", "D-optimality"],
    title="Pareto Front: CDMean vs D-optimality",
    output_file="pareto_front.png"
)
```

### Multiple Sets with Different Types

```python
# Define a custom fitness function
def custom_fitness(int_solutions, data):
    # Calculate fitness based on both sets
    set1 = int_solutions[0]  # First set (UOS)
    set2 = int_solutions[1]  # Second set (OS)
    
    # Implement your own fitness calculation
    return some_fitness_measure(set1, set2, data)

# Run with multiple sets
result = train_sel(
    data=ts_data,
    candidates=[list(range(100)), list(range(100, 200))],  # Two candidate sets
    setsizes=[30, 20],                     # Different sizes
    settypes=["UOS", "OS"],               # Different types
    stat=custom_fitness,
    control=control
)
```

## Converting R Data

If you have the original WheatData from the R package, you can convert it:

```python
from trainselpy.utils import r_data_to_python

# Convert R data to Python format
python_data = r_data_to_python("path/to/WheatData.rda", "wheat_data.pkl")

# Load the converted data
import pickle
with open("wheat_data.pkl", "rb") as f:
    wheat_data = pickle.load(f)
```

## Full Documentation

For more details and advanced usage, see the examples directory and the API documentation.

## Requirements

- numpy
- scipy
- pandas
- scikit-learn
- matplotlib
- joblib

Optional:
- rpy2 (for converting R data)
- seaborn (for advanced plotting)

## License

MIT License

## Acknowledgments

This package is a Python implementation of the TrainSel R package. The original R package was written by Deniz Akdemir, Julio Isidro Sanchez, Simon Rio and Javier Fernandez-Gonzalez. The TrainSelPy package was developed by Deniz Akdemir.