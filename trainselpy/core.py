"""
Core module implementing the main functionality of TrainSelPy.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Callable, Union, Optional, Any, TypedDict
from dataclasses import dataclass
import time
from joblib import Parallel, delayed
from scipy.stats import norm
from scipy.linalg import det, solve
from scipy.spatial.distance import pdist, squareform
import warnings
import random

from trainselpy.optimization_criteria import cdmean_opt
from trainselpy.algorithms import (
    genetic_algorithm,
    island_model_ga
)


class TrainSelData(TypedDict, total=False):
    G: Union[np.ndarray, pd.DataFrame]
    R: Union[np.ndarray, pd.DataFrame]
    lambda_val: float
    labels: pd.DataFrame
    Nind: int
    class_name: str
    X: Optional[np.ndarray]


class ControlParams(TypedDict, total=False):
    size: str
    niterations: int
    minitbefstop: int
    nEliteSaved: int
    nelite: int
    npop: int
    mutprob: float
    mutintensity: float
    crossprob: float
    crossintensity: float
    niterSANN: int
    tempini: float
    tempfin: float
    dynamicNelite: bool
    progress: bool
    parallelizable: bool
    mc_cores: int
    nislands: int
    niterIslands: int
    minitbefstopIslands: int
    nEliteSavedIslands: int
    neliteIslands: int
    npopIslands: int
    niterSANNislands: int
    solution_diversity: bool
    trace: bool
    use_surrogate: bool
    surrogate_start_gen: int
    surrogate_update_freq: int
    surrogate_prescreen_factor: int
    use_surrogate_objective: bool
    surrogate_generation_prob: float
    use_nsga3: bool


def make_data(
    M: Optional[np.ndarray] = None,
    K: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    Vk: Optional[np.ndarray] = None,
    Ve: Optional[np.ndarray] = None,
    lambda_val: Optional[float] = None,
    X: Optional[np.ndarray] = None
) -> TrainSelData:
    """
    Create a data structure for TrainSel optimization.
    
    Parameters
    ----------
    M : ndarray, optional
        Features matrix for samples.
    K : ndarray, optional
        Relationship matrix for samples.
    R : ndarray, optional
        Relationship matrix for errors of samples.
    Vk : ndarray, optional
        Relationship matrix blocks.
    Ve : ndarray, optional
        Relationship matrix errors of blocks.
    lambda_val : float, optional
        Ratio of Ve to Vk.
    X : ndarray, optional
        Design matrix.
        
    Returns
    -------
    Dict[str, Any]
        Data structure for TrainSel optimization.
    """
    if M is None and K is None:
        raise ValueError("At least one of M (features) or K (similarity matrix) must be provided.")
    
    if M is None and K is not None:
        # Use SVD decomposition to obtain a feature matrix from K.
        U, _, _ = np.linalg.svd(K)
        M = U
        if hasattr(K, 'index'):
            M = pd.DataFrame(M, index=K.index)
    
    # Determine names from M
    if hasattr(M, 'index'):
        names_in_M = M.index
    else:
        names_in_M = np.arange(M.shape[0])
        M = pd.DataFrame(M, index=names_in_M)
    
    # Compute K from M if not provided
    if K is None:
        K = np.dot(M, M.T) / M.shape[1]
        K = make_positive_definite(K / np.mean(np.diag(K)))
    
    if not hasattr(K, 'index'):
        K = pd.DataFrame(K, index=names_in_M, columns=names_in_M)
    
    if R is None:
        R = np.eye(K.shape[0])
        R = pd.DataFrame(R, index=K.index, columns=K.columns)
    
    if Vk is not None and Ve is not None:
        if lambda_val is None:
            lambda_val = 1
        
        if not hasattr(Vk, 'index'):
            Vk = pd.DataFrame(Vk, index=np.arange(Vk.shape[0]))
        
        if not hasattr(Ve, 'index'):
            Ve = pd.DataFrame(Ve, index=Vk.index)
        
        # Compute the Kronecker product for block matrices.
        big_K = np.kron(Vk.values, K.values)
        big_R = np.kron(lambda_val * Ve.values, R.values)
        
        # Create combined labels.
        k_idx = K.index
        vk_idx = Vk.index
        combined_names = [f"{a}_{b}" for a in vk_idx for b in k_idx]
        
        labels = pd.DataFrame({
            'intlabel': np.arange(len(combined_names)),
            'names': combined_names
        })
        
        big_K = pd.DataFrame(big_K, index=combined_names, columns=combined_names)
        big_R = pd.DataFrame(big_R, index=combined_names, columns=combined_names)
    else:
        if lambda_val is None:
            lambda_val = 1
        
        big_K = K
        big_R = lambda_val * R
        
        labels = pd.DataFrame({
            'intlabel': np.arange(len(big_K.index)),
            'names': big_K.index
        })
    
    result: TrainSelData = {
        'G': big_K,
        'R': big_R,
        'lambda_val': lambda_val,
        'labels': labels,
        'Nind': K.shape[0],
        'class_name': "TrainSel_Data"
    }
    
    if X is not None:
        result['X'] = X
    
    return result


def train_sel_control(
    size: str = "free",
    niterations: int = 2000,
    minitbefstop: int = 500,
    nEliteSaved: int = 10,
    nelite: int = 200,
    npop: int = 1000,
    mutprob: float = 0.01,
    mutintensity: float = 0.1,
    crossprob: float = 0.5,
    crossintensity: float = 0.75,
    niterSANN: int = 200,
    tempini: float = 100.0,
    tempfin: float = 0.1,
    dynamicNelite: bool = True,
    progress: bool = True,
    parallelizable: bool = False,
    mc_cores: int = 1,
    nislands: int = 1,
    niterIslands: int = 200,
    minitbefstopIslands: int = 20,
    nEliteSavedIslands: int = 3,
    neliteIslands: int = 50,
    npopIslands: int = 200,
    niterSANNislands: int = 30,
    solution_diversity: bool = True,
    trace: bool = False,
    use_surrogate: bool = False,
    surrogate_start_gen: int = 10,
    surrogate_update_freq: int = 5,
    surrogate_prescreen_factor: int = 5,
    use_surrogate_objective: bool = False,
    surrogate_generation_prob: float = 0.0,
    use_nsga3: bool = False
) -> ControlParams:
    """
    Create a control object for the TrainSel function.
    
    Parameters
    ----------
    size : str
        Size of the problem (e.g., "free").
    niterations : int
        Maximum number of iterations.
    minitbefstop : int
        Minimum number of iterations before stopping.
    nEliteSaved : int
        Number of elite solutions to save.
    nelite : int
        Number of elite solutions to carry to the next generation.
    npop : int
        Population size.
    mutprob : float
        Mutation probability.
    mutintensity : float
        Mutation intensity.
    crossprob : float
        Crossover probability.
    crossintensity : float
        Crossover intensity.
    niterSANN : int
        Number of simulated annealing iterations.
    tempini : float
        Initial temperature for simulated annealing.
    tempfin : float
        Final temperature for simulated annealing.
    dynamicNelite : bool
        Whether to adjust the number of elites dynamically.
    progress : bool
        Whether to display progress.
    parallelizable : bool
        Whether to use parallelization.
    mc_cores : int
        Number of cores to use for parallelization.
    nislands : int
        Number of islands for the island model.
    niterIslands : int
        Maximum number of iterations for the island model.
    minitbefstopIslands : int
        Minimum number of iterations before stopping for the island model.
    nEliteSavedIslands : int
        Number of elite solutions to save for the island model.
    neliteIslands : int
        Number of elite solutions to carry to the next generation for the island model.
    npopIslands : int
        Population size for the island model.
    niterSANNislands : int
        Number of simulated annealing iterations for the island model.
    solution_diversity : bool
        Whether to enforce uniqueness of solutions on the Pareto front. When True,
        duplicate solutions (same selected indices) are eliminated, keeping only 
        the one with the best overall fitness, ensuring more diverse solutions.
    trace : bool
        Whether to save the trace of the optimization.
    use_surrogate : bool
        Whether to use surrogate-assisted optimization.
    surrogate_start_gen : int
        Generation to start using surrogate.
    surrogate_update_freq : int
        Frequency of surrogate model updates.
    surrogate_prescreen_factor : int
        Factor for generating extra offspring for pre-screening.
    use_surrogate_objective : bool
        Whether to use the surrogate model as the objective function.
    surrogate_generation_prob : float
        Probability of generating offspring using surrogate optimization.
    use_nsga3 : bool
        Whether to use NSGA-III selection for multi-objective optimization.
        
    Returns
    -------
    Dict[str, Any]
        Control object for the TrainSel function.
    """
    control: ControlParams = {
        "size": size,
        "niterations": niterations,
        "minitbefstop": minitbefstop,
        "nEliteSaved": nEliteSaved,
        "nelite": nelite,
        "npop": npop,
        "mutprob": mutprob,
        "mutintensity": mutintensity,
        "crossprob": crossprob,
        "crossintensity": crossintensity,
        "niterSANN": niterSANN,
        "tempini": tempini,
        "tempfin": tempfin,
        "dynamicNelite": dynamicNelite,
        "progress": progress,
        "parallelizable": parallelizable,
        "mc_cores": mc_cores,
        "nislands": nislands,
        "niterIslands": niterIslands,
        "minitbefstopIslands": minitbefstopIslands,
        "nEliteSavedIslands": nEliteSavedIslands,
        "neliteIslands": neliteIslands,
        "npopIslands": npopIslands,
        "niterSANNislands": niterSANNislands,
        "solution_diversity": solution_diversity,
        "trace": trace,
        "use_surrogate": use_surrogate,
        "surrogate_start_gen": surrogate_start_gen,
        "surrogate_update_freq": surrogate_update_freq,
        "surrogate_prescreen_factor": surrogate_prescreen_factor,
        "use_surrogate_objective": use_surrogate_objective,
        "surrogate_generation_prob": surrogate_generation_prob,
        "use_nsga3": use_nsga3
    }
    
    return control


def set_control_default(size: str = "free", **kwargs) -> ControlParams:
    """
    Set default parameters for the TrainSel control object.
    
    Parameters
    ----------
    size : str
        Size of the problem (default is "free").
    **kwargs
        Additional control parameters.
        
    Returns
    -------
    Dict[str, Any]
        Control object with default parameters.
    """
    # Use the provided size parameter
    return train_sel_control(
        size=size,
        niterations=2000,
        minitbefstop=500,
        nEliteSaved=10,
        nelite=200,
        npop=1000,
        mutprob=0.01,
        mutintensity=0.1,
        crossprob=0.5,
        crossintensity=0.75,
        niterSANN=200,
        tempini=100.0,
        tempfin=0.1,
        dynamicNelite=True,
        progress=True,
        parallelizable=False,
        mc_cores=1,
        nislands=1,
        solution_diversity=True,
        **kwargs
    )


@dataclass
class TrainSelResult:
    """
    Class to hold the results of the TrainSel optimization.
    """
    selected_indices: List[List[int]]  # Selected indices for each set.
    selected_values: List[Any]         # Selected values for each set.
    fitness: float                     # Fitness value.
    fitness_history: List[float]       # History of fitness values.
    execution_time: float              # Execution time in seconds.
    pareto_front: Optional[List[List[float]]] = None  # Pareto front for multi-objective optimization.
    pareto_solutions: Optional[List[Dict[str, Any]]] = None  # Pareto solutions for multi-objective optimization.


def train_sel(
    data: Optional[TrainSelData] = None,
    candidates: Optional[List[List[int]]] = None,
    setsizes: Optional[List[int]] = None,
    ntotal: Optional[int] = None,
    settypes: Optional[List[str]] = None,
    stat: Optional[Callable] = None,
    n_stat: int = 1,
    target: Optional[List[int]] = None,
    control: Optional[ControlParams] = None,
    init_sol: Optional[Dict[str, Any]] = None,
    packages: List[str] = [],
    n_jobs: int = 1,
    verbose: bool = True,
    solution_diversity: Optional[bool] = None
) -> TrainSelResult:
    """
    Optimize the selection of training populations.
    
    Parameters
    ----------
    data : Dict[str, Any], optional
        Data structure for TrainSel optimization.
    candidates : List[List[int]], optional
        List of candidate index sets.
    setsizes : List[int], optional
        List of set sizes to select.
    ntotal : int, optional
        Total number of elements to select.
    settypes : List[str], optional
        List of set types. Options include "UOS", "OS", "UOMS", "OMS", "BOOL", "DBL".
    stat : Callable, optional
        Fitness function. If None, the CDMEAN criterion is used.
    n_stat : int, optional
        Number of objectives for multi-objective optimization.
    target : List[int], optional
        List of target indices.
    control : Dict[str, Any], optional
        Control object for the TrainSel function.
    init_sol : Dict[str, Any], optional
        Initial solution.
    packages : List[str], optional
        List of packages to import in parallel.
    n_jobs : int, optional
        Number of jobs for parallelization.
    verbose : bool, optional
        Whether to display progress messages.
    solution_diversity : bool, optional
        Whether to enforce uniqueness of solutions on the Pareto front. When True,
        duplicate solutions (same selected indices) are eliminated, keeping only 
        the one with the best overall fitness, ensuring more diverse solutions.
        If not provided, uses the value from control or defaults to True.
        
    Returns
    -------
    TrainSelResult
        Results of the optimization.
    """
    if verbose:
        print("Starting TrainSelPy optimization")
    
    start_time = time.time()
    
    # Set defaults
    if ntotal is None:
        ntotal = 0
    
    if target is None:
        target = []
    
    # Use CDMEAN if no alternative fitness function is provided.
    if stat is None:
        stat = lambda sol, d=data: cdmean_opt(sol, d)
    
    if stat is None and data is None:
        raise ValueError("No data provided for CDMEAN optimization")
    
    # Set control parameters if not provided.
    if control is None:
        control = set_control_default()
    
    # Override solution_diversity if explicitly provided
    if solution_diversity is not None:
        control["solution_diversity"] = solution_diversity
    
    # Extract parameters from control.
    nislands = control.get("nislands", 1)
    parallelizable = control.get("parallelizable", False)
    n_cores = control.get("mc_cores", 1)
    solution_diversity_param = control.get("solution_diversity", True)
    
    # Validate candidates and setsizes.
    if candidates is None or setsizes is None:
        raise ValueError("Candidates and setsizes must be provided")
    
    if len(candidates) != len(setsizes):
        raise ValueError("Candidates and setsizes must have the same length")
    
    # Validate settypes.
    if settypes is None:
        settypes = ["UOS"] * len(candidates)
    
    if len(settypes) != len(candidates):
        raise ValueError("Settypes and candidates must have the same length")
    
    # Count the number of double variables.
    n_dbl = sum(1 for st in settypes if st == "DBL")
    
    # Parallel processing setup.
    if parallelizable and n_stat > 1 and nislands == 1:
        raise ValueError("Parallelization for multi-objective optimization is only supported when nislands > 1")
    
    if ntotal is not None and ntotal > 0 and parallelizable:
        raise ValueError("ntotal is not supported when working in parallel")
    
    # Run the optimization.
    if nislands == 1:
        # Single island optimization.
        if parallelizable:
            if n_stat > 1:
                raise ValueError("Parallelization for multi-objective optimization is only supported when nislands > 1")
            
            # Define parallel fitness evaluation function.
            if n_dbl == 0 or n_dbl == len(settypes):
                def parallel_stat(solutions):
                    return Parallel(n_jobs=n_jobs)(
                        delayed(stat)(solution, data) for solution in solutions
                    )
            else:
                def parallel_stat(int_solutions, dbl_solutions):
                    return Parallel(n_jobs=n_jobs)(
                        delayed(stat)(int_sol, dbl_sol, data) 
                        for int_sol, dbl_sol in zip(int_solutions, dbl_solutions)
                    )
            
            result = genetic_algorithm(
                data=data,
                candidates=candidates,
                setsizes=setsizes,
                settypes=settypes,
                stat_func=parallel_stat,
                target=target,
                control=control,
                init_sol=init_sol,
                n_stat=n_stat,
                is_parallel=True,
                solution_diversity=solution_diversity_param
            )
        else:
            result = genetic_algorithm(
                data=data,
                candidates=candidates,
                setsizes=setsizes,
                settypes=settypes,
                stat_func=stat,
                target=target,
                control=control,
                init_sol=init_sol,
                n_stat=n_stat,
                is_parallel=False,
                solution_diversity=solution_diversity_param
            )
    else:
        # Island model optimization.
        inner_control = control.copy()
        inner_control["niterations"] = control.get("niterIslands", 200)
        inner_control["minitbefstop"] = control.get("minitbefstopIslands", 20)
        inner_control["niterSANN"] = control.get("niterSANNislands", 30)
        inner_control["nEliteSaved"] = control.get("nEliteSavedIslands", 3)
        inner_control["nelite"] = control.get("neliteIslands", 50)
        inner_control["npop"] = control.get("npopIslands", 200)
        
        if parallelizable:
            result = island_model_ga(
                data=data,
                candidates=candidates,
                setsizes=setsizes,
                settypes=settypes,
                stat_func=stat,
                n_stat=n_stat,
                target=target,
                control=inner_control,
                init_sol=init_sol,
                n_islands=nislands,
                n_jobs=n_jobs,
                solution_diversity=solution_diversity_param
            )
        else:
            result = island_model_ga(
                data=data,
                candidates=candidates,
                setsizes=setsizes,
                settypes=settypes,
                stat_func=stat,
                n_stat=n_stat,
                target=target,
                control=inner_control,
                init_sol=init_sol,
                n_islands=nislands,
                n_jobs=1,
                solution_diversity=solution_diversity_param
            )
    
    execution_time = time.time() - start_time
    
    sel_result = TrainSelResult(
        selected_indices=result["selected_indices"],
        selected_values=result["selected_values"],
        fitness=result["fitness"],
        fitness_history=result["fitness_history"],
        execution_time=execution_time,
        pareto_front=result.get("pareto_front", None),
        pareto_solutions=result.get("pareto_solutions", None)
    )
    
    if verbose:
        print(f"Optimization completed in {execution_time:.2f} seconds")
        print(f"Final fitness: {sel_result.fitness}")
    
    return sel_result


def time_estimation(
    nind: int, 
    nsel: int, 
    niter: int = 100, 
    control: Optional[ControlParams] = None
) -> float:
    """
    Estimate the time required for optimization.
    
    Parameters
    ----------
    nind : int
        Number of individuals.
    nsel : int
        Number of individuals to select.
    niter : int, optional
        Number of iterations.
    control : Dict[str, Any], optional
        Control object for the TrainSel function.
        
    Returns
    -------
    float
        Estimated time in seconds.
    """
    if control is None:
        control = set_control_default()
    
    npop = control.get("npop", 1000)
    nislands = control.get("nislands", 1)
    
    base_time = 0.001  # Base time per evaluation.
    complexity_factor = 1 + (nsel / nind) * 10  # Complexity increases with the selection ratio.
    
    estimated_time = base_time * npop * niter * complexity_factor
    
    if nislands > 1:
        estimated_time = estimated_time * 0.8 * (1 + 0.1 * nislands)
    
    return estimated_time


def make_positive_definite(matrix, epsilon=1e-6):
    """
    Make a matrix positive definite by adding a small value to the diagonal.
    
    Parameters
    ----------
    matrix : ndarray
        Input matrix.
    epsilon : float, optional
        Small value to add to the diagonal.
        
    Returns
    -------
    ndarray
        Positive definite matrix.
    """
    try:
        np.linalg.cholesky(matrix)
        return matrix
    except np.linalg.LinAlgError:
        n = matrix.shape[0]
        return matrix + np.eye(n) * epsilon
