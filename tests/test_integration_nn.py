import pytest
import numpy as np
from trainselpy.algorithms import genetic_algorithm
from trainselpy.utils import create_mixed_model_data

def simple_fitness(int_vals, dbl_vals, data):
    # Try to maximize sum of elements
    fit = 0
    if int_vals:
        # Check if list of lists or single list
        if isinstance(int_vals[0], list):
            for v in int_vals:
                if v: fit += sum(v)
        else:
            fit += sum(int_vals)
            
    if dbl_vals is not None and len(dbl_vals) > 0:
        if isinstance(dbl_vals[0], list) or (hasattr(dbl_vals[0], '__len__') and not isinstance(dbl_vals[0], (str, bytes))):
             # Handle list of lists or 2D array
            for v in dbl_vals:
                if hasattr(v, 'sum'): fit += v.sum()
                elif v: fit += sum(v)
        else:
            # 1D array or list
            if hasattr(dbl_vals, 'sum'): fit += dbl_vals.sum()
            else: fit += sum(dbl_vals)
    return fit

def test_ga_with_vae_gan():
    # Setup problem
    # 1. Binary: select 5 bits. candidates=[0,1], setsize=5.
    # 2. Permutation: select 3 from 5. UOS. candidates=[0..4], setsize=3.
    # 3. Continuous: 2 variables. setsize=2.
    
    candidates = [
        # Check initialize_population:
        # if type == "BOOL": np.random.choice([0, 1], size=len(cand))
        # Wait, if type is BOOL, `len(cand)` determines how many bits?
        # Yes. So cand should be a list of features to select from? 
        # Or just a list of length N?
        # Let's assume cand is [0, 1, 2, 3, 4] for 5 bits.
        list(range(5)),
        
        list(range(5)), # Perm set candidates
        
        [] # DBL (candidates ignored)
    ]
    
    setsizes = [
        5, # If BOOL, size=len(cand). Wait, initialize_population ignores `setsizes` for BOOL?
           # "if type_ == 'BOOL': sol.int_values.append(np.random.choice([0, 1], size=len(cand)).tolist())"
           # Yes, it uses len(cand). setsizes[i] is ignored for BOOL?
           # Actually logic: "for i, (cand, size, type_) in enumerate(...)"
           # So yes.
        3, # UOS size
        2  # DBL size
    ]
    
    settypes = ["BOOL", "UOS", "DBL"]
    
    control = {
        "npop": 20,
        "niterations": 5,
        "progress": False,
        "use_vae": True,
        "use_gan": True,
        "vae_latent_dim": 4,
        "gan_noise_dim": 4,
        "nn_update_freq": 1, # Train every generation
        "nn_start_gen": 1    # Start immediately
    }
    
    data = {}
    
    # Run GA
    result = genetic_algorithm(
        data=data,
        candidates=candidates,
        setsizes=setsizes,
        settypes=settypes,
        stat_func=simple_fitness,
        control=control
    )
    
    assert result is not None
    assert "fitness" in result
    print(f"Result fitness: {result['fitness']}")

if __name__ == "__main__":
    test_ga_with_vae_gan()
