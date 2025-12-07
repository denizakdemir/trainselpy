
import numpy as np
import matplotlib.pyplot as plt
from trainselpy.algorithms import genetic_algorithm
import time
import os

# --- Problem Definitions ---

def trap_function_5(bits):
    """
    Deceptive Trap Function of order 5.
    Unitation u = number of 1s.
    If u = 5 -> fitness = 5 (Global Peak)
    If u < 5 -> fitness = 4 - u (Local Peak at u=0 with fitness 4)
    """
    u = sum(bits)
    if u == 5:
        return 5.0
    else:
        return 4.0 - u

def evaluate_mixed_fitness(int_vals, dbl_vals, data):
    """
    Mixed Problem:
    1. Binary: 6 blocks of 5-bit Deceptive Traps (30 bits total).
    2. Permutation: 10 elements, target is sorted [0,1,..,9].
    3. Continuous: 10D Rastrigin.
    
    Overall Goal: MAXIMIZE Fitness.
    """
    
    # --- 1. Binary (Trap) ---
    # int_vals[0] contains the binary vector (list of 0/1)
    binary_vec = int_vals[0]
    trap_score = 0
    block_size = 5
    for i in range(0, len(binary_vec), block_size):
        chunk = binary_vec[i:i+block_size]
        if len(chunk) == block_size:
            trap_score += trap_function_5(chunk)
            
    # Max trap score = 6 blocks * 5 = 30.
    
    # --- 2. Permutation ---
    # int_vals[1] contains the permutation indices for set 2 (UOS)
    # Permutation of size 10 from candidates [0..9]
    perm_vec = int_vals[1]
    # Fitness: Count correct absolute positions? 
    # Or negative distance from sorted?
    # Let's maximize: Max Score = 0 (perfect match).
    # Penalty: sum(|index(val) - val|) ? 
    # If perfect: val 0 at index 0, val 1 at index 1...
    # Cost = Sum(|v - i|) for i, v in enumerate(perm_vec)
    perm_cost = sum(abs(v - i) for i, v in enumerate(perm_vec))
    # Normalize or scale? Max dist for 10 elements is high.
    # We want to MAXIMIZE, so let's subtract cost.
    
    # --- 3. Continuous (Rastrigin) ---
    # dbl_vals[0] contains 10D vector
    # Minimize Rastrigin -> Maximize Negative Rastrigin
    # dbl_vals is a list. dbl_vals[0] should be the vector.
    vals = dbl_vals[0]
    if hasattr(vals, '__len__'):
        x = np.array(vals)
    else:
        # Unexpected scalar?
        x = np.array([vals])
        
    n = len(x)
    rast_val = 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    
    # Combined Fitness
    fitness = 10.0 * trap_score - 1.0 * perm_cost - 1.0 * rast_val
    
    # Debug print for final solution? 
    # Only if passed a flag in data?
    if data.get("debug_print", False):
         print(f"Fit: {fitness:.2f} | Trap: {trap_score} | Perm: {perm_cost} | Rast: {rast_val:.2f}")

    return fitness

def run_benchmark():
    print("--- Running Difficult Mixed-Type Benchmark ---")
    
    cand_bin = list(range(30))
    cand_perm = list(range(10))
    cand_dbl = [] 
    
    candidates = [cand_bin, cand_perm, cand_dbl]
    settypes = ["BOOL", "UOS", "DBL"]
    setsizes = [30, 10, 10]
    
    data = {"debug_print": False}
    
    # Common Control
    # Reduced population but more generations to allow learning
    npop = 5000
    ngen = 100
    base_control = {
        "npop": npop,
        "niterations": ngen,
        "progress": True,
        "sigma": 0.5,
        "mutation_rate": 0.1,
        "verbose": False
    }
    
    # --- Run 1: Baseline (No NN) ---
    print("\n[Run 1] Baseline Genetic Algorithm (NSGA-II/standard)...")
    start_time = time.time()
    control_base = base_control.copy()
    control_base["use_vae"] = False
    control_base["use_gan"] = False
    
    # Pass control as kwarg explicitly!
    res_base = genetic_algorithm(
        data, candidates, setsizes, settypes, evaluate_mixed_fitness, control=control_base
    )
    time_base = time.time() - start_time
    print(f"Baseline Final Fitness: {res_base['fitness']:.4f}, Time: {time_base:.2f}s")
    
    # Print breakdown for Baseline
    data["debug_print"] = True
    print("Baseline Breakdown:")
    evaluate_mixed_fitness(res_base['selected_indices'], res_base['selected_values'], data)
    data["debug_print"] = False
    
    # --- Run 2: With VAE + GAN ---
    print("\n[Run 2] Neural-Enhanced Genetic Algorithm (VAE+GAN)...")
    start_time = time.time()
    control_nn = base_control.copy()
    control_nn["use_vae"] = True
    control_nn["use_gan"] = True
    control_nn["vae_latent_dim"] = 8
    control_nn["gan_noise_dim"] = 8
    control_nn["nn_update_freq"] = 2
    control_nn["nn_start_gen"] = 5
    control_nn["nn_offspring_ratio"] = 0.3
    
    # Also set nn_epochs to demonstrate control
    control_nn["nn_epochs"] = 10
    
    res_nn = genetic_algorithm(
        data, candidates, setsizes, settypes, evaluate_mixed_fitness, control=control_nn
    )
    time_nn = time.time() - start_time
    print(f"NN-Enhanced Final Fitness: {res_nn['fitness']:.4f}, Time: {time_nn:.2f}s")
    
    # Print breakdown for NN
    data["debug_print"] = True
    print("NN-Enhanced Breakdown:")
    evaluate_mixed_fitness(res_nn['selected_indices'], res_nn['selected_values'], data)
    data["debug_print"] = False
    
    # --- Comparison ---

    print("\n--- Summary ---")
    print(f"Baseline: {res_base['fitness']:.4f}")
    print(f"NN-Enh  : {res_nn['fitness']:.4f}")
    improvement = res_nn['fitness'] - res_base['fitness']
    print(f"Improvement: {improvement:.4f}")

    # Plotting if history available? (Genetic algorithm returns final solution/result dict)
    # If we want history, we'd need to modify GA to return it or capture logs.
    # The current GA prints progress but doesn't return history list in simple mode?
    # Actually `trainselpy` usually returns dict.
    # Let's check `result` keys.
    # It seems it returns 'solution', 'fitness', etc.
    
    # Just printing final results is enough for the benchmark script.

if __name__ == "__main__":
    run_benchmark()
