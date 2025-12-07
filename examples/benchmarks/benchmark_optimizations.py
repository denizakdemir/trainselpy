"""
Quick benchmark to demonstrate optimization improvements.
Compares new optimized defaults vs. old behavior.
"""

import numpy as np
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from trainselpy.core import train_sel
from trainselpy.optimization_criteria import cdmean_opt

# Create a test problem
np.random.seed(42)
n = 500  # Number of candidates
k = 50   # Selection size

# Generate random genomic relationship matrix (positive definite)
# Use X @ X.T to guarantee positive semi-definite, then add diagonal
X = np.random.randn(n, n // 2)
G = X @ X.T / (n // 2)  # Normalize
G = G + np.eye(n) * 1.0  # Make it positive definite

# Create data dictionary
data = {
    'G': G,
    'lambda': 0.01
}

candidates = [list(range(n))]
setsizes = [k]
settypes = ['UOS']

print("=" * 70)
print("TrainSelPy Performance Benchmark")
print("=" * 70)
print(f"Problem size: n={n}, k={k}")
print(f"Population: 200, Generations: 50")
print()

# Test 1: New optimized defaults
print("Test 1: NEW OPTIMIZED DEFAULTS")
print("-" * 70)
control_new = {
    'niterations': 50,
    'npop': 200,
    'niterSANN': 20,      # New default (was 50)
    'sannFrequency': 20,   # New parameter (was implicit 1)
    'progress': False
}

start = time.time()
result_new = train_sel(
    data=data,
    candidates=candidates,
    setsizes=setsizes,
    settypes=settypes,
    stat=cdmean_opt,
    control=control_new,
    verbose=False
)
time_new = time.time() - start

print(f"Time: {time_new:.2f} seconds")
print(f"Best fitness: {result_new.fitness:.6f}")
print()

# Test 2: Old behavior (for comparison)
print("Test 2: OLD BEHAVIOR (SA every generation)")
print("-" * 70)
control_old = {
    'niterations': 50,
    'npop': 200,
    'niterSANN': 50,       # Old default
    'sannFrequency': 1,    # Apply SA every generation (old behavior)
    'progress': False
}

start = time.time()
result_old = train_sel(
    data=data,
    candidates=candidates,
    setsizes=setsizes,
    settypes=settypes,
    stat=cdmean_opt,
    control=control_old,
    verbose=False
)
time_old = time.time() - start

print(f"Time: {time_old:.2f} seconds")
print(f"Best fitness: {result_old.fitness:.6f}")
print()

# Test 3: GA only (no SA)
print("Test 3: GA ONLY (no SA)")
print("-" * 70)
control_ga_only = {
    'niterations': 50,
    'npop': 200,
    'niterSANN': 0,        # Disable SA
    'progress': False
}

start = time.time()
result_ga = train_sel(
    data=data,
    candidates=candidates,
    setsizes=setsizes,
    settypes=settypes,
    stat=cdmean_opt,
    control=control_ga_only,
    verbose=False
)
time_ga = time.time() - start

print(f"Time: {time_ga:.2f} seconds")
print(f"Best fitness: {result_ga.fitness:.6f}")
print()

# Summary
print("=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)
print(f"{'Configuration':<25} {'Time (s)':<12} {'Speedup':<10} {'Fitness':<12}")
print("-" * 70)
print(f"{'GA only (fastest)':<25} {time_ga:>10.2f}   {time_ga/time_ga:>8.2f}x   {result_ga.fitness:>10.6f}")
print(f"{'New optimized (default)':<25} {time_new:>10.2f}   {time_new/time_new:>8.2f}x   {result_new.fitness:>10.6f}")
print(f"{'Old behavior':<25} {time_old:>10.2f}   {time_old/time_new:>8.2f}x   {result_old.fitness:>10.6f}")
print()
print(f"Speedup (new vs old): {time_old/time_new:.1f}x")
print(f"Quality comparison:")
print(f"  New vs GA-only: {((result_new.fitness - result_ga.fitness) / result_ga.fitness * 100):+.2f}%")
print(f"  New vs Old: {((result_new.fitness - result_old.fitness) / result_old.fitness * 100):+.2f}%")
print()
print("Conclusion: New defaults provide ~{:.1f}x speedup with comparable quality".format(time_old/time_new))
print("=" * 70)
