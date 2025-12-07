
import time
import random
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from trainselpy.solution import Solution
from trainselpy.operators import mutation, crossover

def benchmark_mutation():
    print("Benchmarking Mutation...")
    # Setup
    n_candidates = 10000
    candidates = [list(range(n_candidates))]
    settypes = ["UOS"]
    
    # Create a population
    pop_size = 100
    set_size = 100
    population = []
    for _ in range(pop_size):
        sol = Solution()
        sol.int_values = [sorted(random.sample(candidates[0], set_size))]
        population.append(sol)
        
    # Benchmark
    start_time = time.time()
    # High intensity to force many mutations
    mutation(population, candidates, settypes, mutprob=1.0, mutintensity=0.1)
    end_time = time.time()
    
    print(f"Mutation Time (100 sols, 10k candidates, 100 set size): {end_time - start_time:.4f} seconds")

def benchmark_crossover():
    print("\nBenchmarking Crossover...")
    # Setup
    n_candidates = 10000
    candidates = [list(range(n_candidates))]
    settypes = ["UOS"]
    
    # Create parents
    pop_size = 100
    set_size = 100
    parents = []
    for _ in range(pop_size):
        sol = Solution()
        sol.int_values = [sorted(random.sample(candidates[0], set_size))]
        parents.append(sol)
        
    # Benchmark
    start_time = time.time()
    # High intensity
    offspring = crossover(parents, crossprob=1.0, crossintensity=0.5, settypes=settypes, candidates=candidates)
    end_time = time.time()
    
    print(f"Crossover Time (100 parents, 10k candidates, 100 set size): {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    benchmark_mutation()
    benchmark_crossover()
