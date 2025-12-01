
import time
import numpy as np
from trainselpy import make_data
from trainselpy.optimization_criteria import cdmean_opt as lib_cdmean
from custom_cdmean import custom_cdmean as slow_cdmean

def benchmark():
    # Setup data
    print("Setting up data...")
    # data = make_data(K=None) # Removed
    # Let's look at make_data implementation or just use random data.
    # make_data in trainselpy/utils.py or core.py?
    # It was imported from trainselpy in the example.
    
    # Let's just create synthetic data for benchmark
    N = 5000
    k = 100
    G = np.random.randn(N, N)
    G = G @ G.T # Make it symmetric positive definite
    # Normalize diagonal to 1 for realism
    d = np.diag(G)
    G = G / np.sqrt(np.outer(d, d))
    
    data = {
        "G": G,
        "lambda": 0.01
    }
    
    soln = list(range(k))
    
    print(f"Benchmarking with N={N}, k={k}")
    
    # Time slow_cdmean
    start = time.time()
    for _ in range(10):
        slow_cdmean(soln, data)
    end = time.time()
    print(f"Slow CDMean (10 runs): {end - start:.4f}s")
    
    # Time lib_cdmean
    start = time.time()
    for _ in range(10):
        lib_cdmean(soln, data)
    end = time.time()
    print(f"Lib CDMean (10 runs): {end - start:.4f}s")

if __name__ == "__main__":
    benchmark()
