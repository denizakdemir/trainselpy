
import time
import numpy as np
from trainselpy.optimization_criteria import dopt

def benchmark_dopt():
    N = 1000
    d = 1279
    k = 40
    
    print(f"Benchmarking D-opt with N={N}, d={d}, k={k}")
    
    # Create random feature matrix
    M = np.random.randn(N, d)
    
    data = {"FeatureMat": M}
    soln = list(range(k))
    
    # Time dopt
    start = time.time()
    for _ in range(5):
        dopt(soln, data)
    end = time.time()
    print(f"D-opt (5 runs): {end - start:.4f}s")
    
    # Test dual form manually
    print("Testing dual form speed...")
    start = time.time()
    for _ in range(5):
        X = M[soln, :]
        # XX' + eps I
        Gram = X @ X.T
        Gram.flat[::k+1] += 1e-10
        L = np.linalg.cholesky(Gram)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        # Correction term not included in timing, just the heavy lifting
    end = time.time()
    print(f"Dual form (5 runs): {end - start:.4f}s")

if __name__ == "__main__":
    benchmark_dopt()
