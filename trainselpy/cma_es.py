"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation.
"""

import numpy as np
from typing import List, Tuple, Optional

class CMAESOptimizer:
    """
    CMA-ES optimizer for continuous variables.
    
    References:
    Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial. arXiv:1604.00772.
    """
    def __init__(self, mean: np.ndarray, sigma: float, pop_size: int = None):
        """
        Initialize CMA-ES optimizer.
        
        Parameters
        ----------
        mean : np.ndarray
            Initial mean vector
        sigma : float
            Initial step size (standard deviation)
        pop_size : int, optional
            Population size (lambda). If None, calculated automatically.
        """
        self.N = len(mean)
        self.xmean = np.array(mean, dtype=float)
        self.sigma = sigma
        
        # Strategy parameter setting: Selection
        if pop_size is None:
            self.lam = 4 + int(3 * np.log(self.N))
        else:
            self.lam = pop_size
            
        self.mu = self.lam // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1 / np.sum(self.weights**2)
        
        # Strategy parameter setting: Adaptation
        self.cc = (4 + self.mueff / self.N) / (self.N + 4 + 2 * self.mueff / self.N)
        self.cs = (self.mueff + 2) / (self.N + self.mueff + 5)
        self.c1 = 2 / ((self.N + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.N + 2)**2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.N + 1)) - 1) + self.cs
        
        # Initialize dynamic (internal) strategy parameters and constants
        self.pc = np.zeros(self.N)
        self.ps = np.zeros(self.N)
        self.B = np.eye(self.N)
        self.D = np.ones(self.N)
        self.C = self.B @ np.diag(self.D**2) @ self.B.T
        self.invsqrtC = self.B @ np.diag(self.D**-1) @ self.B.T
        self.eigeneval = 0
        self.chiN = self.N**0.5 * (1 - 1 / (4 * self.N) + 1 / (21 * self.N**2))
        
        # Generation loop
        self.counteval = 0
        
    def ask(self) -> List[np.ndarray]:
        """
        Generate new candidate solutions.
        
        Returns
        -------
        List[np.ndarray]
            List of candidate vectors
        """
        candidates = []
        for _ in range(self.lam):
            z = np.random.randn(self.N)
            candidate = self.xmean + self.sigma * (self.B @ (self.D * z))
            candidates.append(candidate)
        return candidates
    
    def tell(self, candidates: List[np.ndarray], fitness_values: List[float]) -> None:
        """
        Update internal state based on fitness values.
        
        Parameters
        ----------
        candidates : List[np.ndarray]
            List of candidate vectors
        fitness_values : List[float]
            List of fitness values (higher is better for maximization)
        """
        # Sort by fitness (descending because we maximize)
        # Note: Standard CMA-ES minimizes, so we negate fitness or sort descending
        # Here we assume maximization, so we sort descending
        sorted_indices = np.argsort(fitness_values)[::-1]
        
        # Update mean
        xold = self.xmean.copy()
        self.xmean = np.zeros(self.N)
        for i in range(self.mu):
            self.xmean += self.weights[i] * candidates[sorted_indices[i]]
            
        # Update evolution paths
        y = self.xmean - xold
        z = self.invsqrtC @ y / self.sigma
        
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * z
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * (self.counteval / self.lam + 1))) / self.chiN < 1.4 + 2 / (self.N + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y / self.sigma
        
        # Adapt covariance matrix C
        artmp = (1 / self.sigma) * (np.array([candidates[i] for i in sorted_indices[:self.mu]]) - xold).T
        self.C = (1 - self.c1 - self.cmu) * self.C \
                 + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) \
                 + self.cmu * artmp @ np.diag(self.weights) @ artmp.T
                 
        # Adapt step size sigma
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        
        # Decomposition of C
        if self.counteval - self.eigeneval > self.lam / (self.c1 + self.cmu) / self.N / 10:
            self.eigeneval = self.counteval
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(self.D)
            self.invsqrtC = self.B @ np.diag(self.D**-1) @ self.B.T
            
        self.counteval += self.lam

