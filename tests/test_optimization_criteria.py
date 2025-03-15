"""
Tests for the optimization criteria of TrainSelPy, with fixes for CDMean.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add the examples directory to sys.path
test_dir = os.path.dirname(os.path.abspath(__file__))
examples_dir = os.path.join(os.path.dirname(test_dir), 'examples')
sys.path.insert(0, examples_dir)

# Import the fixed CDMean implementation
from custom_cdmean import custom_cdmean

from trainselpy import (
    make_data,
    dopt,
    maximin_opt,
    pev_opt
)

class TestOptimizationCriteriaFixed(unittest.TestCase):
    """
    Test cases for the optimization criteria of TrainSelPy with fixed CDMean.
    """
    
    def setUp(self):
        """Set up test data."""
        # Create a small test dataset
        n_samples = 20
        n_features = 10
        
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Create a marker matrix
        self.M = np.random.choice([-1, 0, 1], size=(n_samples, n_features), p=[0.25, 0.5, 0.25])
        
        # Create a relationship matrix
        self.K = np.dot(self.M, self.M.T) / n_features
        
        # Add a small value to the diagonal to ensure positive definiteness
        self.K += np.eye(n_samples) * 1e-6
        
        # Create names for the samples
        self.names = [f"Sample_{i+1}" for i in range(n_samples)]
        
        # Create dataframes
        self.M_df = pd.DataFrame(self.M, index=self.names)
        self.K_df = pd.DataFrame(self.K, index=self.names, columns=self.names)
        
        # Create a TrainSel data object
        self.data = make_data(K=self.K)
        self.data_df = make_data(K=self.K_df)
        
        # Set up data for other criteria
        self.data["FeatureMat"] = self.M
        self.data_df["FeatureMat"] = self.M_df
        
        # Set up data for PEV
        self.data["Target"] = [0, 1, 2, 3, 4]
        self.data_df["Target"] = [0, 1, 2, 3, 4]
        
        # Set up data for distance matrix
        self.dist_matrix = 1 - np.abs(self.K)
        self.dist_df = pd.DataFrame(self.dist_matrix, index=self.names, columns=self.names)
        self.data["DistMat"] = self.dist_matrix
        self.data_df["DistMat"] = self.dist_df
        
        # Sample solution for testing
        self.soln = [5, 6, 7, 8, 9]
    
    def test_dopt(self):
        """Test D-optimality criterion."""
        # Calculate D-optimality with numpy array
        dopt_val = dopt(self.soln, self.data)
        
        # Calculate D-optimality with dataframe
        dopt_val_df = dopt(self.soln, self.data_df)
        
        # Check that the values are close
        self.assertAlmostEqual(dopt_val, dopt_val_df, places=5)
        
        # Check that the value is a scalar
        self.assertTrue(np.isscalar(dopt_val))
        
        # Check that the value is finite
        self.assertTrue(np.isfinite(dopt_val))
    
    def test_maximin_opt(self):
        """Test maximin criterion."""
        # Calculate maximin with numpy array
        maximin_val = maximin_opt(self.soln, self.data)
        
        # Calculate maximin with dataframe
        maximin_val_df = maximin_opt(self.soln, self.data_df)
        
        # Check that the values are close
        self.assertAlmostEqual(maximin_val, maximin_val_df, places=5)
        
        # Check that the value is a scalar
        self.assertTrue(np.isscalar(maximin_val))
        
        # Check that the value is finite
        self.assertTrue(np.isfinite(maximin_val))
    
    def test_pev_opt(self):
        """Test PEV criterion."""
        # Calculate PEV with numpy array
        pev_val = pev_opt(self.soln, self.data)
        
        # Calculate PEV with dataframe
        pev_val_df = pev_opt(self.soln, self.data_df)
        
        # Check that the values are close
        self.assertAlmostEqual(pev_val, pev_val_df, places=5)
        
        # Check that the value is a scalar
        self.assertTrue(np.isscalar(pev_val))
        
        # Check that the value is finite
        self.assertTrue(np.isfinite(pev_val))
    
    def test_cdmean_fixed(self):
        """Test fixed CDMean criterion."""
        # Calculate CDMean with numpy array
        cdmean_val = custom_cdmean(self.soln, self.data)
        
        # Check that the value is a scalar
        self.assertTrue(np.isscalar(cdmean_val))
        
        # Check that the value is finite
        self.assertTrue(np.isfinite(cdmean_val))

if __name__ == '__main__':
    unittest.main()