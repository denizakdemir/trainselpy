"""
Tests for the core functionality of TrainSelPy.
"""

import unittest
import numpy as np
import pandas as pd
from trainselpy import (
    make_data,
    train_sel,
    set_control_default,
    dopt
)


class TestCore(unittest.TestCase):
    """
    Test cases for the core functionality of TrainSelPy.
    """
    
    def setUp(self):
        """Set up test data."""
        # Create a small test dataset
        n_samples = 50
        n_features = 20
        
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
    
    def test_make_data(self):
        """Test the make_data function."""
        # Test with M only
        data = make_data(M=self.M)
        self.assertIn('G', data)
        self.assertIn('R', data)
        self.assertIn('lambda_val', data)
        self.assertEqual(data['lambda_val'], 1)
        
        # Test with K only
        data = make_data(K=self.K)
        self.assertIn('G', data)
        self.assertIn('R', data)
        self.assertIn('lambda_val', data)
        self.assertEqual(data['lambda_val'], 1)
        
        # Test with both M and K
        data = make_data(M=self.M, K=self.K)
        self.assertIn('G', data)
        self.assertIn('R', data)
        self.assertIn('lambda_val', data)
        self.assertEqual(data['lambda_val'], 1)
        
        # Test with lambda
        data = make_data(M=self.M, lambda_val=2)
        self.assertIn('G', data)
        self.assertIn('R', data)
        self.assertIn('lambda_val', data)
        self.assertEqual(data['lambda_val'], 2)
    
    def test_set_control_default(self):
        """Test the set_control_default function."""
        # Test demo version
        control = set_control_default(size="demo")
        self.assertEqual(control["size"], "demo")
        self.assertIn("niterations", control)
        self.assertIn("nelite", control)
        self.assertIn("npop", control)
        
        # Test full version
        control = set_control_default(size="full")
        self.assertEqual(control["size"], "full")
        self.assertIn("niterations", control)
        self.assertIn("nelite", control)
        self.assertIn("npop", control)
    
    def test_train_sel_simple(self):
        """Test the train_sel function with a simple example."""
        # Create data
        data = make_data(M=self.M)
        data["FeatureMat"] = self.M
        
        # Create control with minimal iterations
        control = set_control_default()
        control["niterations"] = 10
        control["npop"] = 50
        
        # Run a simple optimization
        result = train_sel(
            data=data,
            candidates=[list(range(10))],
            setsizes=[5],
            settypes=["UOS"],
            stat=dopt,
            control=control,
            verbose=False
        )
        
        # Check the result
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.selected_indices)
        self.assertEqual(len(result.selected_indices), 1)
        self.assertEqual(len(result.selected_indices[0]), 5)
        self.assertIsNotNone(result.fitness)
        self.assertIsNotNone(result.fitness_history)
        self.assertGreater(len(result.fitness_history), 0)
    
    def test_train_sel_bool(self):
        """Test the train_sel function with boolean selection."""
        # Create data
        data = make_data(M=self.M)
        data["FeatureMat"] = self.M
        
        # Create control with minimal iterations
        control = set_control_default()
        control["niterations"] = 10
        control["npop"] = 50
        
        # Run a boolean optimization
        result = train_sel(
            data=data,
            candidates=[list(range(20))],
            setsizes=[20],
            settypes=["BOOL"],
            stat=dopt,
            control=control,
            verbose=False
        )
        
        # Check the result
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.selected_indices)
        self.assertEqual(len(result.selected_indices), 1)
        self.assertEqual(len(result.selected_indices[0]), 20)
        for val in result.selected_indices[0]:
            self.assertIn(val, [0, 1])
    
    def test_train_sel_ordered(self):
        """Test the train_sel function with ordered selection."""
        # Create data
        data = make_data(M=self.M)
        data["FeatureMat"] = self.M
        
        # Create control with minimal iterations
        control = set_control_default()
        control["niterations"] = 10
        control["npop"] = 50
        
        # Run an ordered optimization
        result = train_sel(
            data=data,
            candidates=[list(range(20))],
            setsizes=[5],
            settypes=["OS"],
            stat=dopt,
            control=control,
            verbose=False
        )
        
        # Check the result
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.selected_indices)
        self.assertEqual(len(result.selected_indices), 1)
        self.assertEqual(len(result.selected_indices[0]), 5)
        
        # Check that the values are in the expected range
        for val in result.selected_indices[0]:
            self.assertGreaterEqual(val, 0)
            self.assertLess(val, 20)


if __name__ == '__main__':
    unittest.main()