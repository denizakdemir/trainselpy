"""Quick test to verify improvements work correctly."""
import numpy as np
from trainselpy.solution import Solution

# Test 1: Solution copy with numpy arrays
print("Test 1: Solution.copy() with numpy arrays...")
sol = Solution()
sol.int_values = [[1, 2, 3]]
sol.dbl_values = [np.array([0.5, 0.6, 0.7])]  # numpy array!
sol.fitness = 10.0
sol.multi_fitness = [5.0, 5.0]

# Create a copy
sol_copy = sol.copy()

# Modify the original
sol.int_values[0][0] = 999
sol.dbl_values[0][0] = 999.0
sol.fitness = 999.0

# Check that copy is independent
assert sol_copy.int_values[0][0] == 1, "Int values not properly copied!"
assert sol_copy.dbl_values[0][0] == 0.5, "Dbl values not properly copied!"
assert sol_copy.fitness == 10.0, "Fitness not properly copied!"
print("✓ Solution.copy() works correctly with numpy arrays")

# Test 2: Solution initialization with mixed types
print("\nTest 2: Solution initialization with mixed types...")
sol2 = Solution(
    int_values=[[1, 2, 3]],
    dbl_values=[np.array([0.1, 0.2])],  # numpy array
    fitness=5.0
)
assert isinstance(sol2.dbl_values[0], list), "Dbl values not converted to list!"
assert sol2.dbl_values[0] == [0.1, 0.2], "Values not correct!"
print("✓ Solution initialization normalizes numpy arrays to lists")

# Test 3: Test evaluate_fitness helper functions
print("\nTest 3: evaluate_fitness helper functions...")
from trainselpy.algorithms import _prepare_function_args, _prepare_batch_args

# Test single int
sol3 = Solution(int_values=[[1, 2, 3]], dbl_values=[])
args = _prepare_function_args(sol3, has_int=True, has_dbl=False)
assert args == ([1, 2, 3],), f"Expected ([1, 2, 3],), got {args}"

# Test int + dbl
sol4 = Solution(int_values=[[1, 2]], dbl_values=[[0.5, 0.6]])
args = _prepare_function_args(sol4, has_int=True, has_dbl=True)
assert args == ([1, 2], [0.5, 0.6]), f"Expected ([1, 2], [0.5, 0.6]), got {args}"

# Test batch args
pop = [sol3, sol3]
batch_args = _prepare_batch_args(pop, has_int=True, has_dbl=False)
assert batch_args == ([[1, 2, 3], [1, 2, 3]],), f"Unexpected batch args: {batch_args}"

print("✓ evaluate_fitness helper functions work correctly")

print("\n" + "="*60)
print("All quick tests passed! ✓")
print("="*60)
