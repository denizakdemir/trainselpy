"""
Script to convert the WheatData from R format to Python format.

This script demonstrates how to convert R data files to Python format.
It requires rpy2 to be installed.
"""

import os
import sys
import pickle
from trainselpy.utils import r_data_to_python


def main():
    """
    Convert the WheatData from R format to Python format.
    """
    # Check if rpy2 is installed
    try:
        import rpy2
    except ImportError:
        print("Error: rpy2 is required to convert R data to Python format.")
        print("Install it with 'pip install rpy2'")
        sys.exit(1)
    
    # Get the path to the WheatData.rda file
    r_data_path = input("Enter the path to the WheatData.rda file: ")
    
    if not os.path.exists(r_data_path):
        print(f"Error: File {r_data_path} does not exist.")
        sys.exit(1)
    
    # Convert the data
    print(f"Converting {r_data_path} to Python format...")
    python_data = r_data_to_python(r_data_path)
    
    # Display the converted data
    print("\nConverted data:")
    for key, value in python_data.items():
        print(f"{key}: {type(value)}")
        
        # Display additional information based on data type
        if hasattr(value, 'shape'):
            print(f"  Shape: {value.shape}")
        elif hasattr(value, 'size'):
            print(f"  Size: {value.size}")
        
        # For DataFrames and Series
        if hasattr(value, 'index'):
            print(f"  Index: {len(value.index)} items")
        if hasattr(value, 'columns'):
            print(f"  Columns: {len(value.columns)} items")
    
    # Save the converted data
    output_path = input("Enter the path to save the converted data (default: wheat_data.pkl): ")
    if not output_path:
        output_path = "wheat_data.pkl"
    
    with open(output_path, 'wb') as f:
        pickle.dump(python_data, f)
    
    print(f"\nData saved to {output_path}")
    print("\nYou can load this data in your Python code with:")
    print(f"""
import pickle

with open('{output_path}', 'rb') as f:
    wheat_data = pickle.load(f)
    
# Access the data
M = wheat_data['Wheat.M']
K = wheat_data['Wheat.K']
Y = wheat_data['Wheat.Y']
""")


if __name__ == "__main__":
    main()