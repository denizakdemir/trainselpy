"""
Example demonstrating how to use TrainSelPy for clustering problems.

This example shows how to:
1. Generate synthetic data with cluster structure
2. Use TrainSelPy to select representative points from each cluster
3. Visualize the results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
import sys
import os

# Add the parent directory to sys.path to ensure we find our local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import TrainSelPy functions
from trainselpy import (
    make_data,
    train_sel,
    set_control_default
)


def cluster_representatives(solution, data):
    """
    Custom fitness function for selecting cluster representatives.
    
    This function aims to maximize the minimum distance between selected points
    while ensuring coverage of all clusters.
    
    Parameters
    ----------
    solution : List[int]
        Indices of the selected samples
    data : Dict[str, Any]
        Data structure containing distance matrix and cluster labels
        
    Returns
    -------
    float
        Fitness value (higher is better)
    """
    # Get distance matrix and cluster labels
    dist_mat = data["DistMat"]
    cluster_labels = data["ClusterLabels"]
    
    # Extract distances between selected points
    selected_dist = dist_mat[np.ix_(solution, solution)]
    
    # Get the minimum distance between any pair of selected points
    # (excluding self-distances which are 0)
    min_dist = np.inf
    n_selected = len(solution)
    for i in range(n_selected):
        for j in range(i+1, n_selected):
            if selected_dist[i, j] < min_dist:
                min_dist = selected_dist[i, j]
    
    # If all points are from the same cluster, penalize
    selected_clusters = cluster_labels[solution]
    unique_clusters = np.unique(selected_clusters)
    cluster_coverage = len(unique_clusters) / len(np.unique(cluster_labels))
    
    # Combine minimum distance and cluster coverage
    # We want to maximize both
    fitness = min_dist * cluster_coverage
    
    return fitness


def main():
    """Run a clustering example using TrainSelPy."""
    print("TrainSelPy Clustering Example")
    print("----------------------------")
    
    # Generate synthetic data with clusters
    print("\nGenerating synthetic clustered data...")
    n_samples = 500
    n_features = 2
    n_clusters = 5
    
    # Create clustered data
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=42
    )
    
    print(f"Generated dataset with {n_samples} samples, {n_features} features, and {n_clusters} clusters")
    
    # Compute distance matrix
    print("\nComputing distance matrix...")
    dist_mat = euclidean_distances(X)
    
    # Create the TrainSel data object
    print("\nCreating TrainSel data object...")
    ts_data = make_data(M=X)
    
    # Add distance matrix and cluster labels to the data object
    ts_data["DistMat"] = dist_mat
    ts_data["ClusterLabels"] = y
    
    # Set control parameters
    print("\nSetting control parameters...")
    control = set_control_default()
    control["niterations"] = 100  # Reduced for a faster example
    control["npop"] = 100
    
    # Run the selection algorithm to find cluster representatives
    print("\nRunning TrainSel to find cluster representatives...")
    
    # Select one representative per cluster
    result = train_sel(
        data=ts_data,
        candidates=[list(range(n_samples))],  # Select from all samples
        setsizes=[n_clusters],  # Select as many points as clusters
        settypes=["UOS"],  # Unordered set
        stat=cluster_representatives,  # Use our custom fitness function
        control=control,
        verbose=True
    )
    
    # Extract the selected indices
    selected_indices = result.selected_indices[0]
    selected_clusters = y[selected_indices]
    
    print(f"\nSelected {len(selected_indices)} representatives:")
    print(f"Selected indices: {selected_indices}")
    print(f"Clusters of selected points: {selected_clusters}")
    print(f"Final fitness: {result.fitness:.6f}")
    
    # Visualize the results
    print("\nVisualizing results...")
    plt.figure(figsize=(10, 8))
    
    # Plot all points colored by cluster
    for cluster_id in range(n_clusters):
        cluster_points = X[y == cluster_id]
        plt.scatter(
            cluster_points[:, 0], 
            cluster_points[:, 1],
            alpha=0.5,
            label=f"Cluster {cluster_id}"
        )
    
    # Highlight the selected representatives
    plt.scatter(
        X[selected_indices, 0],
        X[selected_indices, 1],
        s=100,
        c='red',
        marker='*',
        label='Representatives'
    )
    
    plt.title('Cluster Representatives Selected by TrainSelPy')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig('cluster_representatives.png')
    print("Visualization saved as 'cluster_representatives.png'")
    
    # Advanced: Try selecting different numbers of representatives
    print("\n\nAdvanced example: Selecting different numbers of representatives...")
    
    # Try different numbers of representatives
    n_reps_list = [n_clusters, n_clusters*2, n_clusters*3]
    results = []
    
    for n_reps in n_reps_list:
        print(f"\nSelecting {n_reps} representatives...")
        result = train_sel(
            data=ts_data,
            candidates=[list(range(n_samples))],
            setsizes=[n_reps],
            settypes=["UOS"],
            stat=cluster_representatives,
            control=control,
            verbose=True
        )
        results.append(result)
        print(f"Final fitness: {result.fitness:.6f}")
    
    # Visualize the different selections
    plt.figure(figsize=(15, 5))
    
    for i, (n_reps, result) in enumerate(zip(n_reps_list, results)):
        plt.subplot(1, 3, i+1)
        
        # Plot all points colored by cluster
        for cluster_id in range(n_clusters):
            cluster_points = X[y == cluster_id]
            plt.scatter(
                cluster_points[:, 0], 
                cluster_points[:, 1],
                alpha=0.3,
                label=f"Cluster {cluster_id}" if i == 0 else ""
            )
        
        # Highlight the selected representatives
        selected = result.selected_indices[0]
        plt.scatter(
            X[selected, 0],
            X[selected, 1],
            s=100,
            c='red',
            marker='*',
            label='Representatives' if i == 0 else ""
        )
        
        plt.title(f'{n_reps} Representatives')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        if i == 0:
            plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('multiple_representatives.png')
    print("Comparison visualization saved as 'multiple_representatives.png'")
    
    # Example with high-dimensional data
    print("\n\nExample with high-dimensional data...")
    
    # Generate high-dimensional data
    n_features_high = 10
    X_high, y_high = make_blobs(
        n_samples=n_samples,
        n_features=n_features_high,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=42
    )
    
    print(f"Generated high-dimensional dataset with {n_samples} samples and {n_features_high} features")
    
    # Compute distance matrix
    dist_mat_high = euclidean_distances(X_high)
    
    # Create the TrainSel data object
    ts_data_high = make_data(M=X_high)
    ts_data_high["DistMat"] = dist_mat_high
    ts_data_high["ClusterLabels"] = y_high
    
    # Run the selection algorithm
    print("\nRunning TrainSel on high-dimensional data...")
    result_high = train_sel(
        data=ts_data_high,
        candidates=[list(range(n_samples))],
        setsizes=[n_clusters],
        settypes=["UOS"],
        stat=cluster_representatives,
        control=control,
        verbose=True
    )
    
    selected_high = result_high.selected_indices[0]
    print(f"\nSelected {len(selected_high)} representatives from high-dimensional data:")
    print(f"Selected indices: {selected_high}")
    print(f"Final fitness: {result_high.fitness:.6f}")
    
    # Visualize using PCA
    print("\nVisualizing high-dimensional data using PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_high)
    
    plt.figure(figsize=(10, 8))
    
    # Plot all points colored by cluster
    for cluster_id in range(n_clusters):
        cluster_points = X_pca[y_high == cluster_id]
        plt.scatter(
            cluster_points[:, 0], 
            cluster_points[:, 1],
            alpha=0.5,
            label=f"Cluster {cluster_id}"
        )
    
    # Highlight the selected representatives
    plt.scatter(
        X_pca[selected_high, 0],
        X_pca[selected_high, 1],
        s=100,
        c='red',
        marker='*',
        label='Representatives'
    )
    
    plt.title('Cluster Representatives in High-Dimensional Data (PCA Projection)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.savefig('high_dim_representatives.png')
    print("High-dimensional visualization saved as 'high_dim_representatives.png'")
    
    print("\nClustering example completed.")


if __name__ == "__main__":
    main()
