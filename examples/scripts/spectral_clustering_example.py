"""
Example demonstrating how to use TrainSelPy for spectral clustering.

Spectral clustering is a technique that uses the eigenvalues of a similarity matrix
to reduce dimensionality before clustering in fewer dimensions. It's particularly
effective for identifying non-convex clusters where traditional methods like k-means fail.

This example shows how to:
1. Implement spectral clustering using TrainSelPy
2. Handle non-convex clusters (like concentric circles)
3. Compare with traditional clustering methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import sys
import os
import time

# Add the parent directory to sys.path to ensure we find our local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import TrainSelPy functions
from trainselpy import (
    make_data,
    train_sel,
    set_control_default
)


def spectral_embedding(X, n_components=2, gamma=1.0):
    """
    Compute the spectral embedding for the data.
    
    Parameters
    ----------
    X : ndarray
        Data points
    n_components : int
        Number of components to keep
    gamma : float
        Parameter for RBF kernel
        
    Returns
    -------
    ndarray
        Embedded data points
    """
    # Compute similarity matrix using RBF kernel
    similarity = rbf_kernel(X, gamma=gamma)
    
    # Compute the normalized graph Laplacian
    D = np.diag(np.sum(similarity, axis=1))
    L = D - similarity
    D_sqrt_inv = np.diag(1.0 / np.sqrt(np.sum(similarity, axis=1)))
    L_norm = D_sqrt_inv @ L @ D_sqrt_inv
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
    
    # Sort by eigenvalues and select the smallest n_components (excluding the first)
    indices = np.argsort(eigenvalues)[1:n_components+1]
    
    # Return the selected eigenvectors
    return eigenvectors[:, indices]


def spectral_clustering_fitness(solution, data):
    """
    Fitness function for spectral clustering.
    
    This function aims to maximize the separation between clusters in the spectral embedding.
    
    Parameters
    ----------
    solution : List[int]
        Indices of the selected cluster centers
    data : Dict[str, Any]
        Data structure containing embedded data
        
    Returns
    -------
    float
        Fitness value (higher is better)
    """
    # Get embedded data and distance matrix
    embedded_data = data["EmbeddedData"]
    dist_mat = data["DistMat"]
    
    # For each point, find the distance to the closest center in the embedded space
    min_distances = np.min(dist_mat[:, solution], axis=1)
    
    # Return negative sum of distances (since TrainSelPy maximizes fitness)
    return -np.sum(min_distances)


def assign_clusters(X, centers):
    """
    Assign each point to the nearest center.
    
    Parameters
    ----------
    X : ndarray
        Data points
    centers : ndarray
        Center points indices
        
    Returns
    -------
    ndarray
        Cluster assignments
    """
    # Compute distances from each point to each center
    distances = np.zeros((X.shape[0], len(centers)))
    for i, center_idx in enumerate(centers):
        distances[:, i] = np.sqrt(np.sum((X - X[center_idx])**2, axis=1))
    
    # Assign each point to the nearest center
    return np.argmin(distances, axis=1)


def main():
    """Run a spectral clustering example using TrainSelPy."""
    print("TrainSelPy Spectral Clustering Example")
    print("------------------------------------")
    
    # Generate synthetic data with non-convex clusters
    print("\nGenerating synthetic data with non-convex clusters...")
    
    # Create concentric circles
    n_samples = 500
    X_circles, y_circles = make_circles(
        n_samples=n_samples,
        factor=0.5,
        noise=0.05,
        random_state=42
    )
    
    # Create moons
    X_moons, y_moons = make_moons(
        n_samples=n_samples,
        noise=0.05,
        random_state=42
    )
    
    print(f"Generated two datasets with {n_samples} samples each")
    
    # Process both datasets
    datasets = [
        ("Concentric Circles", X_circles, y_circles),
        ("Two Moons", X_moons, y_moons)
    ]
    
    for dataset_name, X, true_labels in datasets:
        print(f"\n\nProcessing {dataset_name} dataset...")
        n_clusters = len(np.unique(true_labels))
        
        # Compute spectral embedding
        print("\nComputing spectral embedding...")
        gamma = 10.0 if dataset_name == "Concentric Circles" else 5.0
        embedded_data = spectral_embedding(X, n_components=2, gamma=gamma)
        
        # Compute distance matrix in the embedded space
        print("Computing distance matrix in embedded space...")
        dist_mat = np.zeros((embedded_data.shape[0], embedded_data.shape[0]))
        for i in range(embedded_data.shape[0]):
            for j in range(embedded_data.shape[0]):
                dist_mat[i, j] = np.sqrt(np.sum((embedded_data[i] - embedded_data[j])**2))
        
        # Create the TrainSel data object
        print("\nCreating TrainSel data object...")
        ts_data = make_data(M=X)
        ts_data["EmbeddedData"] = embedded_data
        ts_data["DistMat"] = dist_mat
        
        # Set control parameters
        print("\nSetting control parameters...")
        control = set_control_default()
        control["niterations"] = 100  # Reduced for a faster example
        control["npop"] = 100
        
        # Run spectral clustering using TrainSelPy
        print("\nRunning spectral clustering using TrainSelPy...")
        
        start_time = time.time()
        result = train_sel(
            data=ts_data,
            candidates=[list(range(n_samples))],
            setsizes=[n_clusters],
            settypes=["UOS"],
            stat=spectral_clustering_fitness,
            control=control,
            verbose=True
        )
        trainsel_time = time.time() - start_time
        
        # Extract the selected centers
        centers = result.selected_indices[0]
        
        # Assign clusters based on centers in the embedded space
        trainsel_labels = assign_clusters(embedded_data, centers)
        
        print(f"\nTrainSelPy spectral clustering completed in {trainsel_time:.2f} seconds")
        print(f"Selected centers: {centers}")
        
        # Run standard k-means for comparison
        print("\nRunning standard k-means for comparison...")
        start_time = time.time()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X)
        kmeans_time = time.time() - start_time
        
        print(f"K-means clustering completed in {kmeans_time:.2f} seconds")
        
        # Run scikit-learn's spectral clustering for comparison
        print("\nRunning scikit-learn's spectral clustering for comparison...")
        start_time = time.time()
        sklearn_spectral = SpectralClustering(
            n_clusters=n_clusters,
            gamma=gamma,
            random_state=42,
            assign_labels='kmeans'
        )
        sklearn_labels = sklearn_spectral.fit_predict(X)
        sklearn_time = time.time() - start_time
        
        print(f"Scikit-learn spectral clustering completed in {sklearn_time:.2f} seconds")
        
        # Calculate silhouette scores (in original space)
        trainsel_silhouette = silhouette_score(X, trainsel_labels)
        kmeans_silhouette = silhouette_score(X, kmeans_labels)
        sklearn_silhouette = silhouette_score(X, sklearn_labels)
        
        print(f"\nSilhouette score for TrainSelPy spectral: {trainsel_silhouette:.4f}")
        print(f"Silhouette score for k-means: {kmeans_silhouette:.4f}")
        print(f"Silhouette score for scikit-learn spectral: {sklearn_silhouette:.4f}")
        
        # Visualize the results
        print("\nVisualizing clustering results...")
        plt.figure(figsize=(15, 5))
        
        # Original data with true labels
        plt.subplot(1, 4, 1)
        for cluster_id in range(n_clusters):
            cluster_points = X[true_labels == cluster_id]
            plt.scatter(
                cluster_points[:, 0], 
                cluster_points[:, 1],
                alpha=0.7,
                label=f"Cluster {cluster_id}"
            )
        
        plt.title(f'{dataset_name}\nTrue Labels')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        
        # TrainSelPy spectral clustering
        plt.subplot(1, 4, 2)
        for cluster_id in range(n_clusters):
            cluster_points = X[trainsel_labels == cluster_id]
            plt.scatter(
                cluster_points[:, 0], 
                cluster_points[:, 1],
                alpha=0.7
            )
        
        # Highlight the centers
        plt.scatter(
            X[centers, 0],
            X[centers, 1],
            s=200,
            c='red',
            marker='*',
            edgecolors='black',
            label='Centers'
        )
        
        plt.title(f'TrainSelPy Spectral\nSilhouette: {trainsel_silhouette:.4f}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        
        # K-means
        plt.subplot(1, 4, 3)
        for cluster_id in range(n_clusters):
            cluster_points = X[kmeans_labels == cluster_id]
            plt.scatter(
                cluster_points[:, 0], 
                cluster_points[:, 1],
                alpha=0.7
            )
        
        # Highlight the centroids
        plt.scatter(
            kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200,
            c='blue',
            marker='X',
            edgecolors='black',
            label='Centroids'
        )
        
        plt.title(f'K-Means\nSilhouette: {kmeans_silhouette:.4f}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        
        # Scikit-learn spectral clustering
        plt.subplot(1, 4, 4)
        for cluster_id in range(n_clusters):
            cluster_points = X[sklearn_labels == cluster_id]
            plt.scatter(
                cluster_points[:, 0], 
                cluster_points[:, 1],
                alpha=0.7
            )
        
        plt.title(f'Scikit-learn Spectral\nSilhouette: {sklearn_silhouette:.4f}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        plt.tight_layout()
        plt.savefig(f'spectral_clustering_{dataset_name.lower().replace(" ", "_")}.png')
        print(f"Comparison visualization saved as 'spectral_clustering_{dataset_name.lower().replace(' ', '_')}.png'")
        
        # Visualize the embedded space
        plt.figure(figsize=(15, 5))
        
        # Original data
        plt.subplot(1, 3, 1)
        for cluster_id in range(n_clusters):
            cluster_points = X[true_labels == cluster_id]
            plt.scatter(
                cluster_points[:, 0], 
                cluster_points[:, 1],
                alpha=0.7,
                label=f"Cluster {cluster_id}"
            )
        
        plt.title(f'{dataset_name}\nOriginal Space')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        
        # Embedded space
        plt.subplot(1, 3, 2)
        for cluster_id in range(n_clusters):
            cluster_points = embedded_data[true_labels == cluster_id]
            plt.scatter(
                cluster_points[:, 0], 
                cluster_points[:, 1],
                alpha=0.7,
                label=f"Cluster {cluster_id}"
            )
        
        plt.title('Spectral Embedding\nTrue Labels')
        plt.xlabel('Embedding Dimension 1')
        plt.ylabel('Embedding Dimension 2')
        
        # Embedded space with TrainSelPy clusters
        plt.subplot(1, 3, 3)
        for cluster_id in range(n_clusters):
            cluster_points = embedded_data[trainsel_labels == cluster_id]
            plt.scatter(
                cluster_points[:, 0], 
                cluster_points[:, 1],
                alpha=0.7
            )
        
        # Highlight the centers in embedded space
        plt.scatter(
            embedded_data[centers, 0],
            embedded_data[centers, 1],
            s=200,
            c='red',
            marker='*',
            edgecolors='black',
            label='Centers'
        )
        
        plt.title('Spectral Embedding\nTrainSelPy Clusters')
        plt.xlabel('Embedding Dimension 1')
        plt.ylabel('Embedding Dimension 2')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'spectral_embedding_{dataset_name.lower().replace(" ", "_")}.png')
        print(f"Embedding visualization saved as 'spectral_embedding_{dataset_name.lower().replace(' ', '_')}.png'")
    
    print("\nSpectral clustering example completed.")


if __name__ == "__main__":
    main()
