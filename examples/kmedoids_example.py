"""
Example demonstrating how to use TrainSelPy for k-medoids clustering.

K-medoids is a clustering algorithm that selects actual data points as cluster centers,
unlike k-means which uses the mean of points in a cluster. This makes k-medoids more
robust to outliers and applicable to datasets where means cannot be computed.

This example shows how to:
1. Implement k-medoids clustering using TrainSelPy
2. Compare it with traditional k-means clustering
3. Demonstrate its robustness to outliers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sys
import os
import time

# Add the parent directory to sys.path to ensure we find our local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import TrainSelPy functions
from trainselpy import (
    make_data,
    train_sel,
    set_control_default
)


def kmedoids_fitness(solution, data):
    """
    Fitness function for k-medoids clustering.
    
    This function aims to minimize the sum of distances from each point to its nearest medoid.
    
    Parameters
    ----------
    solution : List[int]
        Indices of the selected medoids
    data : Dict[str, Any]
        Data structure containing distance matrix
        
    Returns
    -------
    float
        Negative sum of distances (negative because TrainSelPy maximizes fitness)
    """
    # Get distance matrix
    dist_mat = data["DistMat"]
    
    # For each point, find the distance to the closest medoid
    min_distances = np.min(dist_mat[:, solution], axis=1)
    
    # Return negative sum of distances (since TrainSelPy maximizes fitness)
    return -np.sum(min_distances)


def assign_clusters(X, medoids):
    """
    Assign each point to the nearest medoid.
    
    Parameters
    ----------
    X : ndarray
        Data points
    medoids : ndarray
        Medoid points
        
    Returns
    -------
    ndarray
        Cluster assignments
    """
    # Compute distances from each point to each medoid
    distances = euclidean_distances(X, X[medoids])
    
    # Assign each point to the nearest medoid
    return np.argmin(distances, axis=1)


def main():
    """Run a k-medoids clustering example using TrainSelPy."""
    print("TrainSelPy K-Medoids Clustering Example")
    print("-------------------------------------")
    
    # Generate synthetic data with clusters
    print("\nGenerating synthetic clustered data...")
    n_samples = 300
    n_features = 2
    n_clusters = 4
    random_state = 42
    
    # Create clustered data
    X, true_labels = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=random_state
    )
    
    # Add some outliers to demonstrate robustness
    n_outliers = 10
    outliers = np.random.uniform(low=-15, high=15, size=(n_outliers, n_features))
    X_with_outliers = np.vstack([X, outliers])
    true_labels_with_outliers = np.append(true_labels, np.ones(n_outliers) * -1)  # -1 for outliers
    
    print(f"Generated dataset with {n_samples} samples, {n_outliers} outliers, and {n_clusters} clusters")
    
    # Compute distance matrix for both datasets
    print("\nComputing distance matrices...")
    dist_mat = euclidean_distances(X)
    dist_mat_with_outliers = euclidean_distances(X_with_outliers)
    
    # Create the TrainSel data objects
    print("\nCreating TrainSel data objects...")
    ts_data = make_data(M=X)
    ts_data["DistMat"] = dist_mat
    
    ts_data_with_outliers = make_data(M=X_with_outliers)
    ts_data_with_outliers["DistMat"] = dist_mat_with_outliers
    
    # Set control parameters
    print("\nSetting control parameters...")
    control = set_control_default()
    control["niterations"] = 100  # Reduced for a faster example
    control["npop"] = 100
    
    # Run k-medoids clustering using TrainSelPy
    print("\nRunning k-medoids clustering using TrainSelPy...")
    
    # Without outliers
    start_time = time.time()
    result = train_sel(
        data=ts_data,
        candidates=[list(range(n_samples))],
        setsizes=[n_clusters],
        settypes=["UOS"],
        stat=kmedoids_fitness,
        control=control,
        verbose=True
    )
    kmedoids_time = time.time() - start_time
    
    # Extract the selected medoids
    medoids = result.selected_indices[0]
    
    # Assign clusters based on medoids
    kmedoids_labels = assign_clusters(X, medoids)
    
    print(f"\nK-medoids clustering completed in {kmedoids_time:.2f} seconds")
    print(f"Selected medoids: {medoids}")
    
    # Run k-means for comparison
    print("\nRunning k-means clustering for comparison...")
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    kmeans_time = time.time() - start_time
    
    print(f"K-means clustering completed in {kmeans_time:.2f} seconds")
    
    # Calculate silhouette scores
    kmedoids_silhouette = silhouette_score(X, kmedoids_labels)
    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    
    print(f"\nSilhouette score for k-medoids: {kmedoids_silhouette:.4f}")
    print(f"Silhouette score for k-means: {kmeans_silhouette:.4f}")
    
    # Visualize the results
    print("\nVisualizing clustering results...")
    plt.figure(figsize=(12, 5))
    
    # K-medoids
    plt.subplot(1, 2, 1)
    for cluster_id in range(n_clusters):
        cluster_points = X[kmedoids_labels == cluster_id]
        plt.scatter(
            cluster_points[:, 0], 
            cluster_points[:, 1],
            alpha=0.5,
            label=f"Cluster {cluster_id}" if cluster_id == 0 else ""
        )
    
    # Highlight the medoids
    plt.scatter(
        X[medoids, 0],
        X[medoids, 1],
        s=200,
        c='red',
        marker='*',
        edgecolors='black',
        label='Medoids'
    )
    
    plt.title(f'K-Medoids Clustering\nSilhouette: {kmedoids_silhouette:.4f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # K-means
    plt.subplot(1, 2, 2)
    for cluster_id in range(n_clusters):
        cluster_points = X[kmeans_labels == cluster_id]
        plt.scatter(
            cluster_points[:, 0], 
            cluster_points[:, 1],
            alpha=0.5
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
    
    plt.title(f'K-Means Clustering\nSilhouette: {kmeans_silhouette:.4f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('kmedoids_vs_kmeans.png')
    print("Comparison visualization saved as 'kmedoids_vs_kmeans.png'")
    
    # Test with outliers
    print("\n\nTesting robustness to outliers...")
    
    # Run k-medoids with outliers
    print("\nRunning k-medoids clustering on data with outliers...")
    result_with_outliers = train_sel(
        data=ts_data_with_outliers,
        candidates=[list(range(len(X_with_outliers)))],
        setsizes=[n_clusters],
        settypes=["UOS"],
        stat=kmedoids_fitness,
        control=control,
        verbose=True
    )
    
    # Extract the selected medoids
    medoids_with_outliers = result_with_outliers.selected_indices[0]
    
    # Assign clusters based on medoids
    kmedoids_labels_with_outliers = assign_clusters(X_with_outliers, medoids_with_outliers)
    
    # Run k-means with outliers
    kmeans_with_outliers = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans_labels_with_outliers = kmeans_with_outliers.fit_predict(X_with_outliers)
    
    # Visualize the results with outliers
    plt.figure(figsize=(12, 5))
    
    # K-medoids with outliers
    plt.subplot(1, 2, 1)
    for cluster_id in range(n_clusters):
        cluster_points = X_with_outliers[kmedoids_labels_with_outliers == cluster_id]
        plt.scatter(
            cluster_points[:, 0], 
            cluster_points[:, 1],
            alpha=0.5
        )
    
    # Highlight the medoids
    plt.scatter(
        X_with_outliers[medoids_with_outliers, 0],
        X_with_outliers[medoids_with_outliers, 1],
        s=200,
        c='red',
        marker='*',
        edgecolors='black',
        label='Medoids'
    )
    
    # Highlight the outliers
    plt.scatter(
        outliers[:, 0],
        outliers[:, 1],
        s=100,
        c='black',
        marker='x',
        label='Outliers'
    )
    
    plt.title('K-Medoids Clustering with Outliers')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # K-means with outliers
    plt.subplot(1, 2, 2)
    for cluster_id in range(n_clusters):
        cluster_points = X_with_outliers[kmeans_labels_with_outliers == cluster_id]
        plt.scatter(
            cluster_points[:, 0], 
            cluster_points[:, 1],
            alpha=0.5
        )
    
    # Highlight the centroids
    plt.scatter(
        kmeans_with_outliers.cluster_centers_[:, 0],
        kmeans_with_outliers.cluster_centers_[:, 1],
        s=200,
        c='blue',
        marker='X',
        edgecolors='black',
        label='Centroids'
    )
    
    # Highlight the outliers
    plt.scatter(
        outliers[:, 0],
        outliers[:, 1],
        s=100,
        c='black',
        marker='x',
        label='Outliers'
    )
    
    plt.title('K-Means Clustering with Outliers')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('kmedoids_vs_kmeans_with_outliers.png')
    print("Outlier comparison visualization saved as 'kmedoids_vs_kmeans_with_outliers.png'")
    
    # Check if any medoids are outliers
    medoid_is_outlier = any(m >= n_samples for m in medoids_with_outliers)
    print(f"\nAre any medoids outliers? {'Yes' if medoid_is_outlier else 'No'}")
    
    # Advanced: Determine optimal number of clusters
    print("\n\nAdvanced example: Determining optimal number of clusters...")
    
    # Try different numbers of clusters
    k_range = range(2, 8)
    silhouette_scores = []
    
    for k in k_range:
        print(f"\nTesting with {k} clusters...")
        result_k = train_sel(
            data=ts_data,
            candidates=[list(range(n_samples))],
            setsizes=[k],
            settypes=["UOS"],
            stat=kmedoids_fitness,
            control=control,
            verbose=False
        )
        
        # Assign clusters
        medoids_k = result_k.selected_indices[0]
        labels_k = assign_clusters(X, medoids_k)
        
        # Calculate silhouette score
        silhouette_k = silhouette_score(X, labels_k)
        silhouette_scores.append(silhouette_k)
        
        print(f"Silhouette score with {k} clusters: {silhouette_k:.4f}")
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(list(k_range), silhouette_scores, 'o-', linewidth=2, markersize=8)
    plt.grid(True)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Optimal Number of Clusters by Silhouette Score')
    plt.xticks(list(k_range))
    plt.savefig('optimal_clusters.png')
    print("\nOptimal clusters visualization saved as 'optimal_clusters.png'")
    
    # Find optimal k
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of clusters based on silhouette score: {optimal_k}")
    
    print("\nK-medoids clustering example completed.")


if __name__ == "__main__":
    main()
