import torch
import cupy as cp
import triton
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from test import testdata_kmeans, testdata_knn, testdata_ann
from concurrent.futures import ThreadPoolExecutor

# np.random.seed(47)
# cp.random.seed(47)

# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def distance_cosine_cpu(X, Y):
    return 1 - np.dot(X, Y) / np.sqrt(np.sum(np.square(X))) / np.sqrt(np.sum(np.square(Y)))

def distance_cosine(X, Y):
    l2norm_kernel = cp.ReductionKernel(
        'T x',  # input params
        'T y',  # output params
        'x * x',  # map
        'a + b',  # reduce
        'y = sqrt(a)',  # post-reduction map
        '0',  # identity value
        'l2norm'  # kernel name
    )
    distance = 1 - cp.dot(X, Y) / l2norm_kernel(X) / l2norm_kernel(Y)
    cp.cuda.Stream.null.synchronize()

    return distance

def distance_l2_cpu(X, Y):
    return np.sqrt(np.sum((X - Y) ** 2))    

def distance_l2(X, Y):
    if X.ndim > 1 and Y.ndim == 1:
        # Batched calculation: distances between each row in X and vector Y
        diff = X - Y 
        return cp.sqrt(cp.sum(diff * diff, axis=1))
    else:
        # Original single vector calculation
        l2norm_kernel = cp.ReductionKernel(
            'T x',  # input params
            'T y',  # output params
            'x * x',  # map
            'a + b',  # reduce
            'y = sqrt(a)',  # post-reduction map
            '0',  # identity value
            'l2norm'  # kernel name
        )
        distance = l2norm_kernel(X - Y)
        cp.cuda.Stream.null.synchronize()
        return distance

def distance_dot_cpu(X, Y):
    return np.dot(X, Y)

def distance_dot(X, Y):
    return cp.dot(X, Y)


def distance_manhattan_cpu(X, Y):
    return np.sum(np.abs(X - Y))

def distance_manhattan(X, Y):
    manhattan_kernel = cp.ReductionKernel(
        'T x',  # input params
        'T y',  # output params
        'abs(x)',  # map
        'a + b',  # reduce
        'y = a',  # post-reduction map
        '0',  # identity value
        'l2norm'  # kernel name
    )
    distance = manhattan_kernel(X - Y)
    cp.cuda.Stream.null.synchronize()
    return distance

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_knn_cpu(N, D, A, X, K):
    return np.argsort(distance_cosine_cpu(A, X))[:K]

def our_knn(N, D, A, X, K):
    A = cp.asarray(A)
    X = cp.asarray(X)
    distances = distance_l2(A, X)
    return cp.argsort(distances)[:K]

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def our_kmeans(N, D, A, K):
    '''
    1. Initialize:
        - Randomly select K points from dataset as initial centroids

    2. REPEAT:
        a. Assignment step:
            - For each data point:
                - Calculate distance to each centroid
                - Assign point to closest centroid's cluster
    
        b. Update step:
            - For each cluster:
                - Calculate mean of all points in cluster
                - Set new centroid position to cluster mean

    3. UNTIL:
        - Centroids no longer move significantly OR
        - Maximum iterations reached

    Input:
        N: Number of vectors
        D: Dimension of vectors
        A[N, D]: A collection of vectors
        K: number of clusters

    Output:
        Result[N]: cluster ID for each vector
    '''
    max_iterations = 10
    A = cp.asarray(A)
    result = cp.zeros(N, dtype=int)
    # new_result = cp.zeros(N, dtype=int)
    # intialise centroids
    centroids = A[cp.random.choice(N, K, replace=False)]
    for _ in range(max_iterations):
        # assign
        distances = cp.zeros((N, K))
        for k in range(K):
            # Vectorized distance calculation between all points and current centroid
            distances[:, k] = distance_l2(A, centroids[k])
        result = cp.argmin(distances, axis=1)

        # convergence?
        # if cp.all(new_result == result):
        #     break
        # result = new_result.copy()

        # update
        for idx, _ in enumerate(centroids):
            centroids[idx] = cp.mean(A[result == idx], axis=0)

    return result, centroids

'''def kmeans(N, D, A, K, max_iters=100, tolerance=1e-4):
    # Initialize
    A = cp.asarray(A)
    centroids = A[cp.random.choice(N, K, replace=False)]  # Randomly initialize centroids
    prev_centroids = cp.zeros_like(centroids)
    labels = cp.zeros(N, dtype=cp.int32)

    for _ in range(max_iters):
        # Assignment step: Assign each point to the nearest centroid
        for i in range(N):
            distances = cp.array([distance_cosine(A[i], centroids[k]) for k in range(K)])
            labels[i] = cp.argmin(distances)  # Assign to closest centroid
        
        # Update step: Recompute centroids as the mean of the points in each cluster
        for k in range(K):
            cluster_points = A[labels == k]
            if cluster_points.shape[0] > 0:  # Avoid division by zero
                new_centroid = cp.mean(cluster_points, axis=0)
                centroids[k] = new_centroid

        # Check for convergence (if centroids have moved less than the tolerance)
        centroid_shift = cp.linalg.norm(centroids - prev_centroids)
        if centroid_shift < tolerance:
            break
        
        # Save the current centroids for the next iteration
        prev_centroids = centroids.copy()

    return labels'''
    
def simple_kmeans(A, K, max_iters=3):
    """Simplified k-means used by our_ann_optimised"""
    centroids = A[cp.random.choice(len(A), K, replace=False)]
    
    for _ in range(max_iters):
        distances = ((A[:, None, :] - centroids[None, :, :])**2).sum(axis=2)
        clusters = cp.argmin(distances, axis=1)
        
        for k in range(K):
            mask = clusters == k
            if cp.any(mask):
                centroids[k] = A[mask].mean(axis=0)
    
    return clusters, centroids

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here


def our_ann_optimized(N, D, A, X, K, k1=5, k2=80, kmeans_iters=3):
    """Optimized ANN with better recall"""
    A = cp.asarray(A, dtype=cp.float32)
    X = cp.asarray(X, dtype=cp.float32)
    
    if X.ndim == 1:
        X = X.reshape(1, -1)

    clusters, centroids = simple_kmeans(A, k1, kmeans_iters)

    query_cluster_dists = ((X[:, None, :] - centroids[None, :, :])**2).sum(axis=2)
    nearest_clusters = cp.argsort(query_cluster_dists, axis=1)[:, :k1]
    
    candidates = []
    for i in range(X.shape[0]):
        for c in nearest_clusters[i]:
            cluster_points = cp.where(clusters == c)[0]
            if len(cluster_points) > 0:
                dists = ((A[cluster_points] - X[i])**2).sum(axis=1)
                top_k = min(k2, len(cluster_points))
                candidates.append(cluster_points[cp.argpartition(dists, top_k-1)[:top_k]])
    
    if not candidates:
        return cp.array([], dtype=cp.int32)
    
    candidates = cp.unique(cp.concatenate(candidates))
    
    dists = ((A[candidates] - X)**2).sum(axis=1)
    top_k = min(K, len(candidates))
    return candidates[cp.argpartition(dists, top_k-1)[:top_k]]

def our_ann(N, D, A, X, K):
    # Ensure all inputs are CuPy arrays
    A = cp.asarray(A)
    X = cp.asarray(X)
    k1 = 5
    k2 = 100
    clusters, centroids = our_kmeans(N, D, A, K)
    k1_cluster_centers = our_knn(K, D, centroids, X, k1)

    candidates = cp.array([], dtype=cp.int32)
    for cluster_id in k1_cluster_centers:
        cluster_vectors_ids = cp.where(clusters == cluster_id)[0]  # Use CuPy for indexing
        closest_vectors_ids = our_knn(len(cluster_vectors_ids), D, A[cluster_vectors_ids], centroids[cluster_id], k2)
        candidates = cp.append(candidates, cluster_vectors_ids[closest_vectors_ids])

    final_indices = our_knn(len(candidates), D, A[candidates], X, K)
    return candidates[final_indices]

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

# Example
def test_kmeans():
    N, D, A, K = testdata_kmeans("test.json")
    kmeans_result = our_kmeans(N, D, A, K)
    print(kmeans_result)

def test_knn_cpu():
    N, D, A, X, K = testdata_knn("test.json")
    knn_result = our_knn_cpu(N, D, A, X, K)
    print(knn_result)

def test_knn():
    N, D, A, X, K = testdata_knn("test.json")
    knn_result = our_knn(N, D, A, X, K)
    print(knn_result)
    return knn_result
    
def test_ann():
    N, D, A, X, K = testdata_ann("test.json")
    ann_result = our_ann(N, D, A, X, K)
    print(ann_result)
    return ann_result
    
def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)

if __name__ == "__main__":
    # Set file path to empty string to generate fresh data
    N, D, A, X, K = testdata_knn("")
    bad_recall_count = 0
    bad_recall_optimized_count = 0
    num_trials = 10  # More descriptive than 'k'
    
    # Arrays to store metrics
    knn_times = []
    ann_times = []
    ann_optimized_times = []
    recall_rates = []
    recall_rates_optimized = []

    for _ in range(num_trials):
        # KNN (ground truth)
        start = time.time()
        knn_result = our_knn(N, D, A, X, K)
        cp.cuda.Device().synchronize()
        knn_time = time.time() - start
        knn_times.append(knn_time)
        print(f'KNN result: {knn_result}')
        print(f'Time elapsed for KNN: {knn_time:.4f}s')

        # Original ANN
        start = time.time()
        ann_result = our_ann(N, D, A, X, K)
        cp.cuda.Device().synchronize()
        ann_time = time.time() - start
        ann_times.append(ann_time)
        print(f'ANN result: {ann_result}')
        print(f'Time elapsed for ANN: {ann_time:.4f}s')
        
        # Optimized ANN
        start = time.time()
        ann_optimized_result = our_ann_optimized(N, D, A, X, K)
        cp.cuda.Device().synchronize()
        ann_optimized_time = time.time() - start
        ann_optimized_times.append(ann_optimized_time)
        print(f'Optimized ANN result: {ann_optimized_result}')
        print(f'Time elapsed for Optimized ANN: {ann_optimized_time:.4f}s')

        # Calculate recall rates
        current_recall = recall_rate(knn_result.tolist(), ann_result.tolist())
        current_recall_optimized = recall_rate(knn_result.tolist(), ann_optimized_result.tolist())
        recall_rates.append(current_recall)
        recall_rates_optimized.append(current_recall_optimized)
        
        # Count bad recalls (below 0.7 threshold)
        if current_recall < 0.7:
            bad_recall_count += 1
        if current_recall_optimized < 0.7:
            bad_recall_optimized_count += 1

        print(f"Recall rate (original): {current_recall:.4f}")
        print(f"Recall rate (optimized): {current_recall_optimized:.4f}\n")

    # Calculate statistics
    bad_recall_rate = bad_recall_count / num_trials
    bad_recall_rate_optimized = bad_recall_optimized_count / num_trials
    
    print("\n=== Final Results ===")
    print(f"Bad recall rate (original): {bad_recall_rate:.4f} ({bad_recall_count}/{num_trials} trials)")
    print(f"Bad recall rate (optimized): {bad_recall_rate_optimized:.4f} ({bad_recall_optimized_count}/{num_trials} trials)")
    print(f"\nAverage recall rate (original): {np.mean(recall_rates):.4f} ± {np.std(recall_rates):.4f}")
    print(f"Average recall rate (optimized): {np.mean(recall_rates_optimized):.4f} ± {np.std(recall_rates_optimized):.4f}")
    print(f"\nAverage processing time:")
    print(f"- KNN: {np.mean(knn_times):.4f}s ± {np.std(knn_times):.4f}")
    print(f"- Original ANN: {np.mean(ann_times):.4f}s ± {np.std(ann_times):.4f}")
    print(f"- Optimized ANN: {np.mean(ann_optimized_times):.4f}s ± {np.std(ann_optimized_times):.4f}")
    # warm up
    '''for _ in range(10):
        test_knn()

    times = []
    for i in range(50):
        start = time.time()
        test_knn()
        times.append(time.time() - start)
        if i % 10 == 0:
            print(time.time() - start)
    # plt.plot([i for i in range(50)], times, label='gpu')
    print(f"cp avg: {sum(times) / len(times)}")
    
    times = []
    for i in range(50):
        start = time.time()
        test_knn_cpu()
        times.append(time.time() - start)
        if i % 10 == 0:
            print(time.time() - start)
    print(f"np avg: {sum(times) / len(times)}")'''

    #plt.legend()
    #plt.show()
    

    # times = np.array([])
    # for i in range(10):
    #     a = np.random.randn(32768)
    #     b = np.random.randn(32768)
    #     start = time.time()
    #     distance_cosine(a, b)
    #     times = np.append(times, time.time() - start)
    
    # print(f'average time: {times.mean()}')