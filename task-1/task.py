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

    
def our_kmeans(A, K, max_iters=3):
    """Simplified k-means used by our_ann"""
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


def our_ann(N, D, A, X, K, k1=10, k2=120, kmeans_iters=3):
    A = cp.asarray(A, dtype=cp.float32)
    X = cp.asarray(X, dtype=cp.float32)
    
    if X.ndim == 1:
        X = X.reshape(1, -1)

    clusters, centroids = our_kmeans(A, k1, kmeans_iters)

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
    N, D, A, X, K = testdata_knn("")
    bad_recall_count = 0
    num_trials = 10
    
    knn_times = []
    ann_times = []
    recall_rates = []

    for _ in range(num_trials):
        # KNN (ground truth)
        start = time.time()
        knn_result = our_knn(N, D, A, X, K)
        cp.cuda.Device().synchronize()
        knn_time = time.time() - start
        knn_times.append(knn_time)
        print(f'KNN result: {knn_result}')
        print(f'Time elapsed for KNN: {knn_time:.4f}s')

        start = time.time()
        ann_result = our_ann(N, D, A, X, K)
        cp.cuda.Device().synchronize()
        ann_time = time.time() - start
        ann_times.append(ann_time)
        print(f'ANN result: {ann_result}')
        print(f'Time elapsed for ANN: {ann_time:.4f}s')
        

        current_recall = recall_rate(knn_result.tolist(), ann_result.tolist())
        recall_rates.append(current_recall)
        
        if current_recall < 0.7:
            bad_recall_count += 1

        print(f"Recall rate (original): {current_recall:.4f}")

    # Calculate statistics
    bad_recall_rate = bad_recall_count / num_trials
    
    print("\n=== Final Results ===")
    print(f"Bad recall rate: {bad_recall_rate:.4f} ({bad_recall_count}/{num_trials} trials)")
    print(f"\nAverage recall rate (original): {np.mean(recall_rates):.4f} ± {np.std(recall_rates):.4f}")
    print(f"\nAverage processing time:")
    print(f"- KNN: {np.mean(knn_times):.4f}s ± {np.std(knn_times):.4f}")
    print(f"- ANN: {np.mean(ann_times):.4f}s ± {np.std(ann_times):.4f}")
    print(f"- Difference in KNN and ANN: {np.mean(knn_times) - np.mean(ann_times):.4f}s")