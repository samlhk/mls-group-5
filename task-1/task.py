import torch
import cupy as cp
import triton
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from test import testdata_kmeans, testdata_knn, testdata_ann
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
    l2norm_kernel = cp.ReductionKernel(
        'T x',  # input params
        'T y',  # output params
        'abs(x)',  # map
        'a + b',  # reduce
        'y = a',  # post-reduction map
        '0',  # identity value
        'l2norm'  # kernel name
    )
    distance = l2norm_kernel(X - Y)
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
    distances = cp.array([distance_cosine(A[i], X) for i in range(A.shape[0])])
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
    max_iterations = 100
    A = cp.asarray(A)
    result = cp.zeros(N, dtype=int)
    new_result = cp.zeros(N, dtype=int)
    # intialise centroids
    centroids = A[cp.random.choice(N, K, replace=False)]
    for _ in range(max_iterations):
        # assign
        for idx, a in enumerate(A):
            distances = cp.array([distance_cosine(a, centroid) for centroid in centroids])
            new_result[idx] = cp.argmin(distances)

        # convergence?
        if cp.all(new_result == result):
            break
        result = new_result.copy()

        # update
        for idx, _ in enumerate(centroids):
            centroids[idx] = cp.mean(A[result == idx], axis=0)

    return result

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

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_ann(N, D, A, X, K):
    pass

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

# Example
def test_kmeans():
    N, D, A, K = testdata_kmeans("")
    kmeans_result = test_kmeans(N, D, A, K)
    print(kmeans_result)

def test_knn_cpu():
    N, D, A, X, K = testdata_knn("")
    knn_result = our_knn_cpu(N, D, A, X, K)
    # print(knn_result)

def test_knn():
    N, D, A, X, K = testdata_knn("")
    knn_result = our_knn(N, D, A, X, K)
    print(knn_result)
    
def test_ann():
    N, D, A, X, K = testdata_ann("test_file.json")
    ann_result = our_ann(N, D, A, X, K)
    print(ann_result)
    
def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)

if __name__ == "__main__":
    # test_kmeans()

    start = time.time()
    test_kmeans()
    print(time.time() - start)

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