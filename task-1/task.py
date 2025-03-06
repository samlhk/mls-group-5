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
    max_i = 10
    A = cp.asarray(A)
    # intialise centroids
    centroids = A[cp.random.choice(N, K, replace=False)]
    for _ in range(max_i):
        # assign
        distances = cp.linalg.norm(A[:, None, :] - centroids[None, :, :], axis=2)  # (N, K)
        labels = cp.argmin(distances, axis=1)  # Shape (N,)

        # update
        new_centroids = cp.array([A[labels == k].mean(axis=0) for k in range(K)])

        # check if centroids are no longer moving significantly?
        # something something
        
        centroids = new_centroids

    return labels

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
    kmeans_result = our_kmeans(N, D, A, K)
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