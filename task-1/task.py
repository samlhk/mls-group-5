import torch
import cupy as cp
import triton
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def distance_cosine(X, Y):

    return 1 - np.dot(X, Y) / np.sqrt(sum(pow(element, 2) for element in X)) / np.sqrt(sum(pow(element, 2) for element in Y))

    # X = cp.asarray(X)
    # Y = cp.asarray(Y)
    # l2norm_kernel = cp.ReductionKernel(
    #     'T x',  # input params
    #     'T y',  # output params
    #     'x * x',  # map
    #     'a + b',  # reduce
    #     'y = sqrt(a)',  # post-reduction map
    #     '0',  # identity value
    #     'l2norm'  # kernel name
    # )
    # distance = 1 - cp.dot(X, Y) / l2norm_kernel(X) / l2norm_kernel(Y)
    # cp.cuda.Stream.null.synchronize()

    return distance
    

def distance_l2(X, Y):

    # return np.sqrt(np.sum((X - Y) ** 2))

    X = cp.asarray(X)
    Y = cp.asarray(Y)
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

def distance_dot(X, Y):

    return np.dot(X, Y)

    X = cp.asarray(X)
    Y = cp.asarray(Y)
    return cp.dot(X, Y)

def distance_manhattan(X, Y):
    
    # return np.sum(np.abs(X - Y))

    X = cp.asarray(X)
    Y = cp.asarray(Y)
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

def our_knn(N, D, A, X, K):
    pass

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def our_kmeans(N, D, A, K):
    pass

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
    N, D, A, K = testdata_kmeans("test_file.json")
    kmeans_result = our_kmeans(N, D, A, K)
    print(kmeans_result)

def test_knn():
    N, D, A, X, K = testdata_knn("test_file.json")
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

    # warm up
    a = np.random.randn(10000000)
    b = np.random.randn(10000000)
    distance_manhattan(a, b)

    times = np.array([])
    for i in range(10):
        a = np.random.randn(32768)
        b = np.random.randn(32768)
        start = time.time()
        distance_manhattan(a, b)
        times = np.append(times, time.time() - start)
    
    print(f'average time: {times.mean()}')
