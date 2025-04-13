# MLS Group 5 code and materials
 This repository is a clone of the MLS instructions repo with our implementation for the tasks. Below describe the relevant files we implemented and how to run them. These instructions assume you are running the code in the teaching cluster and have set up the conda environment based on the task specific README.mds (these are not modified by us).

 ## Task 1

 ### task.py

 Implementation of distance functions, our knn, kmeans and ann algorithms. Running this file gives you a comparison between our knn and ann algorithms.

 ```
 python task.py
 ```

 ## Task 2

 ### download.py

 Helper script to download models for the RAG pipeline. **Run this before `serving_rag.py`.**

 ```
 python download.py
 ```

 ### serving_rag.py

 Starts RAG API server. Before running, modify `embed_model_path` and `chat_model_path` with the correct hash values. You can find these by checking the path after running `download.py`. Note that the `/rag` endpoint implements the request queue and batching while the `/rag_basic` endpoint is the original implementation.

 ```
 python serving_rag.py
 ```

 ### test_batch_latency.py

 Test script that sends x concurrent requests to the API. Before running, change the IP address in `url` to match that of the machine you are running the server in (check with `hostname -i`).

 ```
 python test_batch_latency.py
 ```


 ### test_performance.py

 Test script that sends requests to the API simulating in varying request rates and concurrency levels. Before running, change the IP address in `base_url` to match that of the machine you are running the server in (check with `hostname -i`). It stores its results in `data.npy` and produces a graph at `latency.pdf`.

 ```
 python test_performance.py
 ```