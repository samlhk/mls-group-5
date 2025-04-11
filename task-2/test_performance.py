import time
import threading
import requests
import numpy as np
import matplotlib.pyplot as plt

url = "http://192.168.16.21:8002/rag"
query = {"query": "Which animals can hover in the air?"}

rounds = 1
peak_rps = 16
peak_concurrency_level = 16

latencies = np.array([])

def send_request():
    global latencies
    start = time.time()
    res = requests.post(url, json=query)
    elapsed = time.time() - start
    assert res.status_code == 200
    latencies = np.append(latencies, elapsed)

for round in range(rounds):
    print(f'Round {round + 1}: Testing performance')
    rps = 1
    concurrency_level = 1
    start = time.time()
    threads = []
    while rps <= peak_rps and concurrency_level <= peak_concurrency_level:
        # constant requests
        for _ in range(rps):
            thread = threading.Thread(target=send_request)
            thread.daemon = True
            thread.start()
            threads = np.append(threads, thread)
            time.sleep(1 / rps)
        rps *= 2

        # request spikes
        for _ in range(concurrency_level):
            thread = threading.Thread(target=send_request)
            thread.daemon = True
            thread.start()
            threads = np.append(threads, thread)
        concurrency_level *= 2

    for thread in threads:
        thread.join()
    elapsed = time.time() - start
    print(f'Round {round + 1}: Received all {len(threads)} requests in {elapsed} seconds -> throughput: {len(threads)/elapsed} RPS')

plt.hist(latencies)
plt.savefig('latency_batch.pdf')

