import time
import threading
import requests
import numpy as np

url = "http://192.168.16.21:8002/rag"
query = {"query": "Which animals can hover in the air?"}

concurrency_level = 32
rounds = 20

batch_latencies = np.array([])

def send_request():
    res = requests.post(url, json=query)
    assert res.status_code == 200

for round in range(rounds):
    print(f'Round {round + 1}: Sending {concurrency_level} concurrent requests')
    threads = []
    for _ in range(concurrency_level):
        thread = threading.Thread(target=send_request)
        thread.daemon = True
        threads.append(thread)
    start = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.time() - start
    print(f'Round {round + 1}: Received all requests in {elapsed} seconds')
    batch_latencies = np.append(batch_latencies, elapsed)

print(f'Average batch latency: {np.round(batch_latencies.mean() * 1000)} milliseconds')
print(f'Average request latency: {np.round(batch_latencies.mean() * 1000 / concurrency_level)} milliseconds')

