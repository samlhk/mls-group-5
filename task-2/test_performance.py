import time
import threading
import requests
import numpy as np
import matplotlib.pyplot as plt

base_url = "http://192.168.47.132:8002/"
query = {"query": "Which animals can hover in the air?"}

endpoint = ""
rounds = 10
peak_rps = 64
peak_concurrency_level = 64

basic_latencies = []
batched_latencies = []

def send_request():
    global latencies
    start = time.time()
    res = requests.post(base_url + endpoint, json=query)
    elapsed = time.time() - start
    assert res.status_code == 200
    if endpoint == "rag_basic":
        basic_latencies.append(elapsed)
    else:
        batched_latencies.append(elapsed)

for case in ["rag_basic", "rag"]:
    endpoint = case
    throughputs = np.array([])
    print(f"Evaluating {'basic' if case == 'rag_basic' else 'batched'} pipeline")
    for round in range(rounds):
        print(f'Round {round + 1}: Testing performance')
        rps = 16
        concurrency_level = 16
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
        throughput = len(threads) / elapsed
        print(f'Round {round + 1}: Received all {len(threads)} requests in {elapsed} seconds -> throughput: {throughput} RPS')
        throughputs = np.append(throughputs, throughput)

    print(f'Average throughput ({endpoint}): {np.round(throughputs.mean(), 3)} RPS')


with open('data.npy', 'wb') as f:
    np.save(f, basic_latencies)
    np.save(f, batched_latencies)

# with open('data.npy', 'rb') as f:
#     basic_latencies = np.load(f)
#     batched_latencies = np.load(f)

plt.ecdf(basic_latencies, label='Original RAG Pipeline')
plt.ecdf(batched_latencies, label='Batched RAG Pipeline')

plt.legend()
plt.xlabel('Latency (s)')
plt.ylabel('Cumulative Probability')
plt.title('Empirical CDF of Response Latency')

plt.savefig('latency.pdf')

