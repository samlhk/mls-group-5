#!/bin/bash

URL="http://192.168.47.131:8001/rag"
DATA='{"query": "Which animals can hover in the air?"}'
HEADER="Content-Type: application/json"

# Number of concurrent requests to test
CONCURRENCY_LEVELS=(32)
ROUNDS=20  # How many batches to run at each concurrency level

for CONCURRENCY in "${CONCURRENCY_LEVELS[@]}"; do
  echo "Testing $CONCURRENCY concurrent requests, $ROUNDS rounds"
  TOTAL_TIME=0

  for ((round = 1; round <= ROUNDS; round++)); do
    echo "  Round $round"
    START_TIME=$(date +%s%N | cut -b1-13)

    for ((i = 1; i <= CONCURRENCY; i++)); do
      curl -s -o /dev/null -w "%{http_code} " -X POST "$URL" \
        -H "$HEADER" -d "$DATA" &
    done

    wait  # Wait for all concurrent requests to finish
    END_TIME=$(date +%s%N | cut -b1-13)
    ELAPSED=$((END_TIME - START_TIME))
    TOTAL_TIME=$((TOTAL_TIME+ELAPSED))
    echo "Finished $CONCURRENCY concurrent requests in $ELAPSED seconds"
    echo  # New line after response codes
  done

  AVERAGE=$((TOTAL_TIME/ROUNDS))
  echo "Average batch takes $AVERAGE seconds"

  echo "-----------------------------------------"
done
